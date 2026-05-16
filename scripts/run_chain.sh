#!/bin/bash
# Chain: cohort pull → fits (J=1, J=5, Xu) → LOCO-23 → knot J=4/J=6
#      → subgroup × 3 → estimand extraction → outputs → archive → shutdown.
# K = treatment-bin count (20); J = spline knots (1 scalar, 5 primary).
# Reference bin = 7 (cohort Q25, geometric center ≈ 9.3 J/min).
#
# Device: set DEVICE env var to "gpu" (default, V100 production) or "cpu" (local
# debug). CPU mode forces JAX_PLATFORMS=cpu and chain_method=sequential, and
# skips the auto-shutdown step at the end.
#
# Usage:
#   DEVICE=gpu  bash scripts/run_chain.sh    # V100 production
#   DEVICE=cpu  bash scripts/run_chain.sh    # local CPU smoke (slow, but runs)

# Fail fast on Stage 1 / 2 (cohort + main fits); allow LOCO/sensitivity stages
# to continue past individual failures (collected per-stage, summarized at end).
set -uo pipefail
ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
DEVICE=${DEVICE:-gpu}
# Use sequential chain method on single-V100 to ensure clean R-hat / ESS
# diagnostics (vectorized vmap shares step-size adaptation across chains
# and was observed to inflate R-hat; we accept ~4x longer fit time for
# convergence-quality fits suitable for publication).
case "$DEVICE" in
  gpu) CHAIN_METHOD=sequential ;;
  cpu) CHAIN_METHOD=sequential ; export JAX_PLATFORMS=cpu ;;
  *)   echo "[ABORT] DEVICE must be gpu|cpu (got: $DEVICE)"; exit 1 ;;
esac
XU_CHAIN_METHOD=sequential

# ROOT: repository root. Override via $REPO env var or pass as first arg.
# Default falls back to legacy CI layout for backward compatibility.
ROOT="${REPO:-$HOME/hmm-gformula-ci}"
if [ ! -d "$ROOT" ]; then
  # Auto-detect: script is in scripts/, repo root is parent
  SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
  ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
echo "[INFO] ROOT=$ROOT (override with REPO=/path/to/fre-nice-gformula)"
RES=$ROOT/_draft/results/extracted/main
GF=$RES/g_formula                                   # post-fit estimand outputs
LOCO=$ROOT/_draft/results/extracted/loco
SUBG=$ROOT/_draft/results/extracted/subgroup
SENS=$ROOT/_draft/results/extracted/sens            # knot J=4, J=6
LOG=$ROOT/_draft/results/chain.log
mkdir -p "$RES" "$GF" "$LOCO" "$SUBG" "$SENS"
cd "$ROOT"

REFBIN=7                                             # cohort Q25 — drop directly via --ref-bin
NAT=0.00656                                          # cohort observational off-MV per-day mortality

echo "[$(ts)] === CHAIN START (device=$DEVICE chain_method=$CHAIN_METHOD ref-bin=$REFBIN) ===" | tee -a "$LOG"

# Best-effort partial archive used by the wall-clock backstop. Pulled out
# so both the normal Stage 9 path and the backstop path can call it.
# GCS upload (optional — for cloud-instance backup). Set GCS_BUCKET to enable.
# Default empty = no upload. External users should set their own bucket:
#   export GCS_BUCKET=your-bucket-name
GCS_BUCKET=${GCS_BUCKET:-}
backstop_partial_save() {
  local TS_P="$1"
  local PARTIAL="$ROOT/results/chain_partial_${TS_P}.tar.gz"
  echo "[$(ts)] === BACKSTOP: building partial archive $PARTIAL ===" >> "$LOG"
  ( tar czf "$PARTIAL" -C "$ROOT/results" . 2>>"$LOG" ) || true
  if [ -f "$PARTIAL" ]; then
    timeout 600 gsutil cp "$PARTIAL" "gs://$GCS_BUCKET/" 2>>"$LOG" || true
    echo "[$(ts)] === BACKSTOP: partial archive uploaded (best effort) ===" >> "$LOG"
  fi
}
export -f backstop_partial_save 2>/dev/null || true

if [ "$DEVICE" = "gpu" ]; then
  nvidia-smi --query-gpu=name,memory.free --format=csv | tee -a "$LOG"
  # Wall-clock cost backstop. Conservative 12h cap (chain ETA ~10h with
  # LOCO-23 + knot + subgroup at J=5 primary settings). At 10h, if the
  # log file is still actively being written (mtime within 5 min), give
  # an additional 2h grace; otherwise treat as hung and shutdown.
  # On any path: best-effort partial archive + GCS upload before shutdown
  # so partial results survive a hang.
  (
    sleep 64800  # 18h primary cap (sequential chain method needs 14-16h for LOCO)
    last_active=$(stat -c %Y "$LOG" 2>/dev/null || echo 0)
    now=$(date +%s)
    age=$((now - last_active))
    echo "[$(ts)] === BACKSTOP T+18h: log idle ${age}s ===" >> "$LOG"
    if [ "$age" -lt 300 ]; then
      echo "[$(ts)] === BACKSTOP: chain still active, extending 2h ===" >> "$LOG"
      sleep 7200  # +2h grace = 20h hard cap
    fi
    echo "[$(ts)] === BACKSTOP TRIGGERED — partial save + shutdown ===" >> "$LOG"
    backstop_partial_save "$(date -u +%Y%m%d-%H%M%S)"
    sudo shutdown -h now
  ) &
  BACKSTOP_PID=$!
  echo "[$(ts)] === 20h backstop scheduled (pid=$BACKSTOP_PID; 18h primary + 2h grace) ===" | tee -a "$LOG"
else
  python3 -c "import jax; print('[device] jax devices:', jax.devices())" | tee -a "$LOG"
fi

# --------------------------------------------------------------------------
# Stage 1: Pull cohort + validate
# --------------------------------------------------------------------------
echo "[$(ts)] === Stage 1: Pull cohort from BigQuery (or use SCP-uploaded CSV) ===" | tee -a "$LOG"
mkdir -p data
if [ -s "data/ards_cohort.csv" ]; then
  echo "[$(ts)] === Stage 1: data/ards_cohort.csv already present (size $(du -h data/ards_cohort.csv | cut -f1)); skipping BQ pull ===" | tee -a "$LOG"
else
  # External users: set BQ_TABLE to your own MIMIC-IV ARDS cohort BigQuery table.
  # Default placeholder requires user override; see src/data/bq_extract/cohort.sql
  # for the cohort definition SQL.
  BQ_TABLE=${BQ_TABLE:-YOUR_GCP_PROJECT.YOUR_DATASET.cohort}
  if [ "$BQ_TABLE" = "YOUR_GCP_PROJECT.YOUR_DATASET.cohort" ]; then
    echo "[ABORT] BQ_TABLE not set. Either upload data/ards_cohort.csv directly, or"
    echo "        export BQ_TABLE=your_project.your_dataset.cohort"
    exit 1
  fi
  timeout 1800 bq --quiet query --use_legacy_sql=false --format=csv --max_rows=2000000 \
    "SELECT * FROM \`${BQ_TABLE}\` ORDER BY stay_id, day_num" \
    > data/ards_cohort.csv 2>>"$LOG"
fi

python3 - <<'PY' 2>&1 | tee -a "$LOG"
import pandas as pd
df = pd.read_csv("data/ards_cohort.csv")
print(f"[validate] rows={len(df):,}, stays={df.stay_id.nunique():,}")
both = df.dropna(subset=["ppeak", "pplat"])
peak_ge_plat = (both["ppeak"] >= both["pplat"]).mean()
print(f"[validate] P(Peak >= Plat | both measured) = {peak_ge_plat:.4f}")
mp = df["mp_j_min"].dropna()
print(f"[validate] MP rows = {len(mp):,}, median {mp.median():.2f} J/min")
n = df["stay_id"].nunique()
assert 17500 <= n <= 18000, f"Cohort size {n} outside expected 17500-18000"
assert peak_ge_plat >= 0.80
print(f"[OK] cohort validated (N={n}).")
PY
[ $? -ne 0 ] && { echo "[ABORT] validation failed"; exit 1; }

# --------------------------------------------------------------------------
# Stage 2: Main fits
# --------------------------------------------------------------------------
echo "[$(ts)] === Stage 2a: Fit J=1 NICE ===" | tee -a "$LOG"
if [ -f "$RES/fre_nice_J1_state.npz" ]; then
  echo "[$(ts)] === Stage 2a: state file present — skipping ===" | tee -a "$LOG"
else
  timeout 5400 python3 -u scripts/run_bayesian_main.py \
    --csv data/ards_cohort.csv --out-dir "$RES" \
    --inference nuts --chain-method $CHAIN_METHOD --n-warmup 1000 --n-samples 1000 --n-chains 4 \
    --target-accept 0.95 --n-posterior-subset 200 \
    --ref-bin $REFBIN --methods J1 --phase fit --record-loglik \
    >> "$LOG" 2>&1
fi

echo "[$(ts)] === Stage 2b: Fit J=5 FRE-NICE (primary) ===" | tee -a "$LOG"
if [ -f "$RES/fre_nice_J5_state.npz" ]; then
  echo "[$(ts)] === Stage 2b: state file present — skipping ===" | tee -a "$LOG"
else
  timeout 5400 python3 -u scripts/run_bayesian_main.py \
    --csv data/ards_cohort.csv --out-dir "$RES" \
    --inference nuts --chain-method $CHAIN_METHOD --n-warmup 2000 --n-samples 1000 --n-chains 4 \
    --target-accept 0.99 --n-posterior-subset 200 \
    --ref-bin $REFBIN --methods J5 --phase fit --record-loglik \
    >> "$LOG" 2>&1
fi

echo "[$(ts)] === Stage 2c: Fit Xu joint GLMM ===" | tee -a "$LOG"
if [ -f "$RES/xu_bayesian_state.npz" ]; then
  echo "[$(ts)] === Stage 2c: state file present — skipping ===" | tee -a "$LOG"
else
  # Xu uses sequential chain method (vectorized risks OOM on 16GB V100).
  # Timeout extended to 2h (7200s) since sequential 2 chains × 1500 warmup
  # can take up to 90-110 min depending on warmup adaptation.
  timeout 7200 python3 -u scripts/run_bayesian_main.py \
    --csv data/ards_cohort.csv --out-dir "$RES" \
    --inference nuts --chain-method $XU_CHAIN_METHOD --n-warmup 1500 --n-samples 750 --n-chains 2 \
    --target-accept 0.95 --n-posterior-subset 200 \
    --ref-bin $REFBIN --methods xu --phase fit --record-loglik \
    >> "$LOG" 2>&1
fi

# --------------------------------------------------------------------------
# Stage 3: LOCO-23 sensitivity (J=5 only)
# --------------------------------------------------------------------------
TV_COLS=(pf_ratio paco2 lactate map_mmhg heart_rate gcs_total creatinine
         temperature_c ph_arterial hemoglobin sofa_daily pressor_day prone_day)
STATIC_COLS=(anchor_age gender_M bmi_imputed charlson_index bmi_missing)
CCI_COLS=(cci_chf cci_renal cci_cancer cci_metastatic cci_liver_severe)

run_loco_one () {
  local label="$1" outdir="$2" excl_flag="$3" col="$4"
  echo "[$(ts)] === LOCO ${label}: ${col} ===" | tee -a "$LOG"
  if [ -f "$outdir/fre_nice_J5_state.npz" ]; then
    echo "[$(ts)] === LOCO ${label}: ${col} — state file present, skipping ===" | tee -a "$LOG"
    return 0
  fi
  # J=5 LOCO uses primary convergence settings (warmup 2000, target 0.99) so
  # divergence rates match Stage 2b; otherwise CrIs would be wider for LOCO.
  timeout 5400 python3 -u scripts/run_bayesian_main.py \
    --csv data/ards_cohort.csv --out-dir "$outdir" \
    --inference nuts --chain-method $CHAIN_METHOD --n-warmup 2000 --n-samples 1000 --n-chains 4 \
    --target-accept 0.99 --ref-bin $REFBIN \
    --methods J5 --phase fit --record-loglik "$excl_flag" "$col" \
    >> "$LOG" 2>&1
}
for col in "${TV_COLS[@]}";     do run_loco_one TV     "$LOCO/loco_tv_${col}"     --exclude-tv "$col"; done
for col in "${STATIC_COLS[@]}"; do run_loco_one static "$LOCO/loco_static_${col}" --exclude-static "$col"; done
for col in "${CCI_COLS[@]}";    do run_loco_one CCI    "$LOCO/loco_cci_${col}"    --exclude-static "$col"; done

# --------------------------------------------------------------------------
# Stage 4: Knot sensitivity (J=4, J=6)
# --------------------------------------------------------------------------
for jj in J4 J6; do
  echo "[$(ts)] === Stage 4: Knot sensitivity ${jj} ===" | tee -a "$LOG"
  prefix_lc=$(echo $jj | tr 'A-Z' 'a-z')
  if [ -f "$SENS/fre_nice_${prefix_lc}_state.npz" ]; then
    echo "[$(ts)] === Stage 4 ${jj}: state file present — skipping ===" | tee -a "$LOG"
    continue
  fi
  timeout 5400 python3 -u scripts/run_bayesian_main.py \
    --csv data/ards_cohort.csv --out-dir "$SENS" \
    --inference nuts --chain-method $CHAIN_METHOD --n-warmup 1500 --n-samples 1000 --n-chains 4 \
    --target-accept 0.95 --ref-bin $REFBIN \
    --methods $jj --phase fit --record-loglik \
    >> "$LOG" 2>&1
done

# --------------------------------------------------------------------------
# Stage 5: Subgroup stratified fits (J=5 × severity strata)
# --------------------------------------------------------------------------
for sev in mild moderate severe; do
  echo "[$(ts)] === Stage 5: Subgroup ${sev} ===" | tee -a "$LOG"
  if [ -f "$SUBG/severity_${sev}/fre_nice_J5_state.npz" ]; then
    echo "[$(ts)] === Stage 5 ${sev}: state file present — skipping ===" | tee -a "$LOG"
    continue
  fi
  timeout 5400 python3 -u scripts/run_bayesian_main.py \
    --csv data/ards_cohort.csv --out-dir "$SUBG/severity_${sev}" \
    --inference nuts --chain-method $CHAIN_METHOD --n-warmup 1000 --n-samples 1000 --n-chains 4 \
    --target-accept 0.95 --ref-bin $REFBIN \
    --severity ${sev} \
    --methods J5 --phase fit --record-loglik \
    >> "$LOG" 2>&1
done

# --------------------------------------------------------------------------
# Stage 6: Estimand extraction (forward L sim + post-hoc β-based HR)
# --------------------------------------------------------------------------
for state in fre_nice_J1_state.npz fre_nice_J5_state.npz xu_bayesian_state.npz; do
  if [ ! -f "$RES/$state" ]; then
    echo "[ABORT] Stage 2 missing $RES/$state — main fit failed; not entering Stage 6." | tee -a "$LOG"
    exit 1
  fi
done

echo "[$(ts)] === Stage 6a: Marginal cumulative-hazard HR (primary, J=1, J=5) ===" | tee -a "$LOG"
for prefix in fre_nice_J1 fre_nice_J5; do
  timeout 1800 python3 -u scripts/run_g_formula.py \
    --csv data/ards_cohort.csv --state-dir "$RES" --prefix "$prefix" \
    --model fre_nice --estimand per_day_hr --hr-definition cumulative_hazard \
    --ref-bin $REFBIN --n-posterior-subset 200 --n-b-draws 5 \
    --out-dir "$GF" >> "$LOG" 2>&1
done

echo "[$(ts)] === Stage 6b: Cumulative incidence (sensitivity, J=5) ===" | tee -a "$LOG"
# sustained 28d (no suffix) and trunc-7d + natural-course splice (suffix _trunc7nat)
timeout 1800 python3 -u scripts/run_g_formula.py \
  --csv data/ards_cohort.csv --state-dir "$RES" --prefix fre_nice_J5 \
  --model fre_nice --estimand cumulative --ref-bin $REFBIN \
  --n-posterior-subset 200 --n-b-draws 5 \
  --out-dir "$GF" >> "$LOG" 2>&1
timeout 1800 python3 -u scripts/run_g_formula.py \
  --csv data/ards_cohort.csv --state-dir "$RES" --prefix fre_nice_J5 \
  --model fre_nice --estimand cumulative --ref-bin $REFBIN \
  --n-posterior-subset 200 --n-b-draws 5 \
  --truncate-at-day 7 --natural-course-rate $NAT --out-suffix _trunc7nat \
  --out-dir "$GF" >> "$LOG" 2>&1

echo "[$(ts)] === Stage 6c: Conditional HR triangulation (no GPU) ===" | tee -a "$LOG"
for prefix in fre_nice_J1 fre_nice_J5; do
  timeout 1800 python3 -u scripts/run_g_formula.py \
    --csv data/ards_cohort.csv --state-dir "$RES" --prefix "$prefix" \
    --model fre_nice --estimand conditional --ref-bin $REFBIN \
    --out-dir "$GF" >> "$LOG" 2>&1
done
timeout 1800 python3 -u scripts/run_g_formula.py \
  --csv data/ards_cohort.csv --state-dir "$RES" --prefix xu_bayesian \
  --model xu --estimand conditional --ref-bin $REFBIN \
  --out-dir "$GF" >> "$LOG" 2>&1

echo "[$(ts)] === Stage 6d: PPC natural course — freq-weighted approximation (J=5) ===" | tee -a "$LOG"
timeout 1800 python3 -u scripts/run_g_formula.py \
  --csv data/ards_cohort.csv --state-dir "$RES" --prefix fre_nice_J5 \
  --model fre_nice --estimand ppc --ref-bin $REFBIN \
  --n-posterior-subset 200 --n-b-draws 5 \
  --out-dir "$GF" >> "$LOG" 2>&1

echo "[$(ts)] === Stage 6d-bis: PPC per-stay forward sim — valid PPC (J=5) ===" | tee -a "$LOG"
timeout 1800 python3 -u scripts/run_g_formula.py \
  --csv data/ards_cohort.csv --state-dir "$RES" --prefix fre_nice_J5 \
  --model fre_nice --estimand ppc_per_stay --ref-bin $REFBIN \
  --n-posterior-subset 200 --n-b-draws 5 \
  --out-dir "$GF" >> "$LOG" 2>&1

echo "[$(ts)] === Stage 6d-rmst: RMST(28) per bin (J=5) ===" | tee -a "$LOG"
timeout 1800 python3 -u scripts/run_g_formula.py \
  --csv data/ards_cohort.csv --state-dir "$RES" --prefix fre_nice_J5 \
  --model fre_nice --estimand rmst --ref-bin $REFBIN \
  --n-posterior-subset 200 --n-b-draws 5 \
  --out-dir "$GF" >> "$LOG" 2>&1

echo "[$(ts)] === Stage 6d-phcurve: per-day HR(t) curve / PH diagnostic (J=5) ===" | tee -a "$LOG"
timeout 1800 python3 -u scripts/run_g_formula.py \
  --csv data/ards_cohort.csv --state-dir "$RES" --prefix fre_nice_J5 \
  --model fre_nice --estimand per_day_hr_curve --ref-bin $REFBIN \
  --n-posterior-subset 200 --n-b-draws 5 \
  --out-dir "$GF" >> "$LOG" 2>&1

echo "[$(ts)] === Stage 6e: LOCO per_day_hr extraction (23 refits) ===" | tee -a "$LOG"
for d in "$LOCO"/loco_*; do
  name=$(basename "$d")
  excl_flag=""; col=""
  case "$name" in
    loco_tv_*)     excl_flag="--exclude-tv";     col="${name#loco_tv_}";;
    loco_static_*) excl_flag="--exclude-static"; col="${name#loco_static_}";;
    loco_cci_*)    excl_flag="--exclude-static"; col="${name#loco_cci_}";;
  esac
  timeout 1800 python3 -u scripts/run_g_formula.py \
    --csv data/ards_cohort.csv --state-dir "$d" --prefix fre_nice_J5 \
    --model fre_nice --estimand per_day_hr --hr-definition cumulative_hazard \
    --ref-bin $REFBIN --n-posterior-subset 200 --n-b-draws 5 \
    $excl_flag $col --out-dir "$d" >> "$LOG" 2>&1
done

echo "[$(ts)] === Stage 6f: Subgroup per_day_hr extraction ===" | tee -a "$LOG"
for sev in mild moderate severe; do
  timeout 1800 python3 -u scripts/run_g_formula.py \
    --csv data/ards_cohort.csv --state-dir "$SUBG/severity_${sev}" \
    --severity ${sev} \
    --prefix fre_nice_J5 --model fre_nice --estimand per_day_hr \
    --hr-definition cumulative_hazard --ref-bin $REFBIN \
    --n-posterior-subset 200 --n-b-draws 5 \
    --out-dir "$SUBG/severity_${sev}" >> "$LOG" 2>&1
done

# --------------------------------------------------------------------------
# Stage 7-8: Manuscript outputs (tables + figures, all consolidated)
# --------------------------------------------------------------------------
echo "[$(ts)] === Stage 7-8: Generate all manuscript outputs ===" | tee -a "$LOG"
timeout 1800 python3 -u scripts/make_outputs.py \
  --csv data/ards_cohort.csv --state-dir "$RES" --gf-dir "$GF" \
  --loco-dir "$LOCO" --sens-dir "$ROOT/results" \
  --out-dir "$ROOT/results/outputs" \
  --ref-bin $REFBIN --n-bins 20 --output all \
  >> "$LOG" 2>&1

# --------------------------------------------------------------------------
# Stage 9: Archive + auto-shutdown
# --------------------------------------------------------------------------
TS=$(date -u +%Y%m%dT%H%M%SZ)
ARCHIVE="$ROOT/results/chain_${TS}.tar.gz"
tar czf "$ARCHIVE" -C "$ROOT/results" main loco subgroup sens outputs chain.log
echo "[$(ts)] === Archive: $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1)) ===" | tee -a "$LOG"

# ----- Stage 9b: Upload archive to GCS (optional — only if GCS_BUCKET set) -----
# All gsutil calls are wrapped in `timeout` so a stuck network can never block
# the chain from terminating (cost-bleed defense). $GCS_BUCKET defined above.
if [ -z "$GCS_BUCKET" ]; then
  echo "[$(ts)] === Stage 9b: GCS_BUCKET unset; archive stays on local disk ===" | tee -a "$LOG"
  GCS_OK=0
else
  echo "[$(ts)] === Stage 9b: Upload archive to gs://$GCS_BUCKET/ ===" | tee -a "$LOG"
  timeout 600 gsutil cp "$ARCHIVE" "gs://$GCS_BUCKET/" 2>&1 | tee -a "$LOG" \
    && echo "[$(ts)] === GCS upload OK: gs://$GCS_BUCKET/$(basename "$ARCHIVE") ===" | tee -a "$LOG" \
    || echo "[$(ts)] === GCS upload FAILED — archive remains on local disk ===" | tee -a "$LOG"
  # Verify upload exists before allowing shutdown
  if timeout 60 gsutil stat "gs://$GCS_BUCKET/$(basename "$ARCHIVE")" >/dev/null 2>&1; then
    echo "[$(ts)] === GCS object verified ===" | tee -a "$LOG"
    GCS_OK=1
  else
    echo "[$(ts)] === GCS verify FAILED or timed out — keeping instance alive for manual recovery ===" | tee -a "$LOG"
    GCS_OK=0
  fi
fi

if [ "$DEVICE" = "gpu" ]; then
  nvidia-smi --query-gpu=memory.free --format=csv | tee -a "$LOG"
  if [ "$GCS_OK" = "1" ]; then
    echo "[$(ts)] === CHAIN COMPLETE. Auto-shutdown in 120s (archive in GCS) ===" | tee -a "$LOG"
    # Cancel the backstop subshell since we have a verified archive in GCS.
    # The 120s grace is for log-flush; backstop kill avoids it firing if the
    # shutdown command happens to take longer than 120s for any reason.
    if [ -n "${BACKSTOP_PID:-}" ]; then
      kill "$BACKSTOP_PID" 2>/dev/null || true
    fi
    sleep 120
    sudo shutdown -h now
  else
    echo "[$(ts)] === CHAIN COMPLETE but GCS upload not verified ===" | tee -a "$LOG"
    echo "[$(ts)] === The 12h backstop will produce a partial archive within ~2h ===" | tee -a "$LOG"
    echo "[$(ts)] === If the user can intervene: SCP $ARCHIVE then 'sudo shutdown -h now' ===" | tee -a "$LOG"
    # Trigger backstop early — primary archive failed to verify, so don't
    # waste compute waiting for the 10h primary cap.
    if [ -n "${BACKSTOP_PID:-}" ]; then
      echo "[$(ts)] === Triggering early backstop (kill primary sleep) ===" | tee -a "$LOG"
      kill "$BACKSTOP_PID" 2>/dev/null || true
    fi
    # Fallback: best-effort partial save + shutdown immediately
    backstop_partial_save "$(date -u +%Y%m%d-%H%M%S)"
    sleep 60
    sudo shutdown -h now
  fi
else
  echo "[$(ts)] === CHAIN COMPLETE (CPU mode; no auto-shutdown) ===" | tee -a "$LOG"
fi
