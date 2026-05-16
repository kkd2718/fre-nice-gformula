#!/bin/bash
# Bayesian standard NICE (J=0, no random effect) LOCO sensitivity chain.
#
# Designed to run IN PARALLEL with the main run_chain.sh (which is doing
# J=5 FRE-NICE LOCO + Stage 4-9). Outputs go to results/loco_j0/ so they
# do not conflict with main chain's results/loco/. Main chain's Stage 9
# archive picks up loco_j0/ automatically (tar -C "$ROOT/results" .).
#
# Methodological purpose: comparing LOCO HR(bin12 vs ref) under J=0 (no RE)
# vs J=5 (functional RE) demonstrates whether the random effect adds
# robustness when TV-confounder adjustment is imperfect.
#
# Does NOT shutdown the instance (main chain handles that).
# Does NOT create archive (main chain's Stage 9 picks up these outputs).

set -uo pipefail
ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }

# ROOT: repository root. Override via $REPO env var.
# Default falls back to legacy CI layout for backward compatibility.
ROOT="${REPO:-$HOME/hmm-gformula-ci}"
if [ ! -d "$ROOT" ]; then
  SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
  ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
echo "[INFO] ROOT=$ROOT (override with REPO=/path/to/fre-nice-gformula)"
LOCO_J0=$ROOT/_draft/results/extracted/loco_j0
LOG=$ROOT/_draft/results/j0_chain.log
mkdir -p "$LOCO_J0"
cd "$ROOT"

REFBIN=7

echo "[$(ts)] === J=0 BAYESIAN STANDARD NICE LOCO CHAIN START ===" | tee -a "$LOG"
nvidia-smi --query-gpu=name,memory.free --format=csv | tee -a "$LOG"

# Optional: fit a J=0 main baseline first (used in supplementary cross-method
# WAIC table; small marginal cost ~5 min). Skip if state file already exists.
echo "[$(ts)] === J=0 main fit (full cohort, no LOCO exclusion) ===" | tee -a "$LOG"
if [ -f "$ROOT/results/main/fre_nice_J0_state.npz" ]; then
  echo "[$(ts)] === J=0 main: state file present, skipping ===" | tee -a "$LOG"
else
  timeout 1800 python3 -u scripts/run_bayesian_main.py \
    --csv data/ards_cohort.csv --out-dir "$ROOT/results/main" \
    --inference nuts --chain-method sequential --n-warmup 1000 --n-samples 1000 --n-chains 4 \
    --target-accept 0.95 --ref-bin $REFBIN \
    --methods J0 --phase fit --record-loglik \
    >> "$LOG" 2>&1
fi

# LOCO loop — same 23 conditions as main chain's J=5 LOCO
TV_COLS=(pf_ratio paco2 lactate map_mmhg heart_rate gcs_total creatinine
         temperature_c ph_arterial hemoglobin sofa_daily pressor_day prone_day)
STATIC_COLS=(anchor_age gender_M bmi_imputed charlson_index bmi_missing)
CCI_COLS=(cci_chf cci_renal cci_cancer cci_metastatic cci_liver_severe)

run_loco_j0 () {
  local label="$1" outdir="$2" excl_flag="$3" col="$4"
  echo "[$(ts)] === J=0 LOCO ${label}: ${col} ===" | tee -a "$LOG"
  if [ -f "$outdir/fre_nice_J0_state.npz" ]; then
    echo "[$(ts)] === J=0 LOCO ${label}: ${col} — state file present, skipping ===" | tee -a "$LOG"
    return 0
  fi
  timeout 1800 python3 -u scripts/run_bayesian_main.py \
    --csv data/ards_cohort.csv --out-dir "$outdir" \
    --inference nuts --chain-method sequential --n-warmup 1000 --n-samples 1000 --n-chains 4 \
    --target-accept 0.95 --ref-bin $REFBIN \
    --methods J0 --phase fit --record-loglik "$excl_flag" "$col" \
    >> "$LOG" 2>&1
}

for col in "${TV_COLS[@]}";     do run_loco_j0 TV     "$LOCO_J0/loco_tv_${col}"     --exclude-tv "$col"; done
for col in "${STATIC_COLS[@]}"; do run_loco_j0 static "$LOCO_J0/loco_static_${col}" --exclude-static "$col"; done
for col in "${CCI_COLS[@]}";    do run_loco_j0 CCI    "$LOCO_J0/loco_cci_${col}"    --exclude-static "$col"; done

# Summary
COUNT=$(find "$LOCO_J0" -name "fre_nice_J0_state.npz" 2>/dev/null | wc -l)
echo "[$(ts)] === J=0 LOCO CHAIN COMPLETE: $COUNT/23 LOCO state files ===" | tee -a "$LOG"

# Best-effort upload of partial archive to GCS (in case main chain crashes
# before its Stage 9). Main chain's full archive will supersede this.
GCS_BUCKET=${GCS_BUCKET:-ards-mp-results-pivotal-nebula}
TS_J0=$(date -u +%Y%m%d-%H%M%S)
J0_ARCHIVE=$ROOT/results/j0_loco_${TS_J0}.tar.gz
tar czf "$J0_ARCHIVE" -C "$ROOT/results" loco_j0 j0_chain.log 2>>"$LOG" || true
timeout 300 gsutil cp "$J0_ARCHIVE" "gs://$GCS_BUCKET/" 2>>"$LOG" || true
echo "[$(ts)] === J=0 partial archive uploaded (best effort): $(basename "$J0_ARCHIVE") ===" | tee -a "$LOG"
