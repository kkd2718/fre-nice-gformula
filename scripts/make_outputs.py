"""Manuscript tables + figures from cohort CSV, fit states, and g_formula NPZs.

CLI: python scripts/make_outputs.py --csv ... --state-dir ... --gf-dir ...
     --output {table1,fig1,table2,fig3,table3,fig4,
               waic,positivity,ppc,diagnostics,e_value,all}

Refs: Hernán-Robins 2020 §21; Vehtari 2017 WAIC/LOO; VanderWeele-Ding 2017 E-value.
"""
from __future__ import annotations
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.data.ards import ARDSConfig, load_ards_cohort
from src.benchmarks.g_formula import Estimands, load_state


# ============================================================================
# Plotting style
# ============================================================================
plt.rcParams.update({
    "font.family": ["Malgun Gothic", "Arial"],
    "axes.unicode_minus": False, "figure.dpi": 200,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})
NAVY, ACCENT, LIGHT, GRAY = "#1F3A5F", "#B31B1B", "#F2F5FA", "#7A7A7A"


# ============================================================================
# Markdown helpers
# ============================================================================

def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    lines = ["| " + " | ".join(headers) + " |", sep]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines) + "\n"


def _med_iqr(x: np.ndarray) -> str:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if not len(x):
        return "—"
    return f"{np.median(x):.1f} ({np.percentile(x, 25):.1f}–{np.percentile(x, 75):.1f})"


def _n_pct(mask: pd.Series) -> str:
    n, tot = int(mask.sum()), len(mask)
    return f"{n} ({100 * n / tot:.1f})"


def _hr_cell(mean: float, lo: float, hi: float, is_ref: bool) -> str:
    return "1.00 (—)" if is_ref else f"{mean:.2f} ({lo:.2f}–{hi:.2f})"


def _risk_cell(mean: float, lo: float, hi: float) -> str:
    return f"{100 * mean:.1f}% ({100 * lo:.1f}–{100 * hi:.1f})"


# ============================================================================
# WAIC / PSIS-LOO computation (Vehtari 2017)
# ============================================================================

def _waic_loo(log_lik: np.ndarray) -> dict:
    """WAIC + PSIS-LOO via ArviZ (Vehtari et al. 2017, generalized Pareto fit)."""
    import arviz as az
    nz = ~(np.abs(log_lik).sum(axis=0) == 0)
    ll = log_lik[:, nz]                                     # (S, N_obs)
    # ArviZ expects (chains, draws, *obs_dims). Pack S as a single chain.
    idata = az.from_dict(log_likelihood={"y": ll[None, :, :]})
    waic = az.waic(idata, pointwise=True)
    loo = az.loo(idata, pointwise=True)
    return {
        "elpd_waic": float(waic.elpd_waic), "waic": -2 * float(waic.elpd_waic),
        "waic_se": 2 * float(waic.se), "p_waic": float(waic.p_waic),
        "elpd_loo": float(loo.elpd_loo), "loo": -2 * float(loo.elpd_loo),
        "loo_se": 2 * float(loo.se), "p_loo": float(loo.p_loo),
        "pareto_k_max": float(loo.pareto_k.max()) if hasattr(loo, "pareto_k") else float("nan"),
        "n_obs": int(ll.shape[1]),
    }


# ============================================================================
# Output builder
# ============================================================================

@dataclass
class OutputConfig:
    csv: Path
    state_dir: Path
    gf_dir: Path                                   # g_formula output dir
    out_dir: Path
    loco_dir: Path | None = None
    sens_dir: Path | None = None
    ref_bin: int = 7
    n_bins: int = 20


class OutputBuilder:
    """Produces all manuscript tables and figures from a single config."""

    def __init__(self, cfg: OutputConfig):
        self.cfg = cfg
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        self._df = None
        self._cohort = None

    # -------- lazy properties --------
    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.read_csv(self.cfg.csv)
        return self._df

    @property
    def cohort(self):
        if self._cohort is None:
            self._cohort = load_ards_cohort(ARDSConfig(
                csv_path=self.cfg.csv, n_bins=self.cfg.n_bins))
        return self._cohort

    def _gf(self, prefix: str, estimand: str, suffix: str = "") -> dict:
        """Load g_formula output NPZ as dict."""
        path = self.cfg.gf_dir / f"{prefix}_{estimand}{suffix}.npz"
        if not path.exists():
            raise FileNotFoundError(f"g_formula output missing: {path}")
        return dict(np.load(path, allow_pickle=True))

    def _centers(self) -> np.ndarray:
        edges = self.cohort.bin_edges_mp
        return np.array([
            np.sqrt(edges[k] * edges[k + 1])
            if (np.isfinite(edges[k]) and np.isfinite(edges[k + 1])
                and edges[k] > 0) else float("nan")
            for k in range(len(edges) - 1)
        ])

    # -------- Table 1: baseline characteristics --------
    def table1_baseline(self) -> Path:
        by_stay = (self.df.sort_values(["stay_id", "day_num"])
                   .groupby("stay_id").first().reset_index())
        sev = by_stay.get("severity", pd.Series(["unknown"] * len(by_stay)))
        strata = {"Overall": pd.Series(True, index=by_stay.index)}
        for s in ("mild", "moderate", "severe"):
            strata[s.capitalize()] = sev == s

        def col(fn, m): return fn(by_stay.loc[m])
        rows = [
            ["N"]                + [str(int(m.sum())) for m in strata.values()],
            ["Age (yr)"]         + [col(lambda d: _med_iqr(d["anchor_age"]), m) for m in strata.values()],
            ["Male, N (%)"]      + [col(lambda d: _n_pct(d["gender"] == "M"), m) for m in strata.values()],
            ["BMI"]              + [col(lambda d: _med_iqr(d.get("bmi_imputed", pd.Series([]))), m) for m in strata.values()],
            ["Charlson index"]   + [col(lambda d: _med_iqr(d.get("charlson_index", pd.Series([]))), m) for m in strata.values()],
            ["PaO₂/FiO₂ (d1)"]   + [col(lambda d: _med_iqr(d.get("pf_ratio", pd.Series([]))), m) for m in strata.values()],
            ["PaCO₂ (d1)"]       + [col(lambda d: _med_iqr(d.get("paco2", pd.Series([]))), m) for m in strata.values()],
            ["Lactate (d1)"]     + [col(lambda d: _med_iqr(d.get("lactate", pd.Series([]))), m) for m in strata.values()],
            ["Heart rate (d1)"]  + [col(lambda d: _med_iqr(d.get("heart_rate", pd.Series([]))), m) for m in strata.values()],
            ["MAP (d1)"]         + [col(lambda d: _med_iqr(d.get("map_mmhg", pd.Series([]))), m) for m in strata.values()],
            ["GCS total (d1)"]   + [col(lambda d: _med_iqr(d.get("gcs_total", pd.Series([]))), m) for m in strata.values()],
            ["Creatinine (d1)"]  + [col(lambda d: _med_iqr(d.get("creatinine", pd.Series([]))), m) for m in strata.values()],
            ["MP (d1, J/min)"]   + [col(lambda d: _med_iqr(d.get("mp_j_min", pd.Series([]))), m) for m in strata.values()],
            ["28-day mortality, N (%)"]
                + [col(lambda d: _n_pct(d.get("mortality_28d", pd.Series([])).fillna(0) == 1), m) for m in strata.values()],
        ]
        md = "# Table 1. Baseline characteristics\n\n"
        md += f"_N = {len(by_stay)} ICU stays. Continuous: median (IQR). Categorical: N (%). Stratified by Berlin ARDS severity._\n\n"
        md += _md_table(["Characteristic", "Overall", "Mild", "Moderate", "Severe"], rows)

        path = self.cfg.out_dir / "table1_baseline.md"
        path.write_text(md, encoding="utf-8")
        print(f"  wrote {path}")
        return path

    # -------- Fig 1: STARD flow diagram --------
    def fig1_flow(self, upstream_counts: dict | None = None) -> Path:
        """STARD-style cohort flow with explicit ECMO exclusion step.

        upstream_counts keys: all_icu, all_pts, adult_los1, adult_pts,
        invasive_mv, n_ecmo_excluded, n_pre_berlin (= invasive_mv − ecmo).
        """
        c = upstream_counts or {"all_icu": 94458, "all_pts": 65366,
                                 "adult_los1": 74829, "adult_pts": 54551,
                                 "invasive_mv": 36188, "n_ecmo_excluded": 89}
        by_stay = (self.df.sort_values(["stay_id", "day_num"])
                   .groupby("stay_id").first().reset_index())
        n_final = len(by_stay)
        sev = by_stay["severity"].value_counts()

        fig, ax = plt.subplots(figsize=(11, 10.5))
        ax.set_xlim(0, 11); ax.set_ylim(-1.0, 13); ax.axis("off")
        ax.text(5.5, 12.4, "MIMIC-IV (v3.1) ARDS Cohort Derivation",
                ha="center", va="center", fontsize=15, fontweight="bold", color=NAVY)
        ax.text(5.5, 11.95,
                "Berlin Definition (P/F ≤ 300  AND  PEEP ≥ 5 cmH₂O) on first observed day",
                ha="center", va="center", fontsize=10, style="italic", color=GRAY)
        ax.plot([1, 10], [11.55, 11.55], color=NAVY, lw=0.8, alpha=0.4)

        def box(x, y, w, h, title, lines, fc=LIGHT, ec=NAVY, lw=1.6):
            ax.add_patch(FancyBboxPatch((x, y), w, h,
                boxstyle="round,pad=0.04,rounding_size=0.12",
                lw=lw, ec=ec, fc=fc, zorder=2))
            cx = x + w / 2; ty = y + h - 0.32
            ax.text(cx, ty, title, ha="center", va="center",
                    fontsize=11, fontweight="bold", color=NAVY, zorder=3)
            n = len(lines); body_top = ty - 0.38; body_bot = y + 0.18
            total_h = (n - 1) * 0.30
            first_y = min((body_top + body_bot + total_h) / 2, body_top)
            for i, L in enumerate(lines):
                ax.text(cx, first_y - i * 0.30, L, ha="center", va="center",
                        fontsize=9.5, color="#222", zorder=3)

        def varr(x, ytop, ybot):
            ax.plot([x, x], [ytop, ybot + 0.12], color=NAVY, lw=1.4, zorder=1)
            ax.add_patch(FancyArrowPatch((x, ybot + 0.12), (x, ybot),
                arrowstyle="->", mutation_scale=14, color=NAVY, lw=1.4, zorder=1))

        def harr(x_main, x_excl, y):
            ax.plot([x_main, x_excl - 0.12], [y, y], color=NAVY, lw=1.4, zorder=1)
            ax.add_patch(FancyArrowPatch((x_excl - 0.12, y), (x_excl, y),
                arrowstyle="->", mutation_scale=14, color=NAVY, lw=1.4, zorder=1))

        MX, MW, EX, EW = 1.5, 5.0, 7.5, 3.2
        H, HF = 1.30, 1.55
        y_lst = [11.20 - H, 11.20 - 2*H - 0.55, 11.20 - 3*H - 1.10]
        y_final = y_lst[2] - 0.55 - HF

        box(MX, y_lst[0], MW, H, "All MIMIC-IV (v3.1) ICU stays",
            [f"N = {c['all_icu']:,} ICU stays  ({c['all_pts']:,} unique patients)"])
        box(MX, y_lst[1], MW, H, "Adult ICU stays (≥18 yr, LOS ≥ 1 day)",
            [f"N = {c['adult_los1']:,} ICU stays  ({c['adult_pts']:,} unique patients)"])
        box(MX, y_lst[2], MW, H, "Invasive mechanically ventilated stays",
            [f"N = {c['invasive_mv']:,} ICU stays",
             "(Tidal volume + Peak/Plateau pressure measured)"])
        box(MX, y_final, MW, HF, "Berlin ARDS analysis cohort",
            [f"N = {n_final:,} ICU stays  ({by_stay['subject_id'].nunique():,} unique patients)",
             "P/F ≤ 300  AND  PEEP ≥ 5 cmH₂O  on first observed day",
             "28-day window, MIMIC-IV v3.1 BigQuery"],
            fc="#E8F0FA", lw=2.0)

        # Excluded boxes (Berlin failure + ECMO exclusion shown separately)
        excl1 = c["all_icu"] - c["adult_los1"]
        excl2 = c["adult_los1"] - c["invasive_mv"]
        n_ecmo = c.get("n_ecmo_excluded", 0)
        n_berlin_fail = c["invasive_mv"] - n_final - n_ecmo
        for i, (label, n) in enumerate([
            ("Age <18 yr OR LOS <1 day", excl1),
            ("Not invasively ventilated", excl2),
            (f"First obs. day not Berlin (n={n_berlin_fail:,}); "
             f"ECMO during stay (n={n_ecmo:,})", n_berlin_fail + n_ecmo),
        ]):
            yc = (y_lst[i] + (y_lst[i+1] if i < 2 else y_final) + (H if i < 2 else HF)) / 2
            box(EX, yc - 0.55, EW, 1.10, "Excluded",
                [f"• {label}", f"• total n = {n:,} stays"],
                fc="#FCEBEB", ec=ACCENT, lw=1.2)

        for ytop, ybot, h in [(y_lst[0], y_lst[1], H), (y_lst[1], y_lst[2], H),
                              (y_lst[2], y_final, HF)]:
            varr(MX + MW / 2, ytop, ybot + h)

        # severity strata
        sw = 2.6; sy_top = y_final - 1.5; sy = sy_top - 1.20
        for j, (label, key) in enumerate([("Mild ARDS", "mild"),
                                           ("Moderate ARDS", "moderate"),
                                           ("Severe ARDS", "severe")]):
            x = 0.5 + j * (sw + 0.4)
            n = int(sev.get(key, 0))
            box(x, sy, sw, 1.20, label,
                [f"P/F {200 if key=='mild' else (100 if key=='moderate' else 0)}–"
                 f"{300 if key=='mild' else (200 if key=='moderate' else 100)}",
                 f"N = {n:,}"],
                fc=("#FFF4E0" if key == "severe" else LIGHT),
                ec=(ACCENT if key == "severe" else NAVY),
                lw=(2.0 if key == "severe" else 1.6))

        path = self.cfg.out_dir / "figures" / "fig1_flow_diagram.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout(); plt.savefig(path, facecolor="white"); plt.close()
        print(f"  wrote {path}")
        return path

    # -------- Table 2: cross-method per-day HR (FRE-NICE × Xu) --------
    def table2_cross_method(self,
                             prefixes: tuple[str, ...] = (
                                 "fre_nice_J1", "fre_nice_J5"),
                             include_xu_conditional: bool = True) -> Path:
        """[DESIGN] Marginal per-day HR for FRE-NICE methods + conditional HR for Xu.

        Xu marginal-via-forward-sim is in xu_glmm_bayesian.py legacy path; this
        table reports Xu's conditional HR (= exp(post_beta_A_k - post_beta_A_ref))
        for triangulation, clearly labeled.
        """
        centers = self._centers()
        methods: dict[str, dict] = {}
        for p in prefixes:
            d = self._gf(p, "per_day_hr")
            methods[f"{p} (marginal)"] = d
        if include_xu_conditional:
            xu_state = load_state(self.cfg.state_dir / "xu_bayesian_state.npz", "xu")
            xu_cond = Estimands.conditional_hr_from_beta(
                xu_state, self.cfg.ref_bin, self.cfg.n_bins, "xu")
            methods["Xu joint GLMM (conditional)"] = xu_cond

        headers = ["Bin", "MP (J/min)"] + list(methods.keys())
        rows = []
        for k in range(self.cfg.n_bins):
            row = [str(k) + (" (ref)" if k == self.cfg.ref_bin else ""),
                   f"{centers[k]:.1f}"]
            for nm, d in methods.items():
                row.append(_hr_cell(d["hr_mean"][k], d["hr_ci_low"][k],
                                     d["hr_ci_high"][k], k == self.cfg.ref_bin))
            rows.append(row)

        md = (f"# Table 2. Per-day hazard ratio HR(a vs bin {self.cfg.ref_bin}, "
              f"≈{centers[self.cfg.ref_bin]:.1f} J/min)\n\n"
              "_FRE-NICE methods report **marginal** per-day HR via NICE forward "
              "L simulation (primary estimand). Xu joint GLMM reports **conditional** "
              "HR = exp(β_A^k − β_A^ref) for triangulation; under joint GLMM the "
              "two estimands agree when L equations are correctly specified._\n\n")
        md += _md_table(headers, rows)
        path = self.cfg.out_dir / "table2_cross_method.md"
        path.write_text(md, encoding="utf-8")
        print(f"  wrote {path}")
        return path

    # -------- Fig 3: dose-response curves --------
    def fig3_dose_response(self,
                            prefixes: tuple[str, ...] = (
                                "fre_nice_J1", "fre_nice_J5")) -> Path:
        centers = self._centers()
        fig, ax = plt.subplots(figsize=(8, 5))
        for p, color in zip(prefixes, [NAVY, ACCENT]):
            d = self._gf(p, "per_day_hr")
            ax.plot(centers, d["hr_mean"], "-o", color=color, label=p, lw=1.5, ms=4)
            ax.fill_between(centers, d["hr_ci_low"], d["hr_ci_high"],
                            color=color, alpha=0.15)
        ax.axhline(1.0, color=GRAY, ls=":", lw=1.0)
        ax.axvline(centers[self.cfg.ref_bin], color=GRAY, ls=":", lw=1.0,
                   label=f"ref bin {self.cfg.ref_bin}")
        ax.axvline(17.0, color=ACCENT, ls="--", lw=1.0, alpha=0.4,
                   label="Costa 2021 cutoff (17 J/min)")
        ax.set_xlabel("Mechanical power (J/min)")
        ax.set_ylabel(f"Marginal per-day HR (vs bin {self.cfg.ref_bin})")
        ax.set_title("Dose-response: MP → 28-day mortality (FRE-NICE g-formula)")
        ax.legend(); ax.grid(alpha=0.3)

        path = self.cfg.out_dir / "figures" / "fig3_dose_response.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout(); plt.savefig(path, facecolor="white"); plt.close()
        print(f"  wrote {path}")
        return path

    # -------- Table 3: LOCO 17 × J=5 --------
    def table3_loco(self, base_prefix: str = "fre_nice_J5") -> Path:
        if not self.cfg.loco_dir or not self.cfg.loco_dir.exists():
            raise FileNotFoundError("loco_dir not set or missing")
        centers = self._centers()
        # Each LOCO sub-dir contains its own per_day_hr.npz
        loco_results = {}
        for sub in sorted(self.cfg.loco_dir.glob("loco_*")):
            f = sub / f"{base_prefix}_per_day_hr.npz"
            if f.exists():
                loco_results[sub.name] = dict(np.load(f, allow_pickle=True))

        headers = ["Bin", "MP (J/min)"] + list(loco_results.keys())
        rows = []
        for k in range(self.cfg.n_bins):
            row = [str(k) + (" (ref)" if k == self.cfg.ref_bin else ""),
                   f"{centers[k]:.1f}"]
            for nm, d in loco_results.items():
                row.append(_hr_cell(d["hr_mean"][k], d["hr_ci_low"][k],
                                     d["hr_ci_high"][k], k == self.cfg.ref_bin))
            rows.append(row)

        md = (f"# Table 3. LOCO sensitivity (J=5 FRE-NICE, marginal per-day HR)\n\n"
              f"_Reference: bin {self.cfg.ref_bin}. Each column is J=5 FRE-NICE "
              "refit excluding the named confounder. Stable estimates across LOCO "
              "indicate robustness to single-confounder misspecification._\n\n")
        md += _md_table(headers, rows)
        path = self.cfg.out_dir / "table3_loco.md"
        path.write_text(md, encoding="utf-8")
        print(f"  wrote {path}")
        return path

    # -------- Fig 4: subgroup forest --------
    def fig4_subgroup(self) -> Path:
        if not self.cfg.sens_dir:
            raise FileNotFoundError("sens_dir not set")
        centers = self._centers()
        results = {}
        for sub in sorted((self.cfg.sens_dir / "subgroup").glob("severity_*")):
            f = sub / "fre_nice_J5_per_day_hr.npz"
            if f.exists():
                results[sub.name.replace("severity_", "")] = dict(
                    np.load(f, allow_pickle=True))

        if not results:
            raise FileNotFoundError("No subgroup outputs found")

        # Forest plot at the bin nearest to the Costa 2021 17 J/min cutoff
        target_k = int(np.nanargmin(np.abs(centers - 17.0)))
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, (name, d) in enumerate(results.items()):
            ax.errorbar(d["hr_mean"][target_k], i,
                        xerr=[[d["hr_mean"][target_k] - d["hr_ci_low"][target_k]],
                              [d["hr_ci_high"][target_k] - d["hr_mean"][target_k]]],
                        fmt="o", color=NAVY, capsize=4)
        ax.axvline(1.0, color=GRAY, ls=":", lw=1.0)
        ax.set_yticks(range(len(results))); ax.set_yticklabels(list(results.keys()))
        ax.set_xlabel(f"HR at MP {centers[target_k]:.1f} J/min (vs bin {self.cfg.ref_bin})")
        ax.set_title("Subgroup forest plot")
        path = self.cfg.out_dir / "figures" / "fig4_subgroup_forest.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout(); plt.savefig(path, facecolor="white"); plt.close()
        print(f"  wrote {path}")
        return path

    # -------- WAIC / PSIS-LOO helper --------
    def _waic_rows(self, prefixes: tuple[str, ...]) -> list:
        """Look up state files (state_dir then sens_dir/sens) and compute WAIC/LOO stats."""
        search_dirs = [self.cfg.state_dir]
        if self.cfg.sens_dir is not None:
            search_dirs.append(self.cfg.sens_dir / "sens")
            search_dirs.append(self.cfg.sens_dir)  # fallback if user passes sens dir directly
        rows = []
        for p in prefixes:
            sp = None
            for d in search_dirs:
                cand = d / f"{p}_state.npz"
                if cand.exists():
                    sp = cand; break
            if sp is None:
                continue
            z = np.load(sp, allow_pickle=True)
            ll_key = "log_lik_subject" if "log_lik_subject" in z.files else "log_lik"
            stats = _waic_loo(z[ll_key])
            rows.append([p, f"{stats['waic']:.0f} ± {stats['waic_se']:.0f}",
                         f"{stats['loo']:.0f} ± {stats['loo_se']:.0f}",
                         f"{stats['p_waic']:.0f}", f"{stats['p_loo']:.0f}",
                         f"{stats['pareto_k_max']:.2f}", str(stats['n_obs'])])
        return rows

    # -------- Table S: WAIC / PSIS-LOO method comparison --------
    def table_waic_methods(self, prefixes: tuple[str, ...] = (
            "fre_nice_J1", "fre_nice_J5", "xu_bayesian")) -> Path:
        """Cross-method comparison: J=1 NICE vs J=5 FRE-NICE vs Xu joint GLMM.
        Different RE structures, same cohort/covariates."""
        rows = self._waic_rows(prefixes)
        md = "# Table S. WAIC / PSIS-LOO — method comparison\n\n"
        md += ("_Cross-method comparison of subject-level (leave-one-cluster-out) "
               "WAIC and PSIS-LOO. Lower = better. Pareto k̂ < 0.7 indicates "
               "reliable LOO estimate (Vehtari et al. 2017 §3.2)._\n\n")
        md += _md_table(["Method", "WAIC ± SE", "PSIS-LOO ± SE",
                         "p_WAIC", "p_LOO", "max k̂", "N obs"], rows)
        path = self.cfg.out_dir / "table_waic_methods.md"
        path.write_text(md, encoding="utf-8")
        print(f"  wrote {path}")
        return path

    # -------- Table S: WAIC / PSIS-LOO knot sensitivity (FRE-NICE family) --------
    def table_waic_knots(self, prefixes: tuple[str, ...] = (
            "fre_nice_J1", "fre_nice_J4", "fre_nice_J5", "fre_nice_J6")) -> Path:
        """Within-FRE-NICE knot sensitivity: J=1 (scalar RE) → J=4/5/6
        (functional RE with increasing spline rank). Confirms primary J=5
        is not over- or under-parameterised."""
        rows = self._waic_rows(prefixes)
        md = "# Table S. WAIC / PSIS-LOO — knot sensitivity (FRE-NICE family)\n\n"
        md += ("_Knot-rank sensitivity within the FRE-NICE family. J=1 is the "
               "scalar random intercept; J=4/5/6 are functional REs with "
               "natural cubic spline bases at the knot positions documented "
               "in Methods §Models. Primary J=5 is selected if its WAIC "
               "(or its LOO ELPD) is competitive with J=4 and J=6._\n\n")
        md += _md_table(["Spline rank J", "WAIC ± SE", "PSIS-LOO ± SE",
                         "p_WAIC", "p_LOO", "max k̂", "N obs"], rows)
        path = self.cfg.out_dir / "table_waic_knots.md"
        path.write_text(md, encoding="utf-8")
        print(f"  wrote {path}")
        return path

    # -------- Backward-compatible alias --------
    def table_waic(self, prefixes: tuple[str, ...] = (
            "fre_nice_J1", "fre_nice_J5", "xu_bayesian")) -> Path:
        """Backward-compatible alias for table_waic_methods."""
        return self.table_waic_methods(prefixes)

    # -------- Table S: positivity per bin --------
    def table_positivity(self) -> Path:
        A_bin = self.cohort.A_bin.numpy()
        L = self.cohort.L_dyn.numpy()
        C = self.cohort.C_static.numpy()
        at_risk = self.cohort.at_risk.numpy().squeeze(-1)
        bins = A_bin.argmax(-1).astype(int); bins[at_risk == 0] = -1
        centers = self._centers()
        # Resolve column positions from feature_layout (handles LOCO exclusions safely).
        sc = self.cohort.feature_layout["static_cols"]
        tc = self.cohort.feature_layout["tv_cols"]
        i_age = sc.index("anchor_age") if "anchor_age" in sc else None
        i_charlson = sc.index("charlson_index") if "charlson_index" in sc else None
        i_pf = tc.index("pf_ratio") if "pf_ratio" in tc else None
        # L is (N, T, p_dyn); C is (N, p_stat); mask is (N, T).
        # For C (per-stay), reduce to (N,) "ever in bin k" for masking.
        rows = []
        for k in range(self.cfg.n_bins):
            mask = bins == k                                         # (N, T)
            n = int(mask.sum())
            if not n:
                rows.append([str(k), f"{centers[k]:.1f}", "0", "—", "—", "—"])
                continue
            stay_in_bin = mask.any(axis=1)                           # (N,)
            pf_z = float(L[mask][:, i_pf].mean()) if i_pf is not None else float("nan")
            age_z = (float(C[stay_in_bin, i_age].mean())
                     if i_age is not None else float("nan"))
            char_z = (float(C[stay_in_bin, i_charlson].mean())
                      if i_charlson is not None else float("nan"))
            rows.append([str(k), f"{centers[k]:.1f}", str(n),
                         f"{pf_z:.2f}", f"{age_z:.2f}", f"{char_z:.2f}"])
        md = "# Table S. Positivity diagnostic per MP bin\n\n"
        md += ("_Standardized covariate means by bin (z-score). |z| > 0.5 may "
               "signal positivity violation._\n\n")
        md += _md_table(["Bin", "MP (J/min)", "N stay-days",
                         "PF z-mean", "Age z-mean", "Charlson z-mean"], rows)
        path = self.cfg.out_dir / "table_positivity.md"
        path.write_text(md, encoding="utf-8")
        print(f"  wrote {path}")
        return path

    # -------- Fig S: PPC natural course (freq-weighted + per-stay overlay) --------
    def fig_ppc(self, prefix: str = "fre_nice_J5") -> Path:
        d_freq = self._gf(prefix, "ppc")
        cum_freq = d_freq["cum_per_day"]                                # (S, T)
        # Per-stay PPC if available — main panel for valid PPC
        try:
            d_stay = self._gf(prefix, "ppc_per_stay")
            cum_stay = d_stay["cum_per_day"]
            has_stay = True
        except FileNotFoundError:
            has_stay = False

        Y = self.cohort.Y.numpy().squeeze(-1)
        T = Y.shape[1]
        obs_per_day = np.array([(Y[:, :t+1].max(axis=1) == 1).mean() for t in range(T)])

        fig, ax = plt.subplots(figsize=(7, 4))
        days = np.arange(T)
        ax.plot(days, obs_per_day, "k-", label="Observed", lw=2)
        if has_stay:
            cs_m = cum_stay.mean(axis=0); cs_lo = np.quantile(cum_stay, 0.025, axis=0); cs_hi = np.quantile(cum_stay, 0.975, axis=0)
            ax.plot(days, cs_m, color=NAVY, label="Predicted — per-stay PPC (valid)", lw=1.8)
            ax.fill_between(days, cs_lo, cs_hi, color=NAVY, alpha=0.20)
        cf_m = cum_freq.mean(axis=0); cf_lo = np.quantile(cum_freq, 0.025, axis=0); cf_hi = np.quantile(cum_freq, 0.975, axis=0)
        ax.plot(days, cf_m, color=GRAY, ls="--", label="Predicted — freq-weighted (approx.)", lw=1.2)
        ax.fill_between(days, cf_lo, cf_hi, color=GRAY, alpha=0.10)
        ax.set_xlabel("Day"); ax.set_ylabel("Cumulative mortality")
        ax.set_title("Posterior predictive check — natural-course validation")
        ax.legend(); ax.grid(alpha=0.3)
        path = self.cfg.out_dir / "figures" / "figS_ppc_natural_course.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout(); plt.savefig(path, facecolor="white"); plt.close()
        print(f"  wrote {path}")
        return path

    # -------- Fig S: per-day HR(t) curve (PH diagnostic) --------
    def fig_per_day_hr_curve(self, prefix: str = "fre_nice_J5") -> Path:
        d = self._gf(prefix, "per_day_hr_curve")
        hr_t = d["hr_t_mean"]                                          # (T, n_bins)
        T = hr_t.shape[0]
        days = np.arange(T)
        centers = self._centers()
        # Show three contrasts: bin 12 (Costa cutoff), bin 16 (high), bin 3 (low).
        targets = [k for k in (3, 12, 16) if k != self.cfg.ref_bin]
        fig, ax = plt.subplots(figsize=(7, 4))
        for k in targets:
            ax.plot(days, hr_t[:, k], lw=1.6,
                    label=f"Bin {k} ({centers[k]:.1f} J/min)")
        ax.axhline(1.0, color=GRAY, ls=":", lw=1.0)
        ax.set_xlabel("Day"); ax.set_ylabel(f"HR(t) vs bin {self.cfg.ref_bin}")
        ax.set_title("Per-day HR(t) — proportional-hazards diagnostic")
        ax.legend(); ax.grid(alpha=0.3)
        path = self.cfg.out_dir / "figures" / "figS_per_day_hr_curve.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout(); plt.savefig(path, facecolor="white"); plt.close()
        print(f"  wrote {path}")
        return path

    # -------- Table S: RMST(28) per bin --------
    def table_rmst(self, prefix: str = "fre_nice_J5") -> Path:
        d = self._gf(prefix, "rmst")
        centers = self._centers()
        rows = []
        for k in range(self.cfg.n_bins):
            r = d["rmst_mean"][k]; rl = d["rmst_ci_low"][k]; rh = d["rmst_ci_high"][k]
            dl = d["delta_rmst_mean"][k]
            dlo = d["delta_rmst_ci_low"][k]; dhi = d["delta_rmst_ci_high"][k]
            ref_text = "(0)" if k == self.cfg.ref_bin else f"{dl:+.2f} ({dlo:+.2f},{dhi:+.2f})"
            mark = " (ref)" if k == self.cfg.ref_bin else ""
            rows.append([f"{k}{mark}", f"{centers[k]:.1f}",
                         f"{r:.2f} ({rl:.2f}–{rh:.2f})", ref_text])
        md = "# Table S. RMST(28) and Δ-RMST per MP bin\n\n"
        md += ("_Restricted Mean Survival Time at day 28 (in days). "
               "Δ-RMST = RMST_k − RMST_ref. Negative Δ = days of survival lost "
               "relative to lung-protective Q25 reference. Robust to non-PH "
               "violations._\n\n")
        md += _md_table(["Bin", "MP (J/min)", "RMST (95% CrI)",
                         "Δ-RMST vs ref"], rows)
        path = self.cfg.out_dir / "table_rmst.md"
        path.write_text(md, encoding="utf-8")
        print(f"  wrote {path}")
        return path

    # -------- Table S: posterior diagnostics (R-hat / ESS) --------
    def table_diagnostics(self, prefixes: tuple[str, ...] = (
            "fre_nice_J1", "fre_nice_J5", "xu_bayesian")) -> Path:
        rows = []
        for p in prefixes:
            sp = self.cfg.state_dir / f"{p}_state.npz"
            if not sp.exists():
                continue
            z = np.load(sp, allow_pickle=True)
            if "diagnostics_keys" not in z.files:
                rows.append([p, "(no diagnostics saved)"])
                continue
            keys = list(z["diagnostics_keys"])
            vals = list(z["diagnostics_values"])
            row = [p] + [f"{k}: {v}" for k, v in zip(keys, vals)]
            rows.append(row)
        # Variable column count — flatten
        max_cols = max(len(r) for r in rows)
        rows = [r + [""] * (max_cols - len(r)) for r in rows]
        headers = ["Method"] + [f"diag_{i}" for i in range(max_cols - 1)]
        md = "# Table S. Posterior diagnostics\n\n"
        md += _md_table(headers, rows)
        path = self.cfg.out_dir / "table_diagnostics.md"
        path.write_text(md, encoding="utf-8")
        print(f"  wrote {path}")
        return path

    # -------- Table S: E-value at canonical contrast --------
    def table_e_value(self, prefix: str = "fre_nice_J5") -> Path:
        """E = HR + sqrt(HR(HR-1)) (VanderWeele & Ding 2017).
        For HR < 1 the bound is computed on 1/HR; direction column reports
        whether the unmeasured-confounder threshold applies to harm or
        protection.
        """
        d = self._gf(prefix, "per_day_hr")
        centers = self._centers()
        rows = []
        for k in range(self.cfg.n_bins):
            if k == self.cfg.ref_bin:
                continue
            hr = d["hr_mean"][k]
            direction = "harm" if hr > 1 else "protective"
            hr_eff = max(hr, 1.0 / hr)
            e = hr_eff + np.sqrt(hr_eff * (hr_eff - 1)) if hr_eff > 1 else 1.0
            rows.append([str(k), f"{centers[k]:.1f}",
                         f"{hr:.2f}", direction, f"{e:.2f}"])
        md = ("# Table S. E-value (unmeasured confounder bound)\n\n"
              "_VanderWeele & Ding 2017. E-value = minimum risk-ratio of an "
              "unmeasured confounder to both exposure and outcome required to "
              "fully explain the observed HR. For protective bins (HR<1) the "
              "bound is computed on 1/HR._\n\n")
        md += _md_table(["Bin vs ref", "MP (J/min)", "HR",
                         "Direction", "E-value"], rows)
        path = self.cfg.out_dir / "table_e_value.md"
        path.write_text(md, encoding="utf-8")
        print(f"  wrote {path}")
        return path

    # -------- All --------
    def all(self) -> None:
        self.table1_baseline()
        self.fig1_flow()
        self.table2_cross_method()
        self.fig3_dose_response()
        self.table_waic_methods()
        try: self.table_waic_knots()
        except Exception as e: print(f"  skip table_waic_knots: {e}")
        self.table_positivity()
        try: self.fig_ppc()
        except FileNotFoundError as e: print(f"  skip fig_ppc: {e}")
        try: self.fig_per_day_hr_curve()
        except FileNotFoundError as e: print(f"  skip fig_per_day_hr_curve: {e}")
        try: self.table_rmst()
        except FileNotFoundError as e: print(f"  skip table_rmst: {e}")
        try: self.table_diagnostics()
        except Exception as e: print(f"  skip table_diagnostics: {e}")
        try: self.table_e_value()
        except FileNotFoundError as e: print(f"  skip table_e_value: {e}")
        try: self.table3_loco()
        except (FileNotFoundError, AttributeError) as e:
            print(f"  skip table3_loco: {e}")
        try: self.fig4_subgroup()
        except (FileNotFoundError, AttributeError) as e:
            print(f"  skip fig4_subgroup: {e}")


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--state-dir", required=True, type=Path)
    p.add_argument("--gf-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--loco-dir", type=Path)
    p.add_argument("--sens-dir", type=Path)
    p.add_argument("--ref-bin", type=int, default=7)
    p.add_argument("--n-bins", type=int, default=20)
    p.add_argument("--output", default="all",
                   choices=["table1", "fig1", "table2", "fig3", "table3",
                            "fig4", "waic", "waic_methods", "waic_knots",
                            "positivity", "ppc", "ph_curve", "rmst",
                            "diagnostics", "e_value", "all"])
    args = p.parse_args()

    builder = OutputBuilder(OutputConfig(
        csv=args.csv, state_dir=args.state_dir, gf_dir=args.gf_dir,
        out_dir=args.out_dir, loco_dir=args.loco_dir, sens_dir=args.sens_dir,
        ref_bin=args.ref_bin, n_bins=args.n_bins,
    ))

    {
        "table1": builder.table1_baseline,
        "fig1":   builder.fig1_flow,
        "table2": builder.table2_cross_method,
        "fig3":   builder.fig3_dose_response,
        "table3": builder.table3_loco,
        "fig4":   builder.fig4_subgroup,
        "waic":   builder.table_waic_methods,
        "waic_methods": builder.table_waic_methods,
        "waic_knots":   builder.table_waic_knots,
        "positivity": builder.table_positivity,
        "ppc":    builder.fig_ppc,
        "ph_curve": builder.fig_per_day_hr_curve,
        "rmst":   builder.table_rmst,
        "diagnostics": builder.table_diagnostics,
        "e_value": builder.table_e_value,
        "all":    builder.all,
    }[args.output]()


if __name__ == "__main__":
    main()
