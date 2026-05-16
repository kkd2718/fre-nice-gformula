"""Post-fit g-formula estimand CLI.

Estimands:
  per_day_hr   marginal cumulative-hazard ratio at T=28 (primary)
  cumulative   cumulative incidence under sustained / truncated regimes
  conditional  post-hoc β-based HR (no forward sim)
  ppc          natural-course validation (frequency-weighted approximation)

LOCO via --exclude-tv VAR / --exclude-static VAR.
Subgroup via --severity {mild,moderate,severe}.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.data.ards import ARDSConfig, load_ards_cohort
from src.benchmarks.g_formula import (
    GFormulaConfig, FreNiceForwardSim, Estimands, load_state,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--state-dir", required=True, type=Path)
    p.add_argument("--prefix", required=True,
                   help="Filename prefix, e.g. fre_nice_J1 or xu_bayesian.")
    p.add_argument("--model", choices=["fre_nice", "xu"], required=True)
    p.add_argument("--estimand",
                   choices=["per_day_hr", "cumulative", "conditional", "ppc",
                            "ppc_per_stay", "rmst", "per_day_hr_curve"],
                   required=True)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--ref-bin", type=int, default=7)
    p.add_argument("--n-bins", type=int, default=20)
    p.add_argument("--n-posterior-subset", type=int, default=200)
    p.add_argument("--n-b-draws", type=int, default=5)
    p.add_argument("--m-subset", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--truncate-at-day", type=int, default=None)
    p.add_argument("--natural-course-rate", type=float, default=0.0)
    p.add_argument("--exclude-tv", default=None,
                   help="Name of TV covariate to exclude (LOCO sensitivity).")
    p.add_argument("--exclude-static", default=None,
                   help="Name of static covariate to exclude (LOCO sensitivity).")
    p.add_argument("--out-suffix", default="",
                   help="Append to output filenames (e.g. '_trunc7zero').")
    p.add_argument("--severity", default=None,
                   choices=["mild", "moderate", "severe"],
                   help="Restrict cohort to one Berlin severity stratum "
                        "(bin edges still computed from full cohort).")
    p.add_argument("--hr-definition", default="cumulative_hazard",
                   choices=["cumulative_hazard", "mean_of_ratios", "ratio_of_means"],
                   help="HR definition for --estimand per_day_hr. "
                        "Manuscript primary: cumulative_hazard.")
    return p.parse_args()


def _save_outputs(estimand: str, result: dict, centers: np.ndarray,
                   ref_bin: int, out_dir: Path, prefix: str, suffix: str) -> None:
    """Save NPZ + Markdown for any estimand result."""
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{prefix}_{estimand}{suffix}.npz"
    np.savez(npz_path, bin_centers_J_min=centers, ref_bin=ref_bin, **{
        k: v for k, v in result.items() if isinstance(v, np.ndarray)
    })
    print(f"  wrote {npz_path}")

    md_path = out_dir / f"{prefix}_{estimand}{suffix}.md"

    # PPC variants are single-regime — write summary not bin table.
    if estimand in ("ppc", "ppc_per_stay"):
        m, lo, hi = float(result["risk_mean"][0]), float(result["risk_ci_low"][0]), float(result["risk_ci_high"][0])
        label_kind = ("freq-weighted (approximation)" if estimand == "ppc"
                      else "per-stay forward sim (valid PPC)")
        md_path.write_text(
            f"# {prefix} — PPC natural course ({label_kind})\n\n"
            f"Day-28 cumulative mortality (predicted): "
            f"{100*m:.1f}% ({100*lo:.1f}–{100*hi:.1f})\n",
            encoding="utf-8")
        print(f"  wrote {md_path}")
        return

    # per_day_hr_curve: HR(t) by day for each bin — wide table not useful in MD;
    # NPZ already saved upstream. Write a compact summary instead.
    if estimand == "per_day_hr_curve":
        T = result["hr_t_mean"].shape[0]
        lines = [
            f"# {prefix} — per-day HR(t) curve",
            f"_Reference bin: {ref_bin} ≈ {centers[ref_bin]:.1f} J/min. "
            f"PH-violation diagnostic: HR(t) constant in t ⇒ PH holds._", "",
            f"| Bin | MP | HR(d=1) | HR(d=7) | HR(d=14) | HR(d=21) | HR(d=27) |",
            "|---|---|---|---|---|---|---|",
        ]
        marks = [min(d, T - 1) for d in (0, 6, 13, 20, 26)]
        for k in range(len(centers)):
            if k == ref_bin:
                lines.append(f"| {k} (ref) | {centers[k]:.1f} | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |")
                continue
            cells = [f"{result['hr_t_mean'][d, k]:.2f}" for d in marks]
            lines.append(f"| {k} | {centers[k]:.1f} | " + " | ".join(cells) + " |")
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  wrote {md_path}")
        return

    if estimand == "rmst":
        lines = [
            f"# {prefix} — RMST(28) per bin",
            f"_Reference bin: {ref_bin} ≈ {centers[ref_bin]:.1f} J/min. "
            f"RMST in days; Δ-RMST vs reference (95% CrI)._", "",
            "| Bin | MP | RMST | Δ-RMST vs ref |",
            "|---|---|---|---|",
        ]
        for k in range(len(centers)):
            r = result["rmst_mean"][k]; rl = result["rmst_ci_low"][k]; rh = result["rmst_ci_high"][k]
            d = result["delta_rmst_mean"][k]; dl = result["delta_rmst_ci_low"][k]; dh = result["delta_rmst_ci_high"][k]
            ref_text = "(0)" if k == ref_bin else f"{d:+.2f} ({dl:+.2f},{dh:+.2f})"
            mark = " (ref)" if k == ref_bin else ""
            lines.append(f"| {k}{mark} | {centers[k]:.1f} | {r:.2f} ({rl:.2f}–{rh:.2f}) | {ref_text} |")
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  wrote {md_path}")
        return

    if estimand in ("per_day_hr", "conditional"):
        label = "HR (95% CrI)"
        mean, lo, hi = result["hr_mean"], result["hr_ci_low"], result["hr_ci_high"]
    elif estimand == "cumulative":
        label = "Risk % (95% CrI)"
        mean, lo, hi = result["risk_mean"], result["risk_ci_low"], result["risk_ci_high"]
    else:
        return

    lines = [
        f"# {prefix} — {estimand}{suffix}",
        f"_Reference bin: {ref_bin} (≈ {centers[ref_bin]:.1f} J/min). "
        f"Posterior 95% credible intervals._", "",
        f"| Bin | MP center (J/min) | {label} |",
        "|---|---|---|",
    ]
    for k in range(len(centers)):
        marker = " (ref)" if k == ref_bin else ""
        if estimand == "cumulative":
            lines.append(f"| {k}{marker} | {centers[k]:.1f} | "
                         f"{100*mean[k]:.1f}% ({100*lo[k]:.1f}–{100*hi[k]:.1f}) |")
        else:
            ref_text = "1.00 (—)" if k == ref_bin else \
                       f"{mean[k]:.2f} ({lo[k]:.2f}–{hi[k]:.2f})"
            lines.append(f"| {k}{marker} | {centers[k]:.1f} | {ref_text} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {md_path}")


def main() -> None:
    args = _parse_args()
    t0 = time.time()

    # Build cohort with optional LOCO exclusion
    excl_tv = (args.exclude_tv,) if args.exclude_tv else ()
    excl_static = (args.exclude_static,) if args.exclude_static else ()
    cohort = load_ards_cohort(ARDSConfig(
        csv_path=args.csv, n_bins=args.n_bins,
        exclude_tv_cols=excl_tv, exclude_static_cols=excl_static,
        severity=args.severity,
    ))
    edges = cohort.bin_edges_mp
    centers = np.array([
        np.sqrt(edges[k] * edges[k + 1])
        if (np.isfinite(edges[k]) and np.isfinite(edges[k + 1]) and edges[k] > 0)
        else float("nan") for k in range(len(edges) - 1)
    ])
    print(f"[cohort] N={cohort.Y.shape[0]} stays, T={cohort.Y.shape[1]}, "
          f"K_A={cohort.feature_layout['n_bins']}, "
          f"p_dyn={cohort.L_dyn.shape[2]}, p_stat={cohort.C_static.shape[1]}")

    cfg = GFormulaConfig(
        ref_bin=args.ref_bin, n_bins=args.n_bins,
        n_posterior_subset=args.n_posterior_subset,
        n_b_draws=args.n_b_draws, M_subset=args.m_subset,
        seed=args.seed, truncate_at_day=args.truncate_at_day,
        natural_course_rate=args.natural_course_rate,
    )

    state_path = args.state_dir / f"{args.prefix}_state.npz"
    state = load_state(state_path, args.model)
    print(f"[state] {state_path.name} loaded.")

    if args.estimand == "conditional":
        result = Estimands.conditional_hr_from_beta(
            state, ref_bin=args.ref_bin, n_bins=args.n_bins, model=args.model,
        )
    else:
        if args.model != "fre_nice":
            raise NotImplementedError(
                "Forward L sim for Xu joint GLMM remains in xu_glmm_bayesian.py "
                "(numpy, deterministic L, subject-level RE). "
                "Use --estimand conditional for Xu, or call the legacy chain "
                "for Xu cumulative incidence (xu_*_risks.npz)."
            )
        sim = FreNiceForwardSim(state, cohort, cfg)
        if args.estimand == "ppc":
            print(f"[forward] PPC: simulating natural-course (freq-weighted approximation)")
            ppc = sim.simulate_natural_course(cohort)
            T_idx = ppc.cum_per_day.shape[1] - 1
            risk_per_sample = ppc.cum_per_day[:, T_idx]
            result = {
                "risk_mean": np.array([risk_per_sample.mean()]),
                "risk_ci_low": np.array([np.quantile(risk_per_sample, 0.025)]),
                "risk_ci_high": np.array([np.quantile(risk_per_sample, 0.975)]),
                "h_per_day": ppc.h_per_day,
                "cum_per_day": ppc.cum_per_day,
            }
        elif args.estimand == "ppc_per_stay":
            print(f"[forward] PPC: per-stay forward sim under observed A_t (valid PPC)")
            ppc = sim.simulate_per_stay_ppc(cohort)
            T_idx = ppc.cum_per_day.shape[1] - 1
            risk_per_sample = ppc.cum_per_day[:, T_idx]
            result = {
                "risk_mean": np.array([risk_per_sample.mean()]),
                "risk_ci_low": np.array([np.quantile(risk_per_sample, 0.025)]),
                "risk_ci_high": np.array([np.quantile(risk_per_sample, 0.975)]),
                "h_per_day": ppc.h_per_day,
                "cum_per_day": ppc.cum_per_day,
            }
        else:
            print(f"[forward] sustained-regime forward sim across {args.n_bins} bins")
            per_bin = sim.simulate_all_bins()
            if args.estimand == "per_day_hr":
                result = Estimands.marginal_per_day_hr(
                    per_bin, args.ref_bin, definition=args.hr_definition)
            elif args.estimand == "rmst":
                result = Estimands.rmst_28(per_bin, args.ref_bin)
            elif args.estimand == "per_day_hr_curve":
                result = Estimands.per_day_hr_curve(per_bin, args.ref_bin)
            else:                                                       # cumulative
                result = Estimands.cumulative_incidence(per_bin)

    _save_outputs(
        args.estimand, result, centers, args.ref_bin,
        args.out_dir, args.prefix, args.out_suffix,
    )
    print(f"[done] total {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
