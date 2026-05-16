"""Bayesian main analysis: Xu + J=1 + J=5 FRE-NICE on full cohort
(K = treatment bin count, J = spline knot count for the functional RE).

Supports phase-based execution for pipelining (overlap dose_response CPU work
with the next method's NUTS GPU fit):

    --phase fit   : run NUTS only, save posterior + state, exit
    --phase dose  : load posterior + state, run dose_response, save risks
    --phase full  : (default) fit then dose_response inline

Outputs (to <out_dir>):
    {prefix}_state.npz      — posterior + benchmark state (after fit)
    {prefix}_risks.npz      — dose-response output (after dose)
    bayesian_table2.md      — combined markdown (after all dose phases)
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.ards import ARDSConfig, load_ards_cohort
from src.benchmarks import (
    XuGLMMBayesian, XuBayesianConfig,
    FRENICEBayesianBenchmark, FRENICEBayesianConfig,
)


# ----------------------------------------------------------------------
# State persistence helpers
# ----------------------------------------------------------------------
def _save_state_xu(bench: XuGLMMBayesian, prefix: str, out_dir: Path) -> None:
    arr = {"n_groups": bench._n_groups, "joint_glmm": int(bench.config.joint_glmm)}
    # Persist all posterior keys (joint GLMM stores beta_0/beta_A/...; legacy stores beta)
    for k, v in bench._posterior.items():
        arr[f"post_{k}"] = v
    if bench._log_lik is not None:
        arr["log_lik_subject"] = bench._log_lik
    if bench._diagnostics is not None:
        items = list(bench._diagnostics.items())
        arr["diagnostics_keys"] = np.array([k for k, _ in items], dtype=object)
        arr["diagnostics_values"] = np.array([str(v) for _, v in items], dtype=object)
    np.savez(out_dir / f"{prefix}_state.npz", **arr)


def _load_state_xu(bench: XuGLMMBayesian, cohort, prefix: str, out_dir: Path) -> None:
    z = np.load(out_dir / f"{prefix}_state.npz", allow_pickle=True)
    post: dict = {}
    for f in z.files:
        if f.startswith("post_"):
            post[f[len("post_"):]] = z[f]
    if not post:  # fall through to legacy schema
        post = {"beta": z["beta"], "sigma_b": z["sigma_b"]}
    bench._posterior = post
    bench._n_groups = int(z["n_groups"])
    if "joint_glmm" in z.files:
        bench.config.joint_glmm = bool(int(z["joint_glmm"]))
    if bench.config.joint_glmm:
        bench._joint_state = bench._build_joint_arrays(cohort)


def _save_state_fre(bench: FRENICEBayesianBenchmark, prefix: str, out_dir: Path) -> None:
    arr = {
        "beta": bench._posterior["beta"],
        "L_chol": bench._posterior["L_chol"],
        "tau": bench._posterior["tau"],
        "B_basis": bench._B_basis,
        "sd_L": np.array(bench._sd_L),
        "n_bins": bench._n_bins, "n_dyn": bench._n_dyn,
        "n_static": bench._n_static, "t_max": bench._t_max,
        "n_groups": bench._n_groups,
        "L_has_RE_col": int(bench._L_has_RE_col),
        # Persist j5_joint flag so downstream forward sim can route correctly
        # without depending on the CLI re-passing --methods J5_joint.
        # (State-file key uses upper-case J5_joint to match the CLI/prefix
        #  convention; the in-memory config field is lowercase j5_joint.)
        "J5_joint": int(bench.config.j5_joint),
    }
    arr["beta_L"] = np.stack(bench._beta_L) if bench._beta_L else np.zeros((0, 0))
    if bench._b_hat is not None:
        arr["b_hat"] = bench._b_hat
    if bench._lambda_L is not None:
        arr["lambda_L"] = bench._lambda_L
    # Persist full posteriors for J5_joint so downstream consumers (e.g.,
    # uncertainty-propagated forward sim) can use them. Only saved when
    # present; absent for J=0/J=1/J=5 standard runs.
    if "beta_L_bayes" in bench._posterior:
        arr["beta_L_bayes"] = bench._posterior["beta_L_bayes"]
    if "sigma_L_bayes" in bench._posterior:
        arr["sigma_L_bayes"] = bench._posterior["sigma_L_bayes"]
    if "lambda_L_post" in bench._posterior:
        arr["lambda_L_post"] = bench._posterior["lambda_L_post"]
    elif bench._posterior is not None and "lambda_L" in bench._posterior:
        arr["lambda_L_post"] = bench._posterior["lambda_L"]
    if bench._log_lik is not None:
        arr["log_lik_subject"] = bench._log_lik
    if bench._diagnostics is not None:
        items = list(bench._diagnostics.items())
        arr["diagnostics_keys"] = np.array([k for k, _ in items], dtype=object)
        arr["diagnostics_values"] = np.array([str(v) for _, v in items], dtype=object)
    np.savez(out_dir / f"{prefix}_state.npz", **arr)


def _load_state_fre(
    bench: FRENICEBayesianBenchmark, cohort, prefix: str, out_dir: Path,
) -> None:
    z = np.load(out_dir / f"{prefix}_state.npz", allow_pickle=True)
    bench._posterior = {
        "beta": z["beta"],
        "L_chol": z["L_chol"],
        "tau": z["tau"],
    }
    bench._B_basis = z["B_basis"]
    bench._beta_L = [z["beta_L"][j] for j in range(z["beta_L"].shape[0])]
    bench._sd_L = list(z["sd_L"])
    bench._n_bins = int(z["n_bins"])
    bench._n_dyn = int(z["n_dyn"])
    bench._n_static = int(z["n_static"])
    bench._t_max = int(z["t_max"])
    bench._n_groups = int(z["n_groups"])
    if "L_has_RE_col" in z.files:
        bench._L_has_RE_col = bool(int(z["L_has_RE_col"]))
    if "b_hat" in z.files:
        bench._b_hat = z["b_hat"]
    # Restore j5_joint state — both the config flag (so forward sim takes the
    # joint branch) and the lambda_L point estimate used in that branch.
    if "J5_joint" in z.files:
        bench.config.j5_joint = bool(int(z["J5_joint"]))
    if "lambda_L" in z.files:
        bench._lambda_L = z["lambda_L"]
    # Restore full posteriors for downstream uncertainty-propagated estimands.
    if "beta_L_bayes" in z.files:
        bench._posterior["beta_L_bayes"] = z["beta_L_bayes"]
    if "sigma_L_bayes" in z.files:
        bench._posterior["sigma_L_bayes"] = z["sigma_L_bayes"]
    if "lambda_L_post" in z.files:
        bench._posterior["lambda_L_post"] = z["lambda_L_post"]


def _save_risks(result, prefix: str, out_dir: Path) -> None:
    np.savez(
        out_dir / f"{prefix}_risks.npz",
        bin_centers_J_min=np.array(result.bin_centers_J_min),
        bins=np.array(result.bins),
        risk_mean=result.risk_mean,
        risk_ci_low=result.risk_ci_low,
        risk_ci_high=result.risk_ci_high,
        risk_raw=result.risk_raw,
    )


def _md_table(centers, ref_bin, results: dict[str, "Tuple"]) -> list[str]:
    methods = list(results.keys())
    md = [
        "# Bayesian 4-method comparison (Table 2 surface)",
        "",
        f"_Reference bin: {ref_bin} (≈ {centers[ref_bin]:.1f} J/min). "
        "Posterior 95% credible intervals._",
        "",
    ]
    headers = ["MP bin", "Center (J/min)"]
    for m in methods:
        headers.extend([f"{m} risk %", f"{m} 95% CI"])
    md.append("| " + " | ".join(headers) + " |")
    md.append("|" + "|".join(["---"] * len(headers)) + "|")
    n_bins = len(results[methods[0]].risk_mean)
    for k in range(n_bins):
        c = centers[k] if k < len(centers) else float("nan")
        row = [str(k), f"{c:.1f}"]
        for m in methods:
            r = results[m]
            row.append(f"{100*r.risk_mean[k]:.1f}")
            row.append(f"({100*r.risk_ci_low[k]:.1f}–{100*r.risk_ci_high[k]:.1f})")
        md.append("| " + " | ".join(row) + " |")
    return md


# ----------------------------------------------------------------------
# Per-method runners (each handles fit / dose / full)
# ----------------------------------------------------------------------
def run_xu(cohort, target_bins, args, out_dir: Path):
    cfg = XuBayesianConfig(
        inference=args.inference,
        n_warmup=args.n_warmup, n_samples=args.n_samples,
        n_chains=args.n_chains, chain_method=args.chain_method,
        target_accept=args.target_accept,
        svi_steps=args.svi_steps, svi_lr=args.svi_lr,
        svi_n_posterior_draws=args.svi_posterior_draws,
        n_b_draws=args.n_b_draws, n_posterior_subset=args.n_posterior_subset,
        sigma_b_prior=args.sigma_b_prior,
        record_loglik=args.record_loglik,
        holdout_subj_ids=tuple(args.holdout_subj_ids) if args.holdout_subj_ids else None,
        joint_glmm=not args.xu_legacy,
        seed=args.seed,
    )
    bench = XuGLMMBayesian(cfg)
    if args.phase in ("fit", "full"):
        print("\n=== Xu Bayesian — FIT ===")
        t0 = time.time()
        bench.fit(cohort)
        print(f"  fit time: {(time.time()-t0)/60:.1f} min")
        _save_state_xu(bench, "xu_bayesian", out_dir)
    if args.phase in ("dose", "full"):
        print("\n=== Xu Bayesian — DOSE ===")
        if args.phase == "dose":
            _load_state_xu(bench, cohort, "xu_bayesian", out_dir)
        t0 = time.time()
        result = bench.dose_response(
            cohort, target_bins=target_bins, refit=False,
            truncate_at_day=args.truncate_at_day,
            natural_course_rate=args.natural_course_rate,
        )
        print(f"  dose time: {(time.time()-t0)/60:.1f} min")
        out_prefix = f"xu_bayesian{args.prefix_suffix}"
        _save_risks(result, out_prefix, out_dir)
        return result
    return None


def run_fre_nice(knots, prefix, cohort, target_bins, args, out_dir: Path,
                 seed_offset: int = 1, ref_bin: int = 7,
                 j5_joint: bool = False):
    cfg = FRENICEBayesianConfig(
        ref_bin=ref_bin,                                              # plumb from cohort
        knots=knots, inference=args.inference,
        n_warmup=args.n_warmup, n_samples=args.n_samples,
        n_chains=args.n_chains, chain_method=args.chain_method,
        target_accept=args.target_accept,
        svi_steps=args.svi_steps, svi_lr=args.svi_lr,
        svi_n_posterior_draws=args.svi_posterior_draws,
        n_posterior_subset=args.n_posterior_subset,
        share_RE_on_L=args.share_RE_on_L,
        j5_joint=j5_joint,
        sigma_b_prior=args.sigma_b_prior,
        record_loglik=args.record_loglik,
        holdout_subj_ids=tuple(args.holdout_subj_ids) if args.holdout_subj_ids else None,
        n_b_draws_per_post=5, seed=args.seed + seed_offset,
    )
    bench = FRENICEBayesianBenchmark(cfg)
    if args.phase in ("fit", "full"):
        print(f"\n=== {prefix} — FIT ===")
        t0 = time.time()
        bench.fit(cohort)
        print(f"  fit time: {(time.time()-t0)/60:.1f} min")
        _save_state_fre(bench, prefix, out_dir)
    if args.phase in ("dose", "full"):
        print(f"\n=== {prefix} — DOSE ===")
        if args.phase == "dose":
            _load_state_fre(bench, cohort, prefix, out_dir)
        t0 = time.time()
        result = bench.dose_response(cohort, target_bins=target_bins, refit=False)
        print(f"  dose time: {(time.time()-t0)/60:.1f} min")
        _save_risks(result, prefix, out_dir)
        return result
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--n-bins", type=int, default=20)
    parser.add_argument("--inference", choices=["nuts", "svi"], default="nuts")
    parser.add_argument("--phase", choices=["fit", "dose", "full"], default="full")
    # Default config matches thesis §Ⅱ.5.1 (primary M3 settings):
    #   n_warmup=2000, n_samples=1000, n_chains=4, target_accept=0.99.
    # For lighter exploratory runs override via CLI (e.g., --n-warmup 1000).
    parser.add_argument("--n-warmup", type=int, default=2000)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-chains", type=int, default=4)
    parser.add_argument("--chain-method", choices=["parallel", "sequential", "vectorized"],
                        default="parallel",
                        help="numpyro chain method. Use 'sequential' on CPU to avoid "
                             "multi-process contention; 'parallel' on GPU.")
    parser.add_argument("--target-accept", type=float, default=0.99)
    parser.add_argument("--svi-steps", type=int, default=8000)
    parser.add_argument("--svi-lr", type=float, default=5e-3)
    parser.add_argument("--svi-posterior-draws", type=int, default=2000)
    parser.add_argument("--n-b-draws", type=int, default=50)
    parser.add_argument("--n-posterior-subset", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference-mp", type=float, default=None,
                        help="MP value mapped to nearest bin → ref_bin "
                             "(used only if --ref-bin not given).")
    parser.add_argument("--ref-bin", type=int, default=None,
                        help="Reference bin index. If unset, --reference-mp "
                             "is mapped to the nearest bin via argmin; if both "
                             "unset, defaults to 7 (cohort Q25).")
    parser.add_argument("--methods", nargs="+",
                        default=["xu", "J1", "J5"],
                        help="Subset of methods to run. J=spline knot count; "
                             "K1/K5/K4/K6 accepted as backward-compat aliases.")
    parser.add_argument("--share-RE-on-L", action="store_true",
                        help="Spec ②: share FRE between Y and L equations "
                             "(refit L with extra column lambda_j * b^T B(t))")
    parser.add_argument("--J5-knots", "--K5-knots", dest="K5_knots",
                        nargs="+", type=float,
                        default=[0.0, 3.0, 7.0, 14.0, 21.0],
                        help="Knot positions (in days) for J=5 specification.")
    parser.add_argument("--exclude-tv", nargs="*", default=[],
                        help="TV covariates to exclude (LOCO sensitivity)")
    parser.add_argument("--exclude-static", nargs="*", default=[],
                        help="Static covariates to exclude (LOCO sensitivity)")
    parser.add_argument("--sigma-b-prior", default="halfcauchy",
                        choices=["halfcauchy", "gamma", "invgamma"],
                        help="Prior on sigma_b (RE scale) for sensitivity analysis")
    parser.add_argument("--record-loglik", action="store_true",
                        help="Record per-observation log_lik for WAIC/PSIS-LOO")
    parser.add_argument("--xu-legacy", action="store_true",
                        help="Use legacy Xu outcome-only RE + observed-L plug-in (NOT Xu 2024 faithful; "
                             "default is joint GLMM with forward L sim)")
    parser.add_argument("--truncate-at-day", type=int, default=None,
                        help="Counterfactual truncation: A_t = e_k for t < N, A_t = 0 for t >= N. "
                             "Default None = sustained 28d regime.")
    parser.add_argument("--natural-course-rate", type=float, default=0.0,
                        help="If > 0 and truncate_at_day set: per-day mortality after "
                             "truncate_at_day forced to this value (cohort observational off-MV rate). "
                             "Use with Option B fit (at_risk = alive AND on-MV).")
    parser.add_argument("--prefix-suffix", type=str, default="",
                        help="Suffix appended to output prefixes (e.g., '_trunc7nat') so different regimes don't overwrite.")
    parser.add_argument("--holdout-subj-ids-file", type=Path, default=None,
                        help="Path to .npy file with subject indices to hold out from fit (for PPC)")
    parser.add_argument("--severity", default=None,
                        choices=["mild", "moderate", "severe"],
                        help="Restrict cohort to one Berlin severity stratum "
                             "(bin edges still computed from full cohort).")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve holdout subject IDs from optional file
    if args.holdout_subj_ids_file is not None and args.holdout_subj_ids_file.exists():
        args.holdout_subj_ids = list(map(int, np.load(args.holdout_subj_ids_file)))
        print(f"  [holdout] loaded {len(args.holdout_subj_ids)} held-out subjects from "
              f"{args.holdout_subj_ids_file}")
    else:
        args.holdout_subj_ids = None

    cohort = load_ards_cohort(ARDSConfig(
        csv_path=args.csv, n_bins=args.n_bins,
        exclude_tv_cols=tuple(args.exclude_tv),
        exclude_static_cols=tuple(args.exclude_static),
        severity=args.severity,
    ))
    target_bins = list(range(args.n_bins))
    edges = cohort.bin_edges_mp
    centers = np.array([
        np.sqrt(edges[k] * edges[k + 1])
        if (np.isfinite(edges[k]) and np.isfinite(edges[k + 1]) and edges[k] > 0)
        else float("nan")
        for k in range(len(edges) - 1)
    ])
    if args.ref_bin is not None:
        ref_bin = int(args.ref_bin)
    elif args.reference_mp is not None:
        ref_bin = int(np.nanargmin(np.abs(centers - args.reference_mp)))
    else:
        ref_bin = 7
    print(f"  ref_bin={ref_bin}, MP center {centers[ref_bin]:.2f} J/min")
    print(f"Cohort: N={cohort.Y.shape[0]} stays, "
          f"G={len(np.unique(cohort.subject_ids))} subjects. "
          f"ref_bin={ref_bin}, phase={args.phase}")

    results = {}
    if "xu" in args.methods:
        r = run_xu(cohort, target_bins, args, args.out_dir)
        if r is not None:
            results["Xu Bayesian"] = r
    # Spline knot rank denoted J (treatment bin count uses K — J is the random
    # effect basis dimension). Backwards-compatible flags accepted via aliases.
    if "J1" in args.methods or "K1" in args.methods:
        r = run_fre_nice(
            (14.0,), "fre_nice_J1", cohort, target_bins, args, args.out_dir,
            ref_bin=ref_bin, seed_offset=1,
        )
        if r is not None:
            results["FRE-NICE J=1"] = r
    if "J5" in args.methods or "K5" in args.methods:
        r = run_fre_nice(
            tuple(args.K5_knots), "fre_nice_J5", cohort,
            target_bins, args, args.out_dir, ref_bin=ref_bin, seed_offset=2,
        )
        if r is not None:
            results["FRE-NICE J=5"] = r
    if "J4" in args.methods or "K4" in args.methods:
        r = run_fre_nice(
            (0.0, 7.0, 14.0, 21.0), "fre_nice_J4", cohort,
            target_bins, args, args.out_dir, ref_bin=ref_bin, seed_offset=3,
        )
        if r is not None:
            results["FRE-NICE J=4"] = r
    if "J6" in args.methods or "K6" in args.methods:
        r = run_fre_nice(
            (0.0, 3.0, 7.0, 14.0, 21.0, 27.0), "fre_nice_J6", cohort,
            target_bins, args, args.out_dir, ref_bin=ref_bin, seed_offset=4,
        )
        if r is not None:
            results["FRE-NICE J=6"] = r
    if "J0" in args.methods or "K0" in args.methods:
        # Bayesian standard NICE — no random effect (knots=()). Used as
        # methodological baseline against J=5 FRE-NICE in LOCO sensitivity.
        r = run_fre_nice(
            (), "fre_nice_J0", cohort, target_bins, args, args.out_dir,
            ref_bin=ref_bin, seed_offset=5,
        )
        if r is not None:
            results["Bayesian standard NICE (J=0)"] = r
    if "J5_joint" in args.methods or "J5joint" in args.methods:
        # J=5 with joint shared functional RE on Y AND L equations.
        # Same J=5 spline basis as FRE-NICE primary, but the same b_i additionally
        # drives each L equation via per-L scaling lambda_j.
        # Manuscript footnote: this design was previously labelled "Spec II".
        r = run_fre_nice(
            tuple(args.K5_knots), "fre_nice_J5_joint", cohort,
            target_bins, args, args.out_dir, ref_bin=ref_bin, seed_offset=6,
            j5_joint=True,
        )
        if r is not None:
            results["FRE-NICE J=5 joint (shared RE on Y+L)"] = r

    if results and args.phase != "fit":
        md = _md_table(centers, ref_bin, results)
        (args.out_dir / "bayesian_table2.md").write_text(
            "\n".join(md), encoding="utf-8",
        )
        print(f"\nWrote {args.out_dir / 'bayesian_table2.md'}")


if __name__ == "__main__":
    main()
