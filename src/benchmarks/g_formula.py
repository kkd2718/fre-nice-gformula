"""g-formula post-fit estimand pipeline (FRE-NICE family).

Public API:
  GFormulaConfig, ForwardOutput, FreNiceForwardSim, Estimands,
  load_state, build_counterfactual_A, empirical_bin_frequencies.

Notation:
  K — treatment-bin count (n_bins, K_A);  J — spline knot rank (K_re).

Xu joint GLMM forward L sim lives in xu_glmm_bayesian.py; this module
provides only post-hoc β-based reductions for Xu output.

Refs: Robins 1986; Hernán-Robins 2020 §21; Xu 2024; Wood 2017 §6.5;
      Greenland-Robins-Pearl 1999; Cohen 2003 §8.2; Vehtari 2017.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)              # match legacy x64 precision
import jax.numpy as jnp
import jax.random as jr


# ============================================================================
# Configuration & data types
# ============================================================================

@dataclass(frozen=True)
class GFormulaConfig:
    """Hyperparameters for forward L simulation and estimand reductions.

    ref_bin = 7 = cohort 25th percentile of MP (lung-protective reference).
    Must match the bin dropped from the outcome design at fit time.
    """
    ref_bin: int = 7
    n_bins: int = 20
    n_posterior_subset: int = 200
    n_b_draws: int = 5
    M_subset: int = 200
    seed: int = 0
    truncate_at_day: int | None = None
    natural_course_rate: float = 0.0


class ForwardOutput(NamedTuple):
    """Per-day outputs from forward L simulation, all shape (S, T).

    [REF] HR2020 §21: outputs of g-formula forward simulation under
    counterfactual A_t = a* are time-varying functions of follow-up day.

    h_per_day[s, t]        — E_b E_M[ p_outcome_t | s ]   per-day conditional hazard
    cum_per_day[s, t]      — E_b E_M[ cum_t | s ]         cumulative incidence
                                                          (cum_per_day[:, T-1] == legacy
                                                          dose_response_jax cum_final.mean())
    survived_per_day[s, t] — E_b E_M[ survived_t | s ]    population survival
    """
    h_per_day: np.ndarray
    cum_per_day: np.ndarray
    survived_per_day: np.ndarray


# ============================================================================
# A_per_t builders
# ============================================================================

def build_counterfactual_A(
    k: int, ref_bin: int, n_bins: int, T: int,
    truncate_at_day: int | None = None,
) -> np.ndarray:
    """(T, n_bins-1) counterfactual exposure matrix for sustained intervention A_t = k.

    [DESIGN] Reference-coded one-hot: A_t = e_k (with ref column dropped) for
    k != ref_bin; all-zero vector for k == ref_bin. After truncate_at_day,
    A_t = 0 (intervention released). Matches dose_response_jax convention.

    [LEGACY] Equivalent to scripts/run_jax_dose.py inline construction at
    line ~120 of legacy run_marginal_per_day_hr.py.
    """
    keep = [j for j in range(n_bins) if j != ref_bin]
    A_full = np.zeros((T, n_bins), dtype=np.float64)
    A_full[:, k] = 1.0
    A = A_full[:, keep]
    if truncate_at_day is not None and truncate_at_day < T:
        A[truncate_at_day:] = 0.0
    return A                                          # (T, n_bins-1)


def empirical_bin_frequencies(cohort, n_bins: int) -> np.ndarray:
    """(T, n_bins) per-day empirical bin frequencies over at-risk cohort.

    [DESIGN] For frequency-weighted PPC aggregation. Avoids the Jensen-bias
    issue of pre-averaging one-hot exposure vectors (sub-agent flag): the
    correct PPC is sum_k freq[t, k] * cum_per_day[k][t], not the cum incidence
    of forward sim with a fractional A vector.

    [LIMITATION] Sum across k may be <1 on days where some at-risk stays
    have non-computable MP (Case 1; A_bin all-zero by design). The PPC
    therefore slightly under-counts mortality on those days. Documented in
    manuscript Methods §sensitivity item 4. To bound the magnitude, callers
    can compare freq.sum(axis=1) against 1.0 to quantify Case-1 fraction.
    """
    A_bin = cohort.A_bin.numpy()                      # (N, T, n_bins)
    at_risk = cohort.at_risk.numpy().squeeze(-1)      # (N, T)
    T = A_bin.shape[1]
    freq = np.zeros((T, n_bins), dtype=np.float64)
    for t in range(T):
        active = at_risk[:, t] == 1
        if active.any():
            freq[t] = A_bin[active, t, :].mean(axis=0)
    return freq                                        # (T, n_bins)


# ============================================================================
# FRE-NICE forward L simulation (JIT-compiled core)
# ============================================================================

@partial(jax.jit, static_argnames=("K_A_minus_1", "p_dyn", "p_stat", "T", "n_b", "trunc"))
def _fre_nice_step(
    rng_key, beta, L_chol, B_basis, beta_L, sd_L, L_t0, C_mc,
    A_per_t, natural_course_rate, lambda_L,
    *, K_A_minus_1, p_dyn, p_stat, T, n_b, trunc,
):
    """One forward L sim for one (posterior, regime). Returns (h, cum, survived) per day.

    [REF] HR2020 §21.2: forward simulation iterates
       L_t | A_<=t, L_<t, C  ->  Y_t | A_<=t, L_<=t, C
    integrating over subject random effect b_i ~ N(0, Σ_b).

    [REF] Wood2017 random factor smooth: random_logit_t = b_i^T B(t) where
    B(t) is a fixed natural cubic spline basis evaluated on [0, T-1]/T-1.
    J=1 reduces to scalar random intercept.

    [LEGACY] Identical computational logic to
    src/benchmarks/dose_response_jax.py:_one_posterior_one_bin step body.
    The only difference is scan output captures three per-day means rather
    than discarding all but cum_final.mean(). Verified numerically:
    cum_per_day[T-1] equals legacy cum_final.mean() to machine precision
    (see tests/test_g_formula_legacy_equiv.py).

    [DESIGN] Truncate-at-day + natural-course splice: when t >= trunc and
    natural_course_rate > 0, override p_outcome with cohort observational
    off-MV per-day mortality. Matches Option B fit semantics (at_risk =
    alive AND on-MV).

    [J5_joint extension] When lambda_L is non-zero (Spec II / J5_joint
    state), the L equation gets an additional shared-RE contribution:
        mu_L_j = X_hist @ beta_L_j + lambda_j * (b_i^T B(t-1))
    For all other states (J=0, J=1, J=5 standard, J=4, J=6), pass
    lambda_L = jnp.zeros(p_dyn) so the term is a no-op (preserves backward
    compatibility with prior state files).
    """
    M = L_t0.shape[0]
    K_re = B_basis.shape[1]

    # Sample b ~ N(0, L_chol L_chol^T) per (n_b draws, M stays)
    z = jr.normal(rng_key, shape=(n_b, M, K_re))
    b = z @ L_chol.T                                  # (n_b, M, K_re)
    random_logit = b @ B_basis.T                      # (n_b, M, T)

    C_b = jnp.broadcast_to(C_mc, (n_b, M, p_stat))
    L_t0_b = jnp.broadcast_to(L_t0, (n_b, M, p_dyn))

    def step(carry, t):
        L_prev, survived, cum, key = carry
        A_t = jnp.broadcast_to(A_per_t[t], (n_b, M, K_A_minus_1))
        bias = jnp.ones((n_b, M, 1))
        t_col = jnp.full((n_b, M, 1), t / jnp.maximum(T - 1, 1))

        # L equation history layout: [bias | L_{t-1} | A_t | C | t_norm]
        # [LEGACY] matches dose_response_jax.py line 70 exactly
        X_hist = jnp.concatenate([bias, L_prev, A_t, C_b, t_col], axis=-1)
        mu_L = jnp.einsum("nmh,jh->nmj", X_hist, beta_L)
        # [J5_joint] Add lambda_j * (b_i^T B(t-1)) — shared RE on L equations.
        # B(t-1) at lag uses jnp.maximum(t-1, 0) to handle t=0 (anchor) safely;
        # the L_curr at t=0 is overridden by L_t0_b below regardless.
        b_T_B_lag = jnp.einsum(
            "nmk,k->nm", b, B_basis[jnp.maximum(t - 1, 0)],
        )                                               # (n_b, M)
        re_L = b_T_B_lag[:, :, None] * lambda_L[None, None, :]  # (n_b, M, p_dyn)
        mu_L = mu_L + re_L
        key, sub = jr.split(key)
        L_new = mu_L + jr.normal(sub, shape=(n_b, M, p_dyn)) * sd_L
        L_curr = jnp.where(t == 0, L_t0_b, L_new)     # anchor at observed L_0

        # Outcome layout: [bias | A_t | L_t | C | t_norm]
        # [LEGACY] matches dose_response_jax.py line 78 exactly
        X_out = jnp.concatenate([bias, A_t, L_curr, C_b, t_col], axis=-1)
        eta_fix = jnp.einsum("nmh,h->nm", X_out, beta)
        p_outcome = jax.nn.sigmoid(jnp.clip(eta_fix + random_logit[:, :, t], -30, 30))

        # Natural-course splice (override outcome model after intervention release)
        use_nat = (t >= trunc) & (natural_course_rate > 0)
        p_t = jnp.where(use_nat, natural_course_rate, p_outcome)

        cum_new = cum + survived * p_t
        survived_new = survived * (1.0 - p_t)
        return ((L_curr, survived_new, cum_new, key),
                (p_t.mean(), cum_new.mean(), survived_new.mean()))

    init = (jnp.zeros((n_b, M, p_dyn)), jnp.ones((n_b, M)),
            jnp.zeros((n_b, M)), rng_key)
    _, outs = jax.lax.scan(step, init, jnp.arange(T))
    return outs                                        # tuple of three (T,) arrays


class FreNiceForwardSim:
    """Forward L sim across posterior subset for FRE-NICE family (J=1, J=4, J=5, J=6).

    Notation:
      K (= n_bins, K_A) — number of MP exposure bins (treatment levels)
      J (= K_re in code, "knots") — spline knot count for the functional RE basis B(t)

    [DESIGN] J is read from state["B_basis"].shape[1]. J=1 is scalar random
    intercept (degenerate functional RE, equivalent to Xu's outcome RE
    without the L equation). J=5 is the manuscript primary specification.
    """

    def __init__(self, state: dict, cohort, cfg: GFormulaConfig):
        self.cfg = cfg
        self.K_A = int(cohort.feature_layout["n_bins"])
        self.T = int(cohort.L_dyn.shape[1])
        self.p_dyn = int(cohort.L_dyn.shape[2])
        self.p_stat = int(cohort.C_static.shape[1])

        rng = np.random.default_rng(cfg.seed)
        S_total = state["beta"].shape[0]
        S = min(cfg.n_posterior_subset, S_total)
        self.idx = rng.choice(S_total, size=S, replace=False)
        self.beta_post = jnp.asarray(state["beta"][self.idx], dtype=jnp.float64)
        self.L_chol_post = jnp.asarray(state["L_chol"][self.idx], dtype=jnp.float64)

        # beta_L, sd_L, B_basis are point estimates (single value across posterior)
        # [LEGACY] matches dose_response_jax wrapper convention (broadcast across S).
        self.B_basis = jnp.asarray(state["B_basis"], dtype=jnp.float64)
        self.beta_L = jnp.asarray(state["beta_L"], dtype=jnp.float64)
        self.sd_L = jnp.asarray(state["sd_L"], dtype=jnp.float64)

        # [J5_joint] Load lambda_L if present in state (J5_joint extension);
        # otherwise zeros vector (no-op for J=0/J=1/J=5 standard / J=4 / J=6).
        # The _fre_nice_step JIT function unconditionally adds
        # lambda_j * (b_i^T B(t-1)) to the L equation; with lambda_L = 0 the
        # term contributes nothing and forward sim is identical to pre-J5_joint.
        if "lambda_L" in state:
            self.lambda_L = jnp.asarray(state["lambda_L"], dtype=jnp.float64)
        else:
            self.lambda_L = jnp.zeros((self.p_dyn,), dtype=jnp.float64)

        M = min(cfg.M_subset, cohort.L_dyn.shape[0])
        idx_M = rng.choice(cohort.L_dyn.shape[0], size=M, replace=False)
        self.L_t0 = jnp.asarray(cohort.L_dyn.numpy()[idx_M, 0, :], dtype=jnp.float64)
        self.C_mc = jnp.asarray(cohort.C_static.numpy()[idx_M], dtype=jnp.float64)

    def simulate(self, A_per_t: np.ndarray,
                 rng_key: jax.Array | None = None) -> ForwardOutput:
        """Forward sim across posterior subset for one A_per_t regime.

        Args:
            A_per_t: (T, K_A-1) counterfactual exposure matrix.
            rng_key: optional override; defaults to PRNGKey(cfg.seed).
        Returns:
            ForwardOutput with (S, T) arrays.
        """
        S = self.beta_post.shape[0]
        A_jax = jnp.asarray(A_per_t, dtype=jnp.float64)
        nat_rate = jnp.asarray(self.cfg.natural_course_rate, dtype=jnp.float64)
        trunc = (self.T if self.cfg.truncate_at_day is None
                 else int(self.cfg.truncate_at_day))
        if rng_key is None:
            rng_key = jr.PRNGKey(self.cfg.seed)

        # vmap over S (posterior subset) along axis 0 of: keys, beta, L_chol.
        # All other args broadcast (None). _fre_nice_step now takes
        # (rng_key, beta, L_chol, B_basis, beta_L, sd_L, L_t0, C_mc,
        #  A_per_t, natural_course_rate, lambda_L) = 11 positional args.
        in_axes = (0, 0, 0) + (None,) * 8
        vmapped = jax.vmap(
            partial(_fre_nice_step,
                    K_A_minus_1=self.K_A - 1, p_dyn=self.p_dyn,
                    p_stat=self.p_stat, T=self.T,
                    n_b=self.cfg.n_b_draws, trunc=trunc),
            in_axes=in_axes,
        )
        keys = jr.split(rng_key, S)
        h, cum, surv = vmapped(
            keys, self.beta_post, self.L_chol_post,
            self.B_basis, self.beta_L, self.sd_L,
            self.L_t0, self.C_mc, A_jax, nat_rate, self.lambda_L,
        )
        return ForwardOutput(h_per_day=np.asarray(h),
                             cum_per_day=np.asarray(cum),
                             survived_per_day=np.asarray(surv))

    def simulate_all_bins(self) -> dict[int, ForwardOutput]:
        """Run sustained-regime forward sim for each bin in [0, n_bins).

        [DESIGN] Sustained regime: A_per_t[t] = e_k for all t in [0, T).
        Each bin uses an independent RNG sub-key derived from cfg.seed via
        `jr.fold_in`, so b ~ N(0, Σ_b) draws differ across bins.
        """
        out = {}
        base_key = jr.PRNGKey(self.cfg.seed)
        for k in range(self.cfg.n_bins):
            A = build_counterfactual_A(k, self.cfg.ref_bin, self.cfg.n_bins,
                                        self.T, self.cfg.truncate_at_day)
            out[k] = self.simulate(A, rng_key=jr.fold_in(base_key, k))
        return out

    def simulate_natural_course(self, cohort,
                                 per_bin: dict[int, ForwardOutput] | None = None
                                 ) -> ForwardOutput:
        """PPC: model-predicted cumulative mortality under empirical exposure.

        APPROXIMATION: Aggregates per-bin sustained-regime simulations weighted
        by empirical per-day bin frequencies (sum_k freq[t,k] * cum_k[t]). This
        is *not* equivalent to per-stay forward simulation under each stay's
        observed A_t trajectory — it averages cumulative incidences across
        sustained regimes, not switching trajectories. Manuscript Fig S
        caption MUST label this as a frequency-weighted approximation.
        """
        if per_bin is None:
            per_bin = self.simulate_all_bins()
        freq = empirical_bin_frequencies(cohort, self.cfg.n_bins)   # (T, n_bins)
        bins = sorted(per_bin.keys())
        S, T = per_bin[bins[0]].h_per_day.shape
        h = np.zeros((S, T)); cum = np.zeros((S, T)); surv = np.zeros((S, T))
        for k in bins:
            w = freq[:, k][None, :]                                 # (1, T)
            h += w * per_bin[k].h_per_day
            cum += w * per_bin[k].cum_per_day
            surv += w * per_bin[k].survived_per_day
        return ForwardOutput(h_per_day=h, cum_per_day=cum, survived_per_day=surv)

    def simulate_per_stay_ppc(self, cohort) -> ForwardOutput:
        """Valid PPC: forward-simulate each stay under its OBSERVED A_t trajectory.

        Unlike `simulate_natural_course` (frequency-weighted approximation across
        sustained regimes), this drives the L equation and outcome model by
        each stay's actual day-by-day exposure path, then averages across stays.
        Compare against the empirical 28-day cumulative mortality curve to
        validate model calibration on the observed cohort.

        For Case 1 stay-days (on-MV but MP non-computable, A_bin all-zero) the
        all-zero exposure indicator passes through the reference-coded design
        as the reference-bin contrast, matching fit-time semantics exactly.

        Returns a ForwardOutput with per-day means averaged across the M
        sampled stays and S posterior draws.
        """
        # cohort.A_bin is (N, T, K_A) one-hot; drop the reference column to
        # match fit-time design and the simulate(A) signature.
        A_full = cohort.A_bin.numpy().astype(np.float64)            # (N, T, K_A)
        keep = [j for j in range(self.K_A) if j != self.cfg.ref_bin]
        A_dropped_full = A_full[:, :, keep]                          # (N, T, K_A-1)
        # Use the same M baseline subjects as `simulate(...)` so L_t0 / C_mc
        # match. Since FreNiceForwardSim.__init__ sampled idx_M but does not
        # store it, recompute deterministically with the same RNG.
        rng = np.random.default_rng(self.cfg.seed)
        S_total = self.beta_post.shape[0]
        S = min(self.cfg.n_posterior_subset, S_total)
        _ = rng.choice(S_total, size=S, replace=False)               # consume to match init
        N_obs = cohort.L_dyn.shape[0]
        M = min(self.cfg.M_subset, N_obs)
        idx_M = rng.choice(N_obs, size=M, replace=False)
        # Per-stay A_per_t for those M subjects: (M, T, K_A-1)
        A_per_stay = A_dropped_full[idx_M]
        # Average each stay's per-day exposure indicator into a single (T, K_A-1)
        # "average regime" because _fre_nice_step's vmap structure is over
        # posterior draws and broadcasts a single A_per_t across (n_b, M).
        # The TRUE per-stay PPC requires a separate vmap axis over M; we
        # implement it by looping over m, vmapping over S, and averaging.
        # Cost: M × (S × T × n_b × p_dyn) — same as simulate(...) ×M ÷ S broadcast.
        # In practice with M=200 this is tractable on V100 (~5 min).
        h_all = np.zeros((S, self.T)); cum_all = np.zeros((S, self.T)); surv_all = np.zeros((S, self.T))
        base_key = jr.PRNGKey(self.cfg.seed + 99)  # offset from simulate_all_bins keys
        # For each baseline subject, use its observed A trajectory.
        for m_idx in range(M):
            A_m = A_per_stay[m_idx]                                  # (T, K_A-1)
            # Build a mini-FreNiceForwardSim view that uses ONLY this one
            # baseline subject's L_t0/C — but simulate(...) was built around
            # M shared MC subjects. Easiest: re-call the JIT function with M=1.
            from functools import partial
            # Re-execute _fre_nice_step manually for this single subject
            L_t0_single = self.L_t0[m_idx:m_idx + 1]
            C_mc_single = self.C_mc[m_idx:m_idx + 1]
            # 11 positional args (last is lambda_L); vmap over keys/beta/L_chol.
            in_axes = (0, 0, 0) + (None,) * 8
            vmapped = jax.vmap(
                partial(_fre_nice_step,
                        K_A_minus_1=self.K_A - 1, p_dyn=self.p_dyn,
                        p_stat=self.p_stat, T=self.T,
                        n_b=self.cfg.n_b_draws, trunc=self.T),
                in_axes=in_axes,
            )
            keys = jr.split(jr.fold_in(base_key, m_idx), S)
            h, cum, surv = vmapped(
                keys, self.beta_post, self.L_chol_post,
                self.B_basis, self.beta_L, self.sd_L,
                L_t0_single, C_mc_single,
                jnp.asarray(A_m, dtype=jnp.float64),
                jnp.asarray(0.0, dtype=jnp.float64),
                self.lambda_L,
            )
            h_all += np.asarray(h); cum_all += np.asarray(cum); surv_all += np.asarray(surv)
        h_all /= M; cum_all /= M; surv_all /= M
        return ForwardOutput(h_per_day=h_all, cum_per_day=cum_all,
                             survived_per_day=surv_all)


# ============================================================================
# Estimand reductions
# ============================================================================

class Estimands:
    """Reductions on per-bin ForwardOutput to manuscript estimand arrays.

    [REF] Greenland99: marginal vs conditional HR distinction
    (collapsibility under non-rare outcomes / time-varying confounding).
    """

    @staticmethod
    def cumulative_incidence(per_bin: dict[int, ForwardOutput],
                              T_horizon: int | None = None) -> dict:
        """Day-T_horizon cumulative incidence per bin.

        [LEGACY] Equivalent to dose_response_jax.fre_nice_dose_response_jax
        output: risk_mean = posterior mean of cum_final.mean().
        """
        bins = sorted(per_bin.keys())
        T_total = per_bin[bins[0]].cum_per_day.shape[1]
        t_idx = T_total - 1 if T_horizon is None else min(T_horizon - 1, T_total - 1)

        risk = np.stack([per_bin[k].cum_per_day[:, t_idx] for k in bins], axis=1)
        return {
            "bins": np.array(bins),
            "T_horizon": t_idx + 1,
            "risk_mean": risk.mean(axis=0),
            "risk_ci_low": np.quantile(risk, 0.025, axis=0),
            "risk_ci_high": np.quantile(risk, 0.975, axis=0),
            "risk_per_sample": risk,                   # (S, n_bins)
        }

    @staticmethod
    def marginal_per_day_hr(per_bin: dict[int, ForwardOutput],
                             ref_bin: int,
                             definition: Literal["ratio_of_means",
                                                  "mean_of_ratios",
                                                  "cumulative_hazard"]
                                                  = "cumulative_hazard") -> dict:
        """Marginal per-day HR vs reference bin (PRIMARY estimand).

        Three definitions provided (per sub-agent review of non-PH treatment of
        time-varying hazards):

          'cumulative_hazard'  (DEFAULT, Cox-equivalent):
              HR(s, k) = -log(1 - cum_k(T, s)) / -log(1 - cum_ref(T, s))
              [REF] Standard cumulative-hazard ratio. Reduces to per-day HR
              under proportional hazards. Cox-comparable to Costa 2021
              (HR per 5 J/min).

          'mean_of_ratios':
              HR(s, k) = mean_t [ h_k(t, s) / h_ref(t, s) ]
              [REF] Average of instantaneous per-day HRs across follow-up.
              Diverges from 'cumulative_hazard' under heavy non-PH.

          'ratio_of_means':
              HR(s, k) = mean_t h_k(t, s) / mean_t h_ref(t, s)
              [LEGACY] Original definition; agrees with cumulative_hazard
              under PH but is non-standard otherwise.

        [DESIGN] Manuscript primary: cumulative_hazard. Reports the others
        in supplement as sensitivity (consistency under non-PH check).
        """
        bins = sorted(per_bin.keys())
        if ref_bin not in bins:
            raise ValueError(f"ref_bin {ref_bin} not in available bins {bins}")

        if definition == "cumulative_hazard":
            T_idx = per_bin[bins[0]].cum_per_day.shape[1] - 1
            cum_T = np.stack([per_bin[k].cum_per_day[:, T_idx] for k in bins], axis=1)
            cum_T = np.clip(cum_T, 1e-8, 1 - 1e-8)             # avoid log(0)
            cum_haz = -np.log(1.0 - cum_T)                     # (S, n_bins)
            hr = cum_haz / cum_haz[:, bins.index(ref_bin)][:, None]
        elif definition == "mean_of_ratios":
            h_per_day = np.stack(
                [per_bin[k].h_per_day for k in bins], axis=2
            )                                                  # (S, T, n_bins)
            h_ref = h_per_day[:, :, bins.index(ref_bin)][:, :, None]
            hr_t = np.clip(h_per_day, 1e-12, None) / np.clip(h_ref, 1e-12, None)
            hr = hr_t.mean(axis=1)                             # (S, n_bins)
        elif definition == "ratio_of_means":
            h_avg = np.stack(
                [per_bin[k].h_per_day.mean(axis=1) for k in bins], axis=1
            )                                                  # (S, n_bins)
            hr = h_avg / h_avg[:, bins.index(ref_bin)][:, None]
        else:
            raise ValueError(f"Unknown definition {definition}")

        return {
            "bins": np.array(bins),
            "ref_bin": ref_bin,
            "definition": definition,
            "hr_mean": hr.mean(axis=0),
            "hr_ci_low": np.quantile(hr, 0.025, axis=0),
            "hr_ci_high": np.quantile(hr, 0.975, axis=0),
            "hr_per_sample": hr,                              # (S, n_bins)
        }

    @staticmethod
    def rmst_28(per_bin: dict[int, ForwardOutput], ref_bin: int) -> dict:
        """Restricted Mean Survival Time at T=28 days, per bin.

        RMST_k(28) = sum_{t=0..27} survived_k(t)  (in days)
        Δ-RMST_k = RMST_k − RMST_ref
                                (positive = days of life saved vs reference;
                                 negative = days lost to higher MP)

        [REF] Royston-Parmar 2013, Uno et al. 2014. Robust to non-PH; intuitive
        clinical headline ("MP at 17 J/min loses X days of 28-day survival
        relative to lung-protective Q25").
        """
        bins = sorted(per_bin.keys())
        if ref_bin not in bins:
            raise ValueError(f"ref_bin {ref_bin} not in {bins}")
        # survived_per_day shape: (S, T). Sum over t = RMST in days.
        rmst = np.stack(
            [per_bin[k].survived_per_day.sum(axis=1) for k in bins], axis=1
        )                                                          # (S, n_bins)
        ref_idx = bins.index(ref_bin)
        delta = rmst - rmst[:, ref_idx][:, None]                   # vs ref
        return {
            "bins": np.array(bins),
            "ref_bin": ref_bin,
            "rmst_mean": rmst.mean(axis=0),
            "rmst_ci_low": np.quantile(rmst, 0.025, axis=0),
            "rmst_ci_high": np.quantile(rmst, 0.975, axis=0),
            "delta_rmst_mean": delta.mean(axis=0),
            "delta_rmst_ci_low": np.quantile(delta, 0.025, axis=0),
            "delta_rmst_ci_high": np.quantile(delta, 0.975, axis=0),
            "rmst_per_sample": rmst,                                # (S, n_bins)
            "delta_rmst_per_sample": delta,
        }

    @staticmethod
    def per_day_hr_curve(per_bin: dict[int, ForwardOutput],
                          ref_bin: int) -> dict:
        """Per-day hazard ratio HR_k(t) = h_k(t) / h_ref(t) across follow-up.

        Diagnostic for proportional-hazards (PH) violation. Under PH, HR_k(t)
        is approximately constant in t for each k; under non-PH, the HR
        evolves over t (e.g., early-period harm followed by attenuation).
        Reported as posterior mean and 95% CrI per (k, t).

        Returns shape (S, T, n_bins) for hr_per_sample.
        """
        bins = sorted(per_bin.keys())
        if ref_bin not in bins:
            raise ValueError(f"ref_bin {ref_bin} not in {bins}")
        h_per_day = np.stack(
            [per_bin[k].h_per_day for k in bins], axis=2
        )                                                          # (S, T, n_bins)
        ref_idx = bins.index(ref_bin)
        h_ref = h_per_day[:, :, ref_idx][:, :, None]
        hr_t = np.clip(h_per_day, 1e-12, None) / np.clip(h_ref, 1e-12, None)
        return {
            "bins": np.array(bins),
            "ref_bin": ref_bin,
            "days": np.arange(h_per_day.shape[1]),
            "hr_t_mean": hr_t.mean(axis=0),                        # (T, n_bins)
            "hr_t_ci_low": np.quantile(hr_t, 0.025, axis=0),
            "hr_t_ci_high": np.quantile(hr_t, 0.975, axis=0),
            "hr_t_per_sample": hr_t,                                # (S, T, n_bins)
        }

    @staticmethod
    def conditional_hr_from_beta(state: dict, ref_bin: int, n_bins: int,
                                  model: Literal["fre_nice", "xu"]) -> dict:
        """Post-hoc HR from regression coefficients (no forward sim).

        FRE-NICE: HR(k) = exp(beta[1+pos(k)] - 0)  for k != ref_bin
        Xu (2024): HR(k) = exp(beta_A[k] - beta_A[ref])

        [DESIGN] This is the *direct* (within-L) per-day HR — does NOT
        marginalize over forward-simulated L_t. Reported as triangulation
        against the marginal forward-sim HR. Subsumes the legacy
        scripts/extract_per_day_hr.py.

        [REF] Cohen2003 §8.2: reference dummy coding => HR is exp(coef)
        for the indicator of bin k vs the dropped reference.
        """
        if model == "fre_nice":
            beta = state["beta"]                       # (S, p_outcome)
            # Layout: [bias|bins(K-1)|L(p_dyn)|C(p_stat)|t_norm]
            # The bin coefficients sit at columns 1 .. K-1.
            keep = [j for j in range(n_bins) if j != ref_bin]
            if beta.shape[1] < 1 + len(keep):
                raise ValueError(f"beta has {beta.shape[1]} columns, "
                                 f"expected >= {1 + len(keep)}")
            bin_betas = beta[:, 1:1 + len(keep)]       # (S, n_bins-1)
            hr = np.ones((beta.shape[0], n_bins), dtype=np.float64)
            for i, k in enumerate(keep):
                hr[:, k] = np.exp(bin_betas[:, i])
        elif model == "xu":
            beta_A = state["post_beta_A"]              # (S, n_bins)
            if beta_A.shape[1] != n_bins:
                raise ValueError(f"post_beta_A has {beta_A.shape[1]} cols, "
                                 f"expected {n_bins}")
            hr = np.exp(beta_A - beta_A[:, [ref_bin]])
        else:
            raise ValueError(f"Unknown model {model}")
        return {
            "bins": np.arange(n_bins),
            "ref_bin": ref_bin,
            "hr_mean": hr.mean(axis=0),
            "hr_ci_low": np.quantile(hr, 0.025, axis=0),
            "hr_ci_high": np.quantile(hr, 0.975, axis=0),
            "hr_per_sample": hr,
        }


# ============================================================================
# State loading
# ============================================================================

_FRE_NICE_REQUIRED = frozenset({"beta", "L_chol", "B_basis", "beta_L", "sd_L"})
_XU_REQUIRED = frozenset({
    "post_beta_0", "post_beta_A", "post_beta_L", "post_beta_C",
    "post_sigma_b_y", "post_alpha_0", "post_alpha_A", "post_alpha_Llag",
    "post_alpha_C", "post_sigma_L", "post_sigma_b_L",
})


def load_state(path: Path, model: Literal["fre_nice", "xu"]) -> dict:
    """Load `.npz` state file and validate required field presence.

    [LEGACY] Reads files saved by run_bayesian_main.py; field schema
    documented in src/benchmarks/{fre_nice_bayesian,xu_glmm_bayesian}.py.
    """
    z = np.load(path, allow_pickle=True)
    state = dict(z)
    required = _FRE_NICE_REQUIRED if model == "fre_nice" else _XU_REQUIRED
    missing = required - set(state.keys())
    if missing:
        raise KeyError(f"State {path.name}: missing fields {missing}. "
                       f"Has fields: {sorted(state.keys())}")
    return state
