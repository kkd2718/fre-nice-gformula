"""Functional Random-Effect NICE g-formula — Bayesian implementation.

Bayesian extension of Robins (1986) parametric g-formula with subject-specific
time-varying random effect on the outcome hazard, parameterized via fixed
natural-cubic-spline basis. Generalizes Xu 2024's scalar random intercept
to a finite-rank Gaussian process random effect on day t.

Generative model
----------------
Outcome:
    beta, eta, xi   ~  N(0, 5^2)
    L_corr          ~  LKJCholesky(K_re, eta = 2)
    tau_k           ~  HalfCauchy(0, 2.5),  k = 1, ..., K_re
    L_chol          =  diag(tau) @ L_corr               (Cholesky of Sigma_b)
    b_i | L_chol    ~  N(0, L_chol L_chol^T)            via non-centered z_i
    logit P(Y_it=1) =  beta_0 + beta_A^T A_it + eta^T L_it + xi^T V_i
                       + b_i^T B(t)

L equations (NICE forward simulation):
    Pooled linear regression (frequentist; same as Standard parametric g-formula)
    fit on observed (L_{t-1}, A_{t-1}, V) -> L_t;  no random effect on L.

Counterfactual under intervention a*
------------------------------------
For each posterior draw s:
    Use (beta^s, eta^s, xi^s, L_chol^s) and per-subject b_i^s
    For each subject i (or sampled MC subjects):
        Sample baseline (L_0, V) from observed empirical
        For t = 1, ..., T-1:
            Forward-sim L_t from pooled regression at A=a*
        For t = 0, ..., T-1:
            logit_t = ... + b_i^s · B(t)  (functional RE contribution)
        Accumulate cumulative hazard.
Posterior mean + 95% credible interval over draws.

Why Bayesian
------------
Consistency with Xu 2024 inference framework (also Bayesian MCMC). LKJ +
half-Cauchy priors prevent the sigma_b collapse seen in frequentist Laplace
+ MoM. Replaces frequentist Laplace ML in spline_glmm_nice.py for primary
analysis (Laplace retained for LOCO sensitivity speed).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoMultivariateNormal
from numpyro import optim as numpyro_optim

from ..data.ards import ARDSCohort
from .base import BenchmarkMethod, DoseResponseResult, bin_centers_J_min
from .standard_gformula import _fit_linear
from ..models.spline_glmm import natural_cubic_basis


def _no_re_model(
    X_outcome: jnp.ndarray,
    y: jnp.ndarray, mask: jnp.ndarray,
    group_idx: jnp.ndarray, n_groups: int,
    record_loglik: bool = False,
) -> None:
    """Bayesian standard NICE — no random effects (J=0 baseline).

    Identical fixed-effect specification to FRE-NICE primary but with
    b_i ≡ 0 (no subject random effect on outcome). Same priors β ~ N(0, 5²),
    same outcome design [bias|bins\ref|L|C|t_norm], same likelihood.

    Used to demonstrate the methodological value of the functional random
    effect: under imperfect TV-confounder adjustment (LOCO sensitivity),
    a no-RE model cannot absorb residual unmeasured between-subject
    heterogeneity, so the HR estimate is more strongly perturbed by
    covariate removal than the FRE-NICE J=5 estimate. The contrast
    isolates the contribution of the RE in robustness to imperfect
    adjustment.
    """
    p = X_outcome.shape[1]
    beta = numpyro.sample("beta", dist.Normal(jnp.zeros(p), 5.0))
    logit = X_outcome @ beta
    log_p = mask * dist.Bernoulli(logits=logit).log_prob(y)
    if record_loglik:
        ll_subject = jax.ops.segment_sum(
            log_p, group_idx, num_segments=n_groups,
        )
        numpyro.deterministic("log_lik_subject", ll_subject)
    numpyro.factor("loglik", log_p.sum())


def _j5_joint_model(
    # Y outcome inputs
    X_outcome_y: jnp.ndarray,            # (N*T, p_y)
    b_basis_per_obs_y: jnp.ndarray,      # (N*T, K_re), B(t) per Y observation
    y: jnp.ndarray,                      # (N*T,)
    mask_y: jnp.ndarray,                 # (N*T,) at-risk indicator for Y
    group_idx_y: jnp.ndarray,            # (N*T,) subject index
    # L equation inputs (stacked over t=1..T-1, dim p_dyn)
    X_hist_L: jnp.ndarray,               # (N_L, p_hist) L regression design
    L_obs: jnp.ndarray,                  # (N_L, p_dyn) observed L_t values
    mask_L: jnp.ndarray,                 # (N_L,) weight = at_risk_{t-1} * at_risk_t
    group_idx_L: jnp.ndarray,            # (N_L,) subject index per L row
    b_basis_per_obs_L: jnp.ndarray,      # (N_L, K_re), B(t-1) per L observation (lag)
    # Common
    n_groups: int, K_re: int, p_dyn: int,
    sigma_b_prior: str = "halfcauchy",
    record_loglik: bool = False,
) -> None:
    """Bayesian Spec II — joint outcome + L equations with shared functional RE.

    Same subject-level functional RE b_i ∈ R^{K_re} drives:
      Y outcome:  logit(p_{i,t}) = X_outcome_y[i,t] β + b_i^T B(t)
      L_j eq.   :  L_{i,t,j}     = X_hist_L[i,t] β_L,j + λ_j (b_i^T B(t-1)) + σ_L,j ε

    Identification: λ_j is the per-L scaling that determines how much of the
    Y-driven RE leaks into L_j's trajectory. λ_j = 0 ⇒ L equation reduces to
    standard FRE-NICE (no L RE). Identifiable because b_i is anchored on the
    Y likelihood (LKJ-correlated tau scale + n_groups subjects each with T
    Y-obs). λ_j is then estimated from L equation residual variance after
    accounting for X_hist_L β_L,j fixed effects.

    Contrast with Xu joint GLMM: Xu has *independent* scalar RE per equation
    (b_i^Y, b_i^{L,1}, ..., b_i^{L,p_dyn} all separately Gaussian), totalling
    n_groups × (1 + p_dyn) RE samples. Spec II has *shared* functional RE,
    n_groups × K_re samples + p_dyn λ scalars. More parsimonious; tests
    whether Y-driving heterogeneity also drives L heterogeneity.

    Refs: Wood (2017) §6.5 random factor smooth; this model is the joint
    (Y, L) generalization within the same b_i.
    """
    # ---- Y coefficients ----
    p_y = X_outcome_y.shape[1]
    beta = numpyro.sample("beta", dist.Normal(jnp.zeros(p_y), 5.0))

    # ---- Shared RE structure (same priors as J=5) ----
    if sigma_b_prior == "halfcauchy":
        tau = numpyro.sample("tau", dist.HalfCauchy(jnp.full((K_re,), 2.5)))
    elif sigma_b_prior == "gamma":
        tau = numpyro.sample(
            "tau", dist.Gamma(jnp.full((K_re,), 2.0), jnp.full((K_re,), 0.5)),
        )
    elif sigma_b_prior == "invgamma":
        tau = numpyro.sample(
            "tau", dist.InverseGamma(
                jnp.full((K_re,), 2.0), jnp.full((K_re,), 1.0),
            ),
        )
    else:
        raise ValueError(f"Unknown sigma_b_prior: {sigma_b_prior}")

    if K_re == 1:
        L_chol = numpyro.deterministic("L_chol", tau.reshape(1, 1))
    else:
        L_corr = numpyro.sample(
            "L_corr", dist.LKJCholesky(K_re, concentration=2.0),
        )
        L_chol = numpyro.deterministic(
            "L_chol", jnp.expand_dims(tau, -1) * L_corr,
        )
    z = numpyro.sample(
        "z_b", dist.Normal(jnp.zeros((n_groups, K_re)), 1.0),
    )
    b = numpyro.deterministic("b", z @ L_chol.T)              # (n_groups, K_re)

    # ---- L equation coefficients ----
    p_hist = X_hist_L.shape[1]
    beta_L = numpyro.sample(
        "beta_L_bayes", dist.Normal(jnp.zeros((p_dyn, p_hist)), 5.0),
    )                                                          # (p_dyn, p_hist)
    sigma_L = numpyro.sample(
        "sigma_L_bayes", dist.HalfCauchy(jnp.ones(p_dyn) * 2.5),
    )                                                          # (p_dyn,)
    # Per-L lambda — N(0, 0.5) tighter prior centered on the "no shared RE"
    # null hypothesis. A Normal(0, 1) prior was found to admit a sign-flip
    # multimodality (b_i ↔ -b_i, lambda_j ↔ -lambda_j) when Y likelihood is
    # weak (binary outcome with low event rate); shrinking toward zero
    # mitigates this and biases the model toward the parsimonious J=5 (no
    # L RE) sub-specification, which is a desirable Bayesian regularization
    # for an ablation analysis.
    lambda_L = numpyro.sample(
        "lambda_L", dist.Normal(jnp.zeros(p_dyn), 0.5),
    )                                                          # (p_dyn,)

    # ---- Y likelihood ----
    re_contrib_y = jnp.sum(b[group_idx_y] * b_basis_per_obs_y, axis=-1)  # (N*T,)
    logit_y = X_outcome_y @ beta + re_contrib_y
    log_p_y = mask_y * dist.Bernoulli(logits=logit_y).log_prob(y)

    # ---- L equation likelihood ----
    # re_b_T_B[i] = b[group_idx_L[i]] · B(t_i - 1)   shape (N_L,)
    re_b_T_B = jnp.sum(b[group_idx_L] * b_basis_per_obs_L, axis=-1)
    # re_L_per_obs[i, j] = lambda_j * re_b_T_B[i]    shape (N_L, p_dyn)
    re_L_per_obs = re_b_T_B[:, None] * lambda_L[None, :]
    # mu_L[i, j] = X_hist_L[i] @ beta_L[j] + re_L_per_obs[i, j]
    mu_L = X_hist_L @ beta_L.T + re_L_per_obs                  # (N_L, p_dyn)
    log_p_L = mask_L[:, None] * dist.Normal(mu_L, sigma_L).log_prob(L_obs)
    # Aggregate L log_lik per row (sum across p_dyn dims)
    log_p_L_per_row = log_p_L.sum(axis=-1)                     # (N_L,)

    if record_loglik:
        # Cluster-level log-lik (subject) combining Y and L contributions.
        # ll_y_subject[i] = sum_t log_p_y of subject i
        # ll_L_subject[i] = sum_t sum_j log_p_L of subject i (per-row sum already)
        ll_y_subject = jax.ops.segment_sum(
            log_p_y, group_idx_y, num_segments=n_groups,
        )
        ll_L_subject = jax.ops.segment_sum(
            log_p_L_per_row, group_idx_L, num_segments=n_groups,
        )
        ll_subject = ll_y_subject + ll_L_subject
        numpyro.deterministic("log_lik_subject", ll_subject)

    numpyro.factor("loglik_y", log_p_y.sum())
    numpyro.factor("loglik_L", log_p_L.sum())


def _fre_nice_model(
    X_outcome: jnp.ndarray, b_basis_per_obs: jnp.ndarray,
    y: jnp.ndarray, mask: jnp.ndarray,
    group_idx: jnp.ndarray, n_groups: int, K_re: int,
    sigma_b_prior: str = "halfcauchy",
    record_loglik: bool = False,
) -> None:
    """numpyro model for FRE-NICE outcome layer.

    X_outcome      : (N_obs, p)   design (intercept + A + L + V, no t-effect)
    b_basis_per_obs: (N_obs, K_re) spline basis evaluated at each row's t

    sigma_b_prior  : prior on tau (per-basis-dim scale of subject RE).
                     "halfcauchy" (default, scale 2.5),
                     "gamma" (Gamma(2, 0.5) — concentration & rate),
                     "invgamma" (InverseGamma(2, 1)).
                     Used for Bayesian sensitivity analysis.
    record_loglik  : if True, record per-observation log_lik as
                     numpyro.deterministic for WAIC / PSIS-LOO computation.
    """
    p = X_outcome.shape[1]
    beta = numpyro.sample("beta", dist.Normal(jnp.zeros(p), 5.0))
    # Sigma_b parameterized via LKJ correlation + scale prior.
    # K_re == 1 special-case: LKJCholesky requires dim >= 2.
    if sigma_b_prior == "halfcauchy":
        tau = numpyro.sample("tau", dist.HalfCauchy(jnp.full((K_re,), 2.5)))
    elif sigma_b_prior == "gamma":
        tau = numpyro.sample(
            "tau", dist.Gamma(jnp.full((K_re,), 2.0), jnp.full((K_re,), 0.5)),
        )
    elif sigma_b_prior == "invgamma":
        tau = numpyro.sample(
            "tau", dist.InverseGamma(
                jnp.full((K_re,), 2.0), jnp.full((K_re,), 1.0),
            ),
        )
    else:
        raise ValueError(f"Unknown sigma_b_prior: {sigma_b_prior}")
    if K_re == 1:
        L_chol = numpyro.deterministic("L_chol", tau.reshape(1, 1))
    else:
        L_corr = numpyro.sample(
            "L_corr", dist.LKJCholesky(K_re, concentration=2.0),
        )
        L_chol = numpyro.deterministic(
            "L_chol", jnp.expand_dims(tau, -1) * L_corr,
        )
    # Non-centered hierarchical: b_i = L_chol @ z_i
    z = numpyro.sample(
        "z_b", dist.Normal(jnp.zeros((n_groups, K_re)), 1.0),
    )
    b = numpyro.deterministic("b", z @ L_chol.T)              # (n_groups, K_re)
    # Per-observation random contribution: b_{group_idx} · B(t_obs)
    re_contrib = jnp.sum(b[group_idx] * b_basis_per_obs, axis=-1)
    logit = X_outcome @ beta + re_contrib
    log_p = mask * dist.Bernoulli(logits=logit).log_prob(y)
    if record_loglik:
        # Aggregate to subject (cluster) level. Per-observation log_lik
        # would be (S, ~500k) which OOMs the GPU; cluster-level (S, n_groups)
        # is the appropriate scale for clustered-data WAIC anyway
        # (Vehtari 2017 §4 — leave-one-cluster-out).
        ll_subject = jax.ops.segment_sum(
            log_p, group_idx, num_segments=n_groups,
        )
        numpyro.deterministic("log_lik_subject", ll_subject)
    numpyro.factor("loglik", log_p.sum())


@dataclass
class FRENICEBayesianConfig:
    """Hyperparameters for FRE-NICE Bayesian benchmark.

    inference: "nuts" (full posterior) or "svi" (variational, faster).

    ref_bin: MP bin index dropped from outcome design one-hot to avoid
    bias-vs-bins collinearity. Default 7 = cohort 25th percentile of MP.

    share_RE_on_L: if True, L equations are augmented with an extra column
      lambda_j * b_i^T B(t), where b_i is the same FRE that drives the outcome.
      This is Spec ② (shared RE across Y and L), an ablation against the
      default Spec ① (Y-only RE). Used to test whether mismatch between
      RE-aware Y model and RE-blind OLS L equations causes systematic
      forward-simulation drift (positive bias vs cohort raw natural course).
      Implementation: two-stage. NUTS Y fit -> extract b_hat (posterior
      mean per subject) -> refit L equations with extra column
      b_hat[i]^T B(t). Forward simulation uses sampled b for each MC draw.
    """
    knots: tuple[float, ...] = (0.0, 3.0, 7.0, 14.0, 21.0)
    inference: str = "nuts"
    ref_bin: int = 7
    share_RE_on_L: bool = False         # legacy two-stage approximation
    j5_joint: bool = False         # NEW: full joint Bayesian (formerly Spec II)
    sigma_b_prior: str = "halfcauchy"   # halfcauchy | gamma | invgamma
    record_loglik: bool = False         # for WAIC / PSIS-LOO
    holdout_subj_ids: tuple[int, ...] | None = None  # for PPC: subjects to mask out of fit
    # NUTS
    n_warmup: int = 1000
    n_samples: int = 1000
    n_chains: int = 4
    chain_method: str = "parallel"
    target_accept: float = 0.9
    # SVI
    svi_steps: int = 8000
    svi_lr: float = 5e-3
    svi_n_posterior_draws: int = 2000
    # Counterfactual
    n_posterior_subset: int = 200
    n_b_draws_per_post: int = 5
    n_mc_subjects: int | None = None
    l2_L: float = 1e-4
    seed: int = 0


class FRENICEBayesianBenchmark(BenchmarkMethod):
    """Bayesian FRE-NICE g-formula via numpyro NUTS."""

    method_name = "fre_nice_bayesian"

    def __init__(self, config: FRENICEBayesianConfig | None = None) -> None:
        self.config = config or FRENICEBayesianConfig()
        # State after fit
        self._posterior: dict | None = None
        self._B_basis: np.ndarray | None = None        # (T, K_re)
        self._beta_L: list[np.ndarray] = []
        self._sd_L: list[float] = []
        self._n_groups: int = 0
        self._n_bins: int = 0
        self._n_dyn: int = 0
        self._n_static: int = 0
        self._t_max: int = 0
        # Spec ② (share_RE_on_L) state: posterior-mean RE per subject
        self._b_hat: np.ndarray | None = None          # (n_groups, K_re) or None
        self._L_has_RE_col: bool = False               # True if β_L includes lambda_j
        self._lambda_L: np.ndarray | None = None       # (p_dyn,) Spec ② per-L lambda
        # Diagnostics (R-hat max, ESS bulk min, ESS tail min, divergent count)
        self._diagnostics: dict | None = None
        # Per-observation log_lik samples (S, N_obs) when record_loglik=True
        self._log_lik: np.ndarray | None = None

    # ----- design assembly -----
    def _drop_ref_bins(self, cov: np.ndarray, K_A: int) -> np.ndarray:
        """Drop the reference-bin column from a (NT, K_A + others) design's bin block."""
        ref = self.config.ref_bin
        if ref is None or not (0 <= ref < K_A):
            return cov
        keep_bins = [k for k in range(K_A) if k != ref]
        cov_bins = cov[:, keep_bins]                               # (NT, K_A-1)
        cov_other = cov[:, K_A:]
        return np.concatenate([cov_bins, cov_other], axis=1)

    def _build_outcome_design(
        self, cohort: ARDSCohort, override_bin: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build outcome-stage features. Returns
        (X_outcome, B_per_obs, y, mask, group_idx)."""
        L = cohort.feature_layout
        K_A = L["n_bins"]
        N, T = cohort.Y.shape[0], cohort.Y.shape[1]
        cov = cohort.covariates.numpy().reshape(N * T, -1).astype(np.float64)
        y = cohort.Y.numpy().reshape(N * T)
        m = cohort.at_risk.numpy().reshape(N * T)
        if override_bin is not None:
            cov = cov.copy()
            cov[:, :K_A] = 0.0
            cov[:, override_bin] = 1.0
        # Drop reference bin column to avoid bias-vs-bins collinearity
        cov = self._drop_ref_bins(cov, K_A)
        bias = np.ones((cov.shape[0], 1), dtype=np.float64)
        X_outcome = np.concatenate([bias, cov], axis=1)
        # B basis tiled to each observation's t
        B = self._B_basis                                          # (T, K_re)
        B_per_obs = np.tile(B, (N, 1))                             # (N*T, K_re)
        _, inv = np.unique(cohort.subject_ids, return_inverse=True)
        group_idx = np.repeat(inv, T)
        return (
            X_outcome.astype(np.float64), B_per_obs.astype(np.float64),
            y.astype(np.float64), m.astype(np.float64), group_idx,
        )

    def _drop_ref_in_A(self, A_onehot: np.ndarray) -> np.ndarray:
        """Drop reference bin column from (N, K_A) one-hot."""
        ref = self.config.ref_bin
        if ref is None or A_onehot.shape[1] <= 1:
            return A_onehot
        return np.delete(A_onehot, ref, axis=1)

    def _build_history_features(
        self, L_prev: np.ndarray, A_prev: np.ndarray, C: np.ndarray,
        t_idx: int, T: int,
    ) -> np.ndarray:
        N = L_prev.shape[0]
        bias = np.ones((N, 1), dtype=np.float64)
        t_col = np.full((N, 1), t_idx / max(T - 1, 1), dtype=np.float64)
        A_dropped = self._drop_ref_in_A(A_prev)
        return np.concatenate([bias, L_prev, A_dropped, C, t_col], axis=1)

    def _build_L_design_for_j5_joint(
        self, cohort: ARDSCohort,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build stacked L-equation design for joint Spec II.

        For each transition t = 1, .., T-1, extract:
          X_hist_L row    = [bias | L_{t-1} | A_{t-1}\\ref | C | t_norm]
          L_obs row       = L_t observed
          mask_L row      = at_risk_{t-1} * at_risk_t
          group_idx_L row = subject index of stay
          B_lag_per_obs   = B(t-1) row of spline basis

        Returns: (X_hist_L, L_obs, mask_L, group_idx_L, B_lag_per_obs).
        Shape: (N*(T-1), p_hist), (N*(T-1), p_dyn), (N*(T-1),),
               (N*(T-1),), (N*(T-1), K_re).
        """
        T = self._t_max
        p_dyn = self._n_dyn
        L_dyn = cohort.L_dyn.numpy().astype(np.float64)            # (N, T, p_dyn)
        A_bin = cohort.A_bin.numpy().astype(np.float64)            # (N, T, K_A)
        C_static = cohort.C_static.numpy().astype(np.float64)      # (N, p_stat)
        at_risk = cohort.at_risk.numpy().astype(np.float64).squeeze(-1)
        _, inv = np.unique(cohort.subject_ids, return_inverse=True)
        N = L_dyn.shape[0]
        K_re = self._B_basis.shape[1]

        rows_X, rows_L, rows_mask, rows_grp, rows_B = [], [], [], [], []
        for t in range(1, T):
            X_t = self._build_history_features(
                L_dyn[:, t - 1, :], A_bin[:, t - 1, :], C_static, t_idx=t, T=T,
            )                                                       # (N, p_hist)
            mask_t = (at_risk[:, t - 1] * at_risk[:, t]).astype(np.float64)
            B_lag = np.broadcast_to(
                self._B_basis[t - 1][None, :], (N, K_re),
            ).astype(np.float64)                                    # (N, K_re)
            rows_X.append(X_t)
            rows_L.append(L_dyn[:, t, :])
            rows_mask.append(mask_t)
            rows_grp.append(inv)                                    # subject idx
            rows_B.append(B_lag)
        X_hist_L = np.vstack(rows_X)                                # (N*(T-1), p_hist)
        L_obs = np.vstack(rows_L)                                   # (N*(T-1), p_dyn)
        mask_L = np.concatenate(rows_mask)                          # (N*(T-1),)
        group_idx_L = np.concatenate(rows_grp).astype(np.int64)
        B_lag_per_obs = np.vstack(rows_B)                           # (N*(T-1), K_re)
        return X_hist_L, L_obs, mask_L, group_idx_L, B_lag_per_obs

    # ----- fit -----
    def fit(self, cohort: ARDSCohort, **kwargs) -> None:
        L = cohort.feature_layout
        K_A, p_dyn, p_stat = L["n_bins"], L["n_dyn"], L["n_static"]
        T = cohort.Y.shape[1]
        self._n_bins, self._n_dyn, self._n_static, self._t_max = (
            K_A, p_dyn, p_stat, T,
        )

        # 1) Spline basis on integer day grid (QR-orthonormalized inside).
        # Empty knots tuple → no random effect (J=0 baseline / Bayesian
        # standard NICE). B_basis stored as (T, 0) zero-width array so
        # downstream save/load shape contracts hold.
        if len(self.config.knots) == 0:
            self._B_basis = np.zeros((T, 0), dtype=np.float64)
            K_re = 0
        else:
            self._B_basis = natural_cubic_basis(
                self.config.knots, np.arange(T),
            )
            K_re = self._B_basis.shape[1]

        # 2) Pooled L equations (frequentist, same as Standard NICE)
        L_dyn = cohort.L_dyn.numpy().astype(np.float64)
        A_bin = cohort.A_bin.numpy().astype(np.float64)
        C_static = cohort.C_static.numpy().astype(np.float64)
        at_risk = cohort.at_risk.numpy().astype(np.float64).squeeze(-1)
        rows, targets, weights = [], [], []
        for t in range(1, T):
            X_t = self._build_history_features(
                L_dyn[:, t - 1, :], A_bin[:, t - 1, :], C_static, t_idx=t, T=T,
            )
            w_t = (at_risk[:, t - 1] * at_risk[:, t]).astype(np.float64)
            rows.append(X_t); targets.append(L_dyn[:, t, :]); weights.append(w_t)
        X_L = np.vstack(rows); Y_L = np.vstack(targets); w_L = np.concatenate(weights)
        self._beta_L, self._sd_L = [], []
        for j in range(p_dyn):
            beta_j, sd_j = _fit_linear(X_L, Y_L[:, j], w_L, l2=self.config.l2_L)
            self._beta_L.append(beta_j)
            self._sd_L.append(max(sd_j, 1e-6))

        # 3) Bayesian outcome fit (NUTS or SVI dispatch)
        X_out, B_per_obs, y, mask, group_idx = self._build_outcome_design(cohort)
        n_groups = int(group_idx.max() + 1)
        self._n_groups = n_groups

        # Holdout: zero out mask for subjects in holdout_subj_ids (rows excluded
        # from likelihood). Used for PPC: fit on 80%, predict held-out 20%.
        if self.config.holdout_subj_ids is not None:
            held = np.asarray(self.config.holdout_subj_ids, dtype=np.int64)
            held_mask = np.isin(group_idx, held)
            mask = mask.copy()
            mask[held_mask] = 0.0
            print(f"  [holdout] {len(held)} subjects masked from fit "
                  f"({held_mask.sum()} of {len(mask)} obs)")

        if self.config.inference == "svi":
            self._fit_outcome_svi(
                X_out, B_per_obs, y, mask, group_idx, n_groups, K_re,
            )
            return

        # J=0 (no-RE) → simpler model with no L_chol/tau sampling
        if K_re == 0:
            kernel = NUTS(_no_re_model, target_accept_prob=self.config.target_accept)
            mcmc = MCMC(
                kernel,
                num_warmup=self.config.n_warmup,
                num_samples=self.config.n_samples,
                num_chains=self.config.n_chains,
                chain_method=self.config.chain_method,
                progress_bar=False,
            )
            rng_key = jr.PRNGKey(self.config.seed)
            mcmc.run(
                rng_key,
                X_outcome=jnp.asarray(X_out),
                y=jnp.asarray(y), mask=jnp.asarray(mask),
                group_idx=jnp.asarray(group_idx),
                n_groups=n_groups,
                record_loglik=self.config.record_loglik,
                extra_fields=("diverging",),
            )
            samples_flat = mcmc.get_samples()
            S = samples_flat["beta"].shape[0]
            self._posterior = {
                "beta": np.asarray(samples_flat["beta"]),         # (S, p)
                # Placeholder zero-width tensors so downstream save/load works
                "L_chol": np.zeros((S, 0, 0), dtype=np.float64),
                "tau": np.zeros((S, 0), dtype=np.float64),
            }
        elif self.config.j5_joint:
            # Spec II joint Bayesian (shared functional RE on Y AND L equations)
            X_hist_L, L_obs_arr, mask_L_arr, group_idx_L_arr, B_lag = (
                self._build_L_design_for_j5_joint(cohort)
            )
            kernel = NUTS(_j5_joint_model, target_accept_prob=self.config.target_accept)
            mcmc = MCMC(
                kernel,
                num_warmup=self.config.n_warmup,
                num_samples=self.config.n_samples,
                num_chains=self.config.n_chains,
                chain_method=self.config.chain_method,
                progress_bar=False,
            )
            rng_key = jr.PRNGKey(self.config.seed)
            mcmc.run(
                rng_key,
                X_outcome_y=jnp.asarray(X_out),
                b_basis_per_obs_y=jnp.asarray(B_per_obs),
                y=jnp.asarray(y), mask_y=jnp.asarray(mask),
                group_idx_y=jnp.asarray(group_idx),
                X_hist_L=jnp.asarray(X_hist_L),
                L_obs=jnp.asarray(L_obs_arr),
                mask_L=jnp.asarray(mask_L_arr),
                group_idx_L=jnp.asarray(group_idx_L_arr),
                b_basis_per_obs_L=jnp.asarray(B_lag),
                n_groups=n_groups, K_re=K_re, p_dyn=p_dyn,
                sigma_b_prior=self.config.sigma_b_prior,
                record_loglik=self.config.record_loglik,
                extra_fields=("diverging",),
            )
            samples_flat = mcmc.get_samples()
            self._posterior = {
                "beta": np.asarray(samples_flat["beta"]),
                "L_chol": np.asarray(samples_flat["L_chol"]),
                "tau": np.asarray(samples_flat["tau"]),
                # J5_joint additional posteriors
                "beta_L_bayes": np.asarray(samples_flat["beta_L_bayes"]),    # (S, p_dyn, p_hist)
                "sigma_L_bayes": np.asarray(samples_flat["sigma_L_bayes"]),  # (S, p_dyn)
                "lambda_L_post": np.asarray(samples_flat["lambda_L"]),       # (S, p_dyn)
            }
            # Override _beta_L / _sd_L from frequentist OLS with posterior MEAN
            # of Bayesian beta_L_bayes / sigma_L_bayes (so forward sim has consistent
            # state). Forward sim with full posterior is supported via _spec_ii_forward.
            beta_L_post_mean = self._posterior["beta_L_bayes"].mean(axis=0)  # (p_dyn, p_hist)
            sigma_L_post_mean = self._posterior["sigma_L_bayes"].mean(axis=0)  # (p_dyn,)
            self._beta_L = [beta_L_post_mean[j] for j in range(p_dyn)]
            self._sd_L = [float(sigma_L_post_mean[j]) for j in range(p_dyn)]
            self._lambda_L = self._posterior["lambda_L"].mean(axis=0)        # (p_dyn,)
            # NB: do NOT set self._L_has_RE_col here. That flag is reserved for
            # the legacy two-stage share_RE_on_L approximation (which augments
            # beta_L with an extra column). Spec II joint uses a separate code
            # path keyed off self.config.j5_joint in forward sim.
        else:
            kernel = NUTS(_fre_nice_model, target_accept_prob=self.config.target_accept)
            mcmc = MCMC(
                kernel,
                num_warmup=self.config.n_warmup,
                num_samples=self.config.n_samples,
                num_chains=self.config.n_chains,
                chain_method=self.config.chain_method,
                progress_bar=False,
            )
            rng_key = jr.PRNGKey(self.config.seed)
            mcmc.run(
                rng_key,
                X_outcome=jnp.asarray(X_out),
                b_basis_per_obs=jnp.asarray(B_per_obs),
                y=jnp.asarray(y), mask=jnp.asarray(mask),
                group_idx=jnp.asarray(group_idx),
                n_groups=n_groups, K_re=K_re,
                sigma_b_prior=self.config.sigma_b_prior,
                record_loglik=self.config.record_loglik,
                extra_fields=("diverging",),
            )
            samples_flat = mcmc.get_samples()
            self._posterior = {
                "beta": np.asarray(samples_flat["beta"]),         # (S, p)
                "L_chol": np.asarray(samples_flat["L_chol"]),     # (S, K_re, K_re)
                "tau": np.asarray(samples_flat["tau"]),           # (S, K_re)
            }
        # Extract posterior mean of subject-level FRE for Spec ② refit
        if "b" in samples_flat:
            self._b_hat = np.asarray(samples_flat["b"]).mean(axis=0)  # (n_groups, K_re)
        # Per-subject log_lik samples (S, n_groups) for cluster-level WAIC/PSIS-LOO
        if "log_lik_subject" in samples_flat:
            self._log_lik = np.asarray(samples_flat["log_lik_subject"])

        # ----- Spec ②: refit L equations with shared RE on L -----
        if self.config.share_RE_on_L and self._b_hat is not None:
            self._refit_L_with_shared_RE(cohort, K_A, K_re)
            self._L_has_RE_col = True
        # Convergence diagnostics: R-hat max, min ESS, divergent count.
        from numpyro.diagnostics import summary
        diag_summary: dict = {}
        try:
            samples_chains = mcmc.get_samples(group_by_chain=True)
            diag = summary(samples_chains, prob=0.95)
            rhat_all, ess_all = [], []
            for k, v in diag.items():
                rhat_all.extend(np.asarray(v.get("r_hat", [])).ravel().tolist())
                ess_all.extend(np.asarray(v.get("n_eff", [])).ravel().tolist())
            diag_summary["r_hat_max"] = float(np.nanmax(rhat_all)) if rhat_all else float("nan")
            diag_summary["r_hat_p95"] = float(np.nanpercentile(rhat_all, 95)) if rhat_all else float("nan")
            diag_summary["ess_min"] = float(np.nanmin(ess_all)) if ess_all else float("nan")
            diag_summary["ess_median"] = float(np.nanmedian(ess_all)) if ess_all else float("nan")
            diag_summary["n_params_with_rhat"] = len(rhat_all)
            tau_diag = diag.get("tau", {})
            r_hat_max_tau = float(np.asarray(tau_diag.get("r_hat", [np.nan])).max())
        except Exception as e:
            diag_summary = {"error": str(e)}
            r_hat_max_tau = float("nan")
        try:
            extra = mcmc.get_extra_fields(group_by_chain=False)
            div_count = int(np.asarray(extra.get("diverging", [0])).sum()) if extra else 0
            diag_summary["divergent_count"] = div_count
        except Exception:
            diag_summary["divergent_count"] = -1
        self._diagnostics = diag_summary
        print(
            f"  [FRE-NICE Bayesian fit] tau (per-basis-dim scale): "
            f"mean = {self._posterior['tau'].mean(axis=0)}, "
            f"R-hat (tau) max = {r_hat_max_tau:.3f}, "
            f"R-hat global max = {diag_summary.get('r_hat_max', float('nan')):.3f}, "
            f"ESS min = {diag_summary.get('ess_min', float('nan')):.0f}, "
            f"divergent = {diag_summary.get('divergent_count', -1)}"
        )

    # ----- Spec ② helpers: share RE on L -----
    def _refit_L_with_shared_RE(
        self, cohort: ARDSCohort, K_A: int, K_re: int,
    ) -> None:
        """Refit pooled L equations with extra column lambda_j * b_hat[i]^T B(t).

        b_hat[i] is the posterior mean of the FRE for subject i (extracted
        from NUTS posterior over deterministic site 'b'). Shape (n_groups, K_re).
        For each obs row (i, t): extra feature = b_hat[i] @ B(t).
        Refit β_L_j with this augmented design (β_L now (p_hist + 1)-dim).
        """
        T = self._t_max
        p_dyn = self._n_dyn
        L_dyn = cohort.L_dyn.numpy().astype(np.float64)
        # Use FULL A_bin (K_A); _build_history_features drops ref once internally.
        # Pre-dropping here would cause double-drop (drops 2 columns instead of 1).
        A_bin = cohort.A_bin.numpy().astype(np.float64)
        C_static = cohort.C_static.numpy().astype(np.float64)
        at_risk = cohort.at_risk.numpy().astype(np.float64).squeeze(-1)

        # Group index per stay (i in [0, n_groups)), repeat across T
        _, inv = np.unique(cohort.subject_ids, return_inverse=True)
        # Per-row extra column: b_hat[group_i]^T B(t)
        # Pre-compute b_hat[i, :] @ B[t, :] for all (i, t) -> (N, T)
        N = L_dyn.shape[0]
        b_hat_per_subject = self._b_hat[inv]                            # (N, K_re)
        re_col_full = b_hat_per_subject @ self._B_basis.T              # (N, T)

        rows, targets, weights = [], [], []
        for t in range(1, T):
            X_t = self._build_history_features(
                L_dyn[:, t - 1, :], A_bin[:, t - 1, :], C_static, t_idx=t, T=T,
            )
            # Extra column for L equation at time t (uses B(t-1) since L_t depends on history)
            extra = re_col_full[:, t - 1].reshape(N, 1)               # (N, 1)
            X_t = np.concatenate([X_t, extra], axis=1)
            w_t = (at_risk[:, t - 1] * at_risk[:, t]).astype(np.float64)
            rows.append(X_t)
            targets.append(L_dyn[:, t, :])
            weights.append(w_t)
        X_L = np.vstack(rows); Y_L = np.vstack(targets); w_L = np.concatenate(weights)
        self._beta_L, self._sd_L = [], []
        for j in range(p_dyn):
            beta_j, sd_j = _fit_linear(X_L, Y_L[:, j], w_L, l2=self.config.l2_L)
            self._beta_L.append(beta_j)
            self._sd_L.append(max(sd_j, 1e-6))
        # lambda_j = β_L_j[-1]; report and save explicitly
        lambdas = np.array([b[-1] for b in self._beta_L])
        self._lambda_L = lambdas
        print(
            f"  [Spec ② shared RE] refit L: lambda_j (per L_dim) = "
            f"{[f'{l:.4f}' for l in lambdas]} "
            f"(max abs = {np.max(np.abs(lambdas)):.4f})"
        )

    # ----- SVI fit for outcome -----
    def _fit_outcome_svi(
        self, X_out, B_per_obs, y, mask, group_idx, n_groups: int, K_re: int,
    ) -> None:
        """Variational Bayesian inference via AutoMultivariateNormal."""
        guide = AutoMultivariateNormal(_fre_nice_model)
        optimizer = numpyro_optim.Adam(step_size=self.config.svi_lr)
        svi = SVI(_fre_nice_model, guide, optimizer, Trace_ELBO())
        rng_key = jr.PRNGKey(self.config.seed)
        svi_result = svi.run(
            rng_key, self.config.svi_steps,
            X_outcome=jnp.asarray(X_out),
            b_basis_per_obs=jnp.asarray(B_per_obs),
            y=jnp.asarray(y), mask=jnp.asarray(mask),
            group_idx=jnp.asarray(group_idx),
            n_groups=n_groups, K_re=K_re,
            progress_bar=False,
        )
        post_key = jr.PRNGKey(self.config.seed + 1)
        posterior = guide.sample_posterior(
            post_key, svi_result.params,
            sample_shape=(self.config.svi_n_posterior_draws,),
        )
        self._posterior = {
            "beta": np.asarray(posterior["beta"]),
            "L_chol": np.asarray(posterior["L_chol"]),
            "tau": np.asarray(posterior["tau"]),
        }
        elbo_final = float(svi_result.losses[-1])
        print(
            f"  [FRE-NICE SVI fit (K={K_re})] tau mean = "
            f"{self._posterior['tau'].mean(axis=0)}, "
            f"final ELBO loss = {elbo_final:.1f}"
        )

    # ----- counterfactual forward sim -----
    def _counterfactual_per_bin(
        self, cohort: ARDSCohort, k: int, rng: np.random.Generator,
    ) -> np.ndarray:
        """Posterior-distributed risk under A=k, returns (S_subset,)."""
        K_A, p_dyn, p_stat, T = (
            self._n_bins, self._n_dyn, self._n_static, self._t_max,
        )
        N_obs = cohort.L_dyn.shape[0]
        M = self.config.n_mc_subjects or N_obs
        L_obs = cohort.L_dyn.numpy().astype(np.float64)
        C_obs = cohort.C_static.numpy().astype(np.float64)

        S_total = self._posterior["beta"].shape[0]
        S = min(self.config.n_posterior_subset, S_total)
        post_idx = rng.choice(S_total, size=S, replace=False)
        beta_post = self._posterior["beta"][post_idx]            # (S, p)
        L_chol_post = self._posterior["L_chol"][post_idx]        # (S, K_re, K_re)
        K_re = L_chol_post.shape[-1]

        # [J5_joint] When the joint Bayesian L equation is fitted, propagate
        # uncertainty in beta_L, sigma_L, and lambda_L by subsetting their
        # full posterior chains. For non-joint specs (J=0/J=1/J=5/J=4/J=6),
        # these arrays are absent and we fall back to the point-estimate path
        # set by self._beta_L / self._sd_L / self._lambda_L (= None).
        use_joint_post = (
            self.config.j5_joint
            and "beta_L_bayes" in self._posterior
            and "sigma_L_bayes" in self._posterior
            and ("lambda_L_post" in self._posterior or "lambda_L" in self._posterior)
        )
        if use_joint_post:
            beta_L_post = self._posterior["beta_L_bayes"][post_idx]   # (S, p_dyn, p_hist)
            sigma_L_post = self._posterior["sigma_L_bayes"][post_idx] # (S, p_dyn)
            lam_post_key = "lambda_L_post" if "lambda_L_post" in self._posterior else "lambda_L"
            lambda_L_post = self._posterior[lam_post_key][post_idx]   # (S, p_dyn)

        # Pre-sample baseline subjects ONCE per dose_response call
        idx0 = rng.integers(0, N_obs, size=M)
        L_t0 = L_obs[idx0, 0, :].copy()
        C_mc = C_obs[idx0]
        A_onehot = np.zeros((M, K_A), dtype=np.float64)
        A_onehot[:, k] = 1.0

        # NICE forward L simulation: same trajectory across posterior draws
        # for paired RD across bins (RNG shared via L_seeds outside this fn).
        # Inner loop: for each posterior draw, draw b ~ N(0, Σ_b^s), accumulate.
        risk_per_post = np.zeros(S, dtype=np.float64)
        for s_idx in range(S):
            beta = beta_post[s_idx]                              # (p,)
            L_chol = L_chol_post[s_idx]                          # (K_re, K_re)
            # Per-posterior-draw L equation parameters under J5_joint;
            # otherwise reuse the point estimates (legacy path).
            if use_joint_post:
                beta_L_s = beta_L_post[s_idx]                    # (p_dyn, p_hist)
                sigma_L_s = sigma_L_post[s_idx]                  # (p_dyn,)
                lambda_L_s = lambda_L_post[s_idx]                # (p_dyn,)
            risks_b = []
            for _ in range(self.config.n_b_draws_per_post):
                z = rng.standard_normal(size=(M, K_re))
                b_subj = z @ L_chol.T                            # (M, K_re)
                random_logit_full = b_subj @ self._B_basis.T     # (M, T)

                # Forward simulate L
                L_cur = L_t0.copy()
                survived = np.ones(M, dtype=np.float64)
                cum = np.zeros(M, dtype=np.float64)
                for t in range(T):
                    if t > 0:
                        X_hist = self._build_history_features(
                            L_cur, A_onehot, C_mc, t_idx=t, T=T,
                        )
                        L_new = np.empty_like(L_cur)
                        if self.config.j5_joint:
                            # J5_joint: mu_L_j = X_hist @ beta_L_j
                            #          + lambda_j * (b_subj^T B(t-1))
                            # When use_joint_post (full posterior available),
                            # use this draw's β_L_s, σ_L_s, λ_L_s. Otherwise
                            # fall back to point-estimate self._beta_L / _sd_L
                            # / _lambda_L (legacy path during fit-time call).
                            re_lag = b_subj @ self._B_basis[t - 1]   # (M,)
                            if use_joint_post:
                                # Per-posterior-draw L parameters
                                for j in range(p_dyn):
                                    mu_j = (
                                        X_hist @ beta_L_s[j]
                                        + lambda_L_s[j] * re_lag
                                    )
                                    L_new[:, j] = mu_j + rng.normal(
                                        0.0, sigma_L_s[j], size=M,
                                    )
                            else:
                                # Posterior-mean fallback
                                for j in range(p_dyn):
                                    mu_j = (
                                        X_hist @ self._beta_L[j]
                                        + self._lambda_L[j] * re_lag
                                    )
                                    L_new[:, j] = mu_j + rng.normal(
                                        0.0, self._sd_L[j], size=M,
                                    )
                        elif self._L_has_RE_col:
                            # Legacy two-stage share_RE_on_L: extra column in X_hist;
                            # lambda_j is the last element of self._beta_L[j].
                            re_col = (b_subj @ self._B_basis[t - 1]).reshape(M, 1)
                            X_hist_aug = np.concatenate([X_hist, re_col], axis=1)
                            for j in range(p_dyn):
                                mu_j = X_hist_aug @ self._beta_L[j]
                                L_new[:, j] = mu_j + rng.normal(
                                    0.0, self._sd_L[j], size=M,
                                )
                        else:
                            # Standard FRE-NICE / J=0 / J=1: no L RE term
                            for j in range(p_dyn):
                                mu_j = X_hist @ self._beta_L[j]
                                L_new[:, j] = mu_j + rng.normal(
                                    0.0, self._sd_L[j], size=M,
                                )
                        L_cur = L_new
                    # Outcome logit (use posterior beta).
                    # Outcome design layout from _build_outcome_design after
                    # cov reorder + drop ref bin: [bias, bins\\ref, L_dyn,
                    # C_static, t_norm]. Reproduce it here.
                    bias_col = np.ones((M, 1), dtype=np.float64)
                    t_col = np.full((M, 1), t / max(T - 1, 1))
                    A_dropped = self._drop_ref_in_A(A_onehot)
                    X_t = np.concatenate(
                        [bias_col, A_dropped, L_cur, C_mc, t_col], axis=1,
                    )
                    eta_fix = X_t @ beta
                    logit = eta_fix + random_logit_full[:, t]
                    p_t = 1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0)))
                    cum = cum + survived * p_t
                    survived = survived * (1.0 - p_t)
                risks_b.append(cum.mean())
            risk_per_post[s_idx] = float(np.mean(risks_b))
        return risk_per_post

    def dose_response(
        self, cohort: ARDSCohort, target_bins: Sequence[int],
        n_bootstrap: int = 0, seed: int = 0, refit: bool = True,
    ) -> DoseResponseResult:
        """Posterior credible-interval dose-response (no bootstrap)."""
        if refit and self._posterior is None:
            self.fit(cohort)
        rng = np.random.default_rng(seed)
        K = len(target_bins)
        S = min(self.config.n_posterior_subset, self._posterior["beta"].shape[0])
        risk_mat = np.zeros((K, S), dtype=np.float64)
        for ki, k in enumerate(target_bins):
            risk_mat[ki] = self._counterfactual_per_bin(cohort, k, rng)
        return DoseResponseResult(
            bins=list(target_bins),
            bin_centers_J_min=bin_centers_J_min(cohort),
            risk_mean=risk_mat.mean(axis=1),
            risk_ci_low=np.quantile(risk_mat, 0.025, axis=1),
            risk_ci_high=np.quantile(risk_mat, 0.975, axis=1),
            risk_raw=risk_mat,
            method_name=self.method_name,
        )
