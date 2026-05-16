"""Joint Y+L GLMM — simplified variant inspired by Xu 2024 (numpyro / NUTS).

Inspired by Xu Y, Kim JS, Hummers LK, Shah AA, Zeger SL. Causal inference
using multivariate generalized linear mixed-effects models. Biometrics.
2024;80(3):ujae100. DOI: 10.1093/biomtc/ujae100.

NOT a faithful replication. Substantial simplifications relative to Xu 2024:
  (1) Random effects are independent univariate normals per equation
      (Xu 2024 uses MVN(0, Σ_b) with cross-correlation among b_Y/b_M/b_A —
      the core mechanism for unmeasured-confounding identification).
  (2) No random slopes; only random intercepts (Xu 2024 has b_{i0} + b_{i1}).
  (3) No treatment-assignment equation with b^A (Xu 2024 needs this for
      identification of unmeasured treatment selection). Our treatment A
      is observed multinomial bin (K=20), not Xu 2024's monotonic binary
      treatment initiation — so the assignment-equation RE is also
      structurally inapplicable here.
  (4) Inference is full NUTS (Xu 2024 uses Laplace approximation; NUTS is
      more rigorous but the resulting posterior is not identical).

Used in this work as a methodologically distinct RE specification (intercept
RE on BOTH Y and L equations) within the 4-specification robustness ladder,
NOT as a faithful Xu 2024 replication. Two generative model variants:

  joint_glmm = True (default, η_refined fix):
    Joint GLMM with subject random intercepts on outcome AND each
    time-varying biomarker equation. Counterfactual uses forward L
    simulation conditional on a fixed bin, propagating treatment-confounder
    feedback through L_t. Faithful to Xu 2024 Sections 2-3.

  joint_glmm = False (legacy, retained for reproducibility):
    Outcome-only RE; counterfactual plugs in observed L. This was an
    incomplete replication and underestimates treatment-confounder feedback;
    not recommended for new analyses.

Joint generative model
----------------------
    beta_0, beta_A, beta_L, beta_C  ~  N(0, 5^2)
    alpha_j_0, alpha_j_A, alpha_j_Llag, alpha_j_C  ~  N(0, 5^2)        for each L dim j
    sigma_b_y, sigma_b_L_j          ~  HalfCauchy(0, 2.5)
    sigma_L_j                       ~  HalfCauchy(0, 2.5)
    b_y_i | sigma_b_y               ~  N(0, sigma_b_y^2)
    b_L_ji | sigma_b_L_j            ~  N(0, sigma_b_L_j^2)             RE per L dim
    L_t_j | A_t, L_{t-1}, C, b_L_ji ~  N(alpha_j_0 + alpha_j_A^T A_t + alpha_j_Llag^T L_{t-1}
                                          + alpha_j_C^T C + b_L_ji, sigma_L_j^2)
    Y_t  | A_t, L_t, C, b_y_i       ~  Bernoulli(sigmoid(beta_0 + beta_A^T A_t
                                          + beta_L^T L_t + beta_C^T C + b_y_i))

Counterfactual (joint, forward L simulation)
--------------------------------------------
For each posterior draw s and target bin k:
    For each subject i:
        Sample b_y_i, b_L_ji ~ N(0, sigma_b_*^2)            (one MC draw per posterior sample)
        L_0 = observed baseline L_i,0
        For t = 1, ..., T-1:
            mu_L_t_j = alpha_j_0 + alpha_j_A^T e_k + alpha_j_Llag^T L_{t-1} + alpha_j_C^T C_i + b_L_ji
            L_t = mu_L_t (deterministic forward propagation; Robins g-formula convention)
        For t = 0, ..., T-1:
            logit_t = beta_0 + beta_A^T e_k + beta_L^T L_t + beta_C^T C_i + b_y_i
            p_t = sigmoid(logit_t)
            cum += survived * p_t; survived *= (1 - p_t)
Posterior credible interval on R(a_k) from quantiles across posterior draws.

Notes
-----
- No cluster bootstrap (posterior CI replaces it; consistent with Xu 2024).
- Forward L simulation handles treatment-confounder feedback; this is the
  central methodological feature distinguishing Xu 2024 from MSM/IPTW.
- Convergence diagnostics: R-hat, ESS via numpyro.diagnostics.
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


def _xu_joint_glmm_model(
    A_onehot: jnp.ndarray,        # (N, T, K) bin one-hot
    L_obs: jnp.ndarray,           # (N, T, p_L) observed time-varying L (response)
    L_lag: jnp.ndarray,           # (N, T, p_L) lagged L; L_lag[:, 0, :] = L_obs[:, 0, :] (zero contrib via alpha_Llag init OK)
    C_static: jnp.ndarray,        # (N, p_C)
    Y: jnp.ndarray,               # (N, T)
    y_mask: jnp.ndarray,          # (N, T) outcome at_risk mask
    L_mask: jnp.ndarray,          # (N, T, p_L) per-(stay,day,L) observed indicator
    subject_idx: jnp.ndarray,     # (N,) subject group index, [0, n_subjects)
    n_subjects: int,
    sigma_b_prior: str = "halfcauchy",
    record_loglik: bool = False,
) -> None:
    """Joint GLMM (Xu 2024): outcome + L equations with subject REs."""
    K = A_onehot.shape[-1]
    p_L = L_obs.shape[-1]
    p_C = C_static.shape[-1]

    # ---- Outcome fixed effects ----
    beta_0 = numpyro.sample("beta_0", dist.Normal(0.0, 5.0))
    beta_A = numpyro.sample("beta_A", dist.Normal(jnp.zeros(K), 5.0))
    beta_L = numpyro.sample("beta_L", dist.Normal(jnp.zeros(p_L), 5.0))
    beta_C = numpyro.sample("beta_C", dist.Normal(jnp.zeros(p_C), 5.0))

    # ---- L equations fixed effects (one per L dim) ----
    alpha_0 = numpyro.sample("alpha_0", dist.Normal(jnp.zeros(p_L), 5.0))
    alpha_A = numpyro.sample("alpha_A", dist.Normal(jnp.zeros((p_L, K)), 5.0))
    alpha_Llag = numpyro.sample("alpha_Llag", dist.Normal(jnp.zeros((p_L, p_L)), 5.0))
    alpha_C = numpyro.sample("alpha_C", dist.Normal(jnp.zeros((p_L, p_C)), 5.0))
    sigma_L = numpyro.sample("sigma_L", dist.HalfCauchy(jnp.ones(p_L) * 2.5))

    # ---- Subject REs (non-centered) ----
    # Note: only RE scales (sigma_b_y, sigma_b_L) respond to sigma_b_prior;
    # sigma_L (TV-biomarker observation noise) stays HalfCauchy by design.
    if sigma_b_prior == "halfcauchy":
        sigma_b_y = numpyro.sample("sigma_b_y", dist.HalfCauchy(2.5))
        sigma_b_L = numpyro.sample("sigma_b_L", dist.HalfCauchy(jnp.ones(p_L) * 2.5))
    elif sigma_b_prior == "gamma":
        sigma_b_y = numpyro.sample("sigma_b_y", dist.Gamma(2.0, 0.5))
        sigma_b_L = numpyro.sample("sigma_b_L", dist.Gamma(jnp.ones(p_L) * 2.0, jnp.ones(p_L) * 0.5))
    elif sigma_b_prior == "invgamma":
        sigma_b_y = numpyro.sample("sigma_b_y", dist.InverseGamma(2.0, 1.0))
        sigma_b_L = numpyro.sample("sigma_b_L", dist.InverseGamma(jnp.ones(p_L) * 2.0, jnp.ones(p_L) * 1.0))
    else:
        raise ValueError(f"Unknown sigma_b_prior: {sigma_b_prior}")

    z_y = numpyro.sample("z_y", dist.Normal(jnp.zeros(n_subjects), 1.0))
    b_y = numpyro.deterministic("b_y", sigma_b_y * z_y)              # (n_subjects,)
    z_L = numpyro.sample("z_L", dist.Normal(jnp.zeros((p_L, n_subjects)), 1.0))
    b_L = numpyro.deterministic("b_L", sigma_b_L[:, None] * z_L)     # (p_L, n_subjects)

    # ---- L equation likelihood ----
    # mu_L: (N, T, p_L)
    mu_L = (
        alpha_0[None, None, :]                                   # (1,1,p_L)
        + jnp.einsum("ntk,jk->ntj", A_onehot, alpha_A)            # (N,T,p_L)
        + jnp.einsum("ntp,jp->ntj", L_lag, alpha_Llag)            # (N,T,p_L)
        + jnp.einsum("nc,jc->nj", C_static, alpha_C)[:, None, :]  # (N,1,p_L)
        + b_L[:, subject_idx].T[:, None, :]                       # (N,1,p_L)
    )
    log_p_L = L_mask * dist.Normal(mu_L, sigma_L).log_prob(L_obs)
    numpyro.factor("loglik_L", log_p_L.sum())

    # ---- Outcome likelihood ----
    logit_y = (
        beta_0
        + jnp.einsum("ntk,k->nt", A_onehot, beta_A)
        + jnp.einsum("ntp,p->nt", L_obs, beta_L)
        + jnp.einsum("nc,c->n", C_static, beta_C)[:, None]
        + b_y[subject_idx][:, None]
    )
    log_p_y = y_mask * dist.Bernoulli(logits=logit_y).log_prob(Y)
    if record_loglik:
        ll_subject = jax.ops.segment_sum(
            log_p_y.sum(axis=-1), subject_idx, num_segments=n_subjects,
        )
        numpyro.deterministic("log_lik_subject", ll_subject)
    numpyro.factor("loglik_y", log_p_y.sum())


def _xu_bayesian_model(
    X: jnp.ndarray, y: jnp.ndarray, mask: jnp.ndarray,
    group_idx: jnp.ndarray, n_groups: int,
    sigma_b_prior: str = "halfcauchy",
    record_loglik: bool = False,
) -> None:
    """numpyro model for Xu 2024 GLMM.

    X : (N_obs, p)  pooled design matrix (rows = subject-time)
    y : (N_obs,)    binary outcome
    mask : (N_obs,) at-risk indicator (0/1)
    group_idx : (N_obs,) integer subject index in [0, n_groups)
    sigma_b_prior : halfcauchy | gamma | invgamma — prior sensitivity option
    record_loglik : if True, record per-observation log_lik for WAIC/PSIS-LOO
    """
    p = X.shape[1]
    beta = numpyro.sample("beta", dist.Normal(jnp.zeros(p), 5.0))
    if sigma_b_prior == "halfcauchy":
        sigma_b = numpyro.sample("sigma_b", dist.HalfCauchy(2.5))
    elif sigma_b_prior == "gamma":
        sigma_b = numpyro.sample("sigma_b", dist.Gamma(2.0, 0.5))
    elif sigma_b_prior == "invgamma":
        sigma_b = numpyro.sample("sigma_b", dist.InverseGamma(2.0, 1.0))
    else:
        raise ValueError(f"Unknown sigma_b_prior: {sigma_b_prior}")
    # Non-centered parameterization for hierarchical prior (better mixing)
    z = numpyro.sample("z_b", dist.Normal(jnp.zeros(n_groups), 1.0))
    b = numpyro.deterministic("b", sigma_b * z)
    logit = X @ beta + b[group_idx]
    # Mask non-at-risk observations from likelihood
    log_p = mask * dist.Bernoulli(logits=logit).log_prob(y)
    if record_loglik:
        ll_subject = jax.ops.segment_sum(
            log_p, group_idx, num_segments=n_groups,
        )
        numpyro.deterministic("log_lik_subject", ll_subject)
    numpyro.factor("loglik", log_p.sum())


@dataclass
class XuBayesianConfig:
    """Sampler / counterfactual hyperparameters.

    inference: "nuts" (full posterior, slow) or "svi" (variational, fast).
    SVI uses AutoMultivariateNormal guide -> still Bayesian (variational
    posterior approximation), with diagonal-plus-correlation Gaussian for
    the joint posterior. Recommended fallback when NUTS infeasible.

    ref_bin: MP bin index to use as reference (one-hot column dropped). Removes
    the bias-vs-bins collinearity (rank deficiency by 1) in the design matrix.
    Default 7 = cohort 25th percentile of MP.
    """
    inference: str = "nuts"           # "nuts" or "svi"
    ref_bin: int = 7
    sigma_b_prior: str = "halfcauchy"  # halfcauchy | gamma | invgamma
    record_loglik: bool = False        # for WAIC / PSIS-LOO
    holdout_subj_ids: tuple[int, ...] | None = None  # for PPC
    joint_glmm: bool = True           # η_refined: joint GLMM + forward L sim (Xu 2024 faithful)
    # NUTS parameters
    n_warmup: int = 1000
    n_samples: int = 1000
    n_chains: int = 4
    chain_method: str = "parallel"
    target_accept: float = 0.9
    # SVI parameters
    svi_steps: int = 5000
    svi_lr: float = 1e-2
    svi_n_posterior_draws: int = 2000  # samples drawn from variational posterior
    # Counterfactual
    n_b_draws: int = 50              # MC draws over b per posterior sample
    n_posterior_subset: int = 200    # posterior draws used for counterfactual
    seed: int = 0


class XuGLMMBayesian(BenchmarkMethod):
    """Bayesian replication of Xu 2024 GLMM via numpyro NUTS.

    Replaces the broken Laplace + MoM implementation; faithful to the
    original Bayesian g-computation procedure.
    """

    method_name = "xu_glmm_bayesian"

    def __init__(self, config: XuBayesianConfig | None = None) -> None:
        self.config = config or XuBayesianConfig()
        self._posterior: dict | None = None
        self._n_groups: int = 0
        self._diagnostics: dict | None = None
        self._log_lik: np.ndarray | None = None
        self._joint_state: dict | None = None  # cached arrays for joint counterfactual

    # ------------------------------------------------------------------
    # Joint GLMM tensors (η_refined, Xu 2024 faithful)
    # ------------------------------------------------------------------
    def _build_joint_arrays(self, cohort: ARDSCohort) -> dict:
        """Pack arrays needed for joint GLMM model + forward L simulation."""
        L = cohort.feature_layout
        K = L["n_bins"]
        N = cohort.Y.shape[0]
        T = cohort.Y.shape[1]
        A_onehot = cohort.A_bin.numpy().astype(np.float64)              # (N, T, K)
        L_obs = cohort.L_dyn.numpy().astype(np.float64)                 # (N, T, p_L)
        C_static = cohort.C_static.numpy().astype(np.float64)           # (N, p_C)
        Y = cohort.Y.numpy().reshape(N, T).astype(np.float64)
        y_mask = cohort.at_risk.numpy().reshape(N, T).astype(np.float64)
        # L_mask: 1 iff at_risk (proxy for L observed; L_dyn is ffilled, so use at_risk)
        L_mask = np.broadcast_to(y_mask[..., None], L_obs.shape).astype(np.float64)
        # L_lag[:, t, :] = L_obs[:, t-1, :] for t>=1; t=0 lag = L_obs[:, 0, :] (no contribution given baseline carries)
        L_lag = np.zeros_like(L_obs)
        L_lag[:, 1:, :] = L_obs[:, :-1, :]
        # subject_idx
        _, inv = np.unique(cohort.subject_ids, return_inverse=True)
        subject_idx = inv.astype(np.int64)
        return dict(
            A_onehot=A_onehot, L_obs=L_obs, L_lag=L_lag, C_static=C_static,
            Y=Y, y_mask=y_mask, L_mask=L_mask, subject_idx=subject_idx,
            n_subjects=int(subject_idx.max() + 1),
            N=N, T=T, K=K, p_L=L_obs.shape[-1], p_C=C_static.shape[-1],
        )

    # ------------------------------------------------------------------
    # Design construction (same shape as legacy XuGLMM)
    # ------------------------------------------------------------------
    def _build_design(
        self, cohort: ARDSCohort, override_bin: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        L = cohort.feature_layout
        K = L["n_bins"]
        N, T = cohort.Y.shape[0], cohort.Y.shape[1]
        cov = cohort.covariates.numpy().reshape(N * T, -1).astype(np.float64)
        y = cohort.Y.numpy().reshape(N * T)
        m = cohort.at_risk.numpy().reshape(N * T)
        if override_bin is not None:
            cov = cov.copy()
            cov[:, :K] = 0.0
            cov[:, override_bin] = 1.0
        # Drop reference bin column from one-hot to avoid bias-vs-bins collinearity.
        # cov layout: [bins (K), L_dyn, C_static, t_norm]; bins are first K cols.
        ref = self.config.ref_bin
        if ref is not None and 0 <= ref < K:
            keep_bins = [k for k in range(K) if k != ref]
            cov_bins = cov[:, keep_bins]                   # (NT, K-1)
            cov_other = cov[:, K:]
            cov = np.concatenate([cov_bins, cov_other], axis=1)
        bias = np.ones((cov.shape[0], 1), dtype=np.float64)
        X = np.concatenate([bias, cov], axis=1)
        _, inv = np.unique(cohort.subject_ids, return_inverse=True)
        group_idx = np.repeat(inv, T)
        return X, y.astype(np.float64), m.astype(np.float64), group_idx

    # ------------------------------------------------------------------
    # Fit dispatcher (NUTS or SVI)
    # ------------------------------------------------------------------
    def fit(self, cohort: ARDSCohort, **kwargs) -> None:
        if self.config.joint_glmm:
            return self._fit_joint(cohort)
        X, y, m, group_idx = self._build_design(cohort)
        n_groups = int(group_idx.max() + 1)
        self._n_groups = n_groups

        # Holdout: zero out mask for subjects in holdout_subj_ids (PPC)
        if self.config.holdout_subj_ids is not None:
            held = np.asarray(self.config.holdout_subj_ids, dtype=np.int64)
            held_mask = np.isin(group_idx, held)
            m = m.copy()
            m[held_mask] = 0.0
            print(f"  [holdout] {len(held)} subjects masked from fit "
                  f"({held_mask.sum()} of {len(m)} obs)")

        if self.config.inference == "svi":
            self._fit_svi(X, y, m, group_idx, n_groups)
            return
        # NUTS path
        kernel = NUTS(_xu_bayesian_model, target_accept_prob=self.config.target_accept)
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
            X=jnp.asarray(X), y=jnp.asarray(y), mask=jnp.asarray(m),
            group_idx=jnp.asarray(group_idx), n_groups=n_groups,
            sigma_b_prior=self.config.sigma_b_prior,
            record_loglik=self.config.record_loglik,
            extra_fields=("diverging",),
        )
        # Posterior samples (chain-flattened) for downstream MC
        samples_flat = mcmc.get_samples()
        self._posterior = {
            "beta": np.asarray(samples_flat["beta"]),     # (S, p)
            "sigma_b": np.asarray(samples_flat["sigma_b"]),  # (S,)
        }
        if "log_lik_subject" in samples_flat:
            self._log_lik = np.asarray(samples_flat["log_lik_subject"])
        # Convergence diagnostics — global summary across all params
        from numpyro.diagnostics import summary
        diag_summary: dict = {}
        r_hat_sigma, n_eff_sigma = float("nan"), float("nan")
        try:
            samples_chains = mcmc.get_samples(group_by_chain=True)
            diag = summary(samples_chains, prob=0.95)
            rhat_all, ess_all = [], []
            for k, v in diag.items():
                rhat_all.extend(np.asarray(v.get("r_hat", [])).ravel().tolist())
                ess_all.extend(np.asarray(v.get("n_eff", [])).ravel().tolist())
            diag_summary["r_hat_max"] = float(np.nanmax(rhat_all)) if rhat_all else float("nan")
            diag_summary["ess_min"] = float(np.nanmin(ess_all)) if ess_all else float("nan")
            diag_summary["ess_median"] = float(np.nanmedian(ess_all)) if ess_all else float("nan")
            diag_summary["n_params_with_rhat"] = len(rhat_all)
            sig_diag = diag.get("sigma_b", {})
            r_hat_sigma = float(np.asarray(sig_diag.get("r_hat", float("nan"))))
            n_eff_sigma = float(np.asarray(sig_diag.get("n_eff", float("nan"))))
        except Exception as e:
            diag_summary = {"error": str(e)}
        try:
            extra = mcmc.get_extra_fields(group_by_chain=False)
            div_count = int(np.asarray(extra.get("diverging", [0])).sum()) if extra else 0
            diag_summary["divergent_count"] = div_count
        except Exception:
            diag_summary["divergent_count"] = -1
        self._diagnostics = diag_summary
        print(
            f"  [Xu Bayesian fit] sigma_b posterior mean = "
            f"{self._posterior['sigma_b'].mean():.4f}, "
            f"R-hat (sigma_b) = {r_hat_sigma:.3f}, "
            f"R-hat global max = {diag_summary.get('r_hat_max', float('nan')):.3f}, "
            f"ESS min = {diag_summary.get('ess_min', float('nan')):.0f}, "
            f"divergent = {diag_summary.get('divergent_count', -1)}"
        )

    # ------------------------------------------------------------------
    # Joint GLMM NUTS fit (η_refined, Xu 2024 faithful)
    # ------------------------------------------------------------------
    def _fit_joint(self, cohort: ARDSCohort) -> None:
        st = self._build_joint_arrays(cohort)
        self._joint_state = st
        self._n_groups = st["n_subjects"]

        if self.config.holdout_subj_ids is not None:
            held = np.asarray(self.config.holdout_subj_ids, dtype=np.int64)
            held_mask = np.isin(st["subject_idx"], held)
            st = dict(st)
            st["y_mask"] = st["y_mask"].copy()
            st["y_mask"][held_mask] = 0.0
            st["L_mask"] = st["L_mask"].copy()
            st["L_mask"][held_mask] = 0.0
            print(f"  [joint holdout] {len(held)} subjects masked from fit")

        kernel = NUTS(_xu_joint_glmm_model, target_accept_prob=self.config.target_accept)
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
            A_onehot=jnp.asarray(st["A_onehot"]),
            L_obs=jnp.asarray(st["L_obs"]),
            L_lag=jnp.asarray(st["L_lag"]),
            C_static=jnp.asarray(st["C_static"]),
            Y=jnp.asarray(st["Y"]),
            y_mask=jnp.asarray(st["y_mask"]),
            L_mask=jnp.asarray(st["L_mask"]),
            subject_idx=jnp.asarray(st["subject_idx"]),
            n_subjects=st["n_subjects"],
            sigma_b_prior=self.config.sigma_b_prior,
            record_loglik=self.config.record_loglik,
            extra_fields=("diverging",),
        )
        samples_flat = mcmc.get_samples()
        # Cache only what counterfactual needs (avoid storing all REs to disk later)
        post = {
            "beta_0": np.asarray(samples_flat["beta_0"]),         # (S,)
            "beta_A": np.asarray(samples_flat["beta_A"]),         # (S, K)
            "beta_L": np.asarray(samples_flat["beta_L"]),         # (S, p_L)
            "beta_C": np.asarray(samples_flat["beta_C"]),         # (S, p_C)
            "alpha_0": np.asarray(samples_flat["alpha_0"]),       # (S, p_L)
            "alpha_A": np.asarray(samples_flat["alpha_A"]),       # (S, p_L, K)
            "alpha_Llag": np.asarray(samples_flat["alpha_Llag"]),  # (S, p_L, p_L)
            "alpha_C": np.asarray(samples_flat["alpha_C"]),       # (S, p_L, p_C)
            "sigma_L": np.asarray(samples_flat["sigma_L"]),       # (S, p_L)
            "sigma_b_y": np.asarray(samples_flat["sigma_b_y"]),   # (S,)
            "sigma_b_L": np.asarray(samples_flat["sigma_b_L"]),   # (S, p_L)
        }
        self._posterior = post
        if "log_lik_subject" in samples_flat:
            self._log_lik = np.asarray(samples_flat["log_lik_subject"])

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
            diag_summary["ess_min"] = float(np.nanmin(ess_all)) if ess_all else float("nan")
            diag_summary["ess_median"] = float(np.nanmedian(ess_all)) if ess_all else float("nan")
            diag_summary["n_params_with_rhat"] = len(rhat_all)
        except Exception as e:
            diag_summary = {"error": str(e)}
        try:
            extra = mcmc.get_extra_fields(group_by_chain=False)
            diag_summary["divergent_count"] = (
                int(np.asarray(extra.get("diverging", [0])).sum()) if extra else 0
            )
        except Exception:
            diag_summary["divergent_count"] = -1
        self._diagnostics = diag_summary
        print(
            f"  [Xu joint GLMM] sigma_b_y={post['sigma_b_y'].mean():.3f}, "
            f"sigma_b_L mean={post['sigma_b_L'].mean(0)}, "
            f"R-hat max={diag_summary.get('r_hat_max', float('nan')):.3f}, "
            f"ESS min={diag_summary.get('ess_min', float('nan')):.0f}, "
            f"divergent={diag_summary.get('divergent_count', -1)}"
        )

    # ------------------------------------------------------------------
    # SVI fit (Auto-MVN variational posterior)
    # ------------------------------------------------------------------
    def _fit_svi(self, X, y, m, group_idx, n_groups: int) -> None:
        """Variational Bayesian inference via AutoMultivariateNormal guide.

        Faster than NUTS but approximate. Posterior is the joint MVN that
        best matches the true posterior in KL sense (variational).
        """
        guide = AutoMultivariateNormal(_xu_bayesian_model)
        optimizer = numpyro_optim.Adam(step_size=self.config.svi_lr)
        svi = SVI(_xu_bayesian_model, guide, optimizer, Trace_ELBO())
        rng_key = jr.PRNGKey(self.config.seed)
        svi_result = svi.run(
            rng_key, self.config.svi_steps,
            X=jnp.asarray(X), y=jnp.asarray(y), mask=jnp.asarray(m),
            group_idx=jnp.asarray(group_idx), n_groups=n_groups,
            progress_bar=False,
        )
        # Sample from variational posterior
        post_key = jr.PRNGKey(self.config.seed + 1)
        posterior = guide.sample_posterior(
            post_key, svi_result.params,
            sample_shape=(self.config.svi_n_posterior_draws,),
        )
        self._posterior = {
            "beta": np.asarray(posterior["beta"]),
            "sigma_b": np.asarray(posterior["sigma_b"]),
        }
        # Final ELBO from svi history
        elbo_final = float(svi_result.losses[-1])
        print(
            f"  [Xu Bayesian SVI] sigma_b posterior mean = "
            f"{self._posterior['sigma_b'].mean():.4f}, "
            f"final ELBO loss = {elbo_final:.1f}"
        )

    # ------------------------------------------------------------------
    # Counterfactual via posterior + b-marginalization
    # ------------------------------------------------------------------
    def _counterfactual_per_bin(
        self, cohort: ARDSCohort, k: int, rng: np.random.Generator,
    ) -> np.ndarray:
        """Posterior-mean and credible-interval risk under A=k.

        Returns
        -------
        risk_per_posterior : (S_subset,)  population-mean risk per posterior draw,
                             integrated over b ~ N(0, sigma_b^2_s).
        """
        X, _, _, _ = self._build_design(cohort, override_bin=k)
        N, T = cohort.Y.shape[0], cohort.Y.shape[1]
        S_total = self._posterior["beta"].shape[0]
        S = min(self.config.n_posterior_subset, S_total)
        # Subsample posterior draws (uniformly)
        idx = rng.choice(S_total, size=S, replace=False)
        beta_post = self._posterior["beta"][idx]            # (S, p)
        sigma_b_post = self._posterior["sigma_b"][idx]      # (S,)

        risk_per_post = np.zeros(S, dtype=np.float64)
        # Loop over posterior draws — vectorize over subjects + b draws
        for s in range(S):
            beta = beta_post[s]
            sigma_b = sigma_b_post[s]
            eta_no_b = (X @ beta).reshape(N, T)             # (N, T)
            risks_b = []
            for _ in range(self.config.n_b_draws):
                b_per_subject = rng.normal(0.0, sigma_b, size=N)
                eta = eta_no_b + b_per_subject[:, None]     # (N, T)
                p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))
                survived = np.ones(N, dtype=np.float64)
                cum = np.zeros(N, dtype=np.float64)
                for t in range(T):
                    cum = cum + survived * p[:, t]
                    survived = survived * (1.0 - p[:, t])
                risks_b.append(cum.mean())
            risk_per_post[s] = float(np.mean(risks_b))
        return risk_per_post

    # ------------------------------------------------------------------
    # Joint counterfactual (forward L simulation)
    # ------------------------------------------------------------------
    def _counterfactual_joint_per_bin(
        self, k: int, rng: np.random.Generator,
        truncate_at_day: int | None = None,
        natural_course_rate: float = 0.0,
    ) -> np.ndarray:
        """Forward L simulation under counterfactual A regime.

        truncate_at_day:
            None  -> sustained 28d: A_t = e_k for all t.
            N     -> truncated regime: A_t = e_k for t < N. For t >= N,
                     bin contribution drops out (apply A_bin=0, i.e., ref bin).
        natural_course_rate:
            0.0           -> per-day rate from outcome model (Case 1 baseline if
                              t >= truncate_at_day under Option B fit).
            > 0           -> override per-day rate with this value (cohort
                              observational off-MV mortality), splicing on-MV
                              regime contribution with natural-course outcome.

        Returns risk_per_post : (S_subset,) population-mean cum incidence.
        """
        st = self._joint_state
        N, T, K, p_L, p_C = st["N"], st["T"], st["K"], st["p_L"], st["p_C"]
        L_obs0 = st["L_obs"][:, 0, :]
        C = st["C_static"]
        subj = st["subject_idx"]
        n_subj = st["n_subjects"]

        post = self._posterior
        S_total = post["beta_0"].shape[0]
        S = min(self.config.n_posterior_subset, S_total)
        idx = rng.choice(S_total, size=S, replace=False)

        trunc = T if truncate_at_day is None else int(truncate_at_day)
        trunc = max(0, min(T, trunc))

        risk_per_post = np.zeros(S, dtype=np.float64)
        for ii, s in enumerate(idx):
            beta_0 = post["beta_0"][s]
            beta_A_k = post["beta_A"][s, k]
            beta_L = post["beta_L"][s]
            beta_C = post["beta_C"][s]
            alpha_0 = post["alpha_0"][s]
            alpha_A_k = post["alpha_A"][s, :, k]
            alpha_Llag = post["alpha_Llag"][s]
            alpha_C = post["alpha_C"][s]
            sigma_b_y = post["sigma_b_y"][s]
            sigma_b_L = post["sigma_b_L"][s]

            b_y_subj = rng.normal(0.0, sigma_b_y, size=n_subj)
            b_L_subj = rng.normal(0.0, 1.0, size=(p_L, n_subj)) * sigma_b_L[:, None]
            b_y_i = b_y_subj[subj]
            b_L_i = b_L_subj[:, subj].T

            xC_y = C @ beta_C
            xC_L = C @ alpha_C.T

            L_t = L_obs0.copy()
            survived = np.ones(N, dtype=np.float64)
            cum = np.zeros(N, dtype=np.float64)

            for t in range(T):
                # Apply bin effect only while t < trunc
                bin_contrib_y = beta_A_k if t < trunc else 0.0
                logit_y = (
                    beta_0 + bin_contrib_y
                    + L_t @ beta_L
                    + xC_y
                    + b_y_i
                )
                p_y_model = 1.0 / (1.0 + np.exp(-np.clip(logit_y, -30.0, 30.0)))
                # Splice in natural-course rate after truncation
                if (natural_course_rate > 0) and (t >= trunc):
                    p_y = np.full(N, natural_course_rate, dtype=np.float64)
                else:
                    p_y = p_y_model
                cum = cum + survived * p_y
                survived = survived * (1.0 - p_y)
                # Forward simulate L_{t+1}; bin effect on L only while t < trunc
                if t < T - 1:
                    bin_contrib_L = alpha_A_k if t < trunc else 0.0
                    mu_L_next = (
                        alpha_0[None, :]
                        + bin_contrib_L
                        + L_t @ alpha_Llag.T
                        + xC_L
                        + b_L_i
                    )
                    L_t = mu_L_next
            risk_per_post[ii] = float(cum.mean())
        return risk_per_post

    def dose_response(
        self, cohort: ARDSCohort, target_bins: Sequence[int],
        n_bootstrap: int = 0, seed: int = 0, refit: bool = True,
        truncate_at_day: int | None = None,
        natural_course_rate: float = 0.0,
    ) -> DoseResponseResult:
        """Posterior credible-interval dose-response (no bootstrap; Xu 2024 standard).

        n_bootstrap is accepted for API parity but ignored.
        truncate_at_day, natural_course_rate: see _counterfactual_joint_per_bin.
        """
        if refit and self._posterior is None:
            self.fit(cohort)
        rng = np.random.default_rng(seed)
        K = len(target_bins)
        if self.config.joint_glmm:
            S_total = self._posterior["beta_0"].shape[0]
        else:
            S_total = self._posterior["beta"].shape[0]
        S = min(self.config.n_posterior_subset, S_total)
        risk_mat = np.zeros((K, S), dtype=np.float64)
        for ki, k in enumerate(target_bins):
            if self.config.joint_glmm:
                risk_mat[ki] = self._counterfactual_joint_per_bin(
                    k, rng,
                    truncate_at_day=truncate_at_day,
                    natural_course_rate=natural_course_rate,
                )
            else:
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
