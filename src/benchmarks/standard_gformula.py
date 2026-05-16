"""Robins parametric g-formula via the NICE algorithm.

References
----------
- Robins JM (1986). A new approach to causal inference in mortality studies
  with a sustained exposure period. Math Modelling 7:1393-1512.
- Taubman SL, Robins JM, Mittleman MA, Hernan MA (2009). Intervening on risk
  factors for coronary heart disease. IJE 38(6):1599-1611.
- Hernan MA, Robins JM (2020). Causal Inference: What If. CRC Press, ch. 21.
- McGrath S, Lin V, Zhang Z, Petito LC, Logan RW, Hernan MA, Young JG (2020).
  gfoRmula: An R Package for Estimating the Effects of Sustained Treatment
  Strategies via the Parametric g-Formula. Patterns 1(3):100008.

Algorithm (Non-Iterative Conditional Expectation, NICE):
    1. Fit pooled-over-time covariate models  L^(j)_t | L_{t-1}, A_{t-1}, C, t.
    2. Fit pooled hazard model  Y_t | L_t, A_t, C, t.
    3. Monte Carlo simulation under counterfactual intervention A_t = a:
       (i) sample baseline (L_0, C) from observed empirical distribution,
       (ii) for each t: simulate L^(j)_t with residual noise, set A_t = a,
            accumulate discrete-time hazard.
    4. Bootstrap CIs by cluster-resampling patients and refitting all models.
"""
from __future__ import annotations
from typing import Sequence

import numpy as np

from ..data.ards import ARDSCohort
from .base import BenchmarkMethod, DoseResponseResult, bin_centers_J_min
from ._resample import cluster_bootstrap_indices, slice_cohort


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _fit_logistic(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float = 1e-4,
    max_iter: int = 50, tol: float = 1e-6,
) -> np.ndarray:
    n, p = X.shape
    beta = np.zeros(p, dtype=np.float64)
    for _ in range(max_iter):
        eta = X @ beta
        mu = _expit(eta)
        s = mu * (1.0 - mu) + 1e-8
        W = w * s
        grad = X.T @ (w * (mu - y)) + l2 * beta
        H = (X.T * W) @ X + l2 * np.eye(p)
        step = np.linalg.solve(H, grad)
        beta = beta - step
        if np.max(np.abs(step)) < tol:
            break
    return beta


def _fit_linear(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float = 1e-4,
) -> tuple[np.ndarray, float]:
    """Weighted ridge regression returning (beta, residual_sd)."""
    p = X.shape[1]
    XtW = X.T * w
    H = XtW @ X + l2 * np.eye(p)
    rhs = XtW @ y
    beta = np.linalg.solve(H, rhs)
    resid = y - X @ beta
    sd = float(np.sqrt(np.average(resid * resid, weights=w + 1e-12)))
    return beta, sd


class StandardGFormula(BenchmarkMethod):
    """Parametric g-formula via NICE algorithm (Robins 1986; gfoRmula 2020).

    Identification: NUC given measured time-varying covariates and baseline.
    Each time-varying L^(j) is modeled with a pooled linear regression on
    history; the daily hazard is modeled with pooled logistic regression.
    Counterfactual outcomes are obtained by Monte Carlo simulation under the
    intervention A_t = a held constant for all t.
    """

    method_name = "standard_gformula"

    def __init__(
        self, l2: float = 1e-4, n_mc_subjects: int | None = None,
        ref_bin: int = 16,
    ) -> None:
        """ref_bin: index of MP bin to use as reference (column dropped from
        treatment one-hot to avoid bias-vs-bins collinearity). Default 16
        corresponds to ≈ 17 J/min (Costa 2021 cutoff)."""
        self.l2 = l2
        self.n_mc_subjects = n_mc_subjects
        self.ref_bin = ref_bin
        self._beta_L: list[np.ndarray] = []
        self._sd_L: list[float] = []
        self._beta_Y: np.ndarray | None = None
        self._n_dyn: int = 0
        self._n_bins: int = 0
        self._n_static: int = 0
        self._t_max: int = 0

    def _layout(self, cohort: ARDSCohort) -> tuple[int, int, int, int]:
        L = cohort.feature_layout
        return L["n_bins"], L["n_dyn"], L["n_static"], cohort.Y.shape[1]

    def _drop_ref(self, A_onehot: np.ndarray) -> np.ndarray:
        """Drop reference-bin column from (N, K_A) one-hot matrix.

        After dropping, ref_bin is implicit (all dummies = 0). This removes
        the bias-vs-bins collinearity (sum of one-hot = 1 = bias).
        """
        if self.ref_bin is None or A_onehot.shape[1] <= 1:
            return A_onehot
        return np.delete(A_onehot, self.ref_bin, axis=1)

    def _build_history_features(
        self, L_prev: np.ndarray, A_prev_onehot: np.ndarray,
        C: np.ndarray, t_idx: int, T: int,
    ) -> np.ndarray:
        """Per-row history vector: [1, L_{t-1}, A_{t-1} \\ ref, C, t/T]."""
        N = L_prev.shape[0]
        bias = np.ones((N, 1), dtype=np.float64)
        t_col = np.full((N, 1), t_idx / max(T - 1, 1), dtype=np.float64)
        A_dropped = self._drop_ref(A_prev_onehot)
        return np.concatenate([bias, L_prev, A_dropped, C, t_col], axis=1)

    def _build_outcome_features(
        self, L_t: np.ndarray, A_t_onehot: np.ndarray,
        C: np.ndarray, t_idx: int, T: int,
    ) -> np.ndarray:
        """Per-row outcome features: [1, L_t, A_t \\ ref, C, t/T]."""
        N = L_t.shape[0]
        bias = np.ones((N, 1), dtype=np.float64)
        t_col = np.full((N, 1), t_idx / max(T - 1, 1), dtype=np.float64)
        A_dropped = self._drop_ref(A_t_onehot)
        return np.concatenate([bias, L_t, A_dropped, C, t_col], axis=1)

    def fit(self, cohort: ARDSCohort, **kwargs) -> None:
        K, p_dyn, p_stat, T = self._layout(cohort)
        self._n_bins = K
        self._n_dyn = p_dyn
        self._n_static = p_stat
        self._t_max = T

        L_dyn = cohort.L_dyn.numpy().astype(np.float64)
        A_bin = cohort.A_bin.numpy().astype(np.float64)
        C_static = cohort.C_static.numpy().astype(np.float64)
        Y = cohort.Y.numpy().astype(np.float64).squeeze(-1)
        m = cohort.at_risk.numpy().astype(np.float64).squeeze(-1)
        N = L_dyn.shape[0]

        # 1) Pooled covariate models L^(j)_t given history (t = 1, ..., T-1).
        rows_L: list[np.ndarray] = []
        targets_L: list[np.ndarray] = []
        weights_L: list[np.ndarray] = []
        for t in range(1, T):
            X_hist = self._build_history_features(
                L_prev=L_dyn[:, t - 1, :], A_prev_onehot=A_bin[:, t - 1, :],
                C=C_static, t_idx=t, T=T,
            )
            w_row = (m[:, t - 1] * m[:, t]).astype(np.float64)
            rows_L.append(X_hist)
            targets_L.append(L_dyn[:, t, :])
            weights_L.append(w_row)
        X_L = np.vstack(rows_L)
        w_L = np.concatenate(weights_L)
        Y_L = np.vstack(targets_L)
        self._beta_L = []
        self._sd_L = []
        for j in range(p_dyn):
            beta_j, sd_j = _fit_linear(X_L, Y_L[:, j], w_L, l2=self.l2)
            self._beta_L.append(beta_j)
            self._sd_L.append(max(sd_j, 1e-6))

        # 2) Pooled hazard model Y_t | L_t, A_t, C, t for t = 0, ..., T-1.
        rows_Y: list[np.ndarray] = []
        targets_Y: list[np.ndarray] = []
        weights_Y: list[np.ndarray] = []
        for t in range(T):
            X_out = self._build_outcome_features(
                L_t=L_dyn[:, t, :], A_t_onehot=A_bin[:, t, :],
                C=C_static, t_idx=t, T=T,
            )
            rows_Y.append(X_out)
            targets_Y.append(Y[:, t])
            weights_Y.append(m[:, t].astype(np.float64))
        X_Y = np.vstack(rows_Y)
        y_Y = np.concatenate(targets_Y)
        w_Y = np.concatenate(weights_Y)
        self._beta_Y = _fit_logistic(X_Y, y_Y, w_Y, l2=self.l2)

    def _simulate_counterfactual(
        self, cohort: ARDSCohort, intervene_bin: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Run NICE simulation; return per-MC-subject 28-day cumulative incidence."""
        K, p_dyn, p_stat, T = self._n_bins, self._n_dyn, self._n_static, self._t_max
        N_obs = cohort.L_dyn.shape[0]
        M = self.n_mc_subjects or N_obs

        L_obs = cohort.L_dyn.numpy().astype(np.float64)
        C_obs = cohort.C_static.numpy().astype(np.float64)

        # Sample baseline (L_0, C) from observed empirical distribution
        idx0 = rng.integers(0, N_obs, size=M)
        L_t = L_obs[idx0, 0, :].copy()
        C_mc = C_obs[idx0]

        # Counterfactual A is bin = intervene_bin for all t
        A_onehot = np.zeros((M, K), dtype=np.float64)
        A_onehot[:, intervene_bin] = 1.0

        survived = np.ones(M, dtype=np.float64)
        cum = np.zeros(M, dtype=np.float64)

        for t in range(T):
            if t > 0:
                X_hist = self._build_history_features(
                    L_prev=L_t, A_prev_onehot=A_onehot, C=C_mc, t_idx=t, T=T,
                )
                L_new = np.empty_like(L_t)
                for j in range(p_dyn):
                    mu_j = X_hist @ self._beta_L[j]
                    L_new[:, j] = mu_j + rng.normal(0.0, self._sd_L[j], size=M)
                L_t = L_new
            X_out = self._build_outcome_features(
                L_t=L_t, A_t_onehot=A_onehot, C=C_mc, t_idx=t, T=T,
            )
            p_t = _expit(X_out @ self._beta_Y)
            cum = cum + survived * p_t
            survived = survived * (1.0 - p_t)
        return cum

    def dose_response(
        self, cohort: ARDSCohort, target_bins: Sequence[int],
        n_bootstrap: int = 100, seed: int = 0, refit: bool = True,
    ) -> DoseResponseResult:
        """Outer bootstrap × inner bin loop: B fits (not K*B); paired RD across bins.

        For each bootstrap replicate b: cluster-resample patients, refit if needed,
        then evaluate the counterfactual at every target bin against the SAME
        resampled cohort. This both avoids redundant fits (gfoRmula convention)
        and preserves per-replicate correlation between bins for valid paired
        risk-difference CIs.
        """
        rng = np.random.default_rng(seed)
        if not refit and self._beta_Y is None:
            self.fit(cohort)
        K = len(target_bins)
        risk_mat = np.zeros((K, n_bootstrap), dtype=np.float64)
        for b in range(n_bootstrap):
            idx = cluster_bootstrap_indices(cohort.subject_ids, rng)
            boot_cohort = slice_cohort(cohort, idx)
            if refit:
                self.fit(boot_cohort)
            for ki, k in enumerate(target_bins):
                cum = self._simulate_counterfactual(
                    boot_cohort, intervene_bin=k, rng=rng,
                )
                risk_mat[ki, b] = float(cum.mean())
        return DoseResponseResult(
            bins=list(target_bins),
            bin_centers_J_min=bin_centers_J_min(cohort),
            risk_mean=risk_mat.mean(axis=1),
            risk_ci_low=np.quantile(risk_mat, 0.025, axis=1),
            risk_ci_high=np.quantile(risk_mat, 0.975, axis=1),
            risk_raw=risk_mat,
            method_name=self.method_name,
        )
