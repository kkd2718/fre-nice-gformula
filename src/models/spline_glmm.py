"""Spline-GLMM: time-varying generalization of Xu 2024 outcome random effect.

Extends Xu 2024 (scalar b_i ~ N(0, sigma_b^2) added to outcome logit) by
replacing the scalar with a time-varying smooth function expressed in a
fixed natural-cubic-spline basis:

    b_i(t) = b_i^T B(t),       b_i ~ N(0, Sigma_b),  Sigma_b in R^{KxK}

where B(t) is a pre-specified basis evaluated at integer day t = 0..T-1.
With K=1 and B(t) ≡ 1 the model reduces exactly to Xu 2024.

Outcome model (full NICE g-formula compatible):
    logit P(Y_t = 1 | A_t, L_t, V, b_i)
        = beta_0 + beta_A^T A_t + eta^T L_t + xi^T V + b_i^T B(t)

L equations are fit separately as pooled regressions (Standard NICE form);
they are NOT part of this module.

Sigma_b is parameterized via its lower-triangular Cholesky factor L_chol,
so Sigma_b = L_chol @ L_chol.T is positive-definite by construction. The
basis B(t) is fixed at construction (data-independent), which pins the
random-effect scale and avoids the identifiability issues of the previous
SSM specification (see post_defense_revision_plan.md).

References
----------
- Xu R, Kim S, Hummers L, Shah A, Zeger SL (2024). Causal inference using
  multivariate generalized linear mixed-effects models. Biometrics 80(3):ujae100.
- Durbin J, Koopman SJ (2012). Time Series Analysis by State Space Methods.
- Wood SN (2017). Generalized Additive Models: An Introduction with R, 2e.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


# ----------------------------------------------------------------------
# Natural cubic spline basis (pre-computed at integer days 0..T-1)
# ----------------------------------------------------------------------
def natural_cubic_basis(knots: tuple[float, ...], t_grid: np.ndarray) -> np.ndarray:
    """Natural cubic spline basis evaluated at t_grid.

    Constructs the natural cubic regression spline basis (Wood 2017, sec. 5.3.1)
    with given interior+boundary knots. Returns a basis matrix of shape
    (len(t_grid), K) where K = len(knots). The first column is the constant
    intercept; subsequent columns encode the spline structure with linear
    boundary constraints (2nd derivative = 0 at knots[0] and knots[-1]).

    Special cases for the time-invariant random-intercept comparison:
    - K = 1 (any single knot): returns a constant column 1, recovering the
      scalar random intercept exactly (Xu 2024 RE inside NICE framework).
    - K = 2: returns intercept + linear-in-t columns (random slope on time).

    Parameters
    ----------
    knots : ordered sequence of K time points
    t_grid : array of time points to evaluate at

    Returns
    -------
    B : ndarray of shape (len(t_grid), K)
    """
    knots = np.asarray(sorted(set(knots)), dtype=np.float64)
    K = len(knots)
    t = np.asarray(t_grid, dtype=np.float64)
    n = len(t)

    if K == 1:
        # Scalar random intercept (b_i(t) = b_i, time-invariant).
        # QR-normalize for consistency with K >= 3 path.
        B = np.ones((n, 1), dtype=np.float64)
        Q, _ = np.linalg.qr(B)
        return Q
    if K == 2:
        # Linear basis: random intercept + random slope on time.
        B = np.zeros((n, 2), dtype=np.float64)
        B[:, 0] = 1.0
        B[:, 1] = t
        Q, _ = np.linalg.qr(B)
        return Q

    t = np.asarray(t_grid, dtype=np.float64)
    n = len(t)

    # Truncated power basis with natural-spline (linear-tail) reduction.
    # Following Wood (2017, eq. 5.5) reparameterization to K columns:
    #   col 1: constant 1
    #   col 2: t
    #   cols 3..K: d_k(t) - d_{K-1}(t),  k = 1, ..., K-2
    # where d_k(t) = [(t - tau_k)^3 - (t - tau_K)^3] / (tau_K - tau_k),
    # using only the positive part of (t - tau)^3.
    def _tp_cubic(x: np.ndarray, k: float) -> np.ndarray:
        return np.where(x > k, (x - k) ** 3, 0.0)

    tau_last = knots[-1]
    d = np.zeros((n, K - 1), dtype=np.float64)
    for kk in range(K - 1):
        denom = tau_last - knots[kk]
        d[:, kk] = (
            _tp_cubic(t, knots[kk]) - _tp_cubic(t, tau_last)
        ) / denom

    # Reparameterized columns 3..K = d_k - d_{K-1}, k = 0..K-3
    B = np.zeros((n, K), dtype=np.float64)
    B[:, 0] = 1.0
    B[:, 1] = t
    for col, kk in enumerate(range(K - 2)):
        B[:, 2 + col] = d[:, kk] - d[:, K - 2]
    # QR-orthonormalize on the integer-day grid (S1 fix per Opus review).
    # Truncated-power columns can reach values ~130 at t=14 with knots (0,3,7,14,21);
    # combined with N(0, 0.1) Cholesky init this saturates the logit clamp.
    # Orthonormalized basis has columns of unit norm on the grid, removes
    # implicit basis-rotation residual invariance of Sigma_b, and stabilizes
    # the per-subject Newton inner solver.
    Q, _ = np.linalg.qr(B)
    return Q


@dataclass
class SplineGLMMConfig:
    """Hyperparameters for SplineGLMM."""
    n_bins: int                      # K_A: # of MP treatment bins (one-hot)
    n_dyn: int                       # # of time-varying covariates L
    n_static: int                    # # of static covariates V
    knots: tuple[float, ...] = (0.0, 3.0, 7.0, 14.0, 21.0)
    t_max: int = 28
    init_log_diag_chol: float = -1.0   # init log diagonal of Σ_b Cholesky
    init_off_chol: float = 0.0


class SplineGLMM(nn.Module):
    """Y-only outcome GLMM with spline-basis time-varying random effect.

    Fixed effects:
        beta_0          (scalar)
        beta_A          (n_bins,)        treatment one-hot coefficients
        eta             (n_dyn,)         time-varying covariates
        xi              (n_static,)      static covariates

    Random effects per subject:
        b_i in R^K       ~ N(0, Sigma_b)
    Sigma_b is parameterized by Cholesky factor L_chol (lower triangular):
        L_chol diagonal: exp(log_diag_chol) for positivity
        L_chol below-diagonal: free
    so Sigma_b = L_chol @ L_chol.T is symmetric positive-definite.

    The spline basis B is registered as a buffer of shape (t_max, K).
    """

    def __init__(self, config: SplineGLMMConfig) -> None:
        super().__init__()
        self.config = config

        # Pre-compute basis B(t) for t = 0, 1, ..., t_max-1
        t_grid = np.arange(config.t_max, dtype=np.float64)
        B_np = natural_cubic_basis(config.knots, t_grid)
        self.register_buffer("B", torch.from_numpy(B_np).float())  # (T, K_re)
        self.K_re: int = B_np.shape[1]

        # Fixed effects
        self.beta_0 = nn.Parameter(torch.tensor([-3.0]))
        self.beta_A = nn.Parameter(torch.zeros(config.n_bins))
        self.eta = (
            nn.Parameter(torch.zeros(config.n_dyn)) if config.n_dyn > 0 else None
        )
        self.xi = (
            nn.Parameter(torch.zeros(config.n_static)) if config.n_static > 0 else None
        )
        self.beta_time = nn.Parameter(torch.tensor([0.0]))   # baseline time trend

        # Sigma_b Cholesky parameterization
        self.log_diag_chol = nn.Parameter(
            torch.full((self.K_re,), config.init_log_diag_chol)
        )
        # Below-diagonal entries of L_chol stored as flat vector
        n_off = self.K_re * (self.K_re - 1) // 2
        if n_off > 0:
            self.off_chol = nn.Parameter(
                torch.full((n_off,), config.init_off_chol)
            )
        else:
            self.register_parameter("off_chol", None)

        # Indices of below-diagonal entries (row, col with row > col)
        rows, cols = torch.tril_indices(self.K_re, self.K_re, offset=-1)
        self.register_buffer("_chol_rows", rows)
        self.register_buffer("_chol_cols", cols)

    # ------------------------------------------------------------------
    # Sigma_b construction
    # ------------------------------------------------------------------
    def L_chol(self) -> Tensor:
        """Lower-triangular Cholesky factor of Sigma_b. Shape (K_re, K_re)."""
        K = self.K_re
        L = torch.zeros(K, K, dtype=self.log_diag_chol.dtype,
                        device=self.log_diag_chol.device)
        L[range(K), range(K)] = torch.exp(self.log_diag_chol)
        if self.off_chol is not None:
            L[self._chol_rows, self._chol_cols] = self.off_chol
        return L

    def Sigma_b(self) -> Tensor:
        """Sigma_b = L_chol @ L_chol.T. Shape (K_re, K_re)."""
        L = self.L_chol()
        return L @ L.t()

    def Sigma_b_inv_logdet(self) -> tuple[Tensor, Tensor]:
        """Returns (Sigma_b_inv, log|Sigma_b|) computed stably from L_chol."""
        L = self.L_chol()
        # Sigma_b = L L^T, so log|Sigma_b| = 2 sum log diag(L)
        logdet = 2.0 * self.log_diag_chol.sum()
        # Sigma_b^{-1} = L^{-T} L^{-1}, computed via triangular solve
        K = self.K_re
        I = torch.eye(K, dtype=L.dtype, device=L.device)
        Linv = torch.linalg.solve_triangular(L, I, upper=False)
        Sinv = Linv.t() @ Linv
        return Sinv, logdet

    # ------------------------------------------------------------------
    # Logit computation
    # ------------------------------------------------------------------
    def fixed_logit(
        self, A: Tensor, L_dyn: Tensor, V: Tensor, t_norm: Tensor,
    ) -> Tensor:
        """Compute fixed-effects logit (no random effect).

        Parameters
        ----------
        A : (N, T, K_A)         one-hot treatment indicators
        L_dyn : (N, T, p_dyn)
        V : (N, p_stat) or None
        t_norm : (T,) or (N, T) normalized day index in [0, 1]

        Returns
        -------
        logit_fix : (N, T)
        """
        # Squeeze trailing singleton if cohort gives (N, T, 1)
        if t_norm.ndim == 3 and t_norm.shape[-1] == 1:
            t_norm = t_norm.squeeze(-1)
        eta_fix = self.beta_0 + (A * self.beta_A).sum(dim=-1)  # (N, T)
        if self.eta is not None and L_dyn is not None and L_dyn.shape[-1] > 0:
            eta_fix = eta_fix + (L_dyn * self.eta).sum(dim=-1)
        if self.xi is not None and V is not None and V.shape[-1] > 0:
            eta_fix = eta_fix + (V @ self.xi).unsqueeze(-1)
        if t_norm.ndim == 1:
            eta_fix = eta_fix + self.beta_time * t_norm.unsqueeze(0)
        else:
            # t_norm is (N, T) — element-wise broadcast with eta_fix (N, T)
            eta_fix = eta_fix + self.beta_time * t_norm
        return eta_fix

    def random_logit(self, b: Tensor) -> Tensor:
        """Compute random-effect contribution b @ B(t) for all t.

        Parameters
        ----------
        b : (N, K_re)

        Returns
        -------
        random : (N, T)
        """
        # B is (T, K_re); want output (N, T) = b @ B^T
        return b @ self.B.t()

    def total_logit(
        self, b: Tensor, A: Tensor, L_dyn: Tensor, V: Tensor, t_norm: Tensor,
    ) -> Tensor:
        """Total logit = fixed + random."""
        return self.fixed_logit(A, L_dyn, V, t_norm) + self.random_logit(b)

    # ------------------------------------------------------------------
    # Conditional log-likelihood given b_i (Bernoulli with at_risk mask)
    # ------------------------------------------------------------------
    def conditional_loglik_per_subject(
        self,
        b: Tensor,
        Y: Tensor, A: Tensor, L_dyn: Tensor, V: Tensor,
        at_risk: Tensor, t_norm: Tensor,
    ) -> Tensor:
        """log p(Y_i | b_i, ...) summed over t, returned per subject (N,).

        Y, at_risk : (N, T)  - trailing dim already squeezed (or use .squeeze)
        """
        # Squeeze trailing dim if needed
        if Y.ndim == 3:
            Y = Y.squeeze(-1)
        if at_risk.ndim == 3:
            at_risk = at_risk.squeeze(-1)
        logit = self.total_logit(b, A, L_dyn, V, t_norm)        # (N, T)
        # Bernoulli log-lik per (i, t) via stable form
        # log p = -softplus(-logit) if y=1 else -softplus(logit)
        log_p1 = -torch.nn.functional.softplus(-logit)
        log_p0 = -torch.nn.functional.softplus(logit)
        per_t = at_risk * (Y * log_p1 + (1.0 - Y) * log_p0)      # (N, T)
        return per_t.sum(dim=-1)                                 # (N,)
