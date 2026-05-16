"""Common interface for benchmark methods (28-day cumulative incidence)."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..data.ards import ARDSCohort


@dataclass
class DoseResponseResult:
    """Output of a dose-response sweep over MP bins.

    All risks are 28-day cumulative incidence on the same scale:
    1 - prod_t (1 - p_Y_t).
    """
    bins: list[int]
    bin_centers_J_min: list[float]
    risk_mean: np.ndarray
    risk_ci_low: np.ndarray
    risk_ci_high: np.ndarray
    risk_raw: np.ndarray
    method_name: str


class BenchmarkMethod(ABC):
    """Common contract for benchmark methods on the ARDS cohort.

    Implementations: standard_gformula.StandardGFormula,
    xu_glmm.XuGLMM, proposed.VEMSSMBenchmark.
    """

    method_name: str = "benchmark"

    @abstractmethod
    def fit(self, cohort: ARDSCohort, **kwargs) -> None:
        ...

    @abstractmethod
    def dose_response(
        self, cohort: ARDSCohort, target_bins: Sequence[int],
        n_bootstrap: int = 100, seed: int = 0,
    ) -> DoseResponseResult:
        ...


def bin_centers_J_min(cohort: ARDSCohort) -> list[float]:
    """Geometric midpoint of each MP bin in original J/min units."""
    edges = cohort.bin_edges_mp
    centers = []
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        if np.isfinite(lo) and np.isfinite(hi):
            centers.append(float(np.sqrt(lo * hi)))
        elif np.isfinite(hi):
            centers.append(float(hi * 0.9))
        elif np.isfinite(lo):
            centers.append(float(lo * 1.1))
        else:
            centers.append(float("nan"))
    return centers
