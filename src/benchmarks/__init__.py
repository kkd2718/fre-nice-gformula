"""Benchmark methods for the 4-way ARDS-MP causal comparison.

Methods (post-Bayesian-pivot, primary 4-method ladder):
  1. StandardGFormula              — frequentist NICE g-formula, no RE (baseline)
  2. XuGLMMBayesian                — Bayesian GLMM with scalar RE, MSM observed-L
                                     plug-in (Xu 2024)
  3. FRENICEBayesianBenchmark(K=1) — Bayesian, scalar RE, NICE forward-L
                                     (mathematically equivalent in fitting to
                                     XuGLMMBayesian outcome model; differs in
                                     counterfactual computation)
  4. FRENICEBayesianBenchmark(K=5) — Bayesian, functional RE via natural cubic
                                     spline basis, NICE forward-L (proposed
                                     primary method)

Legacy frequentist Laplace variants (xu_glmm.XuGLMM, spline_glmm_nice.*) and
state-space-model proposed.* moved to legacy/scripts_v1/ during round-2
cleanup; not used in primary analysis.
"""
from .base import BenchmarkMethod, DoseResponseResult, bin_centers_J_min
from .standard_gformula import StandardGFormula
from .xu_glmm_bayesian import XuGLMMBayesian, XuBayesianConfig
from .fre_nice_bayesian import (
    FRENICEBayesianBenchmark, FRENICEBayesianConfig,
)

__all__ = [
    "BenchmarkMethod", "DoseResponseResult", "bin_centers_J_min",
    "StandardGFormula",
    "XuGLMMBayesian", "XuBayesianConfig",
    "FRENICEBayesianBenchmark", "FRENICEBayesianConfig",
]
