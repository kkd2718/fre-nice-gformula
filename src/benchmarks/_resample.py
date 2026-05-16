"""Cluster bootstrap helper for ARDSCohort (resample by subject_id)."""
from __future__ import annotations
import numpy as np
import torch

from ..data.ards import ARDSCohort


def cluster_bootstrap_indices(
    subject_ids: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    """Resample unique patients with replacement, then return all stay indices."""
    unique_pat, inverse = np.unique(subject_ids, return_inverse=True)
    sampled = rng.integers(0, len(unique_pat), size=len(unique_pat))
    return np.concatenate([np.where(inverse == p)[0] for p in sampled])


def slice_cohort(cohort: ARDSCohort, idx: np.ndarray) -> ARDSCohort:
    """Build a new ARDSCohort containing only the rows in idx (preserves layout)."""
    idx_t = torch.as_tensor(idx, dtype=torch.long)
    return ARDSCohort(
        Y=cohort.Y[idx_t],
        A_bin=cohort.A_bin[idx_t],
        L_dyn=cohort.L_dyn[idx_t],
        C_static=cohort.C_static[idx_t],
        at_risk=cohort.at_risk[idx_t],
        t_norm=cohort.t_norm[idx_t],
        drivers=cohort.drivers[idx_t],
        covariates=cohort.covariates[idx_t],
        bin_edges_mp=cohort.bin_edges_mp,
        stay_ids=cohort.stay_ids[idx],
        subject_ids=cohort.subject_ids[idx],
        severity_label=cohort.severity_label[idx],
        feature_layout=cohort.feature_layout,
    )
