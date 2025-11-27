from __future__ import annotations
from typing import List
import numpy as np

# ===== Main function to compute ISC (leave-one-out) =====
def compute_isc(
    data_list: List[np.ndarray],
) -> np.ndarray:
    """
    Simple leave-one-out ISC.

    Parameters
    ----------
    data_list : list of arrays
        Each element is (T, F), where:
        - T = timepoints
        - F = features (voxels, parcels, ROIs; for a single ROI F=1).

    Returns
    -------
    isc_mean : (F,) array
        Mean ISC across subjects for each feature.
    """
    n_subj = len(data_list)
    if n_subj < 2:
        raise ValueError("Need at least 2 subjects for ISC.")

    T, F = data_list[0].shape
    isc_all = np.zeros((n_subj, F), dtype=float)

    for i in range(n_subj):
        this = data_list[i]               # (T, F)
        others = [data_list[j] for j in range(n_subj) if j != i]
        mean_others = np.mean(others, axis=0)  # (T, F)

        # Z-score over time, feature-wise, with small epsilon for stability
        this_mean = this.mean(axis=0, keepdims=True)
        this_std = this.std(axis=0, keepdims=True)
        this_z = (this - this_mean) / (this_std + 1e-8)

        others_mean = mean_others.mean(axis=0, keepdims=True)
        others_std = mean_others.std(axis=0, keepdims=True)
        others_z = (mean_others - others_mean) / (others_std + 1e-8)

        # Feature-wise correlation: mean of elementwise product of z-scores
        corr = np.mean(this_z * others_z, axis=0)
        isc_all[i] = corr

    return isc_all.mean(axis=0)           # (F,)
