from __future__ import annotations

from typing import List, Literal, Tuple, Optional, Union
import numpy as np


# ================================================================
# Low level helpers
# ================================================================
# Small helper for stable z-scoring along time (axis=0)
def _zscore_time(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Z-score over time (axis=0), feature-wise.
    x: (T, F)
    """
    mu = np.nanmean(x, axis=0, keepdims=True)
    sd = np.nanstd(x, axis=0, keepdims=True)
    return (x - mu) / (sd + eps)


# Fisher z transform + inverse (optional; useful for inference)
def _fisher_z(r: np.ndarray) -> np.ndarray:
    r = np.clip(r, -0.999999, 0.999999)
    return np.arctanh(r)


def _inv_fisher_z(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


# ================================================================
# Main function to compute ISC
# ================================================================
def compute_isc(
    data_list: List[np.ndarray],
    *,
    method: Literal["loo"] = "loo",
    standardize: Literal["zscore"] = "zscore",
    summary: Literal["mean"] = "mean",
    fisher_z: bool = False,
    nan_policy: Literal["propagate", "omit"] = "omit",
    return_subjectwise: bool = False,
    eps: float = 1e-8,
    dtype: Union[np.dtype, type] = np.float32,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Leave-one-out ISC (subject vs mean-of-others), feature-wise.

    Parameters
    ----------
    data_list : list of arrays
        Each element is (T, F).
    method : {"loo"}
        Currently only leave-one-out.
    standardize : {"zscore"}
        Standardization over time per feature.
    summary : {"mean"}
        How to summarize across subjects (mean is the standard default).
    fisher_z : bool
        If True: apply Fisher z (atanh) before averaging across subjects,
        then inverse transform back to r. Often preferred for inference.
    nan_policy : {"propagate", "omit"}
        - "omit": ignore NaNs in means/stds/correlations (recommended for fMRI)
        - "propagate": if NaNs exist, they will flow through.
    return_subjectwise : bool
        If True, return (isc_subjectwise, isc_mean).
        isc_subjectwise is shape (N, F). This is the key output for future hypothesis testing.
    eps : float
        Numerical stability for std division.
    dtype : dtype
        Internal dtype to reduce memory (float32 is usually enough).

    Returns
    -------
    isc_mean : (F,) array
        Mean ISC across subjects.
    OR (isc_subjectwise, isc_mean)
        isc_subjectwise: (N, F), isc_mean: (F,)
    """
    if method != "loo":
        raise ValueError(f"Unsupported method={method}. Only 'loo' is supported currently.")

    n_subj = len(data_list)
    if n_subj < 2:
        raise ValueError("Need at least 2 subjects for ISC.")

    # strict shape validation
    T, F = data_list[0].shape
    for i, arr in enumerate(data_list):
        if arr.ndim != 2:
            raise ValueError(f"Subject {i} array must be 2D (T, F), got {arr.shape}")
        if arr.shape != (T, F):
            raise ValueError(f"All subjects must share shape (T, F). Subject {i} has {arr.shape}, expected {(T, F)}")

    # stack once + compute leave-one-out mean via sums (faster + cleaner)
    data = np.stack([np.asarray(a, dtype=dtype) for a in data_list], axis=0)  # (N, T, F)

    # handle NaNs in a controlled way
    if nan_policy == "omit":
        sum_all = np.nansum(data, axis=0)                 # (T, F)
        count_all = np.sum(~np.isnan(data), axis=0)       # (T, F)
    elif nan_policy == "propagate":
        sum_all = np.sum(data, axis=0)
        count_all = np.full((T, F), n_subj, dtype=np.int32)
    else:
        raise ValueError(f"Unsupported nan_policy={nan_policy}")

    isc_subjectwise = np.zeros((n_subj, F), dtype=np.float32)

    for i in range(n_subj):
        this = data[i]  # (T, F)

        # leave-one-out mean: (sum - this) / (n-1) (with NaN-aware denominator if omit)
        if nan_policy == "omit":
            denom = np.maximum(count_all - (~np.isnan(this)), 1)  # avoid division by 0
            mean_others = (sum_all - np.nan_to_num(this, nan=0.0)) / denom
        else:
            mean_others = (sum_all - this) / (n_subj - 1)

        # standardization goes through helper; still matches your logic
        if standardize == "zscore":
            this_z = _zscore_time(this, eps=eps)
            others_z = _zscore_time(mean_others, eps=eps)
        else:
            raise ValueError(f"Unsupported standardize={standardize}")

        # robust correlation with NaN handling
        if nan_policy == "omit":
            corr = np.nanmean(this_z * others_z, axis=0)  # (F,)
        else:
            corr = np.mean(this_z * others_z, axis=0)

        isc_subjectwise[i] = corr.astype(np.float32)

    # optional Fisher z aggregation
    if fisher_z:
        z = _fisher_z(isc_subjectwise)
        isc_mean = np.nanmean(z, axis=0) if nan_policy == "omit" else np.mean(z, axis=0)
        isc_mean = _inv_fisher_z(isc_mean).astype(np.float32)
    else:
        isc_mean = (np.nanmean(isc_subjectwise, axis=0) if nan_policy == "omit" else np.mean(isc_subjectwise, axis=0)).astype(np.float32)

    if summary != "mean":
        raise ValueError(f"Unsupported summary={summary}")

    # return subjectwise values for hypothesis testing later
    if return_subjectwise:
        return isc_subjectwise, isc_mean
    return isc_mean


# ================================================================
# Main function to compute ISC - Using BrainIAK
# ================================================================
def compute_isc_brainiak(
    data_list: List[np.ndarray],
    *,
    fisher_z: bool = False,
    return_subjectwise: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute ISC using BrainIAK backend.

    Notes
    -----
    BrainIAK supports both subjectwise and summary ISC depending on parameters.

    Returns
    -------
    isc_mean : (F,) or (isc_subjectwise, isc_mean)
    """
    from brainiak.isc import isc as brainiak_isc

    data = np.stack(data_list, axis=0)          # (N, T, F)
    data = np.transpose(data, (1, 0, 2))        # (T, N, F)

    # pairwise=False corresponds to leave-one-out style ISC
    # summary_statistic='mean' returns a summary across subjects
    isc_mean = brainiak_isc(
        data,
        pairwise=False,
        summary_statistic="mean",
    )

    isc_mean = np.asarray(isc_mean)

    if return_subjectwise:
        isc_subj, isc_mean_native = compute_isc(
            data_list,
            fisher_z=fisher_z,
            return_subjectwise=True,
        )
        return isc_subj, isc_mean_native

    return isc_mean
