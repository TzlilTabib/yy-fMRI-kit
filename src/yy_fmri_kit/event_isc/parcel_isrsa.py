"""
parcel_isrsa.py
===============
IS-RSA (Inter-Subject RSA) using voxel-level spatial patterns within parcels,
following the same two-approach design as ``parcel_pattern_similarity.py``.

Scientific logic
----------------
For each parcel and each condition:
  1. Build an N×N **neural similarity matrix**: for every pair of subjects,
     compute Pearson r between their activation patterns *across voxels* within
     the parcel.  This measures "do these two subjects activate the same spatial
     arrangement of voxels in this parcel?"

  2. Build an N×N **behavioral similarity matrix** from pairwise affective
     polarization scores (affpol_thermo).

  3. Correlate the upper triangles of the two matrices → IS-RSA r per parcel.

  4. Permutation test: shuffle subject labels on the behavioral matrix
     (rows + columns simultaneously), recompute IS-RSA r → null distribution.

  5. FDR correction (Benjamini–Hochberg, q=0.05).

Two approaches for the neural similarity matrix
-----------------------------------------------
**Approach A — mean-then-correlate**
    Average each subject's patterns across all posts within the condition
    → one ``(n_voxels_in_parcel,)`` spatial map per subject.
    Pairwise Pearson r across voxels → N×N matrix.
    Asks: *do pairs of subjects share the same stable condition-level spatial
    representation in this parcel?*

**Approach B — post-wise-then-average**
    For each post separately: compute pairwise Pearson r across voxels, then
    average the resulting N×N matrices over posts.
    Asks: *on average, do subjects show similar spatial activation for the same
    individual post in this parcel?*

Key difference from the old approach (``parcel.py``)
-----------------------------------------------------
The old ``compute_brain_behavior_rsa`` correlated subjects' activation
*timecourses across posts* (one scalar per parcel per post).  The neural
similarity there was based on whether subjects respond consistently to the same
posts (temporal covariance), not on whether their spatial patterns are alike.
Both new approaches here use Pearson r *across voxels* — true spatial pattern
similarity.

Input data
----------
Patterns come from NPZ files produced by ``ParcelPatternExtractor``
(``scripts/extract_parcel_patterns.py``), loaded via
``load_subject_patterns_postwise`` from ``parcel_pattern_similarity.py``.
Shape per condition: ``(n_subjects, n_posts, n_brain_voxels)``

Public API
----------
    make_behavioral_sim(affiliation, subjects, scale_max=100.0)
        → (n_subjects, n_subjects)

    merge_pattern_conditions(patterns_dict, groups)
        → {group_name: (n_subjects, n_merged_posts, n_brain_voxels)}

    compute_neural_similarity_a(patterns_per_post, vox_labels, parcel_ids)
        → neural_sim (n_subjects, n_subjects, n_parcels)

    compute_neural_similarity_b(patterns_per_post, vox_labels, parcel_ids)
        → neural_sim (n_subjects, n_subjects, n_parcels)

    compute_isrsa(neural_sim, beh_sim)
        → rsa_r (n_parcels,)

    permutation_test_isrsa(neural_sim, beh_sim, n_perms, seed, verbose)
        → obs_r (n_parcels,), p_values (n_parcels,), null_dist (n_perms, n_parcels)

    run_isrsa_both(patterns_per_post, vox_labels, parcel_ids, beh_sim, ...)
        → results_a, results_b  (each: dict with obs_r, p_vals, null_dist,
                                        rejected, p_fdr)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import fdrcorrection


__all__ = [
    "make_behavioral_sim",
    "merge_pattern_conditions",
    "compute_neural_similarity_a",
    "compute_neural_similarity_b",
    "compute_isrsa",
    "permutation_test_isrsa",
    "run_isrsa_both",
]


# ---------------------------------------------------------------------------
# Behavioral similarity matrix
# ---------------------------------------------------------------------------

def make_behavioral_sim(
    affiliation: dict[str, float],
    subjects   : list[str],
    scale_max  : float = 100.0,
) -> np.ndarray:
    """
    Build an N×N behavioral similarity matrix from pairwise score differences.

    similarity(i, j) = 1 − |score_i − score_j| / scale_max

    Parameters
    ----------
    affiliation : {bids_id: affpol_thermo_score}
    subjects    : ordered list of subject IDs
    scale_max   : normalisation constant (default 100 — matches the existing
                  make_behavioral_rdm in parcel.py)

    Returns
    -------
    beh_sim : (n_subjects, n_subjects) float64
    """
    n    = len(subjects)
    sims = np.zeros((n, n))
    for i, si in enumerate(subjects):
        for j, sj in enumerate(subjects):
            sims[i, j] = 1.0 - abs(affiliation[si] - affiliation[sj]) / scale_max
    return sims


# ---------------------------------------------------------------------------
# Pattern merging
# ---------------------------------------------------------------------------

def merge_pattern_conditions(
    patterns_dict: dict[str, np.ndarray],
    groups       : dict[str, list[str]],
) -> dict[str, np.ndarray]:
    """
    Concatenate per-condition patterns along the posts axis.

    Parameters
    ----------
    patterns_dict : {run_type: (n_subjects, n_posts, n_brain_voxels)}
    groups        : {group_name: [run_type, ...]}
                    e.g. {"agreed": ["AntiRight", "ProLeft"]}

    Returns
    -------
    merged : {group_name: (n_subjects, n_merged_posts, n_brain_voxels)}
    """
    merged: dict[str, np.ndarray] = {}
    for group_name, run_types in groups.items():
        arrays = [patterns_dict[rt] for rt in run_types if rt in patterns_dict]
        if not arrays:
            raise KeyError(
                f"merge_pattern_conditions: none of {run_types} found in "
                f"patterns_dict (keys: {list(patterns_dict.keys())})"
            )
        merged[group_name] = np.concatenate(arrays, axis=1)
    return merged


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _upper_tri(mat: np.ndarray) -> np.ndarray:
    """Upper triangle (k=1) of a square matrix as a 1-D vector."""
    idx = np.triu_indices(mat.shape[0], k=1)
    return mat[idx]


def _pairwise_pearson_r(pats_p: np.ndarray) -> np.ndarray:
    """
    Vectorised all-pairs Pearson r across the voxel dimension.

    Parameters
    ----------
    pats_p : (n_subjects, n_voxels) — activation patterns in one parcel

    Returns
    -------
    r_mat : (n_subjects, n_subjects)  symmetric, diagonal = 1
    """
    # Mean-centre each subject's pattern across voxels
    pats_c = pats_p - pats_p.mean(axis=1, keepdims=True)
    norms  = np.sqrt((pats_c ** 2).sum(axis=1))          # (n_subjects,)
    denom  = np.outer(norms, norms)                        # (n_subjects, n_subjects)
    cov    = pats_c @ pats_c.T                             # (n_subjects, n_subjects)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_mat = np.where(denom > 1e-12, cov / denom, 0.0)
    return r_mat


def _compute_isrsa_vectorized(
    neural_sim: np.ndarray,   # (n_subjects, n_subjects, n_parcels)
    beh_sim   : np.ndarray,   # (n_subjects, n_subjects)
) -> np.ndarray:              # (n_parcels,)
    """
    Correlate upper triangles of neural and behavioral similarity matrices
    for all parcels simultaneously.
    """
    n_subjects, _, n_parcels = neural_sim.shape
    idx = np.triu_indices(n_subjects, k=1)

    beh_vec = beh_sim[idx]                                  # (n_pairs,)
    neu_mat = neural_sim[idx[0], idx[1], :]                 # (n_pairs, n_parcels)

    beh_c = beh_vec - beh_vec.mean()
    neu_c = neu_mat - neu_mat.mean(axis=0, keepdims=True)   # (n_pairs, n_parcels)

    num     = (beh_c[:, None] * neu_c).sum(axis=0)          # (n_parcels,)
    den_beh = np.sqrt((beh_c ** 2).sum())
    den_neu = np.sqrt((neu_c ** 2).sum(axis=0))              # (n_parcels,)

    rsa_r = np.full(n_parcels, np.nan)
    valid = (~np.isnan(den_neu)) & (den_neu > 1e-12)
    rsa_r[valid] = num[valid] / (den_beh * den_neu[valid])
    return rsa_r


# ---------------------------------------------------------------------------
# Neural similarity matrices
# ---------------------------------------------------------------------------

def compute_neural_similarity_a(
    patterns_per_post  : np.ndarray,
    voxel_parcel_labels: np.ndarray,
    parcel_ids         : np.ndarray,
    min_voxels         : int = 5,
) -> np.ndarray:
    """
    Approach A — mean-then-correlate.

    Average each subject's patterns across posts → one (n_voxels_p,) map per
    subject per parcel, then pairwise Pearson r across voxels.

    Parameters
    ----------
    patterns_per_post   : (n_subjects, n_posts, n_brain_voxels)
    voxel_parcel_labels : (n_brain_voxels,) int32
    parcel_ids          : (n_parcels,) int32
    min_voxels          : parcels with fewer voxels → NaN column

    Returns
    -------
    neural_sim : (n_subjects, n_subjects, n_parcels)
                 NaN slices for parcels below min_voxels
    """
    n_subjects, _, _  = patterns_per_post.shape
    n_parcels = len(parcel_ids)

    mean_patterns = patterns_per_post.mean(axis=1)          # (n_subjects, n_brain_voxels)
    neural_sim    = np.full((n_subjects, n_subjects, n_parcels), np.nan)

    for p_idx, parcel_id in enumerate(parcel_ids):
        mask  = voxel_parcel_labels == parcel_id
        if mask.sum() < min_voxels:
            continue
        pats_p = mean_patterns[:, mask]                      # (n_subjects, n_voxels_p)
        neural_sim[:, :, p_idx] = _pairwise_pearson_r(pats_p)

    return neural_sim


def compute_neural_similarity_b(
    patterns_per_post  : np.ndarray,
    voxel_parcel_labels: np.ndarray,
    parcel_ids         : np.ndarray,
    min_voxels         : int = 5,
) -> np.ndarray:
    """
    Approach B — post-wise-then-average.

    For each post, compute pairwise Pearson r across voxels; average over posts.

    Parameters
    ----------
    patterns_per_post   : (n_subjects, n_posts, n_brain_voxels)
    voxel_parcel_labels : (n_brain_voxels,) int32
    parcel_ids          : (n_parcels,) int32
    min_voxels          : parcels with fewer voxels → NaN column

    Returns
    -------
    neural_sim : (n_subjects, n_subjects, n_parcels)
    """
    n_subjects, n_posts, _ = patterns_per_post.shape
    n_parcels = len(parcel_ids)
    neural_sim = np.full((n_subjects, n_subjects, n_parcels), np.nan)

    for p_idx, parcel_id in enumerate(parcel_ids):
        mask  = voxel_parcel_labels == parcel_id
        if mask.sum() < min_voxels:
            continue
        pats_p = patterns_per_post[:, :, mask]              # (n_subjects, n_posts, n_vox)

        r_per_post = np.zeros((n_subjects, n_subjects, n_posts))
        for k in range(n_posts):
            r_per_post[:, :, k] = _pairwise_pearson_r(pats_p[:, k, :])

        neural_sim[:, :, p_idx] = r_per_post.mean(axis=2)

    return neural_sim


# ---------------------------------------------------------------------------
# IS-RSA computation
# ---------------------------------------------------------------------------

def compute_isrsa(
    neural_sim: np.ndarray,
    beh_sim   : np.ndarray,
) -> np.ndarray:
    """
    Correlate upper triangles of neural and behavioral similarity matrices.

    Parameters
    ----------
    neural_sim : (n_subjects, n_subjects, n_parcels)
    beh_sim    : (n_subjects, n_subjects)

    Returns
    -------
    rsa_r : (n_parcels,)  Pearson r per parcel (NaN for skipped parcels)
    """
    return _compute_isrsa_vectorized(neural_sim, beh_sim)


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def permutation_test_isrsa(
    neural_sim: np.ndarray,
    beh_sim   : np.ndarray,
    n_perms   : int = 1000,
    seed      : int = 42,
    verbose   : bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Permutation test for IS-RSA: shuffle subject labels on the behavioral
    matrix (rows + columns simultaneously), recompute IS-RSA r.

    The neural similarity matrix is precomputed once and reused for all
    permutations — only the behavioral matrix is shuffled.

    Parameters
    ----------
    neural_sim : (n_subjects, n_subjects, n_parcels)
    beh_sim    : (n_subjects, n_subjects)
    n_perms    : number of permutations (default 1000)
    seed       : random seed (default 42)
    verbose    : print progress every 100 perms

    Returns
    -------
    obs_r     : (n_parcels,)          observed IS-RSA r
    p_values  : (n_parcels,)          one-tailed p (proportion of null ≥ obs)
    null_dist : (n_perms, n_parcels)  full null distribution
    """
    n_subjects = beh_sim.shape[0]
    n_parcels  = neural_sim.shape[2]

    obs_r     = compute_isrsa(neural_sim, beh_sim)
    rng       = np.random.default_rng(seed)
    null_dist = np.full((n_perms, n_parcels), np.nan)

    for i in range(n_perms):
        if verbose and (i + 1) % 100 == 0:
            print(f"  Permutation {i + 1}/{n_perms}")
        perm          = rng.permutation(n_subjects)
        beh_perm      = beh_sim[np.ix_(perm, perm)]
        null_dist[i]  = _compute_isrsa_vectorized(neural_sim, beh_perm)

    valid    = ~np.isnan(obs_r)
    p_values = np.full(n_parcels, np.nan)
    p_values[valid] = np.mean(null_dist[:, valid] >= obs_r[valid], axis=0)

    return obs_r, p_values, null_dist


# ---------------------------------------------------------------------------
# Combined runner
# ---------------------------------------------------------------------------

def run_isrsa_both(
    patterns_per_post  : np.ndarray,
    voxel_parcel_labels: np.ndarray,
    parcel_ids         : np.ndarray,
    beh_sim            : np.ndarray,
    n_perms            : int   = 1000,
    seed               : int   = 42,
    min_voxels         : int   = 5,
    fdr_q              : float = 0.05,
    verbose            : bool  = True,
) -> tuple[dict, dict]:
    """
    Run IS-RSA with both approaches (A and B) for one condition.

    Parameters
    ----------
    patterns_per_post   : (n_subjects, n_posts, n_brain_voxels)
    voxel_parcel_labels : (n_brain_voxels,) int32
    parcel_ids          : (n_parcels,) int32
    beh_sim             : (n_subjects, n_subjects) behavioral similarity matrix
    n_perms             : permutations for null distribution (default 1000)
    seed                : random seed (default 42)
    min_voxels          : minimum voxels per parcel (default 5)
    fdr_q               : FDR threshold (default 0.05)
    verbose             : print progress

    Returns
    -------
    results_a, results_b : each is a dict with keys:
        neural_sim  : (n_subjects, n_subjects, n_parcels)
        obs_r       : (n_parcels,)
        p_vals      : (n_parcels,)
        null_dist   : (n_perms, n_parcels)
        rejected    : (n_parcels,) bool
        p_fdr       : (n_parcels,)
    """
    def _run_one(neural_sim, label):
        if verbose:
            print(f"  [{label}] Running {n_perms} permutations ...")
        obs_r, p_vals, null_dist = permutation_test_isrsa(
            neural_sim, beh_sim, n_perms=n_perms, seed=seed, verbose=verbose,
        )
        valid_mask = ~np.isnan(p_vals)
        rejected   = np.zeros(len(p_vals), dtype=bool)
        p_fdr      = np.full(len(p_vals), np.nan)
        if valid_mask.sum() > 0:
            rej_v, p_fdr_v = fdrcorrection(p_vals[valid_mask], alpha=fdr_q)
            rejected[valid_mask] = rej_v
            p_fdr[valid_mask]    = p_fdr_v
        n_sig = int(rejected.sum())
        if verbose:
            print(f"  [{label}] Done — {n_sig} FDR-significant parcels  "
                  f"mean r={float(np.nanmean(obs_r)):.4f}  "
                  f"max r={float(np.nanmax(obs_r)):.4f}")
        return {
            "neural_sim": neural_sim,
            "obs_r"     : obs_r,
            "p_vals"    : p_vals,
            "null_dist" : null_dist,
            "rejected"  : rejected,
            "p_fdr"     : p_fdr,
        }

    if verbose:
        print("  Building Approach A neural similarity (mean-then-correlate) ...")
    neural_sim_a = compute_neural_similarity_a(
        patterns_per_post, voxel_parcel_labels, parcel_ids, min_voxels,
    )
    results_a = _run_one(neural_sim_a, "Approach A")

    if verbose:
        print("  Building Approach B neural similarity (post-wise-then-average) ...")
    neural_sim_b = compute_neural_similarity_b(
        patterns_per_post, voxel_parcel_labels, parcel_ids, min_voxels,
    )
    results_b = _run_one(neural_sim_b, "Approach B")

    return results_a, results_b
