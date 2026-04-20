"""
parcel_pattern_similarity.py
============================
Per-parcel voxel-pattern ISC, following Chen et al. (2017, Nat Neurosci).

Analysis logic
--------------
For each condition and each parcel:

  1. Load each subject's NPZ (produced by ParcelPatternExtractor)
     → (n_posts, n_brain_voxels) per subject
  2. Average across posts  → (n_brain_voxels,)  one spatial pattern per subject
  3. Stack across subjects → (n_subjects, n_brain_voxels)
  4. For each parcel p:
       slice mask = voxel_parcel_labels == parcel_id_p
       patterns_p = patterns[:, mask]               (n_subjects, n_voxels_p)
       Leave-one-out ISC:
         for each subject i:
           this   = patterns_p[i]                   (n_voxels_p,)
           others = mean of all other rows           (n_voxels_p,)
           r_i    = Pearson r across VOXELS          scalar
         isc_mean_p = mean(r_i)
  5. Result: one ISC value per parcel per condition

Key difference from parcel.py (TSV-based)
------------------------------------------
- parcel.py: Pearson r across posts (timecourse similarity), one scalar/parcel/post
- here:      Pearson r across voxels (spatial pattern similarity within parcel)

Permutation test
----------------
Deferred — to be decided with thesis supervisor.  The placeholder
``permutation_test`` raises NotImplementedError and documents the options.

Public API
----------
    load_subject_patterns(npz_dir, subjects, run_type)
        → patterns (n_subjects, n_brain_voxels),
          voxel_parcel_labels (n_brain_voxels,),
          parcel_ids (n_parcels,), parcel_names (n_parcels,),
          loaded_subjects

    compute_parcel_isc(patterns, voxel_parcel_labels, parcel_ids)
        → isc_mean (n_parcels,), isc_subj (n_subjects, n_parcels)

    run_parcel_isc(npz_dir, subjects, run_types)
        → nested results dict

    results_to_dataframe(results_by_condition, parcel_names)
        → tidy DataFrame (one row per condition × parcel)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from yy_fmri_kit.event_isc.utils import natural_sort_key


__all__ = [
    "load_subject_patterns",
    "load_subject_patterns_postwise",
    "compute_parcel_isc",
    "compute_parcel_isc_postwise",
    "run_parcel_isc",
    "run_parcel_isc_postwise",
    "results_to_dataframe",
    "permutation_test",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_subject_patterns(
    npz_dir  : Path,
    subjects : list[str],
    run_type : str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Load each subject's NPZ for one run type, average across posts.

    NPZ files are expected at:
        npz_dir / subject / *task-{run_type}*_desc-parcel_patterns.npz

    Each NPZ must contain:
        data                : (n_posts, n_brain_voxels) float32
        voxel_parcel_labels : (n_brain_voxels,) int32
        parcel_ids          : (n_parcels,) int32
        parcel_names        : (n_parcels,) str

    Parameters
    ----------
    npz_dir  : output directory of ParcelPatternExtractor.batch_extract
    subjects : ordered subject IDs to load
    run_type : condition name, e.g. 'AntiLeft'

    Returns
    -------
    patterns            : (n_loaded, n_brain_voxels) float64
    voxel_parcel_labels : (n_brain_voxels,) int32
    parcel_ids          : (n_parcels,) int32
    parcel_names        : (n_parcels,) str
    loaded_subjects     : list[str]  subjects in row order (missing ones dropped)
    """
    npz_dir = Path(npz_dir)
    patterns : list[np.ndarray] = []
    loaded   : list[str] = []
    missing  : list[str] = []

    voxel_parcel_labels: Optional[np.ndarray] = None
    parcel_ids         : Optional[np.ndarray] = None
    parcel_names       : Optional[np.ndarray] = None

    for sub in subjects:
        sub_dir = npz_dir / sub
        if not sub_dir.exists():
            missing.append(sub)
            continue

        matches = sorted(sub_dir.glob(f"*task-{run_type}*_desc-parcel_patterns.npz"))
        if not matches:
            missing.append(sub)
            continue

        d    = np.load(matches[0], allow_pickle=True)
        data = d["data"].astype(np.float64)          # (n_posts, n_brain_voxels)

        # Average across posts → (n_brain_voxels,)
        patterns.append(data.mean(axis=0))
        loaded.append(sub)

        # Store atlas metadata from first valid subject (identical across all)
        if voxel_parcel_labels is None:
            voxel_parcel_labels = d["voxel_parcel_labels"].astype(np.int32)
            parcel_ids          = d["parcel_ids"].astype(np.int32)
            parcel_names        = d["parcel_names"].astype(object)

    if missing:
        print(f"  [load] {run_type}: missing NPZ for {missing} — excluded")

    if len(loaded) < 2:
        raise ValueError(
            f"[load] {run_type}: only {len(loaded)} subject(s) loaded — "
            "need at least 2 for ISC."
        )

    return (
        np.stack(patterns),      # (n_loaded, n_brain_voxels)
        voxel_parcel_labels,
        parcel_ids,
        parcel_names,
        loaded,
    )


# ---------------------------------------------------------------------------
# ISC computation
# ---------------------------------------------------------------------------

def compute_parcel_isc(
    patterns           : np.ndarray,
    voxel_parcel_labels: np.ndarray,
    parcel_ids         : np.ndarray,
    min_voxels         : int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Leave-one-out ISC across voxels, computed independently for each parcel.

    For each parcel p and subject i:
        this   = patterns[i, mask_p]                       (n_voxels_p,)
        others = mean of patterns[~i, mask_p]              (n_voxels_p,)
        r_i    = Pearson r across VOXELS (spatial pattern similarity)
    isc_mean_p = mean(r_i across subjects)

    Parameters
    ----------
    patterns            : (n_subjects, n_brain_voxels)
    voxel_parcel_labels : (n_brain_voxels,) int32 — column → parcel int ID
    parcel_ids          : (n_parcels,) int32 — ordered parcel IDs to evaluate
    min_voxels          : parcels with fewer voxels get NaN (default 5)

    Returns
    -------
    isc_mean : (n_parcels,)         group-average ISC per parcel (NaN if skipped)
    isc_subj : (n_subjects, n_parcels)  per-subject ISC values
    """
    n_subjects, _ = patterns.shape
    n_parcels     = len(parcel_ids)

    if n_subjects < 2:
        raise ValueError(f"Need ≥ 2 subjects, got {n_subjects}")

    # Pre-compute the sum across subjects once; we subtract one at a time
    total = patterns.sum(axis=0)     # (n_brain_voxels,)

    isc_mean = np.full(n_parcels, np.nan)
    isc_subj = np.full((n_subjects, n_parcels), np.nan)

    for p_idx, parcel_id in enumerate(parcel_ids):
        mask = voxel_parcel_labels == parcel_id
        n_vox = mask.sum()

        if n_vox < min_voxels:
            continue

        pats_p  = patterns[:, mask]   # (n_subjects, n_vox)
        total_p = total[mask]         # (n_vox,)

        subj_r = np.zeros(n_subjects)
        for i in range(n_subjects):
            this   = pats_p[i]
            others = (total_p - this) / (n_subjects - 1)  # leave-one-out mean

            # Pearson r across voxels
            this_c   = this   - this.mean()
            others_c = others - others.mean()
            num = (this_c * others_c).sum()
            den = np.sqrt((this_c ** 2).sum() * (others_c ** 2).sum())
            subj_r[i] = num / den if den > 1e-12 else 0.0

        isc_mean[p_idx] = subj_r.mean()
        isc_subj[:, p_idx] = subj_r

    return isc_mean, isc_subj


# ---------------------------------------------------------------------------
# Full analysis runner
# ---------------------------------------------------------------------------

def run_parcel_isc(
    npz_dir  : Path,
    subjects : list[str],
    run_types: list[str],
    min_voxels: int = 5,
) -> dict:
    """
    Run per-parcel voxel-pattern ISC for all conditions.

    Parameters
    ----------
    npz_dir    : output directory of ParcelPatternExtractor.batch_extract
    subjects   : left-wing subject IDs
    run_types  : list of conditions, e.g. ['AntiLeft', 'AntiRight', ...]
    min_voxels : minimum voxels required for a parcel to be analysed

    Returns
    -------
    results : {
        run_type: {
            'isc_mean'          : (n_parcels,),
            'isc_subj'          : (n_subjects, n_parcels),
            'parcel_ids'        : (n_parcels,) int32,
            'parcel_names'      : (n_parcels,) str,
            'voxel_parcel_labels': (n_brain_voxels,) int32,
            'n_subjects'        : int,
            'subjects'          : list[str],
        }
    }
    """
    npz_dir = Path(npz_dir)
    results : dict = {}

    for run_type in run_types:
        print(f"\nCondition: {run_type}")
        print("-" * 40)

        try:
            patterns, vox_labels, parcel_ids, parcel_names, loaded = \
                load_subject_patterns(npz_dir, subjects, run_type)
        except ValueError as e:
            print(f"  SKIPPED: {e}")
            continue

        n_subjects, n_brain_voxels = patterns.shape
        print(f"  Subjects: {n_subjects}  |  Brain voxels: {n_brain_voxels}  "
              f"|  Parcels: {len(parcel_ids)}")

        isc_mean, isc_subj = compute_parcel_isc(
            patterns, vox_labels, parcel_ids, min_voxels=min_voxels
        )
        n_valid = np.sum(~np.isnan(isc_mean))
        print(f"  ISC computed for {n_valid} parcels  "
              f"(mean across parcels: {np.nanmean(isc_mean):.4f}  "
              f"max: {np.nanmax(isc_mean):.4f})")

        results[run_type] = {
            "isc_mean"           : isc_mean,
            "isc_subj"           : isc_subj,
            "parcel_ids"         : parcel_ids,
            "parcel_names"       : parcel_names,
            "voxel_parcel_labels": vox_labels,
            "n_subjects"         : n_subjects,
            "subjects"           : loaded,
        }

    return results


# ---------------------------------------------------------------------------
# Results packaging
# ---------------------------------------------------------------------------

def results_to_dataframe(
    results_by_condition: dict,
) -> pd.DataFrame:
    """
    Flatten results from run_parcel_isc into a tidy DataFrame.

    One row per condition × parcel.

    Parameters
    ----------
    results_by_condition : output of run_parcel_isc

    Returns
    -------
    pd.DataFrame with columns:
        condition, parcel_id, parcel_name, isc_mean, n_subjects
    """
    rows = []
    for condition, r in results_by_condition.items():
        for p_idx, (pid, pname) in enumerate(
            zip(r["parcel_ids"], r["parcel_names"])
        ):
            rows.append({
                "condition" : condition,
                "parcel_id" : int(pid),
                "parcel_name": str(pname),
                "isc_mean"  : round(float(r["isc_mean"][p_idx]), 6),
                "n_subjects": r["n_subjects"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Approach B: Post-wise pattern ISC (Chen et al. 2017)
# ---------------------------------------------------------------------------

def load_subject_patterns_postwise(
    npz_dir  : Path,
    subjects : list[str],
    run_type : str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Load per-post patterns without averaging across posts.

    Post order is aligned to the intersection of post IDs across all loaded
    subjects (handles cases where a subject is missing a post).

    Parameters
    ----------
    npz_dir  : output directory of ParcelPatternExtractor.batch_extract
    subjects : ordered subject IDs to load
    run_type : condition name, e.g. 'AntiLeft'

    Returns
    -------
    patterns            : (n_subjects, n_posts, n_brain_voxels) float64
    post_ids            : (n_posts,) str   — common post IDs in natural sort order
    voxel_parcel_labels : (n_brain_voxels,) int32
    parcel_ids          : (n_parcels,) int32
    parcel_names        : (n_parcels,) str
    loaded_subjects     : list[str]
    """
    npz_dir = Path(npz_dir)
    all_data : list[tuple[np.ndarray, np.ndarray]] = []
    loaded   : list[str] = []
    missing  : list[str] = []

    voxel_parcel_labels: Optional[np.ndarray] = None
    parcel_ids         : Optional[np.ndarray] = None
    parcel_names       : Optional[np.ndarray] = None

    for sub in subjects:
        sub_dir = npz_dir / sub
        if not sub_dir.exists():
            missing.append(sub)
            continue

        matches = sorted(sub_dir.glob(f"*task-{run_type}*_desc-parcel_patterns.npz"))
        if not matches:
            missing.append(sub)
            continue

        d           = np.load(matches[0], allow_pickle=True)
        data        = d["data"].astype(np.float64)      # (n_posts, n_brain_voxels)
        post_ids_sub = d["post_ids"].astype(str)

        all_data.append((data, post_ids_sub))
        loaded.append(sub)

        if voxel_parcel_labels is None:
            voxel_parcel_labels = d["voxel_parcel_labels"].astype(np.int32)
            parcel_ids          = d["parcel_ids"].astype(np.int32)
            parcel_names        = d["parcel_names"].astype(object)

    if missing:
        print(f"  [load] {run_type}: missing NPZ for {missing} — excluded")

    if len(loaded) < 2:
        raise ValueError(
            f"[load] {run_type}: only {len(loaded)} subject(s) loaded — "
            "need at least 2 for ISC."
        )

    # Intersect post IDs across all subjects (handles missing posts)
    common = sorted(
        set.intersection(*[set(pids.tolist()) for _, pids in all_data]),
        key=natural_sort_key,
    )
    if not common:
        raise ValueError(f"[load] {run_type}: no post IDs common across all subjects")

    n_dropped = len(all_data[0][1]) - len(common)
    if n_dropped:
        print(f"  [load] {run_type}: {n_dropped} post(s) dropped (not present in all subjects)")

    # Align each subject's rows to the common post order
    patterns = []
    for data, post_ids_sub in all_data:
        pid_list = post_ids_sub.tolist()
        idx = [pid_list.index(p) for p in common]
        patterns.append(data[idx])                      # (n_common_posts, n_brain_voxels)

    return (
        np.stack(patterns),             # (n_subjects, n_posts, n_brain_voxels)
        np.array(common, dtype=str),    # (n_posts,)
        voxel_parcel_labels,
        parcel_ids,
        parcel_names,
        loaded,
    )


def compute_parcel_isc_postwise(
    patterns_per_post  : np.ndarray,
    voxel_parcel_labels: np.ndarray,
    parcel_ids         : np.ndarray,
    min_voxels         : int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Post-wise LOO ISC: for each post separately, compute Pearson r across voxels
    between each subject and the leave-one-out group mean.  Average across posts.

    Following Chen et al. (2017): correlations are computed on individual-post
    patterns rather than on the condition-level mean pattern.

    Parameters
    ----------
    patterns_per_post   : (n_subjects, n_posts, n_brain_voxels)
    voxel_parcel_labels : (n_brain_voxels,) int32
    parcel_ids          : (n_parcels,) int32
    min_voxels          : parcels with fewer voxels get NaN (default 5)

    Returns
    -------
    isc_mean : (n_parcels,)              mean over subjects and posts
    isc_subj : (n_subjects, n_parcels)   per-subject ISC (mean over posts)
    """
    n_subjects, n_posts, _ = patterns_per_post.shape
    n_parcels = len(parcel_ids)

    if n_subjects < 2:
        raise ValueError(f"Need ≥ 2 subjects, got {n_subjects}")

    isc_mean = np.full(n_parcels, np.nan)
    isc_subj = np.full((n_subjects, n_parcels), np.nan)

    for p_idx, parcel_id in enumerate(parcel_ids):
        mask  = voxel_parcel_labels == parcel_id
        n_vox = mask.sum()

        if n_vox < min_voxels:
            continue

        pats_p = patterns_per_post[:, :, mask]   # (n_subjects, n_posts, n_vox)

        subj_r_per_post = np.zeros((n_subjects, n_posts))

        for post_i in range(n_posts):
            post_pats  = pats_p[:, post_i, :]         # (n_subjects, n_vox)
            total_post = post_pats.sum(axis=0)         # (n_vox,)

            for i in range(n_subjects):
                this   = post_pats[i]
                others = (total_post - this) / (n_subjects - 1)

                this_c   = this   - this.mean()
                others_c = others - others.mean()
                num = (this_c * others_c).sum()
                den = np.sqrt((this_c ** 2).sum() * (others_c ** 2).sum())
                subj_r_per_post[i, post_i] = num / den if den > 1e-12 else 0.0

        subj_r = subj_r_per_post.mean(axis=1)     # (n_subjects,) — mean over posts
        isc_mean[p_idx]    = subj_r.mean()
        isc_subj[:, p_idx] = subj_r

    return isc_mean, isc_subj


def run_parcel_isc_postwise(
    npz_dir   : Path,
    subjects  : list[str],
    run_types : list[str],
    min_voxels: int = 5,
) -> dict:
    """
    Run post-wise per-parcel ISC for all conditions (Approach B — Chen et al. 2017).

    Identical return structure to run_parcel_isc, with an additional 'n_posts' key.

    Parameters
    ----------
    npz_dir    : output directory of ParcelPatternExtractor.batch_extract
    subjects   : subject IDs
    run_types  : list of conditions
    min_voxels : minimum voxels required for a parcel to be analysed

    Returns
    -------
    results : same structure as run_parcel_isc with extra key 'n_posts'
    """
    npz_dir = Path(npz_dir)
    results : dict = {}

    for run_type in run_types:
        print(f"\nCondition: {run_type}  [post-wise ISC]")
        print("-" * 40)

        try:
            patterns, post_ids, vox_labels, parcel_ids, parcel_names, loaded = \
                load_subject_patterns_postwise(npz_dir, subjects, run_type)
        except ValueError as e:
            print(f"  SKIPPED: {e}")
            continue

        n_subjects, n_posts, n_brain_voxels = patterns.shape
        print(f"  Subjects: {n_subjects}  |  Posts: {n_posts}  "
              f"|  Brain voxels: {n_brain_voxels}  |  Parcels: {len(parcel_ids)}")

        isc_mean, isc_subj = compute_parcel_isc_postwise(
            patterns, vox_labels, parcel_ids, min_voxels=min_voxels
        )
        n_valid = np.sum(~np.isnan(isc_mean))
        print(f"  ISC computed for {n_valid} parcels  "
              f"(mean: {np.nanmean(isc_mean):.4f}  max: {np.nanmax(isc_mean):.4f})")

        results[run_type] = {
            "isc_mean"           : isc_mean,
            "isc_subj"           : isc_subj,
            "parcel_ids"         : parcel_ids,
            "parcel_names"       : parcel_names,
            "voxel_parcel_labels": vox_labels,
            "n_subjects"         : n_subjects,
            "n_posts"            : n_posts,
            "subjects"           : loaded,
        }

    return results


# ---------------------------------------------------------------------------
# Permutation test (placeholder — deferred)
# ---------------------------------------------------------------------------

def permutation_test(
    patterns           : np.ndarray,
    voxel_parcel_labels: np.ndarray,
    parcel_ids         : np.ndarray,
    n_perms            : int = 1000,
    seed               : int = 42,
    method             : str = "post_phase_shift",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Permutation test for per-parcel voxel-pattern ISC.

    NOT YET IMPLEMENTED — to be agreed with thesis supervisor.

    Three candidate methods:
    ------------------------
    'post_phase_shift'  (recommended for event-locked data)
        Treat each subject's 18-post sequence as a short timeseries.
        Apply a random circular shift to the post order (independently per
        subject), re-average to a condition pattern, recompute ISC.
        Breaks inter-subject post correspondence while preserving each
        subject's within-condition post-to-post structure.

    'tr_phase_shift'    (gold standard for continuous ISC)
        Circularly shift each subject's raw BOLD timeseries (before
        post-averaging) by a random number of TRs.  Requires re-loading
        and re-extracting from NIfTI files — computationally heavy.

    'subject_label_shuffle'
        Permute the subject-row order of the (n_subjects, n_brain_voxels)
        matrix. Breaks inter-subject correspondence without any temporal
        structure preservation.  Fastest; used in the ROI analysis.

    Parameters
    ----------
    patterns            : (n_subjects, n_brain_voxels)
    voxel_parcel_labels : (n_brain_voxels,) int32
    parcel_ids          : (n_parcels,) int32
    n_perms             : number of permutations
    seed                : random seed
    method              : one of the three methods above

    Returns
    -------
    p_values  : (n_parcels,)
    null_dist : (n_perms, n_parcels)

    Raises
    ------
    NotImplementedError : always (pending supervisor decision)
    """
    raise NotImplementedError(
        "Permutation test deferred — method to be agreed with thesis supervisor.\n"
        "Candidates: 'post_phase_shift', 'tr_phase_shift', 'subject_label_shuffle'.\n"
        "See function docstring for details."
    )
