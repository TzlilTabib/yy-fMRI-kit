"""
roi_similarity.py
=================
Voxel-pattern ISC within an ROI, following Chen et al. (2017, Nat Neurosci).

Analysis logic
--------------
For each condition and ROI:

  1. Load each subject's NPZ → (n_posts, n_voxels)
  2. Average across posts    → (n_voxels,)  one spatial pattern per subject
  3. Stack across subjects   → (n_subjects, n_voxels)
  4. Leave-one-out ISC:
       for each subject i:
           their_pattern  = data[i]                          (n_voxels,)
           others_mean    = mean of all other subjects        (n_voxels,)
           r_i            = Pearson r across VOXELS           scalar
       isc_mean = mean(r_i across subjects)
  5. Permutation test: shuffle subject rows, recompute ISC → null distribution

Key difference from parcel.py
-------------------------------
  - Correlation is computed ACROSS VOXELS (spatial pattern similarity)
  - NOT across posts (which would only measure activation timecourse covariance)

Public API
----------
    load_subject_patterns(npz_dir, subjects, run_type)
        → patterns (n_subjects, n_voxels), loaded_subjects

    compute_pattern_isc(patterns)
        → isc_mean (scalar), isc_subj (n_subjects,)

    permutation_test(patterns, n_perms, seed)
        → p_value, null_dist (n_perms,)

    run_roi_isc(npz_dir, subjects, run_types, roi_name, n_perms, seed)
        → results dict
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from yy_fmri_kit.event_isc.utils import natural_sort_key


__all__ = [
    "load_subject_patterns",
    "load_subject_patterns_postwise",
    "compute_pattern_isc",
    "compute_pattern_isc_postwise",
    "permutation_test",
    "run_roi_isc",
    "run_roi_isc_postwise",
    "results_to_dataframe",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_subject_patterns(
    npz_dir: Path,
    subjects: list[str],
    run_type: str,
) -> tuple[np.ndarray, list[str]]:
    """
    Load each subject's NPZ for one run type and average across posts.

    Parameters
    ----------
    npz_dir  : directory containing per-subject sub-folders with NPZ files
    subjects : ordered list of subject IDs (e.g. ['sub-1', 'sub-6', ...])
    run_type : e.g. 'AntiLeft'

    Returns
    -------
    patterns        : (n_loaded, n_voxels)  float64 array, one row per subject
    loaded_subjects : list of subject IDs in the same row order
                      (subjects with missing NPZ are silently dropped)
    """
    npz_dir = Path(npz_dir)
    patterns: list[np.ndarray] = []
    loaded: list[str] = []
    missing: list[str] = []

    for sub in subjects:
        sub_dir = npz_dir / sub
        if not sub_dir.exists():
            missing.append(sub)
            continue

        matches = sorted(sub_dir.glob(f"*task-{run_type}*_desc-roi_patterns.npz"))
        if not matches:
            missing.append(sub)
            continue

        d = np.load(matches[0], allow_pickle=True)
        data = d["data"].astype(np.float64)  # (n_posts, n_voxels)

        # Average across posts → (n_voxels,)
        patterns.append(data.mean(axis=0))
        loaded.append(sub)

    if missing:
        print(f"  [load] {run_type}: missing NPZ for {missing} — excluded")

    if len(loaded) < 2:
        raise ValueError(
            f"[load] {run_type}: only {len(loaded)} subject(s) loaded — "
            "need at least 2 for ISC."
        )

    return np.stack(patterns), loaded  # (n_subjects, n_voxels)


# ---------------------------------------------------------------------------
# ISC computation
# ---------------------------------------------------------------------------

def compute_pattern_isc(
    patterns: np.ndarray,
) -> tuple[float, np.ndarray]:
    """
    Leave-one-out ISC across voxels.

    For each subject i:
      - their_pattern = patterns[i]                     (n_voxels,)
      - others_mean   = mean of all other rows          (n_voxels,)
      - r_i           = Pearson r(their_pattern, others_mean)   scalar

    Pearson r is computed ACROSS VOXELS (spatial pattern similarity).

    Parameters
    ----------
    patterns : (n_subjects, n_voxels)

    Returns
    -------
    isc_mean : float   group-average ISC
    isc_subj : (n_subjects,)  per-subject ISC values
    """
    n_subjects, n_voxels = patterns.shape
    if n_subjects < 2:
        raise ValueError(f"Need at least 2 subjects, got {n_subjects}")
    if n_voxels < 2:
        raise ValueError(f"Need at least 2 voxels for pattern correlation, got {n_voxels}")

    total = patterns.sum(axis=0)  # (n_voxels,)
    isc_subj = np.zeros(n_subjects)

    for i in range(n_subjects):
        this   = patterns[i]                          # (n_voxels,)
        others = (total - this) / (n_subjects - 1)   # (n_voxels,)

        # Pearson r across voxels
        this_c   = this   - this.mean()
        others_c = others - others.mean()
        num      = (this_c * others_c).sum()
        den      = np.sqrt((this_c ** 2).sum() * (others_c ** 2).sum())

        isc_subj[i] = num / den if den > 1e-12 else 0.0

    return float(isc_subj.mean()), isc_subj


# ---------------------------------------------------------------------------
# Approach B: Post-wise pattern ISC (Chen et al. 2017)
# ---------------------------------------------------------------------------

def load_subject_patterns_postwise(
    npz_dir : Path,
    subjects: list[str],
    run_type: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load per-post patterns without averaging across posts.

    Post order is aligned to the intersection of post IDs across all loaded
    subjects (handles cases where a subject is missing a post).

    Parameters
    ----------
    npz_dir  : directory containing per-subject sub-folders with NPZ files
    subjects : ordered list of subject IDs
    run_type : e.g. 'AntiLeft'

    Returns
    -------
    patterns    : (n_subjects, n_posts, n_voxels) float64
    post_ids    : (n_posts,) str   — common post IDs in natural sort order
    loaded_subs : list[str]
    """
    npz_dir = Path(npz_dir)
    all_data : list[tuple[np.ndarray, np.ndarray]] = []
    loaded   : list[str] = []
    missing  : list[str] = []

    for sub in subjects:
        sub_dir = npz_dir / sub
        if not sub_dir.exists():
            missing.append(sub)
            continue

        matches = sorted(sub_dir.glob(f"*task-{run_type}*_desc-roi_patterns.npz"))
        if not matches:
            missing.append(sub)
            continue

        d           = np.load(matches[0], allow_pickle=True)
        data        = d["data"].astype(np.float64)     # (n_posts, n_voxels)
        post_ids_sub = d["post_ids"].astype(str)

        all_data.append((data, post_ids_sub))
        loaded.append(sub)

    if missing:
        print(f"  [load] {run_type}: missing NPZ for {missing} — excluded")

    if len(loaded) < 2:
        raise ValueError(
            f"[load] {run_type}: only {len(loaded)} subject(s) loaded — "
            "need at least 2 for ISC."
        )

    # Intersect post IDs across all subjects
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
        patterns.append(data[idx])                     # (n_common_posts, n_voxels)

    return (
        np.stack(patterns),            # (n_subjects, n_posts, n_voxels)
        np.array(common, dtype=str),   # (n_posts,)
        loaded,
    )


def compute_pattern_isc_postwise(
    patterns_per_post: np.ndarray,
) -> tuple[float, np.ndarray]:
    """
    Post-wise LOO ISC across voxels (Chen et al. 2017 style).

    For each post_i and subject i:
        r_i_post = Pearson r(subject_i's pattern for post_i,
                             LOO mean of others' patterns for post_i)
    isc_subj_i = mean(r_i_post across posts)
    isc_mean   = mean(isc_subj_i across subjects)

    Parameters
    ----------
    patterns_per_post : (n_subjects, n_posts, n_voxels)

    Returns
    -------
    isc_mean : float
    isc_subj : (n_subjects,)  per-subject ISC (mean across posts)
    """
    n_subjects, n_posts, n_voxels = patterns_per_post.shape
    if n_subjects < 2:
        raise ValueError(f"Need at least 2 subjects, got {n_subjects}")
    if n_voxels < 2:
        raise ValueError(f"Need at least 2 voxels, got {n_voxels}")

    subj_r_per_post = np.zeros((n_subjects, n_posts))

    for post_i in range(n_posts):
        post_pats  = patterns_per_post[:, post_i, :]   # (n_subjects, n_voxels)
        total_post = post_pats.sum(axis=0)             # (n_voxels,)

        for i in range(n_subjects):
            this   = post_pats[i]
            others = (total_post - this) / (n_subjects - 1)

            this_c   = this   - this.mean()
            others_c = others - others.mean()
            num = (this_c * others_c).sum()
            den = np.sqrt((this_c ** 2).sum() * (others_c ** 2).sum())
            subj_r_per_post[i, post_i] = num / den if den > 1e-12 else 0.0

    isc_subj = subj_r_per_post.mean(axis=1)           # (n_subjects,)
    return float(isc_subj.mean()), isc_subj


def run_roi_isc_postwise(
    npz_dir  : Path,
    subjects : list[str],
    run_types: list[str],
    roi_name : str,
) -> dict:
    """
    Run post-wise voxel-pattern ISC for all conditions within one ROI
    (Approach B — Chen et al. 2017).

    Parameters match run_roi_isc.  Returns the same dict structure minus
    permutation and t-test fields (deferred).

    Returns
    -------
    results : {
        run_type: {
          'isc_mean'  : float,
          'isc_subj'  : (n_subjects,),
          'n_subjects': int,
          'n_posts'   : int,
          'n_voxels'  : int,
          'subjects'  : list[str],
        }
    }
    """
    npz_dir = Path(npz_dir)
    results : dict = {}

    print(f"\nROI: {roi_name.upper()}  ({npz_dir.name})  [post-wise ISC]")
    print("=" * 50)

    for run_type in run_types:
        print(f"\n  Condition: {run_type}")

        try:
            patterns, post_ids, loaded_subs = \
                load_subject_patterns_postwise(npz_dir, subjects, run_type)
        except ValueError as e:
            print(f"    SKIPPED: {e}")
            continue

        n_subjects, n_posts, n_voxels = patterns.shape
        print(f"    Subjects: {n_subjects}  |  Posts: {n_posts}  |  Voxels: {n_voxels}")

        isc_mean, isc_subj = compute_pattern_isc_postwise(patterns)
        print(f"    ISC mean r = {isc_mean:.4f}")

        results[run_type] = {
            "isc_mean"  : isc_mean,
            "isc_subj"  : isc_subj,
            "n_subjects": n_subjects,
            "n_posts"   : n_posts,
            "n_voxels"  : n_voxels,
            "subjects"  : loaded_subs,
        }

    return results


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def permutation_test(
    patterns: np.ndarray,
    n_perms : int = 1000,
    seed    : int = 42,
) -> tuple[float, np.ndarray]:
    """
    Permutation test for pattern ISC.

    Null distribution: shuffle subject rows (breaks inter-subject correspondence
    while preserving each subject's spatial pattern), recompute ISC.

    Parameters
    ----------
    patterns : (n_subjects, n_voxels)
    n_perms  : number of permutations
    seed     : random seed

    Returns
    -------
    p_value   : float  one-tailed p (proportion of null >= observed)
    null_dist : (n_perms,) null ISC values
    """
    observed, _ = compute_pattern_isc(patterns)
    rng = np.random.default_rng(seed)
    null_dist = np.zeros(n_perms)

    for i in range(n_perms):
        perm = rng.permutation(len(patterns))
        null_dist[i], _ = compute_pattern_isc(patterns[perm])

    p_value = float(np.mean(null_dist >= observed))
    return p_value, null_dist


# ---------------------------------------------------------------------------
# Full analysis runner
# ---------------------------------------------------------------------------

def run_roi_isc(
    npz_dir   : Path,
    subjects  : list[str],
    run_types : list[str],
    roi_name  : str,
    n_perms   : int = 1000,
    seed      : int = 42,
) -> dict:
    """
    Run voxel-pattern ISC for all conditions within one ROI.

    Parameters
    ----------
    npz_dir   : directory with per-subject NPZ files
    subjects  : left-wing subject IDs (in any order)
    run_types : list of conditions, e.g. ['AntiLeft', 'AntiRight', ...]
    roi_name  : label for reporting (e.g. 'auditory', 'visual')
    n_perms   : permutations for null distribution
    seed      : random seed

    Returns
    -------
    results : dict with structure:
        {
          run_type: {
            'isc_mean'      : float,
            'isc_subj'      : np.ndarray (n_subjects,),
            'p_perm'        : float,
            'p_ttest'       : float,
            't_stat'        : float,
            'null_dist'     : np.ndarray (n_perms,),
            'n_subjects'    : int,
            'n_voxels'      : int,
            'subjects'      : list[str],
          }
        }
    """
    npz_dir = Path(npz_dir)
    results: dict = {}

    print(f"\nROI: {roi_name.upper()}  ({npz_dir.name})")
    print("=" * 50)

    for run_type in run_types:
        print(f"\n  Condition: {run_type}")

        try:
            patterns, loaded_subs = load_subject_patterns(npz_dir, subjects, run_type)
        except ValueError as e:
            print(f"    SKIPPED: {e}")
            continue

        n_subjects, n_voxels = patterns.shape
        print(f"    Subjects: {n_subjects}  |  Voxels: {n_voxels}")

        isc_mean, isc_subj = compute_pattern_isc(patterns)
        print(f"    ISC mean r = {isc_mean:.4f}")

        p_perm, null_dist = permutation_test(patterns, n_perms=n_perms, seed=seed)
        print(f"    p (permutation, {n_perms} perms) = {p_perm:.4f}")

        # One-sample t-test: is mean ISC > 0?
        t_stat, p_ttest = ttest_1samp(isc_subj, popmean=0, alternative="greater")
        print(f"    p (one-sample t-test vs 0) = {p_ttest:.4f}  (t = {t_stat:.3f})")

        results[run_type] = {
            "isc_mean"  : isc_mean,
            "isc_subj"  : isc_subj,
            "p_perm"    : p_perm,
            "p_ttest"   : p_ttest,
            "t_stat"    : t_stat,
            "null_dist" : null_dist,
            "n_subjects": n_subjects,
            "n_voxels"  : n_voxels,
            "subjects"  : loaded_subs,
        }

    return results


# ---------------------------------------------------------------------------
# Results packaging
# ---------------------------------------------------------------------------

def results_to_dataframe(
    results_by_roi: dict[str, dict],
) -> pd.DataFrame:
    """
    Flatten results from run_roi_isc (possibly for multiple ROIs) into a tidy
    DataFrame — one row per condition × ROI.

    Parameters
    ----------
    results_by_roi : {'auditory': {run_type: {...}}, 'visual': {...}}
                     or a single-ROI dict {'auditory': results}

    Returns
    -------
    pd.DataFrame with columns:
        roi, condition, isc_mean, p_perm, p_ttest, t_stat, n_subjects, n_voxels
    """
    rows = []
    for roi_name, results in results_by_roi.items():
        for condition, r in results.items():
            rows.append({
                "roi"       : roi_name,
                "condition" : condition,
                "isc_mean"  : round(r["isc_mean"], 6),
                "p_perm"    : round(r["p_perm"], 4),
                "p_ttest"   : round(r["p_ttest"], 4),
                "t_stat"    : round(r["t_stat"], 4),
                "n_subjects": r["n_subjects"],
                "n_voxels"  : r["n_voxels"],
            })
    return pd.DataFrame(rows)
