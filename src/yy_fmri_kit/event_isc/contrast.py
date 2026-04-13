"""
contrast.py
===========
Utilities for ISPC contrast analyses on parcellated fMRI data.

Designed to extend (not modify) the existing parcellated-ISC pipeline in
``yy_fmri_kit.event_isc.extraction.parcel``.

Provides:
- ``filter_subjects_by_group``  — select subject IDs by political affiliation
- ``merge_conditions``          — combine multiple run-type pattern dicts into a group
- ``contrast_permutation_test`` — split-pool permutation for A > B contrasts
- ``ttest_contrast``            — paired t-test A > B per parcel (no permutation)
- ``parcels_to_nifti``          — map parcel scalar values into a 3-D NIfTI volume
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from yy_fmri_kit.event_isc.extraction.parcel import compute_isc

__all__ = [
    "filter_subjects_by_group",
    "merge_conditions",
    "contrast_permutation_test",
    "ttest_contrast",
    "parcels_to_nifti",
]

# {post_id: array(n_subjects, n_parcels)}
PostPatterns = Dict[str, np.ndarray]


# ---------------------------------------------------------------------------
# Subject filtering
# ---------------------------------------------------------------------------

def filter_subjects_by_group(
    behavioral_csv: Path | str,
    group: str = "left",
    id_col: str = "bids_id",
    group_col: str = "political_group",
) -> List[str]:
    """
    Return BIDS subject IDs whose political group matches *group*.

    Parameters
    ----------
    behavioral_csv : path to ``merged_behavioral_bids.csv``
    group          : group value to keep (default ``"left"``)
    id_col         : column containing BIDS IDs (default ``"bids_id"``)
    group_col      : column containing group labels (default ``"political_group"``)

    Returns
    -------
    list[str]
        Sorted list of BIDS IDs (e.g. ``["sub-1", "sub-20", ...]``).

    Raises
    ------
    ValueError
        If required columns are missing or no subjects belong to *group*.

    Example
    -------
    left_subs = filter_subjects_by_group(
        "behavioral_analyses/data/250226/merged_behavioral_bids.csv",
        group="left",
    )
    """
    df = pd.read_csv(behavioral_csv)

    for col in (id_col, group_col):
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in {behavioral_csv}. "
                f"Available columns: {df.columns.tolist()}"
            )

    subset = df[df[group_col] == group][id_col].dropna().astype(str).unique()
    if len(subset) == 0:
        raise ValueError(
            f"No subjects found with {group_col}='{group}' in {behavioral_csv}."
        )

    result = sorted(subset.tolist())
    print(
        f"[filter_subjects_by_group] {len(result)} '{group}' subjects: {result}"
    )
    return result


# ---------------------------------------------------------------------------
# Condition merging
# ---------------------------------------------------------------------------

def merge_conditions(
    patterns: Dict[str, PostPatterns],
    condition_map: Dict[str, List[str]],
) -> Dict[str, PostPatterns]:
    """
    Merge per-run-type pattern dicts into labelled condition groups.

    Parameters
    ----------
    patterns      : ``{run_type: {post_id: array(n_subjects, n_parcels)}}``
                    as returned by ``extract_post_patterns``.
    condition_map : ``{"agree": ["ProLeft", "AntiRight"],
                       "disagree": ["AntiLeft", "ProRight"]}``

    Returns
    -------
    dict[str, PostPatterns]
        ``{condition_group: {post_id: array(n_subjects, n_parcels)}}``

    Raises
    ------
    KeyError
        If a run type listed in *condition_map* is absent from *patterns*.
    ValueError
        If the same post ID appears in more than one run type within a group.

    Example
    -------
    merged = merge_conditions(
        patterns,
        {"agree": ["ProLeft", "AntiRight"],
         "disagree": ["AntiLeft", "ProRight"]},
    )
    # merged["agree"] has posts from ProLeft + AntiRight combined
    """
    merged: Dict[str, PostPatterns] = {}

    for group_name, run_types in condition_map.items():
        combined: PostPatterns = {}
        for rt in run_types:
            if rt not in patterns:
                raise KeyError(
                    f"Run type '{rt}' not found in patterns. "
                    f"Available run types: {list(patterns.keys())}"
                )
            for post_id, arr in patterns[rt].items():
                if post_id in combined:
                    raise ValueError(
                        f"Post ID '{post_id}' appears in multiple run types "
                        f"within group '{group_name}'. Post IDs must be disjoint."
                    )
                combined[post_id] = arr
        merged[group_name] = combined
        print(
            f"[merge_conditions] '{group_name}': "
            f"{len(combined)} posts from {run_types}"
        )

    return merged


# ---------------------------------------------------------------------------
# Contrast permutation test
# ---------------------------------------------------------------------------

def contrast_permutation_test(
    patterns_a: PostPatterns,
    patterns_b: PostPatterns,
    n_perms: int = 1000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split-pool permutation test for the contrast ISC_a > ISC_b.

    Observed contrast: ``ISC_agree - ISC_disagree`` per parcel.

    Null distribution: pool all posts from both groups, randomly split into
    groups of the original sizes *n_a* and *n_b*, compute ISC for each
    random group, record the difference.  This tests whether the actual
    condition assignment produces a larger ISC contrast than an arbitrary one.

    Parameters
    ----------
    patterns_a : ``{post_id: array(n_subjects, n_parcels)}`` — AGREE group
    patterns_b : ``{post_id: array(n_subjects, n_parcels)}`` — DISAGREE group
    n_perms    : number of permutations (default 1000)
    seed       : random seed for reproducibility

    Returns
    -------
    obs_contrast : ``(n_parcels,)`` — observed ISC_a − ISC_b
    p_vals       : ``(n_parcels,)`` — one-tailed permutation p-values
    null_dist    : ``(n_perms, n_parcels)`` — full null distribution

    Notes
    -----
    Post IDs must be disjoint between *patterns_a* and *patterns_b*.
    Subjects (rows) must be identical and in the same order in both dicts.

    Example
    -------
    obs, p_vals, null = contrast_permutation_test(
        merged["agree"], merged["disagree"], n_perms=1000
    )
    rejected, p_fdr = fdr_correct(p_vals, q=0.05)
    """
    # Validate non-empty inputs
    if len(patterns_a) == 0 or len(patterns_b) == 0:
        raise ValueError("Both pattern dicts must be non-empty.")

    # Check post IDs are disjoint
    overlap = set(patterns_a) & set(patterns_b)
    if overlap:
        raise ValueError(
            f"Post IDs overlap between the two groups: {sorted(overlap)}. "
            "Patterns must come from distinct conditions."
        )

    # Infer shapes
    n_a = len(patterns_a)
    n_b = len(patterns_b)
    sample_a = next(iter(patterns_a.values()))
    sample_b = next(iter(patterns_b.values()))
    n_subjects_a, n_parcels = sample_a.shape
    n_subjects_b, _ = sample_b.shape
    if n_subjects_a != n_subjects_b:
        raise ValueError(
            f"Subject count mismatch: patterns_a has {n_subjects_a} subjects, "
            f"patterns_b has {n_subjects_b}. Both must have the same subjects."
        )
    n_subjects = n_subjects_a
    n_total = n_a + n_b

    # Observed contrast
    obs_a, _ = compute_isc(patterns_a)
    obs_b, _ = compute_isc(patterns_b)
    obs_contrast = obs_a - obs_b

    # Stack all posts into one array: (n_total, n_subjects, n_parcels)
    post_ids_a = sorted(patterns_a.keys())
    post_ids_b = sorted(patterns_b.keys())
    all_post_ids = post_ids_a + post_ids_b
    all_data = np.stack(
        [patterns_a[p] for p in post_ids_a]
        + [patterns_b[p] for p in post_ids_b]
    )  # (n_total, n_subjects, n_parcels)

    # Null distribution via split-pool permutation
    rng = np.random.default_rng(seed)
    null_dist = np.zeros((n_perms, n_parcels))

    for i in range(n_perms):
        perm_idx = rng.permutation(n_total)
        idx_a = perm_idx[:n_a]
        idx_b = perm_idx[n_a:]

        perm_data_a = all_data[idx_a]  # (n_a, n_subjects, n_parcels)
        perm_data_b = all_data[idx_b]  # (n_b, n_subjects, n_parcels)

        # Build temporary pattern dicts (post IDs are placeholders here)
        perm_patterns_a = {f"_p{j}": perm_data_a[j] for j in range(n_a)}
        perm_patterns_b = {f"_q{j}": perm_data_b[j] for j in range(n_b)}

        perm_isc_a, _ = compute_isc(perm_patterns_a)
        perm_isc_b, _ = compute_isc(perm_patterns_b)
        null_dist[i] = perm_isc_a - perm_isc_b

    # One-tailed p-value: proportion of null >= observed
    p_vals = np.mean(null_dist >= obs_contrast, axis=0)

    print(
        f"[contrast_permutation_test] n_agree={n_a}, n_disagree={n_b}, "
        f"n_perms={n_perms}, n_parcels={n_parcels}\n"
        f"  mean observed contrast: {obs_contrast.mean():.4f}, "
        f"  parcels with p<0.05 (uncorrected): {(p_vals < 0.05).sum()}"
    )

    return obs_contrast, p_vals, null_dist


# ---------------------------------------------------------------------------
# Paired t-test contrast (no permutation)
# ---------------------------------------------------------------------------

def ttest_contrast(
    patterns_a: PostPatterns,
    patterns_b: PostPatterns,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Paired t-test for the contrast ISC_a > ISC_b, per parcel.

    For each parcel the leave-one-out ISC is computed per subject in each
    condition, giving two vectors of length *n_subjects*.  A paired t-test
    (``scipy.stats.ttest_rel``) is then applied across those subject-level
    values, and the two-tailed p is converted to a one-tailed p (a > b).

    Parameters
    ----------
    patterns_a : ``{post_id: array(n_subjects, n_parcels)}``
    patterns_b : ``{post_id: array(n_subjects, n_parcels)}``

    Returns
    -------
    t_stat    : ``(n_parcels,)`` — paired t-statistic
    p_vals    : ``(n_parcels,)`` — one-tailed p-values (a > b)
    isc_mean_a : ``(n_parcels,)`` — group-mean ISC for condition A
    isc_mean_b : ``(n_parcels,)`` — group-mean ISC for condition B

    Raises
    ------
    ValueError
        If the subject counts differ between the two conditions.

    Example
    -------
    t, p, mu_a, mu_b = ttest_contrast(patterns["AntiRight"], patterns["ProLeft"])
    rejected, p_fdr = fdr_correct(p, q=0.05)
    """
    from scipy.stats import ttest_rel

    isc_mean_a, isc_subj_a = compute_isc(patterns_a)  # (n_subs, n_parcels)
    isc_mean_b, isc_subj_b = compute_isc(patterns_b)

    if isc_subj_a.shape[0] != isc_subj_b.shape[0]:
        raise ValueError(
            f"Subject count mismatch: condition A has {isc_subj_a.shape[0]} subjects "
            f"but condition B has {isc_subj_b.shape[0]}. "
            "Both conditions must include the same subjects."
        )

    # Vectorised paired t-test across parcels
    t_stat, p_two = ttest_rel(isc_subj_a, isc_subj_b, axis=0)

    # Convert two-tailed → one-tailed (a > b):
    # if t > 0 the direction is correct → p_one = p_two / 2
    # if t ≤ 0 the effect goes the wrong way → p_one = 1 - p_two / 2
    p_one = np.where(t_stat > 0, p_two / 2, 1.0 - p_two / 2)

    print(
        f"[ttest_contrast] n_subjects={isc_subj_a.shape[0]}, "
        f"n_parcels={isc_subj_a.shape[1]}\n"
        f"  mean t-stat: {t_stat.mean():.3f}, "
        f"  parcels p<0.05 uncorrected: {(p_one < 0.05).sum()}"
    )

    return t_stat, p_one, isc_mean_a, isc_mean_b


# ---------------------------------------------------------------------------
# Parcel values → 3-D NIfTI volume
# ---------------------------------------------------------------------------

def parcels_to_nifti(
    values: np.ndarray,
    parcel_names: List[str],
    atlas_nii: Path | str,
    labels_tsv: Path | str,
    output_path: Path | str,
) -> Path:
    """
    Write a 3-D NIfTI volume where each voxel takes the scalar value of its
    corresponding parcel.

    Uses the integer-labelled atlas NIfTI (e.g. the local Schaefer+Tian atlas)
    and the accompanying labels TSV to map parcel names to atlas integer IDs.
    Voxels belonging to parcels not present in *parcel_names*, or whose value
    is NaN, are stored as NaN in the output volume.  Background (label 0) is
    stored as 0.

    This approach bypasses any atlas-version name-matching issues: the same
    atlas used during parcellation is used here, so every parcel maps
    perfectly — including Tian S3 subcortical parcels.

    Parameters
    ----------
    values       : ``(n_parcels,)`` scalar array (e.g. ISC, t-stat, contrast)
    parcel_names : list of parcel name strings matching ``labels_tsv['name']``
                   and the column headers of the parcellated TSV files.
    atlas_nii    : path to the integer-labelled atlas NIfTI (``*.dseg.nii.gz``)
    labels_tsv   : path to TSV with columns ``id`` (int) and ``name`` (str)
    output_path  : where to write the output NIfTI

    Returns
    -------
    Path
        Absolute path to the saved NIfTI file.

    Example
    -------
    tmp = parcels_to_nifti(
        isc_agree, parcel_names,
        atlas_nii  = "data/atlases/Schaefer2018_tf_2mm_...dseg.nii.gz",
        labels_tsv = "data/atlases/Schaefer2018_...labels.tsv",
        output_path = "/tmp/isc_agree.nii.gz",
    )
    lh, rh = yab.project_vol2surf(str(tmp), interpolation="nearest")
    """
    import nibabel as nib

    atlas_img = nib.load(str(atlas_nii))
    atlas_data = np.round(atlas_img.get_fdata()).astype(np.int32)

    labels_df = pd.read_csv(str(labels_tsv), sep="\t")
    name_to_id: Dict[str, int] = dict(
        zip(labels_df["name"].astype(str), labels_df["id"].astype(int))
    )

    # Float32 volume: NaN everywhere except background (0)
    out_vol = np.full(atlas_data.shape, np.nan, dtype=np.float32)
    out_vol[atlas_data == 0] = 0.0

    n_mapped = 0
    for i, name in enumerate(parcel_names):
        label_id = name_to_id.get(name)
        if label_id is None:
            continue
        out_vol[atlas_data == label_id] = float(values[i])
        n_mapped += 1

    print(
        f"[parcels_to_nifti] mapped {n_mapped}/{len(parcel_names)} parcels "
        f"→ {output_path}"
    )

    out_img = nib.Nifti1Image(out_vol, atlas_img.affine, atlas_img.header)
    nib.save(out_img, str(output_path))
    return Path(output_path)
