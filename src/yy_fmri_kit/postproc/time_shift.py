"""
Time-shift helpers for denoised 4D fMRI data (MNI space).

Pipeline:
- Discover denoised runs using yy_fmri_kit.find_files.iter_subjects / iter_subject_denoised
- Extract auditory ROI mean time series
- Estimate per-subject lags via maximal cross-correlation (leave-one-out group)
- Apply time-shift to 4D NIfTIs
- Save timeshifted images in a parallel directory structure

Intended usage:
---------------
from yy_fmri_kit.time_shift import time_shift_all_denoised

shifted_paths, lag_info = time_shift_all_denoised(
    derivatives_dir="derivatives/denoised",
    roi_mask_img="masks/auditory_roi.nii.gz",
    max_lag_tr=4,
    out_root="derivatives/denoised_timeshifted",
)

Then run your usual parcellation on the timeshifted files.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import nibabel as nib

from yy_fmri_kit.io.find_files import build_denoised_runs_dict
from yy_fmri_kit.postproc.timeshift_core import get_task_from_bold_path
from yy_fmri_kit.visualization.timeshift import plot_hrf_for_run

# Type aliases
Array1D = np.ndarray
ArrayLike = np.ndarray
PathLike = Union[str, Path]

# ================================================================
# DEFAULT AUDITORY ROI MASK (MNI SPACE) 
# ================================================================

def build_default_auditory_roi(
    ref_img_path: PathLike,
    out_dir: PathLike,
    centers_mm: Sequence[Sequence[float]] = (
        (-46.0, -20.0,  8.0),   # left Heschl approx
        ( 46.0, -20.0,  8.0),   # right Heschl approx
    ),
    radius_mm: float = 10.0,
    overwrite: bool = False,
) -> Path:
    """
    Build a bilateral spherical auditory ROI in the space of `ref_img_path`.
    
    Parameters
    ----------
    ref_img_path : 3D or 4D NIfTI in MNI152NLin2009cAsym space
    out_path     : full path to the ROI NIfTI to create
    centers_mm   : list of (x,y,z) MNI coordinates for sphere centers
    radius_mm    : sphere radius in mm
    overwrite    : if True, overwrite existing out_path
    """
    ref_img_path = Path(ref_img_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "auditory_sphere_roi.nii.gz"
    if out_path.exists() and not overwrite:
        return out_path

    ref_img = nib.load(str(ref_img_path))
    data = ref_img.get_fdata()

    if data.ndim == 4:
        nx, ny, nz, _ = data.shape
    elif data.ndim == 3:
        nx, ny, nz = data.shape
    else:
        raise ValueError(f"ref_img must be 3D or 4D, got shape {data.shape}")

    affine = ref_img.affine

    # voxel grid
    i, j, k = np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
    )
    ijk = np.stack([i, j, k], axis=-1).reshape(-1, 3)  # (V, 3)

    # indices -> MNI mm
    xyz_mm = nib.affines.apply_affine(affine, ijk)      # (V, 3)

    mask_flat = np.zeros(ijk.shape[0], dtype=bool)
    centers_mm = np.asarray(centers_mm, dtype=float)

    for center in centers_mm:
        dist = np.linalg.norm(xyz_mm - center[None, :], axis=1)
        mask_flat |= (dist <= radius_mm)

    mask = mask_flat.reshape(nx, ny, nz)

    roi_img = nib.Nifti1Image(mask.astype("uint8"), affine, ref_img.header)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(roi_img, out_path)
    return out_path


# ================================================================
# 1) LOW-LEVEL HELPERS (NIfTI + lag estimation)
# ================================================================

def _load_niimg(img: PathLike | nib.spatialimages.SpatialImage) -> nib.Nifti1Image:
    """Convenience loader: accept path or Nifti1Image."""
    if isinstance(img, nib.spatialimages.SpatialImage):
        return img  # already loaded
    return nib.load(str(img))


def extract_roi_mean_ts_from_4d(
    func_img: PathLike | nib.Nifti1Image,
    roi_mask_img: PathLike | nib.Nifti1Image,
    mask_threshold: float = 0.5,
) -> Array1D:
    """
    Extract mean time series from a 4D functional NIfTI within an ROI mask.

    Parameters
    ----------
    func_img : path or Nifti1Image
        4D fMRI image (X, Y, Z, T).
    roi_mask_img : path or Nifti1Image
        3D ROI mask in the same space as func_img.
    mask_threshold : float
        Values > mask_threshold are considered inside the ROI.

    Returns
    -------
    ts : np.ndarray, shape (T,)
        Mean ROI time series over all voxels in the mask.
    """
    func_img = _load_niimg(func_img)
    roi_mask_img = _load_niimg(roi_mask_img)

    func_data = func_img.get_fdata()  # (X, Y, Z, T)
    mask_data = roi_mask_img.get_fdata()  # (X, Y, Z)

    if func_data.ndim != 4:
        raise ValueError(f"func_img must be 4D, got shape {func_data.shape}")
    if mask_data.shape != func_data.shape[:3]:
        raise ValueError(
            f"Mask shape {mask_data.shape} != func spatial shape {func_data.shape[:3]}"
        )

    mask = mask_data > mask_threshold
    if not np.any(mask):
        raise ValueError("ROI mask is empty after thresholding.")

    X, Y, Z, T = func_data.shape
    func_flat = func_data.reshape(-1, T)  # (V, T)
    mask_flat = mask.reshape(-1)          # (V,)

    ts = func_flat[mask_flat].mean(axis=0)  # (T,)
    return ts


def _align_corr(x: Array1D, y: Array1D, lag: int) -> float:
    """
    Pearson correlation between two 1D time series at a given lag.

    Sign convention:
    - lag > 0: shift x forward (drop first `lag` samples)
               x[lag:] vs y[:-lag]
    - lag < 0: shift x backward (drop last `-lag` samples)
               x[:lag] vs y[-lag:]
    - lag = 0: x vs y as-is
    """
    if lag > 0:
        xs = x[lag:]
        ys = y[:-lag]
    elif lag < 0:
        xs = x[:lag]   # lag is negative; e.g., -2 -> x[:-2]
        ys = y[-lag:]
    else:
        xs = x
        ys = y

    if xs.size < 5:
        return np.nan

    xs = xs - xs.mean()
    ys = ys - ys.mean()
    denom = xs.std() * ys.std()
    if denom == 0:
        return np.nan

    return float((xs @ ys) / (xs.size * denom))


def estimate_subject_lags(
    aud_ts: Dict[str, Array1D],
    max_lag_tr: int = 4,
) -> Dict[str, dict]:
    """
    Estimate optimal time lag (in TR units) for each subject, based on
    auditory ROI time series and leave-one-out group means.

    Parameters
    ----------
    aud_ts : dict
        {sub_id: 1D np.ndarray of shape (T,)}
        All series must have the same length T.
    max_lag_tr : int
        Maximum absolute lag to explore (in TRs). Search range is
        [-max_lag_tr, ..., 0, ..., +max_lag_tr].

    Returns
    -------
    results : dict
        {sub_id: {
            "lag": int,
            "peak_corr": float,
            "lags": np.ndarray (2*max_lag_tr+1,),
            "corrs": np.ndarray (2*max_lag_tr+1,)
        }}

    Notes
    -----
    This implements the "maximal lag" method:
    for each subject we find the lag that maximizes correlation between
    their TS and the leave-one-out group-mean TS.
    """
    sub_ids = list(aud_ts.keys())
    if len(sub_ids) < 2:
        raise ValueError("Need at least 2 subjects to estimate lags.")

    lengths = {sid: ts.shape[0] for sid, ts in aud_ts.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"All time series must have same length, got: {lengths}")

    T = next(iter(lengths.values()))
    lags = np.arange(-max_lag_tr, max_lag_tr + 1, dtype=int)

    results: Dict[str, dict] = {}

    all_data = np.stack([aud_ts[sid] for sid in sub_ids], axis=0)  # (S, T)

    for i, sid in enumerate(sub_ids):
        idx_others = [j for j in range(len(sub_ids)) if j != i]
        group_mean = all_data[idx_others].mean(axis=0)

        subj_ts = aud_ts[sid]
        corrs = np.empty_like(lags, dtype=float)

        for k, lag in enumerate(lags):
            corrs[k] = _align_corr(subj_ts, group_mean, lag)

        valid = ~np.isnan(corrs)
        if not np.any(valid):
            best_lag = 0
            peak_corr = np.nan
        else:
            best_idx = np.nanargmax(corrs)
            best_lag = int(lags[best_idx])
            peak_corr = float(corrs[best_idx])

        results[sid] = {
            "lag": best_lag,
            "peak_corr": peak_corr,
            "lags": lags.copy(),
            "corrs": corrs,
        }

    return results


def _shift_4d_nifti(
    img: nib.Nifti1Image,
    lag: int,
) -> nib.Nifti1Image:
    """
    Shift a 4D NIfTI in time along the last axis according to lag.

    Parameters
    ----------
    img : Nifti1Image
        4D image (X, Y, Z, T).
    lag : int
        Time shift in TR units:
        - lag > 0: drop first `lag` TRs (data[..., lag:])
        - lag < 0: drop last `-lag` TRs (data[..., :lag])
        - lag = 0: unchanged

    Returns
    -------
    shifted_img : Nifti1Image
        Shifted image (X, Y, Z, T - abs(lag)).
    """
    data = img.get_fdata()
    if data.ndim != 4:
        raise ValueError(f"Expected 4D image, got shape {data.shape}")

    if lag > 0:
        data_shifted = data[..., lag:]
    elif lag < 0:
        data_shifted = data[..., :lag]  # lag negative -> data[..., :-|lag|]
    else:
        data_shifted = data

    shifted_img = nib.Nifti1Image(
        data_shifted,
        affine=img.affine,
        header=img.header,
    )
    return shifted_img


def apply_lags_to_nifti_dict(
    func_imgs: Dict[str, PathLike | nib.Nifti1Image],
    lags: Dict[str, int],
) -> Tuple[Dict[str, nib.Nifti1Image], int]:
    """
    Apply subject-specific lags to 4D NIfTI images and crop to a common T.

    Parameters
    ----------
    func_imgs : dict
        {sub_id: path or Nifti1Image} of 4D denoised images for one run/video.
    lags : dict
        {sub_id: lag_in_TRs} from estimate_subject_lags.

    Returns
    -------
    shifted_imgs : dict
        {sub_id: Nifti1Image} 4D images with aligned time axis (same T).
    T_aligned : int
        Final number of timepoints after shifting + cropping.
    """
    shifted_raw: Dict[str, nib.Nifti1Image] = {}
    lengths: Dict[str, int] = {}

    for sid, img in func_imgs.items():
        if sid not in lags:
            raise KeyError(f"No lag found for subject {sid!r}")

        img_loaded = _load_niimg(img)
        shifted_img = _shift_4d_nifti(img_loaded, lag=int(lags[sid]))
        shifted_raw[sid] = shifted_img
        lengths[sid] = shifted_img.get_fdata().shape[-1]

    T_aligned = min(lengths.values())

    shifted_cropped: Dict[str, nib.Nifti1Image] = {}
    for sid, img in shifted_raw.items():
        data = img.get_fdata()
        data_cropped = data[..., :T_aligned]
        shifted_cropped[sid] = nib.Nifti1Image(
            data_cropped,
            affine=img.affine,
            header=img.header,
        )

    return shifted_cropped, T_aligned

# ================================================================
# 2) GLUE HELPER TO ESTIMATE N RUNS ACROSS SUBJECTS
# ================================================================

def _infer_common_run_count(subject_runs: Dict[str, List[Path]]) -> int:
    """
    Infer the maximum number of runs that can be matched across all subjects.

    We assume runs are aligned by index (0..N-1) across subjects.
    Returns the minimum run count over subjects.
    """
    if not subject_runs:
        raise ValueError("subject_runs is empty.")

    counts = {sub: len(runs) for sub, runs in subject_runs.items()}
    min_count = min(counts.values())

    if len(set(counts.values())) != 1:
        print("⚠️ Subjects have different numbers of runs:")
        for sub, c in counts.items():
            print(f"   {sub}: {c} runs")
        print(f"   -> Using min_count={min_count} and ignoring extra runs.")

    return min_count


# ================================================================
# 3) HIGH-LEVEL API: ESTIMATE LAGS + TIME-SHIFT ALL RUNS
# ================================================================

def estimate_lags_for_all_runs(
    subject_runs: Dict[str, List[Path]],
    roi_mask_img: PathLike,
    max_lag_tr: int = 4,
) -> Dict[int, Dict[str, dict]]:
    """
    Estimate per-subject lags for each run index across subjects.

    We assume runs are aligned by index:
        subject_runs[sub][0] = run-0 for that subject, etc.

    Parameters
    ----------
    subject_runs : dict
        {sub: [Path_to_run0, Path_to_run1, ...]}
    roi_mask_img : path-like
        3D auditory ROI mask in MNI space (same as denoised funcs).
    max_lag_tr : int
        Max absolute lag to explore (TR units).

    Returns
    -------
    lag_info : dict
        {run_index: {sub_id: {lag, peak_corr, lags, corrs}}}
    """
    n_runs_common = _infer_common_run_count(subject_runs)
    lag_info_all: Dict[int, Dict[str, dict]] = {}

    roi_mask_img = Path(roi_mask_img)

    for run_idx in range(n_runs_common):
        # Collect TS for this run across subjects
        aud_ts: Dict[str, Array1D] = {}
        for sid, runs in subject_runs.items():
            if run_idx >= len(runs):
                continue
            func_path = runs[run_idx]
            aud_ts[sid] = extract_roi_mean_ts_from_4d(
                func_img=func_path,
                roi_mask_img=roi_mask_img,
            )

        print(f"Estimating lags for run index {run_idx} ({len(aud_ts)} subjects)")

        lag_info = estimate_subject_lags(aud_ts, max_lag_tr=max_lag_tr)
        lag_info_all[run_idx] = lag_info

    return lag_info_all


def time_shift_all_denoised(
    derivatives_dir: PathLike,
    *,
    roi_mask_img: PathLike | None = None,
    use_default_auditory_roi: bool = False, # uses HCP-MMP1 (Glasser) auditory areas
    default_roi_out_dir: PathLike | None = None,
    denoise_folder: str = "",
    space: str = "MNI152NLin2009cAsym",
    desc_keywords: Sequence[str] = ("denoised", "clean", "nltoolsClean", "preproc"),
    suffix: str = "bold",
    max_lag_tr: int = 4,
    out_root: Optional[PathLike] = None,
    subjects: Optional[Sequence[str]] = None,
) -> Tuple[Dict[int, Dict[str, Path]], Dict[int, Dict[str, dict]]]:
    """
    Full pipeline: discover denoised runs, estimate lags, time-shift and save.

    Parameters
    ----------
    derivatives_dir : path-like
        Root folder where denoised data lives.
        If denoise_folder is non-empty, we look under derivatives_dir / denoise_folder.
    roi_mask_img : path-like
        3D auditory ROI mask.
    use_default_auditory_roi : bool
        If True, build and use the default HCP-MMP1 auditory ROI mask.
        If roi_mask_img is also provided, this argument is ignored.
    default_roi_out_dir : path-like or None
        If use_default_auditory_roi is True, this is the directory
        where the default auditory ROI mask will be saved.
        If None, current directory is used.
    denoise_folder, space, desc_keywords, suffix : see build_denoised_runs_dict.
    max_lag_tr : int
        Max absolute lag (in TR units) to explore.
    out_root : path-like or None
        Root folder for saving timeshifted files.
        If None, defaults to derivatives_dir (timeshifted files overwrite nothing;
        they get new filenames in a parallel structure).
    subjects : optional sequence of subject IDs.
        If None, subjects are inferred via iter_subjects().

    Returns
    -------
    shifted_paths : dict
        {run_index: {sub_id: Path_to_timeshifted_file}}
    lag_info : dict
        {run_index: {sub_id: {lag, peak_corr, ...}}}
    """
    derivatives_dir = Path(derivatives_dir).resolve()
    if out_root is None:
        out_root = derivatives_dir
    out_root = Path(out_root).resolve()

        # --- Decide which ROI mask to use ---
    if roi_mask_img is not None:
        roi_mask_path = Path(roi_mask_img).resolve()
    elif use_default_auditory_roi:
        roi_mask_path = build_default_auditory_roi(
            out_dir=default_roi_out_dir or out_root
        )
    else:
        raise ValueError("Must pass either roi_mask_img or use_default_auditory_roi=True.")
    print(f"Using auditory ROI mask at: {roi_mask_path}")

    # 1) Discover runs
    subject_runs = build_denoised_runs_dict(
        derivatives_dir=derivatives_dir,
        denoise_folder=denoise_folder,
        space=space,
        desc_keywords=desc_keywords,
        suffix=suffix,
        subjects=subjects,
    )
    if not subject_runs:
        raise RuntimeError("No denoised runs found for any subject.")

    n_runs_common = _infer_common_run_count(subject_runs)
    print(f"Found {len(subject_runs)} subjects, up to {n_runs_common} runs in common.")

    # 2) Estimate lags per run
    lag_info_all = estimate_lags_for_all_runs(
        subject_runs=subject_runs,
        roi_mask_img=roi_mask_path,
        max_lag_tr=max_lag_tr,
    )

    # 3) Time-shift & save NIfTIs
    shifted_paths: Dict[int, Dict[str, Path]] = {}

    for run_idx in range(n_runs_common):
        # Build dict of func_imgs for this run across subjects
        func_imgs: Dict[str, Path] = {
            sid: runs[run_idx]
            for sid, runs in subject_runs.items()
            if run_idx < len(runs)
        }

        lag_info_run = lag_info_all[run_idx]
        subject_lags = {sid: lag_info_run[sid]["lag"] for sid in lag_info_run}

        print(
            f"Applying time-shifts for run index {run_idx}: "
            f"{len(func_imgs)} subjects, max |lag|={max_lag_tr} TRs"
        )

        shifted_imgs, T_aligned = apply_lags_to_nifti_dict(func_imgs, subject_lags)
        print(f"Run {run_idx}: aligned T = {T_aligned} TRs")

        # Save with parallel directory structure
        shifted_paths[run_idx] = {}
        for sid, img in shifted_imgs.items():
            original_path = Path(func_imgs[sid])

            try:
                rel = original_path.relative_to(derivatives_dir)
            except ValueError:
                # If not under derivatives_dir for some reason, just use filename
                rel = Path(original_path.name)

            # e.g., sub-01_task-XXX_space-MNI..._bold.nii.gz
            stem = original_path.stem  # may still have ".nii" if .nii.gz
            if stem.endswith(".nii"):
                stem = stem[:-4]

            new_name = stem + "_timeshift.nii.gz"
            out_path = out_root / rel.parent / new_name
            out_path.parent.mkdir(parents=True, exist_ok=True)

            nib.save(img, out_path)
            shifted_paths[run_idx][sid] = out_path

    return shifted_paths, lag_info_all


# ================================================================
# TIME SHIFTING USING THE FIRST AUDITORY PEAK (ALTERNATIVE)
# ================================================================

import pandas as pd

def analyze_hrf_all_runs(
    derivatives_dir: str | Path,
    tasks: list[str] | None,
    mask_path: str | Path,
    TR: float,
    denoise_folder: str = "",
    mark_onset: bool = False,
    onset_sec: float | None = None,
    find_peaks: bool = True,
    subjects: list[str] | None = None,
    save_png: Path | None = None,
    save_csv: Path | None = None,
) -> pd.DataFrame:
    """
    For all subjects/runs:
      - extract ROI timecourse
      - find first HRF peak after onset (in a 0-10 sec window from onset_sec)
      - optionally save HRF plots
      - return and optionally save a CSV summary of peaks

    Saving logic
    ------------
    - If save_png is not None:
        plots are saved under:
            save_png / sub-XXXX / <sub>_<task>_run-XX.png
    - If save_csv is None and save_png is not None:
        CSV is saved as:
            save_png / "hrf_peaks.csv"
      Otherwise, if save_csv is given explicitly, that path is used.
    """
    derivatives_dir = Path(derivatives_dir).resolve()
    mask_path = Path(mask_path).resolve()
    # run all tasks if tasks is None
    runs_filter = None if tasks is None else set(t.lower() for t in tasks) 
    subject_runs = build_denoised_runs_dict(
        derivatives_dir,
        denoise_folder=denoise_folder,
        subjects=subjects,
    )

    save_root: Path | None = None
    if save_png is not None:
        save_root = Path(save_png).resolve()
        save_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for sub, runs in subject_runs.items():
        safe_sub = sub.replace(" ", "_")

        # Folder for this subject's plots
        sub_dir: Path | None = None
        if save_root is not None:
            sub_dir = save_root / safe_sub
            sub_dir.mkdir(parents=True, exist_ok=True)

        for run_idx, bold_path in enumerate(runs):
            task = get_task_from_bold_path(bold_path) or "unknownTask"
            # ---- Task filter ----
            if runs_filter is not None and task.lower() not in runs_filter:  # NEW
                continue  # NEW

            title = f"{sub} – {task}\n{bold_path.name}"
            print(f"Analyzing {title}")

            safe_task = task.replace(" ", "_")
            if sub_dir is not None:
                save_name = f"{safe_sub}_{safe_task}_run-{run_idx+1}.png"
            else:
                save_name = None

            find_peaks = find_peaks and (onset_sec is not None)
            
            peak_info = plot_hrf_for_run(
                bold_path=bold_path,
                mask_path=mask_path,
                TR=TR,
                mark_onset=mark_onset,
                onset_sec=onset_sec,
                zscore=True,
                title=title,
                save_dir=sub_dir,
                save_name=save_name,
                show=(save_png is None),
                mark_peak=find_peaks,
            )

            if peak_info is None:
                continue

            rows.append(
                dict(
                    sub=sub,
                    task=task,
                    bold_path=str(bold_path),
                    peak_time_sec=peak_info["peak_time_sec"],
                    peak_latency_sec=peak_info["peak_latency_sec"],
                    peak_value=peak_info["peak_value"],
                )
            )

    df = pd.DataFrame(rows)
    # ---- Save CSV summary in the *parent folder* of all subjects ----
    if save_csv is None and save_root is not None:
        # parent folder of all sub-XXX directories
        save_csv = save_root / "hrf_peaks.csv"

    if save_csv is not None:
        save_csv = Path(save_csv).resolve()
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_csv, index=False)
        print(f"Saved peak summary to {save_csv}")

    if save_root is not None:
        print(f"Saved HRF plots under: {save_root} (one subfolder per subject)")

    return df
