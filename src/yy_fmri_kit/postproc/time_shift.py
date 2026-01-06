"""
Time-shift helpers for denoised 4D fMRI data (MNI space).

UPDATED PIPELINE (matches political 2021 MATLAB scripts-style HRF delay + cropping)
----------------------------------------------------------
- Discover denoised runs (4 videos per participant) using build_denoised_runs_dict
- Extract auditory ROI mean time series (per subject, per run)
- Detect HRF delay per run using Option B:
    "first auditory peak after stimulus onset within a search window"
- Compute ONE subject-specific HRF delay = average across the 4 runs
- Crop each run like the MATLAB code:
    data_locked = data[:, onset_tr + subject_delay : onset_tr + subject_delay + movie_len_tr]
- Save cropped (timeshifted) images in a parallel directory structure

CHANGES from previous version:
------------------------------
- We now estimate an absolute HRF delay relative to stimulus onset (MATLAB logic),
  then crop fixed-length stimulus windows per task/video.

Intended usage:
---------------
from yy_fmri_kit.time_shift import time_shift_all_denoised

shifted_paths, delay_info, subject_delay_tr = time_shift_all_denoised(
    derivatives_dir="derivatives/denoised",
    roi_mask_img="masks/auditory_roi.nii.gz",
    TR=1.0,
    onset_tr=8,
    task_movie_len_tr={"seker": 85, "neutral": 151, "bibi": 235, "X": 123},
    out_root="derivatives/denoised_timeshifted",
)

Then run voxelwise or parcelwise ISC on the shifted files (do NOT re-shift later).
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
PathLike = Union[str, Path]


# ================================================================
# 1) LOW-LEVEL HELPERS (NIfTI + ROI extraction)
# ================================================================

def _load_niimg(img: PathLike | nib.spatialimages.SpatialImage) -> nib.Nifti1Image:
    """Convenience loader: accept path or NIfTI image."""
    if isinstance(img, nib.spatialimages.SpatialImage):
        return img
    return nib.load(str(img))


def extract_roi_mean_ts_from_4d(
    func_img: PathLike | nib.Nifti1Image,
    roi_mask_img: PathLike | nib.Nifti1Image,
    mask_threshold: float = 0.5,
) -> Array1D:
    """
    Extract mean time series from a 4D functional NIfTI within an ROI mask.

    Returns
    -------
    ts : np.ndarray, shape (T,)
        Mean ROI time series.
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

    _, _, _, T = func_data.shape
    func_flat = func_data.reshape(-1, T)  # (V, T)
    mask_flat = mask.reshape(-1)          # (V,)

    return func_flat[mask_flat].mean(axis=0)


# ================================================================
# 2) HRF delay detection (Option B)
# ================================================================

def _zscore_1d(x: Array1D) -> Array1D:
    """small helper used for peak detection stability."""
    x = np.asarray(x, dtype=float)
    sd = x.std()
    if sd == 0:
        return x * 0.0
    return (x - x.mean()) / sd


def find_first_peak_after_onset(
    ts: Array1D,
    *,
    TR: float,
    onset_tr: int,
    search_window_sec: Tuple[float, float] = (2.0, 12.0),
    zscore: bool = True,
    smooth_tr: int = 0,
) -> dict:
    """
    CHANGED/NEW (Option B):
    Find the FIRST peak after stimulus onset within a time window.

    This is used to estimate HRF delay per subject/run, *relative to onset*.

    Returns
    -------
    dict with:
        - peak_tr (int): absolute TR index in the run
        - peak_latency_tr (int): peak_tr - onset_tr
        - peak_latency_sec (float)
        - peak_value (float): value at peak (after preprocessing)
        - window_tr (tuple[int,int]): TR window searched (start,end)
    """
    ts2 = _zscore_1d(ts) if zscore else np.asarray(ts, dtype=float)

    # optional lightweight smoothing (no scipy dependency)
    if smooth_tr and smooth_tr > 1:
        k = int(smooth_tr)
        kernel = np.ones(k, dtype=float) / k
        ts2 = np.convolve(ts2, kernel, mode="same")

    start_sec, end_sec = search_window_sec
    if end_sec <= start_sec:
        raise ValueError("search_window_sec must be (start < end).")

    w0 = onset_tr + int(np.round(start_sec / TR))
    w1 = onset_tr + int(np.round(end_sec / TR))
    w0 = max(w0, 0)
    w1 = min(w1, ts2.shape[0])  # exclusive

    if w1 - w0 < 3:
        raise ValueError(
            f"Peak window too small after clipping: [{w0},{w1}). "
            f"Check TR/onset_tr/search_window_sec."
        )

    seg = ts2[w0:w1]

    # "first local maximum" (fallback to global max if none)
    peak_rel = None
    for i in range(1, len(seg) - 1):
        if seg[i] > seg[i - 1] and seg[i] >= seg[i + 1]:
            peak_rel = i
            break
    if peak_rel is None:
        peak_rel = int(np.argmax(seg))

    peak_tr = int(w0 + peak_rel)
    peak_lat_tr = int(peak_tr - onset_tr)

    # HRF delay can't really be negative; guard in case of weird inputs
    peak_lat_tr = max(0, peak_lat_tr)

    return dict(
        peak_tr=peak_tr,
        peak_latency_tr=peak_lat_tr,
        peak_latency_sec=float(peak_lat_tr * TR),
        peak_value=float(ts2[peak_tr]),
        window_tr=(w0, w1),
    )


def estimate_subject_delay_tr_for_run(
    aud_ts: Dict[str, Array1D],
    *,
    TR: float,
    onset_tr: int,
    search_window_sec: Tuple[float, float] = (2.0, 12.0),
    zscore: bool = True,
    smooth_tr: int = 0,
) -> Dict[str, dict]:
    """
    For one run (video), estimate each subject's absolute HRF delay (in TRs)
    as the first peak latency after onset.

    Returns
    -------
    {sub_id: {
        "delay_tr": int,
        "delay_sec": float,
        "peak_tr": int,
        "peak_value": float,
        "window_tr": (int,int),
    }}
    """
    out: Dict[str, dict] = {}
    for sid, ts in aud_ts.items():
        info = find_first_peak_after_onset(
            ts,
            TR=TR,
            onset_tr=onset_tr,
            search_window_sec=search_window_sec,
            zscore=zscore,
            smooth_tr=smooth_tr,
        )
        out[sid] = dict(
            delay_tr=int(info["peak_latency_tr"]),
            delay_sec=float(info["peak_latency_sec"]),
            peak_tr=int(info["peak_tr"]),
            peak_value=float(info["peak_value"]),
            window_tr=info["window_tr"],
        )
    return out


def average_subject_delay_across_runs(
    delay_info_all: Dict[int, Dict[str, dict]],
    *,
    n_runs: int = 4,
    max_delay_tr: int | None = None,
) -> Dict[str, int]:
    """
    Compute ONE subject-specific HRF delay (TR) by averaging per-run delays
    across the first `n_runs` runs (videos).

    Returns
    -------
    {sub_id: delay_tr_int}
    """
    run_idxs = sorted(delay_info_all.keys())[:n_runs]
    if not run_idxs:
        raise ValueError("delay_info_all is empty.")

    delays_by_sub: Dict[str, List[int]] = {}
    for r in run_idxs:
        for sid, info in delay_info_all[r].items():
            delays_by_sub.setdefault(sid, []).append(int(info["delay_tr"]))

    out: Dict[str, int] = {}
    for sid, ds in delays_by_sub.items():
        if len(ds) == 0:
            continue
        d = int(np.round(np.mean(ds)))
        if max_delay_tr is not None:
            d = int(np.clip(d, 0, max_delay_tr))
        out[sid] = d
    return out


# ================================================================
# 3) Cropping like MATLAB (fixed window per task)
# ================================================================

def crop_4d_by_onset_and_delay(
    img: nib.Nifti1Image,
    *,
    onset_tr: int,
    delay_tr: int,
    movie_len_tr: int,
) -> nib.Nifti1Image:
    """
        data_locked = data[:, onset_tr + delay_tr : onset_tr + delay_tr + movie_len_tr]

    Returns a 4D NIfTI with exactly `movie_len_tr` timepoints.
    """
    data = img.get_fdata()
    if data.ndim != 4:
        raise ValueError(f"Expected 4D image, got shape {data.shape}")

    T = int(data.shape[-1])
    start = int(onset_tr + delay_tr)
    end = int(start + movie_len_tr)

    if start < 0 or end > T:
        raise ValueError(
            f"Crop out of bounds: start={start}, end={end}, T={T}. "
            f"(onset_tr={onset_tr}, delay_tr={delay_tr}, movie_len_tr={movie_len_tr})"
        )

    data2 = data[..., start:end]
    return nib.Nifti1Image(data2, affine=img.affine, header=img.header)


def crop_run_dict_subject_delays(
    func_imgs: Dict[str, PathLike | nib.Nifti1Image],
    *,
    subject_delay_tr: Dict[str, int],
    onset_tr: int,
    movie_len_tr: int,
    on_out_of_bounds: str = "skip",  # "skip" | "error"
) -> Tuple[Dict[str, nib.Nifti1Image], List[str]]:
    """
    Apply subject-specific delay-based cropping to a dict of run images.

    Returns
    -------
    cropped_imgs : {sub_id: cropped 4D NIfTI}
    skipped : list[sub_id] that were skipped due to out-of-bounds (if on_out_of_bounds="skip")
    """
    cropped: Dict[str, nib.Nifti1Image] = {}
    skipped: List[str] = []

    for sid, img in func_imgs.items():
        if sid not in subject_delay_tr:
            skipped.append(sid)
            continue

        ni = _load_niimg(img)
        try:
            cropped[sid] = crop_4d_by_onset_and_delay(
                ni,
                onset_tr=onset_tr,
                delay_tr=int(subject_delay_tr[sid]),
                movie_len_tr=int(movie_len_tr),
            )
        except ValueError as e:
            if on_out_of_bounds == "error":
                raise
            print(f"⚠️ Skipping {sid} (crop failed): {e}")
            skipped.append(sid)

    return cropped, skipped


# ================================================================
# 4) GLUE: runs across subjects
# ================================================================

def _infer_common_run_count(subject_runs: Dict[str, List[Path]]) -> int:
    """align runs by index and take min count."""
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


def estimate_delay_for_all_runs(
    subject_runs: Dict[str, List[Path]],
    *,
    roi_mask_img: PathLike,
    TR: float,
    onset_tr: int,
    search_window_sec: Tuple[float, float] = (2.0, 12.0),
    zscore: bool = True,
    smooth_tr: int = 0,
) -> Dict[int, Dict[str, dict]]:
    """
    For each run index, compute per-subject HRF delay via first peak latency.

    Returns
    -------
    delay_info_all : {run_idx: {sub_id: {... delay info ...}}}
    """
    n_runs_common = _infer_common_run_count(subject_runs)
    delay_info_all: Dict[int, Dict[str, dict]] = {}

    roi_mask_img = Path(roi_mask_img)

    for run_idx in range(n_runs_common):
        aud_ts: Dict[str, Array1D] = {}
        for sid, runs in subject_runs.items():
            if run_idx >= len(runs):
                continue
            func_path = runs[run_idx]
            aud_ts[sid] = extract_roi_mean_ts_from_4d(
                func_img=func_path,
                roi_mask_img=roi_mask_img,
            )

        print(f"Estimating HRF delay (peak-based) for run index {run_idx} ({len(aud_ts)} subjects)")

        delay_info_all[run_idx] = estimate_subject_delay_tr_for_run(
            aud_ts,
            TR=TR,
            onset_tr=onset_tr,
            search_window_sec=search_window_sec,
            zscore=zscore,
            smooth_tr=smooth_tr,
        )

    return delay_info_all


# ================================================================
# 5) HIGH-LEVEL API: MATLAB-style delay + crop + save
# ================================================================

def time_shift_all_denoised(
    derivatives_dir: PathLike,
    *,
    roi_mask_img: PathLike,
    TR: float,  # required for peak detection
    onset_tr: int,  # blank TRs before stimulus start (MATLAB's 8)
    task_movie_len_tr: Dict[str, int],  # task->movie length in TRs (excluding blanks/ratings)
    denoise_folder: str = "",
    space: str = "MNI152NLin2009cAsym",
    desc_keywords: Sequence[str] = ("denoised", "clean", "nltoolsClean", "preproc"),
    suffix: str = "bold",
    out_root: Optional[PathLike] = None,
    subjects: Optional[Sequence[str]] = None,
    search_window_sec: Tuple[float, float] = (2.0, 12.0),
    smooth_tr: int = 0, 
    zscore: bool = True,
    n_runs_to_average: int = 4,
    max_delay_tr: int | None = None,  # optional clipping
    on_out_of_bounds: str = "skip",  # "skip" or "error"
) -> Tuple[Dict[int, Dict[str, Path]], Dict[int, Dict[str, dict]], Dict[str, int]]:
    """
    - Estimate HRF delay per run from first peak after onset
    - Average delay across 4 runs -> one delay per subject
    - Crop each run using onset_tr + subject_delay and fixed task length

    Returns
    -------
    shifted_paths : {run_idx: {sub_id: saved_path}}
    delay_info_all : {run_idx: {sub_id: per-run peak/delay info}}
    subject_delay_tr : {sub_id: averaged delay (TR)}
    """
    derivatives_dir = Path(derivatives_dir).resolve()
    roi_mask_path = Path(roi_mask_img).resolve()

    if out_root is None:
        out_root = derivatives_dir
    out_root = Path(out_root).resolve()

    # 1) Discover runs
    subject_runs = build_denoised_runs_dict(
        derivatives_dir=derivatives_dir,
        denoise_folder=denoise_folder,
        space=space,
        desc_keywords=desc_keywords,
        subjects=subjects,
    )
    if not subject_runs:
        raise RuntimeError("No denoised runs found for any subject.")

    n_runs_common = _infer_common_run_count(subject_runs)
    print(f"Found {len(subject_runs)} subjects, up to {n_runs_common} runs in common.")
    print(f"Using auditory ROI mask at: {roi_mask_path}")

    # 2) Estimate per-run delays (peak-based)
    delay_info_all = estimate_delay_for_all_runs(
        subject_runs,
        roi_mask_img=roi_mask_path,
        TR=TR,
        onset_tr=onset_tr,
        search_window_sec=search_window_sec,
        zscore=zscore,
        smooth_tr=smooth_tr,
    )

    # 3) Average delays across the first 4 runs/videos -> one per subject
    subject_delay_tr = average_subject_delay_across_runs(
        delay_info_all,
        n_runs=n_runs_to_average,
        max_delay_tr=max_delay_tr,
    )
    print(f"Computed subject-specific HRF delays (TR) from {n_runs_to_average} runs.")

    # 4) Crop + save each run
    shifted_paths: Dict[int, Dict[str, Path]] = {}

    for run_idx in range(n_runs_common):
        func_imgs: Dict[str, Path] = {
            sid: runs[run_idx]
            for sid, runs in subject_runs.items()
            if run_idx < len(runs)
        }

        # Task name is used to choose movie length (TR)
        # We assume run index corresponds to a single task across subjects.
        # If that isn't true in your dataset, you should determine movie_len per subject path.
        example_path = next(iter(func_imgs.values()))
        task = get_task_from_bold_path(example_path) or "unknownTask"

        if task not in task_movie_len_tr:
            print(f"Skipping run {run_idx} (task={task}) not in task_movie_len_tr")
            continue
        movie_len_tr = int(task_movie_len_tr[task])

        print(f"Run {run_idx}: task={task}, movie_len_tr={movie_len_tr}. Cropping using subject_delay_tr.")

        cropped_imgs, skipped = crop_run_dict_subject_delays(
            func_imgs,
            subject_delay_tr=subject_delay_tr,
            onset_tr=onset_tr,
            movie_len_tr=movie_len_tr,
            on_out_of_bounds=on_out_of_bounds,
        )
        print(f"Run {run_idx}: saved {len(cropped_imgs)} subjects. Skipped {len(skipped)}.")

        # Save with parallel directory structure
        shifted_paths[run_idx] = {}
        for sid, img in cropped_imgs.items():
            original_path = Path(func_imgs[sid])

            try:
                rel = original_path.relative_to(derivatives_dir)
            except ValueError:
                rel = Path(original_path.name)

            stem = original_path.stem
            if stem.endswith(".nii"):
                stem = stem[:-4]

            # CHANGED: name reflects MATLAB-style locked/cropped window
            new_name = stem + "_locked.nii.gz"
            out_path = out_root / rel.parent / new_name
            out_path.parent.mkdir(parents=True, exist_ok=True)

            nib.save(img, out_path)
            shifted_paths[run_idx][sid] = out_path

    return shifted_paths, delay_info_all, subject_delay_tr


# ================================================================
# 6) OPTIONAL QA: HRF plots (not used for shifting)
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
    Kept as QA/visualization. Not used by time_shift_all_denoised().
    """
    derivatives_dir = Path(derivatives_dir).resolve()
    mask_path = Path(mask_path).resolve()

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

        sub_dir: Path | None = None
        if save_root is not None:
            sub_dir = save_root / safe_sub
            sub_dir.mkdir(parents=True, exist_ok=True)

        for run_idx, bold_path in enumerate(runs):
            task = get_task_from_bold_path(bold_path) or "unknownTask"
            if runs_filter is not None and task.lower() not in runs_filter:
                continue

            title = f"{sub} – {task}\n{bold_path.name}"
            print(f"Analyzing {title}")

            safe_task = task.replace(" ", "_")
            save_name = f"{safe_sub}_{safe_task}_run-{run_idx+1}.png" if sub_dir is not None else None

            do_find_peaks = find_peaks and (onset_sec is not None)

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
                mark_peak=do_find_peaks,
            )

            if peak_info is None:
                continue

            rows.append(
                dict(
                    sub=sub,
                    task=task,
                    bold_path=str(bold_path),
                    peak_time_sec=peak_info.get("peak_time_sec", np.nan),
                    peak_latency_sec=peak_info.get("peak_latency_sec", np.nan),
                    peak_value=peak_info.get("peak_value", np.nan),
                )
            )

    df = pd.DataFrame(rows)

    if save_csv is None and save_root is not None:
        save_csv = save_root / "hrf_peaks.csv"

    if save_csv is not None:
        save_csv = Path(save_csv).resolve()
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_csv, index=False)
        print(f"Saved peak summary to {save_csv}")

    if save_root is not None:
        print(f"Saved HRF plots under: {save_root} (one subfolder per subject)")

    return df
