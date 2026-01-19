"""
Time-shift helpers for denoised 4D fMRI data (MNI space).

UPDATED PIPELINE (matches political 2021 MATLAB scripts-style HRF delay + cropping)
----------------------------------------------------------
- Discover denoised runs using build_denoised_runs_dict
- 🔴 Organize runs by TASK name (not run index) to handle subjects with extra/missing tasks
- Extract auditory ROI mean time series (per subject, per task)
- Detect HRF delay per task using Option B:
    "first auditory peak after stimulus onset within a search window"
- 🔴 Compute ONE subject-specific HRF delay = average across SELECTED tasks
- Crop each task/run like the MATLAB code:
    data_locked = data[:, onset_tr + subject_delay : onset_tr + subject_delay + movie_len_tr[task]]
- Save cropped (timeshifted) images in a parallel directory structure

Intended usage:
---------------
from yy_fmri_kit.time_shift import time_shift_all_denoised

shifted_paths, delay_info_by_task, subject_delay_tr = time_shift_all_denoised(
    derivatives_dir="derivatives/denoised",
    roi_mask_img="masks/auditory_roi.nii.gz",
    TR=1.0,
    onset_tr=8,
    task_movie_len_tr={"AntiLeft": 294, "AntiRight": 289, "ProLeft": 249, "ProRight": 207},
    tasks=["AntiLeft","AntiRight","ProLeft","ProRight"],   # task selection
    out_root="derivatives/denoised_timeshifted",
)

Then run voxelwise or parcelwise ISC on the cropped/locked files (do NOT re-shift later).
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import nibabel as nib
import pandas as pd

from yy_fmri_kit.io.find_files import build_denoised_runs_dict
from yy_fmri_kit.postproc.timeshift_core import get_task_from_bold_path
from yy_fmri_kit.visualization.hrf import plot_hrf_for_run
from yy_fmri_kit.postproc.extract_ts import extract_roi_mean_ts_from_4d, _load_niimg

# Type aliases
Array1D = np.ndarray
PathLike = Union[str, Path]

# ================================================================
# 2) TASK-KEYED ORGANIZATION (critical for task selection + missing tasks)
# ================================================================

def _subject_runs_to_task_map(
    subject_runs: Dict[str, List[Path]],
) -> Dict[str, Dict[str, Path]]:
    """
    Convert {sub: [paths...]} -> {sub: {task: path}}.

    This avoids assuming all subjects have the same number/order of runs.
    If a subject is missing a task, they are simply skipped for that task.
    """
    out: Dict[str, Dict[str, Path]] = {}
    for sub, runs in subject_runs.items():
        tmap: Dict[str, Path] = {}
        for p in runs:
            task = get_task_from_bold_path(p) or "unknownTask"
            tmap[task] = Path(p)  # last one wins if duplicates
        out[sub] = tmap
    return out


def _select_tasks(
    subject_task_map: Dict[str, Dict[str, Path]],
    tasks: Optional[Sequence[str]],
) -> Dict[str, Dict[str, Path]]:
    """
    Keep only selected tasks for BOTH estimation and cropping.
    If tasks is None -> keep all tasks.
    """
    if tasks is None:
        return subject_task_map
    allow = set(tasks)
    return {
        sub: {t: p for t, p in tmap.items() if t in allow}
        for sub, tmap in subject_task_map.items()
    }


def _available_tasks(
    subject_task_map: Dict[str, Dict[str, Path]],
    tasks: Optional[Sequence[str]],
) -> List[str]:
    """
    Determine tasks to process.

    If tasks is provided: keep only tasks that exist for at least 1 subject.
    (No ">=2" requirement because this is not a group method anymore.)
    """
    if tasks is not None:
        out: List[str] = []
        for t in tasks:
            n_have = sum(t in tmap for tmap in subject_task_map.values())
            if n_have >= 1:
                out.append(t)
        return out

    # infer all tasks found across subjects
    all_tasks = set()
    for tmap in subject_task_map.values():
        all_tasks.update(tmap.keys())
    return sorted(all_tasks)


# ================================================================
# 3) HRF delay detection
# ================================================================

def _zscore_1d(x: Array1D) -> Array1D:
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
    Find FIRST peak after stimulus onset within a time window.

    Returns dict with:
      - peak_tr
      - peak_latency_tr  (relative to onset_tr)
      - peak_latency_sec
      - peak_value
      - window_tr
    """
    ts2 = _zscore_1d(ts) if zscore else np.asarray(ts, dtype=float)

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
        raise ValueError(f"Peak window too small after clipping: [{w0},{w1}).")

    seg = ts2[w0:w1]

    peak_rel = None
    for i in range(1, len(seg) - 1):
        if seg[i] > seg[i - 1] and seg[i] >= seg[i + 1]:
            peak_rel = i
            break
    if peak_rel is None:
        peak_rel = int(np.argmax(seg))

    peak_tr = int(w0 + peak_rel)
    peak_lat_tr = int(peak_tr - onset_tr)
    peak_lat_tr = max(0, peak_lat_tr)

    return dict(
        peak_tr=peak_tr,
        peak_latency_tr=peak_lat_tr,
        peak_latency_sec=float(peak_lat_tr * TR),
        peak_value=float(ts2[peak_tr]),
        window_tr=(w0, w1),
    )


def estimate_delay_for_selected_tasks(
    subject_task_map: Dict[str, Dict[str, Path]],
    *,
    roi_mask_img: PathLike,
    TR: float,
    onset_tr: int,
    tasks: Optional[Sequence[str]] = None,
    search_window_sec: Tuple[float, float] = (2.0, 12.0),
    zscore: bool = True,
    smooth_tr: int = 0,
) -> Dict[str, Dict[str, dict]]:
    """
    Estimate per-subject HRF delay (TR) for each task, using first peak latency.

    Returns
    -------
    delay_info_by_task : {task: {sub_id: {... delay info ...}}}
    """
    roi_mask_img = Path(roi_mask_img).resolve()
    tasks_to_process = _available_tasks(subject_task_map, tasks)

    delay_info_by_task: Dict[str, Dict[str, dict]] = {}

    for task in tasks_to_process:
        per_sub: Dict[str, dict] = {}

        for sid, tmap in subject_task_map.items():
            if task not in tmap:
                continue

            func_path = tmap[task]
            ts = extract_roi_mean_ts_from_4d(func_path, roi_mask_img)

            info = find_first_peak_after_onset(
                ts,
                TR=TR,
                onset_tr=onset_tr,
                search_window_sec=search_window_sec,
                zscore=zscore,
                smooth_tr=smooth_tr,
            )

            per_sub[sid] = dict(
                delay_tr=int(info["peak_latency_tr"]),
                delay_sec=float(info["peak_latency_sec"]),
                peak_tr=int(info["peak_tr"]),
                peak_value=float(info["peak_value"]),
                window_tr=info["window_tr"],
            )

        delay_info_by_task[task] = per_sub
        print(f"Estimated delays for task '{task}' ({len(per_sub)} subjects)")

    return delay_info_by_task


def average_subject_delay_across_tasks(
    delay_info_by_task: Dict[str, Dict[str, dict]],
    *,
    tasks: Optional[Sequence[str]] = None,
    max_delay_tr: int | None = None,
) -> Dict[str, int]:
    """
    Compute one delay per subject by averaging delays across tasks.

    If tasks is provided, average only across those tasks (that exist in dict).
    Subjects missing a task are averaged over the tasks they DO have.
    """
    tasks_in = list(delay_info_by_task.keys())
    tasks_use = [t for t in (tasks or tasks_in) if t in delay_info_by_task]

    delays_by_sub: Dict[str, List[int]] = {}
    for task in tasks_use:
        for sid, info in delay_info_by_task[task].items():
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
# 4) Cropping
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

    Returns 4D NIfTI with exactly `movie_len_tr` timepoints.
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

    return nib.Nifti1Image(data[..., start:end], affine=img.affine, header=img.header)


def crop_task_images_for_subjects(
    subject_task_map: Dict[str, Dict[str, Path]],
    *,
    task: str,
    subject_delay_tr: Dict[str, int],
    onset_tr: int,
    movie_len_tr: int,
    on_out_of_bounds: str = "skip",
) -> Tuple[Dict[str, nib.Nifti1Image], List[str]]:
    """
    Crop one task across all subjects that have it, using subject_delay_tr.
    """
    cropped: Dict[str, nib.Nifti1Image] = {}
    skipped: List[str] = []

    for sid, tmap in subject_task_map.items():
        if task not in tmap:
            continue
        if sid not in subject_delay_tr:
            skipped.append(sid)
            continue

        img = _load_niimg(tmap[task])
        try:
            cropped[sid] = crop_4d_by_onset_and_delay(
                img,
                onset_tr=onset_tr,
                delay_tr=int(subject_delay_tr[sid]),
                movie_len_tr=int(movie_len_tr),
            )
        except ValueError as e:
            if on_out_of_bounds == "error":
                raise
            print(f"⚠️ Skipping {sid} for task {task} (crop failed): {e}")
            skipped.append(sid)

    return cropped, skipped


# ================================================================
# 5) HIGH-LEVEL API: delay (per task) -> avg per subject -> crop per task -> save
# ================================================================

def time_shift_all_denoised(
    derivatives_dir: PathLike,
    *,
    roi_mask_img: PathLike,
    TR: float,
    onset_tr: int,
    task_movie_len_tr: Dict[str, int],
    tasks: Optional[Sequence[str]] = None,  # task selection
    denoise_folder: str = "",
    space: str = "MNI152NLin2009cAsym",
    desc_keywords: Sequence[str] = ("denoised", "clean", "nltoolsClean", "preproc"),
    out_root: Optional[PathLike] = None,
    subjects: Optional[Sequence[str]] = None,
    search_window_sec: Tuple[float, float] = (2.0, 12.0),
    smooth_tr: int = 0,
    zscore: bool = True,
    max_delay_tr: int | None = None,
    on_out_of_bounds: str = "skip",
) -> Tuple[Dict[str, Dict[str, Path]], Dict[str, Dict[str, dict]], Dict[str, int]]:
    """
    Task-keyed pipeline:
    1) find per-task per-subject delay (peak latency after onset)
    2) average delays across selected tasks -> one delay per subject
    3) crop each task using subject delay and task-specific movie_len_tr
    4) save as *_locked.nii.gz

    Returns
    -------
    shifted_paths : {task: {sub_id: saved_path}}
    delay_info_by_task : {task: {sub_id: delay info}}
    subject_delay_tr : {sub_id: averaged delay}
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

    # 2) Convert to task map + apply task selection
    subject_task_map = _subject_runs_to_task_map(subject_runs)
    subject_task_map = _select_tasks(subject_task_map, tasks)
    tasks_to_process = _available_tasks(subject_task_map, tasks)

    print(f"Using auditory ROI mask at: {roi_mask_path}")
    print(f"Tasks to process: {tasks_to_process}")

    # 3) Estimate peak-based delays per task
    delay_info_by_task = estimate_delay_for_selected_tasks(
        subject_task_map,
        roi_mask_img=roi_mask_path,
        TR=TR,
        onset_tr=onset_tr,
        tasks=tasks_to_process,
        search_window_sec=search_window_sec,
        zscore=zscore,
        smooth_tr=smooth_tr,
    )

    # 4) One delay per subject = average across selected tasks
    subject_delay_tr = average_subject_delay_across_tasks(
        delay_info_by_task,
        tasks=tasks_to_process,
        max_delay_tr=max_delay_tr,
    )
    print(f"Computed subject-specific delays for {len(subject_delay_tr)} subjects.")

    # 5) Crop + save per task
    shifted_paths: Dict[str, Dict[str, Path]] = {}

    for task in tasks_to_process:
        if task not in task_movie_len_tr:
            print(f"⚠️ Skipping task '{task}': not found in task_movie_len_tr")
            continue

        movie_len_tr = int(task_movie_len_tr[task])
        print(f"Cropping task '{task}' with movie_len_tr={movie_len_tr} using subject_delay_tr")

        cropped_imgs, skipped = crop_task_images_for_subjects(
            subject_task_map,
            task=task,
            subject_delay_tr=subject_delay_tr,
            onset_tr=onset_tr,
            movie_len_tr=movie_len_tr,
            on_out_of_bounds=on_out_of_bounds,
        )

        shifted_paths[task] = {}
        for sid, img in cropped_imgs.items():
            original_path = subject_task_map[sid][task]

            try:
                rel = original_path.relative_to(derivatives_dir)
            except ValueError:
                rel = Path(original_path.name)

            stem = original_path.stem
            if stem.endswith(".nii"):
                stem = stem[:-4]

            new_name = stem + "_locked.nii.gz"
            out_path = out_root / rel.parent / new_name
            out_path.parent.mkdir(parents=True, exist_ok=True)

            nib.save(img, out_path)
            shifted_paths[task][sid] = out_path

        print(f"Task '{task}': saved {len(cropped_imgs)}. skipped {len(skipped)}.")

    return shifted_paths, delay_info_by_task, subject_delay_tr

# ================================================================
# 6) OPTIONAL QA: HRF plots
# ================================================================

def analyze_hrf_all_runs(
    derivatives_dir: PathLike,
    *,
    roi_mask_img: PathLike,
    TR: float,
    denoise_folder: str = "",
    space: str = "MNI152NLin2009cAsym",
    desc_keywords: Sequence[str] = ("denoised", "clean", "nltoolsClean", "preproc"),
    subjects: Optional[Sequence[str]] = None,
    tasks: Optional[Sequence[str]] = None,
    mark_onset: bool = True,
    onset_tr: Optional[int] = None,         # if provided, onset_sec = onset_tr * TR
    onset_sec: Optional[float] = None,      # overrides onset_tr if both provided
    find_peaks: bool = True,
    zscore: bool = True,
    save_png: Optional[PathLike] = None,    # if None -> show interactively
    save_csv: Optional[PathLike] = None,    # if None and save_png provided -> save next to plots
) -> pd.DataFrame:
    """
    QA/visualization helper (NOT used by time_shift_all_denoised).
    - processes only `tasks` if provided (in that order), otherwise all tasks present in >=2 subjects

    Outputs
    -------
    df : DataFrame with one row per (subject, task) plotted.
    """
    derivatives_dir = Path(derivatives_dir).resolve()
    roi_mask_img = Path(roi_mask_img).resolve()

    # Decide onset seconds
    if onset_sec is None and onset_tr is not None:
        onset_sec = float(onset_tr) * float(TR)

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

    # 2) Convert to task map + apply selection logic
    subject_task_map = _subject_runs_to_task_map(subject_runs)
    subject_task_map = _select_tasks(subject_task_map, tasks)  # filters per subject
    tasks_to_process = _available_tasks(subject_task_map, tasks)  # stable task list

    if len(tasks_to_process) == 0:
        raise RuntimeError(
            "No tasks to process after selection. "
            "If tasks was provided, make sure task names match get_task_from_bold_path()."
        )

    # 3) Output dirs
    save_root: Optional[Path] = None
    if save_png is not None:
        save_root = Path(save_png).resolve()
        save_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    # 4) Loop tasks then subjects (task-keyed, no run-index assumptions)
    for task in tasks_to_process:
        # subjects who have this task
        subs_with_task = [s for s, tmap in subject_task_map.items() if task in tmap]
        if len(subs_with_task) == 0:
            continue

        print(f"QA HRF: task='{task}' ({len(subs_with_task)} subjects)")

        for sub in subs_with_task:
            bold_path = subject_task_map[sub][task]

            # folder per subject (like your old QA), optional
            sub_dir: Optional[Path] = None
            if save_root is not None:
                safe_sub = sub.replace(" ", "_")
                sub_dir = save_root / safe_sub
                sub_dir.mkdir(parents=True, exist_ok=True)

            title = f"{sub} – {task}\n{Path(bold_path).name}"

            # decide output name
            save_name = None
            if sub_dir is not None:
                safe_task = task.replace(" ", "_")
                save_name = f"{safe_sub}_{safe_task}.png"

            do_find_peaks = bool(find_peaks and (onset_sec is not None))

            peak_info = plot_hrf_for_run(
                bold_path=bold_path,
                mask_path=roi_mask_img,
                TR=TR,
                mark_onset=mark_onset and (onset_sec is not None),
                onset_sec=onset_sec,
                zscore=zscore,
                title=title,
                save_dir=sub_dir,
                save_name=save_name,
                show=(save_root is None),
                mark_peak=do_find_peaks,
            )

            # plot_hrf_for_run might return None if it fails gracefully
            if peak_info is None:
                rows.append(
                    dict(
                        sub=sub,
                        task=task,
                        bold_path=str(bold_path),
                        peak_time_sec=np.nan,
                        peak_latency_sec=np.nan,
                        peak_value=np.nan,
                    )
                )
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

    # 5) Save CSV summary (optional)
    if save_csv is None and save_root is not None:
        save_csv = save_root / "hrf_peaks_QA.csv"

    if save_csv is not None:
        save_csv = Path(save_csv).resolve()
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_csv, index=False)
        print(f"Saved QA peak summary to: {save_csv}")

    if save_root is not None:
        print(f"Saved QA plots under: {save_root} (one subfolder per subject)")

    return df

# ================================================================
# Save timeshift info helper (optional)
# ================================================================
def save_timeshift_report(
    out_csv,
    *,
    delay_info_all,
    subject_delay_tr,
    shifted_paths=None,   # optional
):
    rows = []

    # delay_info_all is like: {task: {sub: {delay_tr, ...}}}  OR {run_idx: {sub: {...}}}
    for key, subdict in delay_info_all.items():
        for sub, info in subdict.items():
            rows.append({
                "key": key,  # task name or run index
                "sub": sub,
                "delay_tr_this": info.get("delay_tr", None),
                "delay_sec_this": info.get("delay_sec", None),
                "peak_tr": info.get("peak_tr", None),
                "peak_value": info.get("peak_value", None),
                "window_tr_start": info.get("window_tr", (None, None))[0],
                "window_tr_end": info.get("window_tr", (None, None))[1],
                "subject_delay_tr_mean": subject_delay_tr.get(sub, None),
            })

    df = pd.DataFrame(rows)

    # Optional: add saved file paths if you want
    if shifted_paths is not None:
        # shifted_paths is {key: {sub: path}}
        path_rows = []
        for key, subpaths in shifted_paths.items():
            for sub, p in subpaths.items():
                path_rows.append({"key": key, "sub": sub, "locked_path": str(p)})
        df_paths = pd.DataFrame(path_rows)
        df = df.merge(df_paths, on=["key", "sub"], how="left")

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df