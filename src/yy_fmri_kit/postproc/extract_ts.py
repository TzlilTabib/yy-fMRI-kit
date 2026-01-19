# yy_fmri_kit/postproc/extract_ts.py

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import nibabel as nib

from yy_fmri_kit.io.find_files import build_denoised_runs_dict
from yy_fmri_kit.postproc.timeshift_core import get_task_from_bold_path

Array1D = np.ndarray
PathLike = Union[str, Path]


def _load_niimg(img: PathLike | nib.spatialimages.SpatialImage) -> nib.Nifti1Image:
    if isinstance(img, nib.spatialimages.SpatialImage):
        return img
    return nib.load(str(img))


def extract_roi_mean_ts_from_4d(
    func_img: PathLike | nib.Nifti1Image,
    roi_mask_img: PathLike | nib.Nifti1Image,
    mask_threshold: float = 0.5,
) -> Array1D:
    """
    Extract mean ROI time series from a 4D NIfTI within an ROI mask.
    Requires mask and func to be on the same 3D grid.
    """
    func_img = _load_niimg(func_img)
    roi_mask_img = _load_niimg(roi_mask_img)

    func_data = func_img.get_fdata()      # (X, Y, Z, T)
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

    T = int(func_data.shape[-1])
    func_flat = func_data.reshape(-1, T)
    mask_flat = mask.reshape(-1)

    return func_flat[mask_flat].mean(axis=0)


def extract_roi_ts_for_denoised_runs(
    derivatives_dir: PathLike,
    *,
    roi_mask_img: PathLike,
    tasks: Optional[Sequence[str]] = None,
    subjects: Optional[Sequence[str]] = None,
    denoise_folder: str = "",
    space: str = "MNI152NLin2009cAsym",
    desc_keywords: Sequence[str] = ("denoised", "clean", "nltoolsClean", "preproc"),
    mask_threshold: float = 0.5,
    allow_duplicates: bool = False,
) -> Tuple[Dict[str, Dict[str, Array1D]], pd.DataFrame]:
    """
    Wrapper: discover denoised runs and extract mean ROI TS per (subject, task).

    Returns
    -------
    ts_dict : {sub: {task: ts}}   OR if allow_duplicates: {sub: {task: [ts, ...]}}
    meta_df : DataFrame with columns [sub, task, T, bold_path]
    """
    derivatives_dir = Path(derivatives_dir).resolve()
    roi_mask_img = _load_niimg(roi_mask_img)

    allow_set = set(tasks) if tasks is not None else None

    subject_runs = build_denoised_runs_dict(
        derivatives_dir=derivatives_dir,
        denoise_folder=denoise_folder,
        space=space,
        desc_keywords=desc_keywords,
        subjects=subjects,
    )

    ts_dict: Dict[str, Dict[str, Array1D]] = {}
    rows: List[dict] = []

    for sub, paths in subject_runs.items():
        ts_dict[sub] = {}

        for p in paths:
            p = Path(p)
            task = get_task_from_bold_path(p)
            if task is None:
                continue
            if allow_set is not None and task not in allow_set:
                continue

            ts = extract_roi_mean_ts_from_4d(
                func_img=p,
                roi_mask_img=roi_mask_img,
                mask_threshold=mask_threshold,
            )

            if allow_duplicates:
                ts_dict[sub].setdefault(task, [])
                ts_dict[sub][task].append(ts)
            else:
                # last one wins if duplicates
                ts_dict[sub][task] = ts

            rows.append({"sub": sub, "task": task, "T": len(ts), "bold_path": str(p)})

    meta_df = pd.DataFrame(rows).sort_values(["sub", "task"]).reset_index(drop=True)
    return ts_dict, meta_df
