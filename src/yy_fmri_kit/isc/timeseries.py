# yy_fmri_kit/isc/timeseries.py

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np
import pandas as pd

from yy_fmri_kit.static.isc.config import ISCConfig
from yy_fmri_kit.io.find_files import (
    iter_subject_denoised,
    iter_subjects,
)

# ==== roiwise ====
def extract_roi_timeseries(
    func_img: nib.Nifti1Image,
    roi_img: nib.Nifti1Image,
    *,
    reduce: str = "mean",
    standardize: bool = True,
) -> np.ndarray:
    """
    Extract a single ROI time series from a 4D functional image.

    Parameters
    ----------
    func_img : Nifti1Image
        4D fMRI image (X, Y, Z, T).
    roi_img : Nifti1Image
        3D binary ROI mask (X, Y, Z).
    reduce : {"mean", "median"}
        How to collapse voxels within the ROI.
    standardize : bool
        If True, z-score the time series over time.

    Returns
    -------
    ts : np.ndarray
        ROI time series with shape (T, 1).
    """
    func_data = func_img.get_fdata()              # (X, Y, Z, T)
    roi_data = roi_img.get_fdata().astype(bool)   # (X, Y, Z)

    if func_data.shape[:3] != roi_data.shape:
        raise ValueError(
            f"Functional data and ROI mask shapes differ: "
            f"{func_data.shape[:3]} vs {roi_data.shape}"
        )

    # Flatten over ROI voxels: (V_roi, T)
    roi_voxels = func_data[roi_data, :]           # (V_roi, T)

    if roi_voxels.size == 0:
        raise ValueError("ROI mask contains no voxels in the functional FOV.")

    if reduce == "mean":
        ts = roi_voxels.mean(axis=0)              # (T,)
    elif reduce == "median":
        ts = np.median(roi_voxels, axis=0)
    else:
        raise ValueError(f"Unknown reduce method: {reduce!r}")

    ts = ts.astype(float)

    if standardize:
        ts_mean = ts.mean()
        ts_std = ts.std()
        ts = (ts - ts_mean) / (ts_std + 1e-8)

    # Make it (T, 1) to match (T, F) convention
    return ts[:, np.newaxis]


def load_all_subject_roi_data(
    config: ISCConfig,
    roi_img: nib.Nifti1Image,
) -> Dict[str, np.ndarray]:
    """
    Load a single ROI time series for each subject.

    For now:
    - Uses the FIRST functional run per subject from `build_subject_runs_dict`.
    - Ignores tasks/conditions (just a smoke test).

    Parameters
    ----------
    config : ISCConfig
        Contains `derivatives_dir`, optional `subjects`, etc.
    roi_img : Nifti1Image
        ROI mask (e.g., A1).

    Returns
    -------
    roi_data : dict
        Mapping: subject ID -> ROI time series (T, 1).
    """
    denoised_root = config.derivatives_dir

    # Discover subjects if not passed explicitly
    if config.subjects is None:
        subjects = iter_subjects(denoised_root)  # e.g. ['sub-1', 'sub-2', ...]
    else:
        subjects = config.subjects

    roi_data: Dict[str, np.ndarray] = {}

    for sub in subjects:
        # 🔴 This uses YOUR helper exactly as-is
        runs = iter_subject_denoised(
            derivatives_dir=denoised_root,
            sub=sub,
            space=config.space,
            # you can tweak desc_keywords/suffix here if needed
        )
        if len(runs) == 0:
            raise RuntimeError(f"No functional runs found for subject {sub}.")

        # 🟢 SIMPLE: pick the first run as a smoke test
        first_run_path = Path(runs[0]).resolve() # TODO ---- modify to include all runs
        func_img = nib.load(str(first_run_path))

        ts = extract_roi_timeseries(
            func_img,
            roi_img,
            reduce="mean",
            standardize=True,
        )
        roi_data[sub] = ts

    return roi_data


# ==== parcelwise ====
def find_parcel_timeseries_file(
    config: ISCConfig,
    sub: str,
    task: Optional[str] = None,
) -> Path:
    """
    Locate a parcellated timeseries file for one subject (and optionally a task).

    Assumes something like:
        parcellation_dir/sub-1/sub-1_task-XXX_parc-<atlas>_desc-<desc>.tsv

    You must adapt the glob pattern to your actual naming.
    """
    root = config.parcellation_dir / sub
    if not root.exists():
        raise FileNotFoundError(f"Parcellation dir not found: {root}")
    atlas_tag = config.tf_atlas or config.atlas_name  # TF if exists, else local
    if atlas_tag is None:
        raise ValueError("Need either config.tf_atlas or config.atlas_name (or atlas_nii) to find files.")

    res = config.tf_resolution or config.res
    space = config.space
    res_part = f"_res-{res}" if res is not None else ""


    if task is None:
        pattern = f"{sub}_*task-*_space-{space}{res_part}_*atlas-{atlas_tag}_timeseries*"
    else:
        pattern = f"{sub}_*task-{task}_space-{space}{res_part}_*atlas-{atlas_tag}_timeseries*"

    matches = sorted(root.glob(pattern))
    if len(matches) == 0:
        raise FileNotFoundError(f"No parcellated timeseries files found with pattern {pattern} in {root}")
    if len(matches) > 1:
        raise RuntimeError(
            f"Expected 1 parcellated file for {sub} (task={task}), found {len(matches)}: {matches}"
        )
    return matches[0]

def load_parcel_timeseries(path: Path) -> np.ndarray:
    """
    Load a T × P parcel timeseries from a TSV/CSV file.

    Assumes:
        - rows = timepoints
        - columns = parcels
    """
    df = pd.read_csv(path, sep="\t")  # adjust sep if needed
    data = df.to_numpy(dtype=float)
    return data  # (T, P)

def load_all_subject_parcel_data(
    config: ISCConfig,
    tasks: Optional[List[str]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load parcellated timeseries for all subjects.

    Returns
    -------
    data : dict
        {
          "sub-1": {
             "task-1": (T × P),
             "task-2": (T × P),
          },
          ...
        }
    """
    root = config.parcellation_dir

    # discover subjects
    if config.subjects is None:
        subjects = iter_subjects(root)
    else:
        subjects = config.subjects

    if tasks is None:
        # you can decide later how to auto-discover tasks;
        # for now require tasks explicitly or treat as a single "all" condition
        tasks = ["all"]

    data: Dict[str, Dict[str, np.ndarray]] = {}

    for sub in subjects:
        data[sub] = {}
        for task in tasks:
            # If you really don't have tasks, you can treat task as a dummy label
            task_name_for_file = None if task == "all" else task

            fpath = find_parcel_timeseries_file(config, sub=sub, task=task_name_for_file)
            ts = load_parcel_timeseries(fpath)  # (T, P)
            data[sub][task] = ts

    return data
