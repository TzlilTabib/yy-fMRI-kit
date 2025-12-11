"""
Compute temporal signal-to-noise ratio (tSNR) within an ROI for fMRI data.
Use this after denoising to assess data quality.
"""

from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd

from yy_fmri_kit.io.find_files import build_runs_dict_generic


def compute_roi_tsnr(bold_path: Path, mask_path: Path) -> float:
    """
    Compute mean tSNR within an ROI for a single 4D BOLD file.
    tSNR = mean(signal) / std(signal) over time, averaged across ROI voxels.
    """
    bold_img = nib.load(str(bold_path))
    bold = bold_img.get_fdata()  # (X, Y, Z, T)

    mask_img = nib.load(str(mask_path))
    mask = mask_img.get_fdata().astype(bool)  # (X, Y, Z)

    if bold.shape[:3] != mask.shape:
        raise ValueError(
            f"Shape mismatch: bold {bold.shape[:3]} vs mask {mask.shape} "
            f"for file {bold_path}"
        )

    # reshape: voxels x time
    roi_data = bold[mask]  # (n_voxels, T)
    if roi_data.size == 0:
        raise ValueError(f"ROI mask is empty for {bold_path}")

    # tSNR per voxel
    mean_ts = roi_data.mean(axis=1)
    std_ts = roi_data.std(axis=1, ddof=1)

    valid = std_ts > 0
    if not np.any(valid):
        raise ValueError(f"No non-zero std voxels in ROI for {bold_path}")

    tsnr_vox = mean_ts[valid] / std_ts[valid]
    return float(tsnr_vox.mean())


def tsnr_for_single_pipeline(
    deriv_root: str | Path,
    mask_path: str | Path,
    subjects: list[str],
    denoise_folder: str = "",
    task_filter: str | None = None,
    pipeline_name: str = "pipeline",
    save_csv: str | Path = "tsnr_single_pipeline.csv",
) -> pd.DataFrame:
    """
    Compute ROI tSNR for a set of subjects for ONE pipeline.

    Parameters
    ----------
    deriv_root : derivatives root of this pipeline (on this machine)
    mask_path : ROI mask (same space as denoised bold)
    subjects : list of subject IDs (e.g. ["sub-01", ...])
    denoise_folder : passed to build_denoised_runs_dict
    task_filter : if not None, only include tasks whose name contains this string
    pipeline_name : label to write in 'pipeline' column ("old" or "new")
    save_csv : path to save the resulting CSV

    Returns
    -------
    df : DataFrame with columns:
         subject, pipeline, task, run, tsnr
    """
    deriv_root = Path(deriv_root)
    mask_path = Path(mask_path)
    save_csv = Path(save_csv)

    runs_dict = build_runs_dict_generic(deriv_root)

    rows = []

    for sub in subjects:
        if sub not in runs_dict:
            print(f"⚠️ Subject {sub} not found in runs_dict, skipping.")
            continue
        sub_runs = runs_dict[sub]

        for i, bold_path in enumerate(sub_runs, start=1):
            name = bold_path.name
            task = "unknown"
            if "task-" in name:
                # crude but works for BIDS-like names
                try:
                    after = name.split("task-")[1]
                    task = after.split("_")[0]
                except Exception:
                    pass

            run_id = f"run-{i:02d}"
            if "run-" in name:
                try:
                    after = name.split("run-")[1]
                    run_id = "run-" + after.split("_")[0].split(".")[0]
                except Exception:
                    pass

            # apply optional task_filter
            if task_filter is not None and task_filter not in task:
                continue

            # compute tSNR
            try:
                tsnr = compute_roi_tsnr(bold_path, mask_path)
            except Exception as e:
                print(f"Error for {pipeline_name}, {sub}, {task}, {run_id}: {e}")
                continue

            rows.append(
                {
                    "subject": sub,
                    "pipeline": pipeline_name,
                    "task": task,
                    "run": run_id,
                    "tsnr": tsnr,
                }
            )

    df = pd.DataFrame(rows)

    save_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_csv, index=False)
    print(f"Saved tSNR data for {pipeline_name} to {save_csv}")
    return df