"""
Crop 4D fMRI timeseries to stimulus windows defined on TR grid,

Usage:
from yy_fmri_kit.postproc.cropping import CropSpec, crop_denoised_runs

crop_specs = {
    "AntiLeft":  CropSpec(drop_start_tr=8, drop_end_tr=12),
    "AntiRight": CropSpec(drop_start_tr=8, drop_end_tr=12),
    "ProLeft":   CropSpec(drop_start_tr=8, drop_end_tr=12),
    "ProRight":  CropSpec(drop_start_tr=8, drop_end_tr=12),
}

crop_denoised_runs(
    derivatives_dir=Path("/media/.../denoised"),
    output_dir=Path("/media/.../denoised_cropped"),
    crop_specs=crop_specs,
    tasks=["AntiLeft", "AntiRight"],
    desc_keywords=("nltoolsClean",),  # IMPORTANT to match your BOLD files
    save_desc_suffix="nltoolsClean_cropped",
)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, Sequence

import numpy as np
import nibabel as nib

from yy_fmri_kit.io.find_files import build_denoised_runs_dict

PathLike = Union[str, Path]
ImgLike = Union[PathLike, nib.Nifti1Image]

# =============================================================
# CONFIG-LIKE SPEC
# =============================================================

@dataclass(frozen=True)
class CropSpec:
    """
    Defines a stimulus window crop on the TR grid.

    Parameters
    ----------
    drop_start_tr : int
        Number of TRs to remove from the beginning (e.g., pre-stim).
    drop_end_tr : int
        Number of TRs to remove from the end (e.g., end fixation).
    keep_len_tr : int | None
        If provided, enforce exact cropped length (after dropping start).
        Useful if tasks differ in total length; you can still force the final
        segment length to match expectations.
    """
    drop_start_tr: int = 0
    drop_end_tr: int = 0
    keep_len_tr: Optional[int] = None


# =============================================================
# LOW-LEVEL HELPERS
# =============================================================

def _load_4d(img: ImgLike) -> nib.Nifti1Image:
    if isinstance(img, nib.spatialimages.SpatialImage):
        return img
    return nib.load(str(img))


def compute_crop_bounds(
    n_tr: int,
    spec: CropSpec,
) -> Tuple[int, int]:
    """
    Convert a CropSpec into [start, stop] indices on the TR axis.
    """
    if n_tr <= 0:
        raise ValueError(f"n_tr must be > 0, got {n_tr}")

    start = int(spec.drop_start_tr)
    if start < 0:
        raise ValueError(f"drop_start_tr must be >= 0, got {start}")

    # If keep_len_tr is given, ignore drop_end_tr and compute stop by length.
    if spec.keep_len_tr is not None:
        keep_len = int(spec.keep_len_tr)
        if keep_len <= 0:
            raise ValueError(f"keep_len_tr must be > 0, got {keep_len}")
        stop = start + keep_len
    else:
        end_drop = int(spec.drop_end_tr)
        if end_drop < 0:
            raise ValueError(f"drop_end_tr must be >= 0, got {end_drop}")
        stop = n_tr - end_drop

    if start >= stop:
        raise ValueError(
            f"Invalid crop bounds for n_tr={n_tr}: start={start}, stop={stop}. "
            f"(drop_start_tr={spec.drop_start_tr}, drop_end_tr={spec.drop_end_tr}, keep_len_tr={spec.keep_len_tr})"
        )
    if stop > n_tr:
        raise ValueError(
            f"Crop stop={stop} exceeds n_tr={n_tr}. "
            f"(drop_start_tr={spec.drop_start_tr}, keep_len_tr={spec.keep_len_tr})"
        )

    return start, stop


def crop_timeseries_array(
    ts: np.ndarray,
    spec: CropSpec,
) -> np.ndarray:
    """
    Crop a (T, F) array along time axis.

    Returns a new array view/copy (depending on numpy slicing).
    """
    if ts.ndim != 2:
        raise ValueError(f"Expected (T, F) array, got shape {ts.shape}")
    T = ts.shape[0]
    start, stop = compute_crop_bounds(T, spec)
    return ts[start:stop, :]


def crop_nifti_time(
    img: ImgLike,
    spec: CropSpec,
) -> nib.Nifti1Image:
    """
    Crop a 4D NIfTI (X, Y, Z, T) along the time axis.

    Preserves affine and header.
    """
    img4d = _load_4d(img)
    data = np.asanyarray(img4d.dataobj)
    if data.ndim != 4:
        raise ValueError(f"Expected 4D image, got shape {data.shape}")
    T = data.shape[3]
    start, stop = compute_crop_bounds(T, spec)

    out = np.asarray(data[..., start:stop])
    return nib.Nifti1Image(out, affine=img4d.affine, header=img4d.header)


# =============================================================
# HIGH-LEVEL HELPER
# =============================================================

def crop_denoised_runs(
    *,
    derivatives_dir: Path,
    output_dir: Path,
    crop_specs: Dict[str, CropSpec],
    tasks: Optional[Sequence[str]] = None,
    space: str = "MNI152NLin2009cAsym",
    desc_keywords: Sequence[str] = ("denoised", "clean", "nltoolsClean", "preproc"),
    save_desc_suffix: str = "nltoolsClean_cropped",
    subjects: Optional[Sequence[str]] = None,
    overwrite: bool = False,
) -> None:
    """
    Crop denoised 4D BOLD runs to stimulus windows, task-wise.

    Parameters
    ----------
    derivatives_dir
        Root derivatives directory containing denoised runs.
    output_dir
        Where cropped runs will be written (mirrors sub/ses/func structure).
    crop_specs
        Mapping task -> CropSpec.
    tasks
        Which tasks to process (e.g. ["AntiLeft", "AntiRight"]).
        If None, all tasks found in filenames are considered (must exist in crop_specs).
    space, desc_keywords, subjects
        Passed to build_denoised_runs_dict.
    overwrite
        If False, skip files that already exist in output_dir.
    """

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_runs = build_denoised_runs_dict(
        derivatives_dir=derivatives_dir,
        space=space,
        desc_keywords=desc_keywords,
        subjects=subjects,
    )

    for sub, runs in subject_runs.items():
        for run_path in runs:
            fname = run_path.name

            # ---- Task detection ----
            matched_tasks = [t for t in crop_specs if f"task-{t}" in fname]
            if not matched_tasks:
                continue

            task = matched_tasks[0]

            if tasks is not None and task not in tasks:
                continue

            spec = crop_specs[task]

            # ---- Load + crop ----
            img = nib.load(str(run_path))
            cropped = crop_nifti_time(img, spec)

            # ---- Build output path (mirror structure) ----
            rel = run_path.relative_to(derivatives_dir)

            if save_desc_suffix is not None:
                fname = run_path.name
                if "_desc-" in fname:
                    fname = fname.replace("_desc-", f"_desc-{save_desc_suffix}_", 1)
                else:
                    fname = fname.replace("_bold", f"_desc-{save_desc_suffix}_bold")
            else:
                fname = run_path.name

            out_path = output_dir / rel.parent / fname

            out_path.parent.mkdir(parents=True, exist_ok=True)

            if out_path.exists() and not overwrite:
                print(f"⏭️  Exists, skipping: {out_path.name}")
                continue

            nib.save(cropped, out_path)
            print(f"✂️  Cropped [{task}] → {out_path}")