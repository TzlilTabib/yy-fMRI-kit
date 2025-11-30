from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import nibabel as nib

from yy_fmri_kit.isc.compute import compute_isc
from yy_fmri_kit.io.find_files import iter_subjects, iter_subject_denoised

PathLike = Union[str, Path]
Array1D = np.ndarray


def compute_roi_isc_for_run(
    derivatives_dir: PathLike,
    roi_mask_img: PathLike,
    *,
    run_idx: int = 0,
    subjects: Optional[Sequence[str]] = None,
    space: str = "MNI152NLin2009cAsym",
    desc_keywords=("denoised", "clean", "nltoolsClean", "preproc", "timeshift"),
    mask_threshold: float = 0.5,
) -> tuple[nib.Nifti1Image, Array1D, Dict[str, Path]]:
    """
    Compute voxelwise ISC inside an ROI for a specific run index, automatically
    discovering subjects and denoised/TS runs using your find_files helpers.

    Parameters
    ----------
    derivatives_dir : path-like
        Path to derivatives directory containing denoised or timeshifted runs.
    roi_mask_img : path-like
        3D ROI mask in same MNI space.
    run_idx : int
        Which run index to use across subjects.
    subjects : optional list[str]
        If None → auto-detect via iter_subjects.
    denoise_folder : str
        Subfolder under derivatives_dir. Usually "" for direct layout.
    space : str
        Functional space, e.g. MNI152NLin2009cAsym.
    desc_keywords : tuple[str]
        Strings that must appear in the filename (denoised, timeshift, etc.)
    mask_threshold : float
        Binary threshold for ROI mask.

    Returns
    -------
    isc_img : Nifti1Image (3D)
        Voxelwise ISC map (values only inside ROI).
    isc_vec : np.ndarray (V,)
        ISC values per voxel in ROI.
    func_imgs : dict
        {subject: path_to_used_4D_file} used for ISC.
    """

    derivatives_dir = Path(derivatives_dir).resolve()

    # ------------------------------
    # 1. Auto-discover subjects
    # ------------------------------
    if subjects is None:
        subjects = iter_subjects(derivatives_dir)

    if len(subjects) < 2:
        raise ValueError("Need at least 2 subjects for ISC.")

    # ------------------------------
    # 2. Find each subject's run
    # ------------------------------
    from collections import OrderedDict

    func_imgs = OrderedDict()
    for sub in subjects:
        runs = list(
            iter_subject_denoised(
                derivatives_dir=derivatives_dir,
                sub=sub,
                space=space,
                desc_keywords=desc_keywords,
            )
        )
        if len(runs) == 0:
            print(f"⚠️ No denoised/timeshift runs for {sub}")
            continue

        if run_idx >= len(runs):
            print(f"⚠️ {sub} does not have run index {run_idx}, skipping.")
            continue

        func_imgs[sub] = runs[run_idx]

    if len(func_imgs) < 2:
        raise RuntimeError("Not enough subjects have the requested run to compute ISC.")

    print(f"✓ Found {len(func_imgs)} subjects for run {run_idx}")

    # ------------------------------
    # 3. Load ROI mask
    # ------------------------------
    if isinstance(roi_mask_img, nib.spatialimages.SpatialImage):
        mask_img = roi_mask_img
    else:
        mask_img = nib.load(str(roi_mask_img))
    mask_data = mask_img.get_fdata()
    mask_bool = mask_data > mask_threshold
    if not mask_bool.any():
        raise ValueError("ROI mask is empty.")

    # ------------------------------
    # 4. Load first func image (type check + reference grid)
    # ------------------------------
    first_sub = next(iter(func_imgs))
    first_img = nib.load(str(func_imgs[first_sub]))
    X, Y, Z, T = first_img.shape

    if mask_data.shape != (X, Y, Z):
        raise ValueError(
            f"Mask shape {mask_data.shape} != func shape {(X,Y,Z)}. "
            "Resample mask first."
        )

    # ------------------------------
    # 5. Extract voxelwise TS inside ROI
    # ------------------------------
    voxel_indices = mask_bool.reshape(-1)
    V = int(voxel_indices.sum())

    data_list = []
    for sid, path in func_imgs.items():
        img = nib.load(str(path))
        data = img.get_fdata()  # (X,Y,Z,T)
        if data.shape != (X, Y, Z, T):
            raise ValueError(f"{sid} image shape mismatch: {data.shape}")

        flat = data.reshape(-1, T)               # (XYZ, T)
        roi_ts = flat[voxel_indices, :]          # (V, T)
        data_list.append(roi_ts.T)               # (T, V)

    # ------------------------------
    # 6. Compute ISC per voxel
    # ------------------------------
    isc_vec = compute_isc(data_list)  # (V,)

    # ------------------------------
    # 7. Reconstruct a 3D volume
    # ------------------------------
    isc_vol = np.zeros((X, Y, Z), dtype=float)
    isc_vol[mask_bool] = isc_vec

    isc_img = nib.Nifti1Image(isc_vol, first_img.affine, first_img.header)

    return isc_img, isc_vec, func_imgs
