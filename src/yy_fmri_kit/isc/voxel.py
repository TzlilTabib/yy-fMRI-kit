from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import nibabel as nib

from yy_fmri_kit.static.isc.config import ISCConfig
from yy_fmri_kit.isc.compute import compute_isc


PathLike = Union[str, Path]
ImgLike = Union[PathLike, nib.Nifti1Image]

# =============================================================
# LOW LEVEL HELPERS
# =============================================================
def _load_4d(img: ImgLike) -> nib.Nifti1Image:
    if isinstance(img, nib.spatialimages.SpatialImage):
        return img
    return nib.load(str(img))


def _load_mask(mask: ImgLike) -> Tuple[np.ndarray, nib.Nifti1Image]:
    mimg = _load_4d(mask)
    m = np.asanyarray(mimg.dataobj)
    if m.ndim != 3:
        raise ValueError(f"Mask must be 3D, got shape {m.shape}")
    mask_bool = m.astype(bool)
    if mask_bool.sum() == 0:
        raise ValueError("Mask is empty (0 voxels).")
    return mask_bool, mimg


def _img_to_masked_ts(img: nib.Nifti1Image, mask_bool: np.ndarray) -> np.ndarray:
    data = np.asanyarray(img.dataobj)  # (X,Y,Z,T) typically
    if data.ndim != 4:
        raise ValueError(f"Expected 4D fMRI image, got shape {data.shape}")
    # Ensure time last: nibabel convention is (X,Y,Z,T), so OK.
    # Extract masked voxels -> (V, T) then transpose -> (T, V)
    masked = data[mask_bool]           # (V, T)
    return np.asarray(masked).T        # (T, V)


# =============================================================
# MAIN VOXELWISE FUNCTION
# =============================================================

def run_voxelwise_isc(
    subject_imgs: Dict[str, Dict[str, ImgLike]],
    config: ISCConfig,
    task: str,
    mask_img: Optional[ImgLike] = None,
    chunk_size: int = 50_000,
    fill_value: float = np.nan,
    return_vector: bool = False,
    fisher_z: bool = False,
    nan_policy: str = "omit",
    return_subjectwise: bool = False,
    return_subjectwise_img: bool = False,
) -> Union[nib.Nifti1Image, Tuple[nib.Nifti1Image, np.ndarray]]:
    """
    Compute voxelwise ISC for a given task and return a 3D ISC map in mask space.

    Parameters
    ----------
    subject_imgs : dict
        {'sub-1': {'taskA': /path/to/bold.nii.gz, ...}, ...}
    config : ISCConfig
        Used for defaults (e.g., mask path).
    task : str
        Task key to use from subject_imgs.
    mask_img : ImgLike, optional
        3D mask. If None, tries config.mask_nii.
    fill_value : float
        Value for voxels outside mask in output volume (NaN recommended).
    fisher_z : bool
        If True, Fisher-z correlations before averaging across subjects.
    nan_policy : {"omit", "propagate"}
        NaN handling passed to compute_isc.
    return_subjectwise : bool
        If True, also return isc_subjectwise (N, V).
    return_subjectwise_img : bool
        If True and return_subjectwise=True, also return a 4D NIfTI with subjectwise ISC maps.

    Returns
    -------
    isc_img : 3D NIfTI
    plus optional outputs depending on flags.
    """
    if mask_img is None:
        mask_img = getattr(config, "mask_nii", None)
    if mask_img is None:
        raise ValueError("No mask provided. Pass mask_img or add mask_nii to ISCConfig.")

    mask_bool, mask_ref_img = _load_mask(mask_img)
    V = int(mask_bool.sum())

    # Collect masked time series per subject
    data_list: List[np.ndarray] = []
    ref_shape = None
    ref_T = None

    for sub, tasks in subject_imgs.items():
        if task not in tasks:
            raise KeyError(f"Task {task} not found for subject {sub}")

        img = _load_4d(tasks[task])

        if ref_shape is None:
            ref_shape = img.shape[:3]
        else:
            if img.shape[:3] != ref_shape:
                raise ValueError(f"Grid mismatch for {sub}: {img.shape[:3]} != {ref_shape}")

        ts = _img_to_masked_ts(img, mask_bool)  # (T, V)
        if ref_T is None:
            ref_T = ts.shape[0]
        if ts.shape[1] != V:
            raise RuntimeError("Masked voxel count changed unexpectedly.")
        data_list.append(ts.astype(np.float32))

    # Compute voxelwise ISC
    if return_subjectwise:
        isc_subj, isc_vec = compute_isc(
            data_list,
            fisher_z=fisher_z,
            nan_policy=nan_policy,   # type: ignore[arg-type]
            return_subjectwise=True,
        )  # isc_subj: (N, V), isc_vec: (V,)
    else:
        isc_vec = compute_isc(
            data_list,
            fisher_z=fisher_z,
            nan_policy=nan_policy,   # type: ignore[arg-type]
            return_subjectwise=False,
        )  # (V,)
        isc_subj = None
    # Put back into 3D volume
    out = np.full(mask_bool.shape, fill_value, dtype=np.float32)
    out[mask_bool] = isc_vec.astype(np.float32)

    isc_img = nib.Nifti1Image(out, affine=mask_ref_img.affine, header=mask_ref_img.header)

    subj_img = None
    if return_subjectwise and return_subjectwise_img:
        N = isc_subj.shape[0]
        out4d = np.full(mask_bool.shape + (N,), fill_value, dtype=np.float32)  # (X,Y,Z,N)
        # fill per subject
        for i in range(N):
            out4d[..., i][mask_bool] = isc_subj[i].astype(np.float32)
        subj_img = nib.Nifti1Image(out4d, affine=mask_ref_img.affine, header=mask_ref_img.header)

    if return_subjectwise and return_subjectwise_img:
        return isc_img, isc_vec, isc_subj, subj_img
    if return_subjectwise:
        return isc_img, isc_vec, isc_subj
    return isc_img
