# yy_fmri_kit/isc/roi.py

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional

import nibabel as nib
import numpy as np

from yy_fmri_kit.static.isc.config import ISCConfig
from yy_fmri_kit.helpers.isc.compute import compute_isc


ROI_PATHS: Dict[str, Path] = {
    # Example:
    # "A1": Path("/absolute/path/to/A1_space-MNI152NLin2009cAsym_res-2.nii.gz"),
}


def register_roi(name: str, path: Path) -> None:
    """
    Optional helper to register ROI paths programmatically.
    """
    ROI_PATHS[name] = Path(path).resolve()


def load_roi_mask(
    name: str,
    *,
    path: Optional[Path] = None,
) -> nib.Nifti1Image:
    """
    Load a single ROI mask as a NIfTI image.

    Parameters
    ----------
    name : str
        Logical name of the ROI, e.g. "A1".
    path : Path, optional
        Override path. If provided, `name` is only used for logging.

    Returns
    -------
    roi_img : nib.Nifti1Image
        3D binary mask in the same space as your functional data.
    """
    if path is not None:
        roi_path = Path(path).resolve()
    else:
        if name not in ROI_PATHS:
            raise KeyError(
                f"ROI '{name}' not found in ROI_PATHS and no explicit path given. "
                f"Add it to ROI_PATHS or call `register_roi`."
            )
        roi_path = ROI_PATHS[name].resolve()

    if not roi_path.exists():
        raise FileNotFoundError(f"ROI file not found: {roi_path}")

    roi_img = nib.load(str(roi_path))
    return roi_img


def run_roi_isc(
    roi_data: Dict[str, np.ndarray],
    *,
    config: Optional[ISCConfig] = None,
) -> float:
    """
    Compute ISC for a single ROI (e.g., A1) across subjects.

    Parameters
    ----------
    roi_data : dict
        Mapping: subject ID -> (T, 1) array.
    config : ISCConfig, optional
        Currently unused but kept for future extensions.

    Returns
    -------
    isc_value : float
        Mean ISC for this ROI across all subjects.
    """
    # consistent ordering of subjects (optional but nice)
    subj_ids = sorted(roi_data.keys())
    data_list = [roi_data[sub] for sub in subj_ids]  # list of (T, 1)

    isc_vec = compute_isc(data_list)  # (1,)
    isc_value = float(isc_vec[0])
    return isc_value

