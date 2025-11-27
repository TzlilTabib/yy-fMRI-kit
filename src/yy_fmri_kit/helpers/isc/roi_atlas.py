# yy_fmri_kit/isc/roi_atlas.py

from __future__ import annotations
from pathlib import Path
from typing import Iterable, List

import nibabel as nib
import numpy as np
import pandas as pd

try:
    from templateflow.api import get as tf_get
except ImportError as e:
    tf_get = None


def _load_hcpmmp1_atlas(
    space: str = "MNI152NLin2009cAsym",
    resolution: int = 2,
):
    """
    Load the HCP-MMP1 parcellation and its labels table from TemplateFlow.
    """
    if tf_get is None:
        raise ImportError(
            "TemplateFlow is required but not installed. "
            "Install with `pip install templateflow`."
        )

    atlas_nii = tf_get(
        space,
        atlas="HCPMMP1",
        desc="group",
        resolution=resolution,
        suffix="dseg",
    )
    labels_tsv = tf_get(
        space,
        atlas="HCPMMP1",
        desc="group",
        resolution=resolution,
        suffix="dseg",
        extension="tsv",
    )

    atlas_img = nib.load(str(atlas_nii))
    labels_df = pd.read_csv(labels_tsv, sep="\t")

    return atlas_img, labels_df


def _find_label_column(labels_df: pd.DataFrame) -> str:
    """
    Try to find the column that holds parcel indices.
    Common options: 'index', 'id', 'label'.
    """
    for cand in ["index", "id", "label"]:
        if cand in labels_df.columns:
            return cand
    raise ValueError(
        f"Could not find a label/index column in labels TSV. "
        f"Available columns: {list(labels_df.columns)}"
    )


def _find_name_column(labels_df: pd.DataFrame) -> str:
    """
    Try to find the column that holds parcel names.
    Common options: 'name', 'label_name', 'region'.
    """
    for cand in ["name", "label_name", "region"]:
        if cand in labels_df.columns:
            return cand
    raise ValueError(
        f"Could not find a name column in labels TSV. "
        f"Available columns: {list(labels_df.columns)}"
    )


def build_A1_mask(
    space: str = "MNI152NLin2009cAsym",
    resolution: int = 2,
    *,
    include_belt: bool = True,
    save_to: Path | None = None,
):
    """
    Build a binary ROI mask for primary auditory cortex (A1) based on
    the HCP-MMP1 atlas in TemplateFlow.

    Strategy
    --------
    - Load HCP-MMP1 group dseg and labels.
    - Select labels whose names contain 'A1'.
    - Optionally include auditory belt areas (names containing 'Belt').
    - Create a union mask over those parcels.

    Parameters
    ----------
    space : str
        TemplateFlow space. Should match your fMRIPrep space.
    resolution : int
        TemplateFlow resolution. Should match your BOLD res (e.g., 2).
    include_belt : bool
        If True, also include belt areas (LBelt, MBelt, PBelt etc.)
        for a slightly broader auditory ROI.
    save_to : Path or None
        If provided, save the mask NIfTI to this path.

    Returns
    -------
    roi_img : nib.Nifti1Image
        3D binary mask (0/1) in the same space as your BOLD.
    selected_labels : List[str]
        List of label names that were included in the ROI.
    """
    atlas_img, labels_df = _load_hcpmmp1_atlas(space=space, resolution=resolution)
    atlas_data = atlas_img.get_fdata().astype(int)

    label_col = _find_label_column(labels_df)
    name_col = _find_name_column(labels_df)

    # Normalize names to strings
    labels_df[name_col] = labels_df[name_col].astype(str)

    # 1) always include primary A1 areas
    mask_A1 = labels_df[name_col].str.contains("A1", case=False, na=False)

    # 2) optionally include belt areas (LBelt, MBelt, PBelt, etc.)
    if include_belt:
        mask_belt = labels_df[name_col].str.contains("Belt", case=False, na=False)
        mask_all = mask_A1 | mask_belt
    else:
        mask_all = mask_A1

    selected = labels_df.loc[mask_all]

    if selected.empty:
        raise RuntimeError(
            "No HCP-MMP1 labels matching 'A1' (and 'Belt' if requested) were found. "
            "Open the TSV file from TemplateFlow and inspect the label names."
        )

    label_ids: List[int] = selected[label_col].astype(int).tolist()
    selected_names: List[str] = selected[name_col].tolist()

    # Build binary mask: union over selected labels
    roi_mask = np.isin(atlas_data, label_ids).astype(np.uint8)

    roi_img = nib.Nifti1Image(
        roi_mask,
        affine=atlas_img.affine,
        header=atlas_img.header,
    )

    if save_to is not None:
        save_to = Path(save_to).resolve()
        save_to.parent.mkdir(parents=True, exist_ok=True)
        nib.save(roi_img, str(save_to))

    return roi_img, selected_names
