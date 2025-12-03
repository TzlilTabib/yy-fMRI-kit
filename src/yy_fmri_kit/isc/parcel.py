from __future__ import annotations
from typing import Dict, List

import numpy as np
import pandas as pd

from yy_fmri_kit.static.isc.config import ISCConfig
from yy_fmri_kit.isc.compute import compute_isc
from yy_fmri_kit.postproc.parcellation import _resolve_atlas_and_labels

# ================================================================
#  PARCELWISE ISC MAIN HELPER FUNCTION
# ================================================================

def run_parcelwise_isc(
    subject_data: Dict[str, Dict[str, np.ndarray]],
    config: ISCConfig,
    task: str,
    parcel_labels: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute parcelwise ISC for a given task.

    Parameters
    ----------
    subject_data : dict
        {'sub-1': {'taskA': (T×P), ...}, ...}
    config : ISCConfig
        Config (unused here except for future extensions).
    task : str
        Which task key to use from subject_data (e.g. "all" or "AntiRight").
    parcel_labels : DataFrame, optional
        Parcel metadata with at least one ID column (index or 'label').
        If None, we'll just number parcels 0..P-1.

    Returns
    -------
    df : DataFrame
        Columns: 'parcel', 'ISC', plus any label columns if provided.
    """
    # Collect time series for this task
    data_list: List[np.ndarray] = []
    for sub, tasks in subject_data.items():
        if task not in tasks:
            raise KeyError(f"Task {task} not found for subject {sub}")
        ts = tasks[task]  # (T, P)
        data_list.append(ts)

    # Compute ISC per parcel (feature)
    isc_vec = compute_isc(data_list)  # (P,)
    P = isc_vec.shape[0]

    # 3) Get labels if not provided
    if parcel_labels is None:
        atlas_nii, labels_path = _resolve_atlas_and_labels(
            atlas_nii=config.atlas_nii,
            labels_file=config.labels_file,
            tf_template=config.tf_template,
            tf_atlas=config.tf_atlas,
            tf_desc=config.tf_desc,
            tf_resolution=config.tf_resolution,
            suffix="dseg",
        )
        if labels_path is None:
            raise RuntimeError(
                "_resolve_atlas_and_labels did not return a labels_file. "
                "Pass labels_file or valid TemplateFlow params in ISCConfig."
            )
        parcel_labels = pd.read_csv(labels_path, sep="\t")

    if len(parcel_labels) != P:
        raise ValueError(
            f"parcel_labels has length {len(parcel_labels)} but ISC has {P} parcels."
        )
    
    df = parcel_labels.copy()
    df["ISC"] = isc_vec

    return df
