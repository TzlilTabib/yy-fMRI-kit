from __future__ import annotations
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from yy_fmri_kit.static.isc.config import ISCConfig
from yy_fmri_kit.helpers.isc.compute import compute_isc
from yy_fmri_kit.helpers.preproc.parcellation import _resolve_atlas_and_labels

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


def run_roi_isc_from_parcels(
    subject_data: Dict[str, Dict[str, np.ndarray]],
    config: ISCConfig,
    task: str,
    roi_parcel_indices: List[int],
) -> float:
    """
    Compute ISC for an ROI defined as a subset of parcels.

    For each subject:
        - Take (T×P) parcel TS for the given task.
        - Average across roi_parcel_indices -> (T×1) ROI TS.
    Then:
        - Run compute_isc over these ROI time series.

    Returns
    -------
    isc_value : float
        Mean ISC for this ROI across subjects.
    """
    data_list: List[np.ndarray] = []

    for sub, tasks in subject_data.items():
        if task not in tasks:
            raise KeyError(f"Task {task} not found for subject {sub}")
        ts_parc = tasks[task]  # (T, P)
        # average over ROI parcels → (T,)
        roi_ts = ts_parc[:, roi_parcel_indices].mean(axis=1)
        roi_ts = roi_ts.astype(float)
        # z-score per subject
        roi_ts = (roi_ts - roi_ts.mean()) / (roi_ts.std() + 1e-8)
        roi_ts = roi_ts[:, np.newaxis]  # (T, 1)
        data_list.append(roi_ts)

    isc_vec = compute_isc(data_list)  # (1,)
    return float(isc_vec[0])

def select_parcel_indices(
    labels_df: pd.DataFrame,
    *,
    name_contains: Optional[List[str]] = None,
    name_regex: Optional[str] = None,
    network_in: Optional[List[str]] = None,
    id_in: Optional[List[int]] = None,
    index_in: Optional[List[int]] = None,
    name_col_candidates=("name", "label_name", "region"),
    network_col_candidates=("network", "net", "network_name"),
    id_col_candidates=("index", "label", "id"),
) -> List[int]:
    """
    Generic helper to pick parcel indices based on label table filters.

    Returns 0-based indices (row positions) into labels_df.
    """
    mask = pd.Series(True, index=labels_df.index)

    # name-based filters
    if name_contains is not None or name_regex is not None:
        name_col = next((c for c in name_col_candidates if c in labels_df.columns), None)
        if name_col is None:
            raise ValueError(f"No name column found in labels_df; columns={labels_df.columns.tolist()}")
        names = labels_df[name_col].astype(str)

        if name_contains is not None:
            m_any = pd.Series(False, index=labels_df.index)
            for s in name_contains:
                m_any |= names.str.contains(s, case=False, na=False)
            mask &= m_any

        if name_regex is not None:
            mask &= names.str.contains(name_regex, case=False, regex=True, na=False)

    # network-based filters
    if network_in is not None:
        net_col = next((c for c in network_col_candidates if c in labels_df.columns), None)
        if net_col is None:
            raise ValueError(f"No network column found in labels_df; columns={labels_df.columns.tolist()}")
        nets = labels_df[net_col].astype(str)
        mask &= nets.isin(network_in)

    # id-based filters (Schaefer index/label id)
    if id_in is not None:
        id_col = next((c for c in id_col_candidates if c in labels_df.columns), None)
        if id_col is None:
            raise ValueError(f"No id/index column found in labels_df; columns={labels_df.columns.tolist()}")
        mask &= labels_df[id_col].astype(int).isin(id_in)

    # explicit index list (positions)
    if index_in is not None:
        allowed = set(index_in)
        mask &= labels_df.index.to_series().isin(allowed)

    idx = list(np.where(mask.to_numpy())[0])

    if not idx:
        raise RuntimeError("select_parcel_indices returned an empty set of parcels.")
    return idx
