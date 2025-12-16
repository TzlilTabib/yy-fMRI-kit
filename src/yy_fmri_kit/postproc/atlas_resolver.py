import numpy as np
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union
import nibabel as nib
import re
import pandas as pd

try:
    from templateflow import api as tf_api
except Exception:
    tf_api = None

# ================================================================
# LOW LEVEL HELPERS
# ================================================================

def _as_path_one(p: Union[str, Path, Sequence[Union[str, Path]]]) -> Path:
    # TemplateFlow may return a single path or a list; take the first if list-like
    if isinstance(p, (list, tuple)):
        return Path(str(p[0]))
    return Path(str(p))

# ================================================================
# ATLAS RESOLVER USING TEMPLATEFLOW - FOR PARCELLATION
# ================================================================

def resolve_atlas_and_labels(
    atlas_nii: Optional[Path],
    labels_file: Optional[Path],
    tf_template: Optional[str],
    tf_atlas: Optional[str],
    tf_desc: Optional[str],
    tf_resolution: Optional[int],
    suffix: str = "dseg",
) -> Tuple[Path, Optional[Path]]:
    """
    If atlas_nii/labels_file are provided, use them.
    Otherwise, if TF params are provided, fetch via TemplateFlow.
    """
    if atlas_nii is not None and Path(atlas_nii).exists():
        return Path(atlas_nii), (Path(labels_file) if labels_file else None)

    # Use TemplateFlow if requested
    if tf_template and tf_atlas and tf_desc:
        if tf_api is None:
            raise RuntimeError("templateflow is not installed. Run: pip install templateflow")
        if tf_resolution is None:
            raise ValueError("Please set tf_resolution to 1 (res-01) or 2 (res-02).")
    
    # 1) Fetch the NIfTI atlas
        atlas_path = _as_path_one(tf_api.get(
            template=tf_template,
            atlas=tf_atlas,
            desc=tf_desc,
            suffix=suffix,
            resolution=tf_resolution,   # e.g., 1 or 2
        ))

        # 2) Fetch the matching TSV explicitly
        try:
            labels_tf = tf_api.get(
                template=tf_template,
                atlas=tf_atlas,
                desc=tf_desc,
                suffix=suffix,
                extension="tsv",
            )
            labels_path = _as_path_one(labels_tf) if labels_tf else None
        except Exception:
            labels_path = None
        
        if labels_path is None:
            # Fallback attempt: replace .nii.gz with .tsv next to the NIfTI
            tsv_guess = Path(str(atlas_path).replace(".nii.gz", ".tsv"))
            labels_path = tsv_guess if tsv_guess.exists() else None
        
        print("ATLAS:", atlas_path)
        print("LABELS:", labels_path, "(exists:", Path(labels_path).exists() if labels_path else None, ")")

        return atlas_path, labels_path

    raise FileNotFoundError(
        "No atlas found. Provide either local paths (atlas_nii/labels_file) "
        "or TemplateFlow params (tf_template, tf_atlas, tf_desc, tf_resolution)."
    )

def _read_labels_aligned(atlas_img: nib.Nifti1Image, labels_file: Optional[Path]) -> Optional[Sequence[str]]:
    """Read labels and align them to integer IDs present in atlas_img; fallback to generic names."""
    ids = np.unique(atlas_img.get_fdata().astype(int))
    ids = ids[ids > 0]

    if not labels_file or not Path(labels_file).exists():
        return [f"parcel-{i:03d}" for i in range(len(ids))]
    
    # --- AAL-style txt: lines like "1 Precentral_L" (sometimes with tabs/spaces) ---
    if labels_file.suffix.lower() == ".txt":
        mapping: dict[int, str] = {}
        with open(labels_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = re.split(r"\s+", line)
                if len(parts) < 2:
                    continue
                try:
                    idx = int(parts[0])
                except ValueError:
                    continue
                name = parts[1]
                mapping[idx] = name
        return [mapping.get(int(i), f"parcel-{int(i):03d}") for i in ids]

    # --- TSV/CSV-style labels ---
    sep = "\t" if str(labels_file).lower().endswith(".tsv") else ","
    df = pd.read_csv(labels_file, sep=sep)

    # Try to find ID and Name columns
    id_col_candidates = ["index", "label", "id", "ID", "Label"]
    name_col_candidates = ["name", "label_name", "Label", "Region", "region"]

    id_col = next((c for c in id_col_candidates if c in df.columns), None)
    name_col = next((c for c in name_col_candidates if c in df.columns), None)

    if id_col and name_col:
        mapping = dict(zip(df[id_col].astype(int), df[name_col].astype(str)))
        return [mapping.get(i, f"parcel-{i:03d}") for i in ids]

    # If no explicit ID column but counts match, take a plausible name column by order
    if name_col and len(df) == len(ids):
        return df[name_col].astype(str).tolist()

    # Final fallback
    return [f"parcel-{i:03d}" for i in range(len(ids))]
