import numpy as np
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union
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