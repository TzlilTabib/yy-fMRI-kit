from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple
import nibabel as nib
import numpy as np
import pandas as pd
from typing import Union, Sequence
import re

try:
    from nilearn.maskers import NiftiLabelsMasker
except Exception:
    from nilearn.input_data import NiftiLabelsMasker
from nilearn.image import resample_to_img
try:
    from templateflow import api as tf_api
except Exception:
    tf_api = None
from yy_fmri_kit.io.find_files import iter_subject_denoised, get_session_from_path, iter_subjects
from yy_fmri_kit.postproc.atlas_resolver import resolve_atlas_and_labels


# ==== Main function to parcellate one functional image ====
def parcellate(
    func_nii: Path,
    atlas_nii: Optional[Path] = None,
    labels_file: Optional[Path] = None,
    *,
    # TemplateFlow (optional): use these to auto-fetch if atlas_nii is not provided
    tf_template: Optional[str] = None,       # e.g., 'MNI152NLin2009cAsym'
    tf_atlas: Optional[str] = None,          # e.g., 'Schaefer2018'
    tf_desc: Optional[str] = None,           # e.g., '400Parcels7Networks'
    tf_resolution: Optional[int] = None,     # e.g., 2 (for res-02)
    standardize: str | bool = "zscore",      # 'zscore', 'psc', or False
    tr: Optional[float] = None,
    out_tsv: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Sequence[str]]:
    """
    Apply integer-labeled atlas to a 4D fMRI image and return a parcel time series.
    Returns (DataFrame time x parcels, labels).

    You can either:
      (A) pass local atlas paths: atlas_nii=Path(...), labels_file=Path(...)
      or
      (B) let TemplateFlow fetch them: set tf_template/tf_atlas/tf_desc[/tf_resolution].
    """
    func_img = nib.load(str(func_nii))

    # Resolve atlas paths (local or TemplateFlow)
    atlas_path, labels_path = resolve_atlas_and_labels(
        atlas_nii=atlas_nii,
        labels_file=labels_file,
        tf_template=tf_template,
        tf_atlas=tf_atlas,
        tf_desc=tf_desc,
        tf_resolution=tf_resolution,
        suffix="dseg",
    )
    atlas_img = nib.load(str(atlas_path))

    # Resample atlas to functional grid if shapes/affines differ
    if atlas_img.shape[:3] != func_img.shape[:3] or not np.allclose(atlas_img.affine, func_img.affine):
        atlas_img = resample_to_img(atlas_img, func_img, interpolation="nearest")

    # Infer TR if not provided
    if tr is None and len(func_img.header.get_zooms()) > 3:
        tr = float(func_img.header.get_zooms()[3])

    # Masker (standardize manually for clarity)
    masker = NiftiLabelsMasker(
        labels_img=atlas_img,
        standardize=False,
        detrend=False,
        low_pass=None,
        high_pass=None,
        t_r=tr,
        strategy="mean",
    )
    data = masker.fit_transform(func_img)  # (time, parcels)

    # Optional standardization
    if standardize == "zscore":
        data = (data - data.mean(0, keepdims=True)) / (data.std(0, ddof=1, keepdims=True) + 1e-8)
    elif standardize == "psc":
        mean_ = data.mean(0, keepdims=True) + 1e-8
        data = (data - mean_) / mean_ * 100.0

    # Labels aligned to atlas IDs
    labels = _read_labels_aligned(atlas_img, labels_path)
    ts = pd.DataFrame(data, columns=labels)

    if out_tsv:
        Path(out_tsv).parent.mkdir(parents=True, exist_ok=True)
        ts.to_csv(out_tsv, sep="\t", index=False)

    return ts, labels

# ==== Main function to parcellate all runs of one subject ====
def parcellate_subject(
    derivatives_dir: Path,
    sub: str,
    *,
    # choose either local atlas paths...
    atlas_nii: Optional[Path] = None,
    labels_file: Optional[Path] = None,
    # ...or TemplateFlow params:
    tf_template: Optional[str] = None,     # e.g., "MNI152NLin2009cAsym"
    tf_atlas: Optional[str] = None,        # e.g., "Schaefer2018"
    tf_desc: Optional[str] = None,         # e.g., "400Parcels7Networks"
    tf_resolution: Optional[int] = None,   # 1 or 2
    space: str = "MNI152NLin2009cAsym",
    desc: str = "denoised",
    standardize: str | bool = "zscore",
    out_root: Optional[Path] = None,       # default: derivatives/parcellation
) -> Dict[Path, pd.DataFrame]:
    """
    Auto-discovers all runs of one subject, parcellates each, and saves TSVs under:
      derivatives/parcellation/sub-XX[/ses-YY]/
    Returns: {func_path -> DataFrame}
    """
    funcs = iter_subject_denoised(derivatives_dir, sub, space=space)
    if not funcs:
        raise FileNotFoundError(f"No runs found for {sub} in {derivatives_dir} (space={space}, desc={desc}).")

    # Resolve atlas+labels once (works with local paths or TemplateFlow)
    atlas_path, labels_path = resolve_atlas_and_labels(
        atlas_nii=atlas_nii,
        labels_file=labels_file,
        tf_template=tf_template,
        tf_atlas=tf_atlas,
        tf_desc=tf_desc,
        tf_resolution=tf_resolution,
        suffix="dseg",
    )

    # Where to save
    out_root = Path(out_root) if out_root else Path(derivatives_dir) / "parcellation"
    results: Dict[Path, pd.DataFrame] = {}

    for f in funcs:
        # put outputs under sub-XX[/ses-YY]
        sub_id = sub
        ses = get_session_from_path(f)
        out_dir = out_root / sub_id / (ses if ses else "")
        out_dir.mkdir(parents=True, exist_ok=True)

        # name like original stem + atlas tag
        atlas_tag = Path(atlas_path).name.split("_atlas-")[-1].split("_")[0]  # e.g., "Schaefer2018"
        out_tsv = out_dir / f"{f.stem}_atlas-{atlas_tag}_timeseries.tsv"

        ts, _ = parcellate(
            func_nii=f,
            atlas_nii=atlas_path,
            labels_file=labels_path,
            standardize=standardize,
            out_tsv=out_tsv,
        )
        results[f] = ts

    return results

# ==== Main function to parcellate all subjects ====
def parcellate_group(
    denoised_root: Path | str,
    *,
    subjects: Optional[Iterable[str]] = None,
    atlas_nii: Optional[Path] = None,
    labels_file: Optional[Path] = None,
    tf_template: Optional[str] = None,
    tf_atlas: Optional[str] = None,
    tf_desc: Optional[str] = None,
    tf_resolution: Optional[int] = None,
    space: str = "MNI152NLin2009cAsym",
    desc: str = "denoised",
    standardize: str | bool = "zscore",
    out_root: Optional[Path] = None,
) -> Dict[str, Dict[Path, pd.DataFrame]]:
    denoised_root = Path(denoised_root).resolve()

    # auto-discover subjects from the denoised folder
    if subjects is None:
        subjects = iter_subjects(denoised_root)

    summary = {}
    print("\n=== Running group-level parcellation ===\n")

    for sub in subjects:
        sub_label = str(sub)
        if not sub_label.startswith("sub-"):
            sub_label = f"sub-{sub_label}"

        print(f"\n===== Parcellating {sub_label} =====")
        try:
            results = parcellate_subject(
                derivatives_dir=denoised_root,
                sub=sub_label,
                atlas_nii=atlas_nii,
                labels_file=labels_file,
                tf_template=tf_template,
                tf_atlas=tf_atlas,
                tf_desc=tf_desc,
                tf_resolution=tf_resolution,
                space=space,
                desc=desc,
                standardize=standardize,
                out_root=out_root,
            )
            n_runs = len(results)
            out_dir = (out_root or (denoised_root / "parcellation")) / sub_label

            print(f"   ✓ Success | {n_runs} run(s) saved →  {out_dir}")
            summary[sub_label] = {
                "n_runs": n_runs,
                "out_dir": str(out_dir),
            }

        except Exception as e:
            print(f"   ✗ Failed: {e}")
            summary[sub_label] = {"error": str(e)}

    print("\n=== Parcellation complete ===\n")
    return summary


# ==== Private helpers ====
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

# ==== Combine Schaefer + Tian parcellations into one NIfTI ====
# TODO: still in development/testing
def build_schaefer400_tianS3_combined_nifti(
    *,
    out_dir: Path,
    tian_s3_dseg_nii: Path,
    tian_s3_labels_file: Path | None = None,
    tf_template: str = "MNI152NLin2009cAsym",
    tf_atlas: str = "Schaefer2018",
    tf_desc: str = "400Parcels7Networks",
    tf_resolution: int | None = 2,
    schaefer_offset: int = 400,
) -> tuple[Path, Path]:
    """
    Returns:
      combined_atlas_nii, combined_labels_tsv
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) fetch Schaefer NIfTI dseg + labels via your resolver
    sch_nii, sch_lbl = resolve_atlas_and_labels(
        atlas_nii=None,
        labels_file=None,
        tf_template=tf_template,
        tf_atlas=tf_atlas,
        tf_desc=tf_desc,
        tf_resolution=tf_resolution,
        suffix="dseg",  # IMPORTANT: volumetric label image
    )
    sch_img = nib.load(str(sch_nii))
    ti_img = nib.load(str(tian_s3_dseg_nii))

    # 2) put Tian on Schaefer grid
    if ti_img.shape[:3] != sch_img.shape[:3] or not np.allclose(ti_img.affine, sch_img.affine):
        ti_img = resample_to_img(ti_img, sch_img, interpolation="nearest")

    sch = sch_img.get_fdata().astype(int)
    ti = ti_img.get_fdata().astype(int)

    # 3) merge: keep cortex labels, fill empty voxels with subcortex (+offset)
    combined = sch.copy()
    fill = (combined == 0) & (ti > 0)
    combined[fill] = ti[fill] + schaefer_offset

    combined_nii = out_dir / f"{tf_atlas}_tf_{tf_resolution}mm_{tf_desc}_plus_TianS3.dseg.nii.gz"
    nib.save(nib.Nifti1Image(combined.astype(np.int32), sch_img.affine, sch_img.header), str(combined_nii))

    # 4) build combined labels TSV
    sch_df = _load_labels_any(sch_lbl)
    # enforce Schaefer IDs 1..400 if not present
    if "id" not in sch_df.columns:
        sch_df["id"] = np.arange(1, len(sch_df) + 1)
    sch_df = sch_df[["id", "name"]].copy()

    if tian_s3_labels_file is not None and Path(tian_s3_labels_file).exists():
        ti_df = _load_labels_any(tian_s3_labels_file)
        if "id" not in ti_df.columns:
            ti_df["id"] = np.arange(1, len(ti_df) + 1)
        ti_df = ti_df[["id", "name"]].copy()
        ti_df["id"] = ti_df["id"].astype(int) + schaefer_offset
    else:
        # fallback: make generic names from what exists in Tian volume
        ti_ids = np.unique(ti[ti > 0])
        ti_df = pd.DataFrame({
            "id": (ti_ids + schaefer_offset).astype(int),
            "name": [f"TianS3-{int(i)}" for i in ti_ids],
        })

    combined_labels = pd.concat([sch_df, ti_df], ignore_index=True)
    combined_tsv = out_dir / "Schaefer2018_400Parcels7Networks_plus_TianS3_labels.tsv"
    combined_labels.to_csv(combined_tsv, sep="\t", index=False)

    return combined_nii, combined_tsv


def _load_labels_any(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    # try tsv/csv first
    if path.suffix.lower() == ".tsv":
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path)
    # normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    # common cases
    rename = {}
    if "index" in df.columns and "id" not in df.columns:
        rename["index"] = "id"
    if "label" in df.columns and "name" not in df.columns:
        rename["label"] = "name"
    if "region" in df.columns and "name" not in df.columns:
        rename["region"] = "name"
    df = df.rename(columns=rename)
    if "name" not in df.columns:
        # try the first non-id column as name
        non_id = [c for c in df.columns if c != "id"]
        if non_id:
            df = df.rename(columns={non_id[0]: "name"})
    return df



