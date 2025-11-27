from __future__ import annotations
from pathlib import Path
from tabnanny import verbose
from typing import Dict, List, Optional, Tuple
import numpy as np
from nltools.data import Brain_Data
import nibabel as nib
from nilearn.image import resample_img

try:
    from templateflow.api import get as tf_get
except ImportError as e:
    tf_get = None

from yy_fmri_kit.io.find_files import build_subject_runs_dict, iter_subjects, find_brain_mask

# === Load standard-space group mask from TemplateFlow ===
def load_MNI_mask(
    space: str = "MNI152NLin2009cAsym",
    resolution: int = 2,
    desc: str = "brain",
    suffix: str = "mask",
) -> nib.Nifti1Image:
    """
    Load a single, standard-space group brain mask for voxelwise analyses.

    Parameters
    ----------
    space : str
        TemplateFlow space name. Must match your fMRIPrep outputs
        (e.g., "MNI152NLin2009cAsym").
    resolution : int
        TemplateFlow resolution (1 or 2). Should match your BOLD `res-*`.
    desc : str
        Descriptor for the mask. Common choice is "brain".
    suffix : str
        Suffix for the file type, usually "mask".

    Returns
    -------
    mask_img : nib.Nifti1Image
        3D boolean brain mask in the requested template space.

    Notes
    -----
    - This is intended to be used as a *single mask* for all subjects and runs
      for voxelwise ISC and other group analyses.
    - Assumes your preprocessed BOLD is in the same space & resolution.
    """
    if tf_get is None:
        raise ImportError(
            "TemplateFlow is required to load the group mask. "
            "Install it with `pip install templateflow`."
        )

    mask_path = tf_get(
        space,
        resolution=resolution,
        desc=desc,
        suffix=suffix,
    )
    mask_path = Path(mask_path)

    if not mask_path.exists():
        raise FileNotFoundError(
            f"TemplateFlow mask not found for space={space}, "
            f"resolution={resolution}, desc={desc}, suffix={suffix}."
        )

    mask_img = nib.load(str(mask_path))
    data = mask_img.get_fdata()

    # Ensure boolean mask
    # (TemplateFlow masks are usually 0/1 floats; this guards against weird values)
    bool_data = (data > 0).astype(np.uint8)

    return nib.Nifti1Image(bool_data, affine=mask_img.affine, header=mask_img.header)

# Build group reliability mask from fMRIPrep outputs -----
def build_group_mask(
    fmriprep_root: Path | str,
    brain_mask_img: Path | str,
    groupmask_out_path: Path | str,
    pass_fraction: float = 0.90,
    tsnr_percentile: float = 40.0,
    fixed_thresh: Optional[float] = None,
    verbose: bool = True,
) -> Tuple[Path, Brain_Data]:
    """
    Builds a mask of voxels that have reliable signal across subjects,
    based on preprocessed (pre-denoising) fMRIPrep outputs.

    Logic:
    - For each subject:
        - Load all preprocessed func runs (space-MNI152NLin2009cAsym_res-2_desc-preproc_bold)
        - Compute tSNR per voxel for each run
        - Take voxelwise median tSNR across runs
        - Mark voxels as "reliable" if their tSNR is above a within-subject percentile
    - Group mask:
        - Keep voxels that are reliable in >= pass_fraction of subjects (default 0.90)

    Parameters
    ----------
    fmriprep_root : Path | str
        Root of fMRIPrep derivatives.
    brain_mask_img : Path | str
        3D brain mask in the same space as your func data (e.g. group-level
        MNI mask or one subject's fMRIPrep brain mask).
    out_path : Path | str
        Where to save the resulting group mask NIfTI (binary 0/1).
    pass_fraction : float, default 0.90
        Voxel must be "reliable" in at least this fraction of subjects.
    tsnr_percentile : float, default 40.0
        Within-subject tSNR percentile cutoff (ignored if fixed_thresh is set).
    fixed_thresh : float | None
        If provided, use an absolute tSNR threshold instead of percentile.
    verbose : bool
        Print progress information.

    Returns
    -------
    groupmask_out_path : Path
        Path to the saved NIfTI group mask.
    group_mask_bd : Brain_Data
        Brain_Data object representing the group mask (1 x n_vox, binary).
    """
    fmriprep_root = Path(fmriprep_root).resolve()
    out_path = Path(groupmask_out_path).resolve()

    # Auto-detect brain mask if not provided
    if brain_mask_img is None:
        brain_mask_img = find_brain_mask(fmriprep_root)
        if verbose:
            print(f"[group_mask] Auto-detected brain mask: {brain_mask_img}")

    brain_mask_bd = Brain_Data(str(brain_mask_img))

    # Find runs per subject
    subj_runs: Dict[str, List[Path]] = build_subject_runs_dict(fmriprep_root)
    n_subs_total = len(subj_runs)

    if n_subs_total == 0:
        raise RuntimeError("No subjects with functional runs found in fMRIPrep root.")

    if verbose:
        print(f"Found {len(subj_runs)} subjects with functional runs.")
    
    # --- START Memory & Resampling Fix ---
    # Use the first run of the first subject as the reference image
    first_sub = next(iter(subj_runs))
    first_run_path = str(subj_runs[first_sub][0].resolve())
    
    if verbose:
        print(f"[group_mask] Aligning mask to the reference run: {first_run_path}")

    # Load the reference image's header/affine
    ref_img = nib.load(first_run_path)
    resampled_mask_img = resample_img(
            brain_mask_bd.to_nifti(),
            target_affine=ref_img.affine,
            interpolation='nearest'
        )
        
    resampled_data = resampled_mask_img.get_fdata()
    resampled_data = np.round(resampled_data).astype(np.int8) 

    # Create a new NIfTI image with the corrected data and affine
    aligned_mask_img_final = nib.Nifti1Image(
        resampled_data,
        resampled_mask_img.affine,
        resampled_mask_img.header
    )

    # Create the final aligned Brain_Data mask object for use throughout
    # Brain_Data will use the entire aligned_mask_img_final as its mask.
    # We pass a flattened array of 1s as the dummy data, which is standard for nltools masks.
    n_voxels_in_mask = np.sum(resampled_data > 0)
    dummy_data = np.ones(n_voxels_in_mask) # 1D array of 1s
    aligned_brain_mask_bd = Brain_Data(dummy_data, mask=aligned_mask_img_final)

    if verbose:
        print(f"Mask successfully resampled and aligned. Warning resolved.")
    # --- END ADDED ---

    subj_masks = []
    for sub, runs in subj_runs.items():
        if verbose:
            print(f"  Computing tSNR for {sub} with {len(runs)} runs...")
        subj_tsnr = _subject_tsnr_median(runs, aligned_brain_mask_bd)
        rel_mask = _subject_reliability_mask(
            subj_tsnr,
            tsnr_percentile=tsnr_percentile,
            fixed_thresh=fixed_thresh,
        )
        subj_masks.append(rel_mask.astype(int))

    subj_masks_arr = np.vstack(subj_masks)  # (n_subj, n_vox)
    n_subs = subj_masks_arr.shape[0]
    need = int(np.ceil(pass_fraction * n_subs))

    if verbose:
        print(f"Requiring voxel to be reliable in >= {need}/{n_subs} subjects "
              f"({pass_fraction:.0%}).")

    passes = np.sum(subj_masks_arr, axis=0)  # (n_vox,)
    group_mask_vox = (passes >= need).astype(int)[np.newaxis, :]  # (1, n_vox)
    group_mask_bd = Brain_Data(group_mask_vox, mask=aligned_brain_mask_bd.mask)


    group_mask_bd = Brain_Data(group_mask_vox, mask=aligned_brain_mask_bd.mask)
    group_mask_bd.write_nifti(str(out_path))

    if verbose:
        kept = int(group_mask_vox.sum())
        total = group_mask_vox.size
        print(f"Group mask saved to {out_path}")
        print(f"Voxels kept: {kept} / {total} "
              f"({kept / total:.1%})")

    return out_path, group_mask_bd

# Apply group mask to one denoised subject -----
def apply_group_mask_to_subject(
    subject: str,
    denoised_root: Path | str,
    group_mask: Brain_Data,
    output_root: Path | str,
    zscore: bool = True,
) -> Path:
    """
    Apply the group reliability mask to a single denoised 4D file.
    Optionally z-score afterward.
    """

    denoised_root = Path(denoised_root)
    output_root = Path(output_root)

    # Input file: /denoised_root/sub-01/sub-01_denoised.nii.gz
    in_file = denoised_root / subject / f"{subject}_denoised.nii.gz"
    if not in_file.exists():
        raise FileNotFoundError(f"No denoised file for {subject}: {in_file}")

    # Output dir: /output_root/sub-01/sub-01_masked.nii.gz
    out_dir = output_root / subject
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{subject}_masked.nii.gz"

    # Load data
    bd = Brain_Data(str(in_file))

    # Apply mask
    bd_masked = bd.apply_mask(group_mask)

    # Z-score
    if zscore:
        bd_masked = bd_masked.standardize()

    # Re-expand to full 3D grid & save
    bd_unmasked = bd_masked.unmask()
    bd_unmasked.write_nifti(str(out_file))

    return out_file

# Apply group mask to all denoised subject -----
def apply_group_mask_for_all(
    fmriprep_root: Path | str,
    denoised_root: Path | str,
    mask_path: Path | str,
    masked_derivatives_root: Path | str,
    zscore: bool = True,
):
    """
    Apply group mask to all subjects' denoised files.
    """

    denoised_root = Path(denoised_root)
    output_root = Path(masked_derivatives_root)
    output_root.mkdir(parents=True, exist_ok=True)

    group_mask = Brain_Data(str(mask_path))

    for subj in iter_subjects(fmriprep_root):
        print(f"Applying group mask to {subj}...")

        try:
            out_file = apply_group_mask_to_subject(
                subject=subj,
                denoised_root=denoised_root,
                group_mask=group_mask,
                output_root=output_root,
                zscore=zscore,
            )
        except FileNotFoundError:
            print(f"⚠️ Skipping {subj} — no denoised file found.")
            continue

        print(f"✓ Saved: {out_file}")

    print("\nAll done.")


# === Private helpers ===
def _subject_tsnr_median(runs: List[Path], brain_mask: Brain_Data) -> Brain_Data:
    """
    Returns: Brain_Data with shape (1, n_vox)
    """
    tsnr_maps = []

    for run in runs:
        run_bd = Brain_Data(str(run), mask=brain_mask.mask)

        mean_bd = run_bd.mean()
        std_bd = run_bd.std()
        tsnr_bd = mean_bd / (std_bd + 1e-8)  # Brain_Data (1, n_vox)

        tsnr_maps.append(tsnr_bd.data[0])  # (n_vox,)

    tsnr_array = np.vstack(tsnr_maps)             # (n_runs, n_vox)
    tsnr_median = np.median(tsnr_array, axis=0)   # (n_vox,)

    # Wrap back into Brain_Data (1, n_vox)
    tsnr_median_bd = Brain_Data(tsnr_median[np.newaxis, :], mask=brain_mask.mask)
    return tsnr_median_bd


def _subject_reliability_mask(
    subj_tsnr: Brain_Data,
    tsnr_percentile: float = 40.0,
    fixed_thresh: Optional[float] = None,
) -> np.ndarray:
    """
    Given a subject's median tSNR (Brain_Data, 1 x n_vox), return a 1D boolean mask
    marking which voxels are "reliable" for that subject.

    If fixed_thresh is provided: use subj_tsnr > fixed_thresh.
    Else: threshold by within-subject percentile (e.g. 40th).
    """
    ts = subj_tsnr.data.ravel()  # (n_vox,)

    if fixed_thresh is not None:
        thr = float(fixed_thresh)
    else:
        thr = np.percentile(ts[~np.isnan(ts)], tsnr_percentile)

    reliable = np.isfinite(ts) & (ts > thr)
    return reliable


