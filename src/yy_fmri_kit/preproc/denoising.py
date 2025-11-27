from __future__ import annotations
from typing import Optional, Iterable, List
import pandas as pd
import numpy as np
from pathlib import Path
import re
from nltools.data import Brain_Data, Design_Matrix
from nltools.stats import regress, zscore
from yy_fmri_kit.io.find_files import iter_subjects, _iter_subject_func_runs

__all__ = ["make_motion_covariates", "run_nltools_denoising"]

# --- Function to create the 24-parameter motion model ---
def make_motion_covariates(mc, tr):
    """Creates a 24-parameter motion model (6 MC + derivatives + squared terms)."""
    z_mc = zscore(mc)
    all_mc = pd.concat([z_mc, z_mc**2, z_mc.diff(), z_mc.diff()**2], axis=1)
    all_mc.fillna(value=0, inplace=True)
    return Design_Matrix(all_mc, sampling_freq=1/tr)

print("Setup complete. Defined parameters and motion model function.")

# --- Denoising function ---
# Denoising one scan -----
def run_nltools_denoising(
    func_file: str | Path,
    confounds_file: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
    tr: float = 2.0,
    fwhm: float = 4.0,
    spike_cutoff: float = 5.0,
) -> Path:
    """
    Performs nuisance regression following the Naturalistic Data Analysis approach.
    Smooth → spikes → 24p motion + CSF + poly trends → regress → save.
    - If confounds_file is None: infer from fMRIPrep filename.
    - If output_dir is None: write under <func>/../../../../derivatives/yy_fmri_kit/denoised/sub-XX/func
    """
    func_file = Path(func_file).resolve()
    confounds_file = Path(confounds_file).resolve() if confounds_file else _infer_confounds_path(func_file)
    out_dir = Path(output_dir).resolve() if output_dir else _default_output_dir(func_file)

    print(f"\n--- Denoising ---\nBOLD: {func_file.name}\nConfounds: {confounds_file.name}\nFWHM={fwhm} | TR={tr} | SpikeCutoff={spike_cutoff}")
    
    # 1) load
    data = Brain_Data(str(func_file))
    confounds_df = pd.read_csv(confounds_file, sep="\t")

    # 2) smooth first
    smoothed = data.smooth(fwhm=fwhm)

    # 3) spikes
    spike_dm = _safe_spike_dm(smoothed, spike_cutoff)

    # 4) motion + CSF
    mc = _pick_motion_columns(confounds_df)
    motion_dm = make_motion_covariates(mc, tr)
    if "csf" not in confounds_df.columns:
        raise KeyError("Confounds file is missing 'csf' column.")
    csf_dm = Design_Matrix(pd.DataFrame({"csf": confounds_df["csf"]}), sampling_freq=1 / tr)

    # 5) design matrix
    dm = pd.concat([csf_dm, motion_dm, spike_dm], axis=1)
    final_dm = Design_Matrix(dm, sampling_freq=1 / tr).add_poly(order=2, include_lower=True)

    # 6) regress
    smoothed.X = final_dm
    stats = smoothed.regress()
    clean = stats["residual"]
    clean.data = np.float32(clean.data)

    # 7) save
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = _output_path_for(func_file, out_dir)
    clean.write(str(out_path))
    print(f"✅ Saved: {out_path}")
    return out_path

# Denoising one subject -----
def run_subject_denoising(
    fmriprep_derivatives_root: Path | str,
    denoised_root: Path | str,
    sub_label: str,
    tr: float,
    fwhm: float,
    spike_cutoff: float,
) -> list[Path]:
    """
    Loop over all bold runs for a subject and denoise them.
    Outputs to: <denoised_root>/sub-<label>/func/
    """
    outputs: list[Path] = []
    runs = list(_iter_subject_func_runs(fmriprep_derivatives_root, sub_label))
    if not runs:
        print(f"⚠️ No preproc runs found for {sub_label}")
        return outputs

    subj_out = Path(denoised_root).resolve() / f"{sub_label}" / "func"
    subj_out.mkdir(parents=True, exist_ok=True)

    for func_file in runs:
        conf = _infer_confounds_path(func_file)
        out = run_nltools_denoising(
            func_file=func_file,
            confounds_file=conf,
            output_dir=subj_out,
            tr=tr,
            fwhm=fwhm,
            spike_cutoff=spike_cutoff,
        )
        outputs.append(out)
    return outputs

# Run denoising for all subjects -----
def run_group_denoising(
    fmriprep_derivatives_root: Path | str,
    denoised_root: Path | str,
    tr: float,
    fwhm: float = 4.0,
    spike_cutoff: float = 5.0,
    subjects: Optional[Iterable[str]] = None,
) -> dict[str, list[Path]]:
    """
    Run denoising for all subjects (or a subset).

    Parameters
    ----------
    fmriprep_derivatives_root : Path | str
        Path to fMRIPrep derivatives root (the folder that contains 'sub-*').
    denoised_root : Path | str
        Root where denoised outputs will be written.
        Each subject will go under: <denoised_root>/sub-XX/func/
    tr, fwhm, spike_cutoff : see `run_subject_denoising` / `run_nltools_denoising`.
    subjects : iterable of subject labels (e.g. ['sub-1', 'sub-2']) or None
        If None, subjects are discovered automatically using `iter_subjects`.

    Returns
    -------
    dict[sub_label, list[Path]]
        Mapping from subject label to list of denoised NIfTI paths.
    """
    fmriprep_derivatives_root = Path(fmriprep_derivatives_root).resolve()
    denoised_root = Path(denoised_root).resolve()

    # Use find_files.iter_subjects if no specific subject list was given
    if subjects is None:
        subjects = iter_subjects(fmriprep_derivatives_root)

    all_outputs: dict[str, list[Path]] = {}

    for sub_label in subjects:
        # Normalize labels: allow "1" as well as "sub-1"
        sub_label_str = str(sub_label)
        if not sub_label_str.startswith("sub-"):
            sub_label_str = f"sub-{sub_label_str}"

        print(f"\n===== Denoising {sub_label_str} =====")
        outs = run_subject_denoising(
            fmriprep_derivatives_root=fmriprep_derivatives_root,
            denoised_root=denoised_root,
            sub_label=sub_label_str,
            tr=tr,
            fwhm=fwhm,
            spike_cutoff=spike_cutoff,
        )
        all_outputs[sub_label_str] = outs

    return all_outputs


# ========= Private helpers =========

def _infer_confounds_path(func_file: Path) -> Path:
    """Infer *desc-confounds_timeseries.tsv beside the BOLD run."""
    name = func_file.name[:-7] if func_file.name.endswith(".nii.gz") else func_file.stem
    core = re.sub(r'(?:_space-[^_]+|_res-[^_]+|_desc-[^_]+)*_bold$', '', name)
    tsv = func_file.with_name(f"{core}_desc-confounds_timeseries.tsv")
    if tsv.exists():
        return tsv

    keys = {"sub","ses","task","run","acq","dir","echo"}
    def ents(stem): 
        return {k:v for k,v in (p.split('-',1) for p in stem.split('_') if '-' in p)}
    want = ents(core)

    for c in func_file.parent.glob("*_desc-confounds_timeseries.tsv"):
        if all(ents(c.stem).get(k) == v for k,v in want.items() if k in keys):
            return c
    raise FileNotFoundError(f"Confounds not found for {func_file}")

def _default_output_dir(func_file: Path) -> Path:
    """
    Default under derivatives: .../derivatives/yy_fmri_kit/denoised/sub-XX/func
    Assumes standard fMRIPrep layout: .../derivatives/fmriprep/sub-XX/func/FILE
    """
    sub_dir = func_file.parents[0].parent.name
    root_deriv = func_file.parents[2].parent    # ".../derivatives"
    return root_deriv / "yy_fmri_kit" / "denoised" / sub_dir / "func"

def _pick_motion_columns(confounds_df: pd.DataFrame) -> pd.DataFrame:
    candidates = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
    if not all(c in confounds_df.columns for c in candidates):
        missing = [c for c in candidates if c not in confounds_df.columns]
        raise KeyError(f"Missing motion columns in confounds: {missing}")
    return confounds_df[candidates]

def _safe_spike_dm(data: Brain_Data, cutoff: float) -> pd.DataFrame:
    """
    Use Brain_Data.find_spikes if available; return a plain DataFrame without 'TR'.
    """
    dm = data.find_spikes(global_spike_cutoff=cutoff, diff_spike_cutoff=cutoff)
    return dm.drop(columns=[c for c in dm.columns if c.upper() == "TR"], errors="ignore")

def _output_path_for(func_file: Path, output_dir: Path) -> Path:
    return output_dir / func_file.name.replace("_bold.nii.gz", "_desc-nltoolsClean_bold.nii.gz")
