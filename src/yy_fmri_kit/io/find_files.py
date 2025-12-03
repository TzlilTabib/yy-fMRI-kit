from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union
import re
PathLike = Union[str, Path]


# ==================================================================
# Find fMRIPrep preprocessed subjects, functional runs
# ==================================================================

def iter_subjects(fmriprep_root: Path | str) -> list[str]:
    """
    Return a sorted list of all subject IDs in the fMRIPrep folder.
    Example output: ["sub-01", "sub-02", "sub-03"]
    """
    fmriprep_root = Path(fmriprep_root)

    subjects = [
        d.name
        for d in fmriprep_root.glob("sub-*")
        if d.is_dir()
    ]

    return sorted(subjects)

def _iter_subject_func_runs(fmriprep_root: Path | str, sub_label: str) -> Iterable[Path]:
    """
    Iterate through all functional preprocessed runs for a subject.
        Handles a single 'ses-*' directory if present.
        Example structure:
        ----------
        fmriprep/
          sub-1/
            ses-202505251228/
              func/
                sub-1_ses-202505251228_task-rest_desc-preproc_bold.nii.gz
    """
    fmriprep_root = Path(fmriprep_root).resolve()
    sub_dir = fmriprep_root / sub_label
    if not sub_dir.exists():
        raise FileNotFoundError(f"Subject directory not found: {sub_dir}")
    
    ses_dirs = sorted(sub_dir.glob("ses-*"))
    func_dir = (ses_dirs[0] / "func") if ses_dirs else (sub_dir / "func")

    if not func_dir.exists():
        return []

    pattern = f"{sub_label}_*_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
    return sorted(func_dir.glob(pattern))


def build_subject_runs_dict(fmriprep_root: Path | str) -> dict[str, list[Path]]:
    """
    Automatically find all subjects in the fMRIPrep folder and collect their runs.
    Returns a dict:
        {
          "sub-01": [run1.nii.gz, run2.nii.gz, ...],
          "sub-02": [...],
          ...
        }
    """
    fmriprep_root = Path(fmriprep_root).resolve()
    subject_runs: Dict[str, List[Path]] = {}
    for sub in iter_subjects(fmriprep_root):
        runs = list(_iter_subject_func_runs(fmriprep_root, sub))
        if len(runs) == 0:
            print(f"⚠️ Warning: No functional runs found for {sub}")
            continue
        subject_runs[sub] = runs

    return subject_runs

# ---------- Find brain mask (first subject)----------
def find_brain_mask(fmriprep_root: Path | str, sub: str = None) -> Path:
    """
    Automatically locate a brain mask in MNI space.
    Default: use the first subject.
    """
    fmriprep_root = Path(fmriprep_root).resolve()

    if sub is None:
        subjects = iter_subjects(fmriprep_root)
        if len(subjects) == 0:
            raise RuntimeError("No subjects found in fMRIPrep folder.")
        sub = subjects[0]

    if isinstance(sub, Path):
        sub = sub.name
    sub = str(sub)
    if not sub.startswith("sub-"):
        sub = f"sub-{sub}"

    sub_dir = fmriprep_root / sub
    if not sub_dir.exists():
        raise FileNotFoundError(f"Subject directory not found: {sub_dir}")
    print(f"[find_brain_mask] Searching under: {sub_dir}")
    candidates = sorted(
        sub_dir.rglob(
            f"func/{sub}_*_space-MNI152NLin2009cAsym_*desc-brain_mask.nii.gz"
        )
    )
    print("[find_brain_mask] Candidates:")
    for c in candidates:
        print("  ", c)

    if not candidates:
        raise FileNotFoundError(
            f"No MNI func brain mask found for {sub} under {sub_dir}.\n"
            "Make sure fMRIPrep was run with space-MNI152NLin2009cAsym."
        )

    mask_file = candidates[0]
    print(f"[find_brain_mask] Using mask: {mask_file}")
    return mask_file

# ---------- Find denoised funcs (first subject)----------
def iter_subject_denoised(
    derivatives_dir: Path,
    sub: str,
    *,
    space: str = "MNI152NLin2009cAsym",
    desc_keywords: Sequence[str] = ("denoised", "clean", "nltoolsClean", "preproc", "timeshift"),
) -> Sequence[Path]:
    """
    Find all 4D denoised runs for one subject.
    Searches inside derivatives/<denoise_folder>/sub-XX/func/.
    Skips *_boldref.nii.gz reference images.
    """
    root = Path(derivatives_dir) / sub / "func"

    if not root.exists():
        raise FileNotFoundError(f"Cannot find subject func folder: {root}")

    patt = f"*space-{space}_*.nii.gz"
    all_func = list(root.glob(patt))

    good = []
    for f in all_func:
        name = f.name.lower()
        if "boldref" in name:
            continue
        if "bold" not in name:        # keep only BOLD, not masks etc.
            continue
        if any(k.lower() in name for k in desc_keywords):
            good.append(f)

    return sorted(good)

def build_denoised_runs_dict(
    derivatives_dir: PathLike,
    *,
    denoise_folder: str = "",  # if your denoised_dir already points at /denoised, leave empty
    space: str = "MNI152NLin2009cAsym",
    desc_keywords: Sequence[str] = ("denoised", "clean", "nltoolsClean", "preproc"),
    subjects: Optional[Sequence[str]] = None,
) -> Dict[str, List[Path]]:
    """
    Build a dict {sub: [run1, run2, ...]} of denoised 4D runs for each subject.

    This wraps your existing iter_subjects + iter_subject_denoised.

    Parameters
    ----------
    derivatives_dir : path-like
        Root derivatives folder that contains the denoised data.
        If denoise_folder is non-empty, we look under derivatives_dir / denoise_folder.
    denoise_folder : str
        Optional subfolder under derivatives_dir (e.g., 'denoised').
        If empty, derivatives_dir is used as-is.
    space, desc_keywords, suffix : see iter_subject_denoised docstring.
    subjects : optional sequence of subject IDs (e.g. ["sub-01", "sub-02"]).
        If None, subjects are inferred via iter_subjects().

    Returns
    -------
    subject_runs : dict
        {sub: [Path, Path, ...]} sorted paths per subject.
    """
    derivatives_dir = Path(derivatives_dir).resolve()
    if denoise_folder:
        root = derivatives_dir / denoise_folder
    else:
        root = derivatives_dir

    if subjects is None:
        subjects = iter_subjects(root)

    subject_runs: Dict[str, List[Path]] = {}
    for sub in subjects:
        try:
            runs = list(
                iter_subject_denoised(
                    derivatives_dir=root,
                    sub=sub,
                    space=space,
                    desc_keywords=desc_keywords
                )
            )
        except FileNotFoundError:
            print(f"⚠️ No func folder found for {sub} in {root}")
            continue

        if len(runs) == 0:
            print(f"⚠️ No denoised runs found for {sub}")
            continue

        subject_runs[sub] = sorted(runs)

    return subject_runs

# ==================================================================
# Helpers to extract 'sub-*' and 'ses-*' from paths
# ==================================================================

_SES_RE = re.compile(r"^ses-[^/]+$")
_SUB_RE = re.compile(r"^sub-[^/]+$")

def get_session_from_path(p: Path) -> Optional[str]:
    """
    Walk up the path and return the first 'ses-*' directory name, if any.

    Examples
    --------
    /.../denoised/sub-1/ses-202505251228/func/file.nii.gz -> 'ses-202505251228'
    /.../denoised/sub-1/func/file.nii.gz                   -> None
    """
    for parent in p.parents:
        name = parent.name
        if _SES_RE.match(name):
            return name
    return None


def get_sub_ses_from_path(p: Path) -> tuple[Optional[str], Optional[str]]:
    """
    Infer 'sub-*' and 'ses-*' (if any) from the directory components of a path.

    Returns
    -------
    (sub, ses)
        sub: e.g. 'sub-1' or None
        ses: e.g. 'ses-202505251228' or None
    """
    sub = None
    ses = None
    for parent in p.parents:
        name = parent.name
        if _SUB_RE.match(name) and sub is None:
            sub = name
        elif _SES_RE.match(name) and ses is None:
            ses = name
    return sub, ses
