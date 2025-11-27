from __future__ import annotations
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

# Running fmriprep command template
def run_fmriprep(
    bids_root: Path | str,
    derivatives_root: Path | str,
    work_dir: Path | str,
    fs_license_file: Path | str,
    subject_label: str | Iterable[str] | None,
    template: str,       
    spaces: Iterable[str],
    nthreads: int,
    extra: str = "",
) -> None:
    
    if subject_label is None:
        labels: list[str] = []
    elif isinstance(subject_label, (str, bytes)):
        labels = [str(subject_label)]
    else:
        labels = [str(s) for s in subject_label]
    
    label_cli = "" if not labels else " ".join(labels)

    if not labels:
        work_suffix = ""
    else:
        safe_labels = [l.replace(" ", "_") for l in labels]
        work_suffix = "_".join(safe_labels)

    bids = Path(bids_root).resolve()
    out  = Path(derivatives_root).resolve()
    work = Path(work_dir).resolve() / work_suffix
    fs   = Path(fs_license_file).resolve()

    _ensure_dirs(out, work)

    cmd = template.format(
        bids=str(bids),
        out=str(out),
        work=str(work),
        fs_license=str(fs),
        label=label_cli,
        spaces=_join_spaces(spaces),
        nthreads=nthreads,
        extra=(extra or ""),
    )
    print("\n--- Running fMRIPrep ---\n", cmd, "\n")
    subprocess.run(cmd, shell=True, check=True)

# ========= Private helpers =========
def _ensure_dirs(out: Path, work: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

def _join_spaces(spaces: Iterable[str]) -> str:
    return " ".join(spaces)
