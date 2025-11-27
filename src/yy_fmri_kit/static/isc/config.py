from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List


@dataclass
class ISCConfig:
    """
    Minimal config for ISC analyses.
    Extend as needed later (tasks, conditions, etc.).
    """
    derivatives_dir: Path
    output_dir: Path

    subjects: Optional[List[str]] = None  # None = auto-detect
    desc: str = "denoised"               # matches your BOLD filename part
    space: str = "MNI152NLin2009cAsym"    
    tr: float = 1.0
    tr: float = 2.0                       # seconds, if you need it later

    parcellation_dir: Optional[Path] = None # if different from derivatives_dir
    atlas_nii: Optional[str] = None         # e.g. "Schaefer2018_400Parcels7Networks"
    labels_file: Optional[Path] = None      # TSV/CSV with parcel names
    tf_template: Optional[str] = None       # e.g. "MNI152NLin2009cAsym"
    tf_atlas: Optional[str] = None          # e.g. "Schaefer2018"
    tf_desc: Optional[str] = None           # e.g. "400Parcels7Networks"
    tf_resolution: Optional[int] = None     # e.g. 2

    def __post_init__(self):
        self.derivatives_dir = Path(self.derivatives_dir).resolve()
        self.output_dir = Path(self.output_dir).resolve()

        if self.parcellation_dir is None:
            self.parcellation_dir = self.derivatives_dir
        else:
            self.parcellation_dir = Path(self.parcellation_dir).resolve()

        if self.labels_file is not None:
            self.labels_file = Path(self.labels_file).resolve()

        if self.subjects is not None:
            self.subjects = list(self.subjects)