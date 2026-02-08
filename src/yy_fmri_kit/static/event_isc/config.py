"""
Configuration classes for event-locked ISC analysis.

This module contains dataclass configurations for:
- Pattern extraction from fMRI data
- Cross-subject alignment
- ISC computation and statistics
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


@dataclass
class ExtractionConfig:
    """Configuration for neural pattern extraction from fMRI data."""
    
    # Timing parameters
    tr: float = 1.5  # Repetition time in seconds
    shift_tr: int = 4  # Hemodynamic shift in TRs (typically 4-6 TRs = 6-9 seconds)
    time_unit: Literal["seconds", "tr"] = "seconds"
    
    # Column names in events file
    onset_col: Optional[str] = None  # Auto-detect if None
    duration_col: Optional[str] = None  # Auto-detect if None
    post_col: str = "stim_file"  # Column containing post IDs
    subject_col: str = "subject"
    run_col: str = "run_type"
    
    # Processing parameters
    smoothing_fwhm: Optional[float] = None  # Spatial smoothing (mm)
    detrend: bool = True  # Remove linear trends
    standardize: bool = False  # Z-score voxel time series
    
    # Searchlight parameters (if using searchlight)
    searchlight_radius: float = 3.0  # Radius in mm
    searchlight_n_jobs: int = -1  # Parallel jobs (-1 = all CPUs)
    
    # Valid run types
    valid_run_types: list[str] = field(
        default_factory=lambda: ["AntiLeft", "AntiRight", "ProLeft", "ProRight"]
    )
    
    # Output
    output_precision: Literal["float32", "float64"] = "float32"
    
    def __post_init__(self):
        """Validate configuration."""
        if self.tr <= 0:
            raise ValueError(f"TR must be positive, got {self.tr}")
        if self.shift_tr < 0:
            raise ValueError(f"shift_tr must be non-negative, got {self.shift_tr}")
        if self.searchlight_radius <= 0:
            raise ValueError(f"searchlight_radius must be positive, got {self.searchlight_radius}")


@dataclass
class AlignmentConfig:
    """Configuration for aligning patterns across subjects."""
    
    # Alignment strategy
    strategy: Literal["intersection", "union", "first_subject"] = "intersection"
    
    # Quality control
    min_subjects: int = 2  # Minimum subjects required after alignment
    allow_missing_posts: bool = False  # Allow NaN for missing posts
    
    # Validation
    check_feature_dims: bool = True  # Verify all subjects have same # features
    check_post_order: bool = True  # Warn if post orders differ dramatically
    
    def __post_init__(self):
        """Validate configuration."""
        if self.min_subjects < 2:
            raise ValueError("min_subjects must be >= 2 for ISC analysis")
        if self.strategy == "union" and not self.allow_missing_posts:
            raise ValueError("union strategy requires allow_missing_posts=True")


@dataclass  
class ISCConfig:
    """Configuration for inter-subject correlation analysis."""
    
    # ISC computation
    method: Literal["pairwise", "leave-one-out"] = "leave-one-out"
    
    # Statistical testing
    n_permutations: int = 1000
    alpha: float = 0.05
    correction: Literal["fdr_bh", "bonferroni", "none"] = "fdr_bh"
    
    # Searchlight ISC specific
    searchlight_aggregate: Literal["mean", "median"] = "mean"  # How to aggregate within searchlight
    
    def __post_init__(self):
        """Validate configuration."""
        if self.n_permutations < 100:
            raise ValueError("n_permutations should be >= 100 for reliable stats")
        if not 0 < self.alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")