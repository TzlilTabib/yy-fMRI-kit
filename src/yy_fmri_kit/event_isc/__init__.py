"""
Event-locked inter-subject correlation (ISC) analysis.

This module provides tools for extracting neural patterns from naturalistic
viewing fMRI experiments and aligning them across subjects for ISC analysis.

Main components:
- extraction: Extract ROI and searchlight patterns from fMRI data
- alignment: Align patterns across subjects (handles random post order)
- utils: Helper functions for validation and sorting

Example:
    from yy_fmri_kit.event_isc import (
        ExtractionConfig,
        ROIPatternExtractor,
        PatternAligner,
    )
    from yy_fmri_kit.static.event_isc.config import AlignmentConfig
    
    # Extract patterns
    config = ExtractionConfig(tr=1.5, shift_tr=4)
    extractor = ROIPatternExtractor(config)
    extractor.batch_extract(runs_dict, events_df, output_dir)
    
    # Align across subjects
    aligner = PatternAligner(AlignmentConfig(strategy="intersection"))
    aligned = aligner.align_all_runs(output_dir, ["AntiLeft", "ProLeft"])
"""

# Import config from static module
from yy_fmri_kit.static.event_isc.config import (
    ExtractionConfig,
    AlignmentConfig,
)

# Import extraction classes
from .extraction import (
    BasePatternExtractor,
    ROIPatternExtractor,
    SearchlightPatternExtractor,
)

# Import alignment class
from .alignment import PatternAligner

# Import utilities
from .utils import (
    natural_sort_key,
    infer_run_type_from_filename,
    validate_events_dataframe,
)

__all__ = [
    # Config classes
    "ExtractionConfig",
    "AlignmentConfig",
    # Extraction classes
    "BasePatternExtractor",
    "ROIPatternExtractor", 
    "SearchlightPatternExtractor",
    # Alignment
    "PatternAligner",
    # Utilities
    "natural_sort_key",
    "infer_run_type_from_filename",
    "validate_events_dataframe",
]

__version__ = "1.0.0"