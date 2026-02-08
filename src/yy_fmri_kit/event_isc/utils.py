"""
Utility functions for event-locked ISC analysis.
"""
import re
from pathlib import Path
from typing import Optional


def natural_sort_key(s: str):
    """
    Sort strings with embedded numbers naturally.
    
    Examples:
        ['post2', 'post10', 'post1'] -> ['post1', 'post2', 'post10']
    
    Args:
        s: String to create sort key for
        
    Returns:
        List of mixed int/str for natural sorting
    """
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def infer_run_type_from_filename(path: Path, valid_types: list[str]) -> Optional[str]:
    """
    Extract run type from filename.
    
    Examples:
        'sub01_task-AntiLeft_bold.nii' -> 'AntiLeft'
        'sub02_task-proleft_bold.nii' -> 'ProLeft'
    
    Args:
        path: Path to NIfTI file
        valid_types: List of valid run type names
        
    Returns:
        Run type string or None if not found
    """
    name = path.name.lower()
    for run_type in valid_types:
        if run_type.lower() in name:
            return run_type
    return None


def validate_events_dataframe(df, config):
    """
    Validate events DataFrame has required columns and valid data.
    
    Args:
        df: Events DataFrame
        config: ExtractionConfig with column names
        
    Returns:
        Tuple of (is_valid, warnings_list, errors_list)
    """
    warnings = []
    errors = []
    
    # Check required columns
    required_cols = [config.subject_col, config.run_col, config.post_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return False, warnings, errors
    
    # Check for NaN values
    for col in required_cols:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            warnings.append(f"{col}: {n_missing} missing values")
    
    # Check onset/duration columns exist
    onset_candidates = ["onset", "onset_s", "onset_tr"]
    duration_candidates = ["duration", "duration_s", "duration_tr"]
    
    has_onset = any(col in df.columns for col in onset_candidates)
    has_duration = any(col in df.columns for col in duration_candidates)
    
    if not has_onset:
        errors.append(f"No onset column found. Tried: {onset_candidates}")
    if not has_duration:
        errors.append(f"No duration column found. Tried: {duration_candidates}")
    
    is_valid = len(errors) == 0
    
    return is_valid, warnings, errors