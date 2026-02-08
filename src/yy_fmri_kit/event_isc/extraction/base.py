"""
Base classes and utilities for neural pattern extraction.
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from nilearn import image, input_data

# Import from your package
from yy_fmri_kit.static.event_isc.config import ExtractionConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class BasePatternExtractor:
    """Base class for extracting neural patterns from fMRI data."""
    
    def __init__(self, config: ExtractionConfig):
        """
        Initialize pattern extractor.
        
        Args:
            config: ExtractionConfig with extraction parameters
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def load_and_filter_events(
        self,
        events_df: pd.DataFrame,
        subject: Optional[str] = None,
        run_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load and filter events DataFrame for a specific subject/run.
        
        Args:
            events_df: Full events DataFrame
            subject: Subject ID to filter (None = no filtering)
            run_type: Run type to filter (None = no filtering)
            
        Returns:
            Filtered DataFrame with only valid posts
        """
        df = events_df.copy()
        
        # Filter by subject and run if provided
        if subject is not None:
            df = df[df[self.config.subject_col].astype(str).str.strip() == str(subject).strip()]
        
        if run_type is not None:
            run_col_lower = df[self.config.run_col].astype(str).str.lower()
            df = df[run_col_lower == run_type.lower()]
        
        # Filter out invalid posts (NaN, empty, "nan" string)
        post_vals = df[self.config.post_col].astype(str).str.strip()
        valid_mask = (
            df[self.config.post_col].notna() & 
            (post_vals != "") & 
            (post_vals.str.lower() != "nan")
        )
        df = df[valid_mask].copy()
        
        if df.empty:
            self.logger.warning(f"No valid posts found for subject={subject}, run={run_type}")
        
        return df
    
    def compute_tr_timings(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert onset/duration to TR indices and compute shifted windows.
        
        Args:
            events_df: Events with onset/duration columns
            
        Returns:
            DataFrame with added TR timing columns
        """
        df = events_df.copy()
        
        # Auto-detect column names if not specified
        onset_col = self.config.onset_col
        duration_col = self.config.duration_col
        
        if onset_col is None:
            if "onset_tr" in df.columns:
                onset_col = "onset_tr"
            elif "onset" in df.columns:
                onset_col = "onset"
            else:
                onset_col = "onset_s"
        
        if duration_col is None:
            if "duration_tr" in df.columns:
                duration_col = "duration_tr"
            elif "duration" in df.columns:
                duration_col = "duration"
            else:
                duration_col = "duration_s"
        
        # Convert to TR indices
        if self.config.time_unit == "seconds":
            df["onset_tr"] = (df[onset_col] / self.config.tr).round().astype(int)
            df["duration_tr"] = (df[duration_col] / self.config.tr).round().astype(int)
        else:
            df["onset_tr"] = df[onset_col].astype(int)
            df["duration_tr"] = df[duration_col].astype(int)
        
        # Compute offsets and shifted windows (for hemodynamic delay)
        df["offset_tr"] = df["onset_tr"] + df["duration_tr"]
        df["shifted_onset_tr"] = df["onset_tr"] + self.config.shift_tr
        df["shifted_offset_tr"] = df["offset_tr"] + self.config.shift_tr
        
        return df
    
    def validate_tr_bounds(
        self, 
        events_df: pd.DataFrame, 
        n_trs: int
    ) -> tuple[pd.DataFrame, dict]:
        """
        Validate TR indices are within bounds and return filtered events.
        
        Args:
            events_df: DataFrame with TR timing columns
            n_trs: Total number of TRs in fMRI scan
            
        Returns:
            Tuple of (valid_events_df, validation_stats)
        """
        df = events_df.copy()
        
        stats = {
            "total": len(df),
            "invalid_duration": 0,
            "out_of_bounds": 0,
            "valid": 0
        }
        
        # Check for invalid durations
        invalid_duration = df["shifted_offset_tr"] <= df["shifted_onset_tr"]
        stats["invalid_duration"] = invalid_duration.sum()
        
        # Check for out of bounds
        out_of_bounds = (df["shifted_onset_tr"] < 0) | (df["shifted_offset_tr"] > n_trs)
        stats["out_of_bounds"] = out_of_bounds.sum()
        
        # Keep only valid
        valid_mask = ~invalid_duration & ~out_of_bounds
        valid_df = df[valid_mask].copy()
        stats["valid"] = len(valid_df)
        
        # Log warnings
        if stats["invalid_duration"] > 0:
            self.logger.warning(
                f"Dropped {stats['invalid_duration']} posts with invalid duration "
                f"(shifted_offset <= shifted_onset)"
            )
        
        if stats["out_of_bounds"] > 0:
            self.logger.warning(
                f"Dropped {stats['out_of_bounds']} posts with out-of-bounds TRs "
                f"(n_trs={n_trs}, shift={self.config.shift_tr})"
            )
        
        if stats["valid"] == 0:
            self.logger.error("No valid posts remain after validation!")
        
        return valid_df, stats
    
    def create_masker(self, mask_path: Optional[Path] = None):
        """
        Create NiftiMasker for extracting voxel time series.
        
        Args:
            mask_path: Path to brain mask (None = whole-brain mask)
            
        Returns:
            Configured NiftiMasker
        """
        masker_kwargs = {
            "smoothing_fwhm": self.config.smoothing_fwhm,
            "detrend": self.config.detrend,
            "standardize": self.config.standardize,
        }
        
        if mask_path is not None:
            masker_kwargs["mask_img"] = str(mask_path)
        
        return input_data.NiftiMasker(**masker_kwargs)
    
    def extract_post_pattern(
        self,
        voxel_time_series: np.ndarray,
        start_tr: int,
        end_tr: int
    ) -> np.ndarray:
        """
        Extract neural pattern for a single post by averaging across time window.
        
        Args:
            voxel_time_series: (n_trs, n_voxels) array
            start_tr: Start index (inclusive)
            end_tr: End index (exclusive)
            
        Returns:
            (n_voxels,) pattern vector averaged across time window
        """
        return np.mean(voxel_time_series[start_tr:end_tr, :], axis=0)