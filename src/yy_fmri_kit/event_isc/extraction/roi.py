"""
ROI-based pattern extraction (average across all voxels in mask).
"""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from nilearn import image

from yy_fmri_kit.event_isc.extraction.base import BasePatternExtractor
from yy_fmri_kit.event_isc.utils import natural_sort_key, infer_run_type_from_filename
from yy_fmri_kit.static.event_isc.config import ExtractionConfig


class ROIPatternExtractor(BasePatternExtractor):
    """Extract ROI patterns by averaging BOLD across all voxels in a mask."""
    
    def extract_single_run(
        self,
        nifti_path: Path,
        events_df: pd.DataFrame,
        mask_path: Optional[Path] = None,
        verbose: bool = False
    ) -> dict[str, np.ndarray]:
        """
        Extract ROI patterns for all posts in a single fMRI run.
        
        Args:
            nifti_path: Path to 4D NIfTI file
            events_df: Events DataFrame (already filtered for this subject/run)
            mask_path: Optional brain mask
            verbose: Print per-post debug info
            
        Returns:
            Dictionary mapping post_id -> pattern vector (n_voxels,)
        """
        self.logger.info(f"Processing {nifti_path.name}")
        
        # Load fMRI image
        img = image.load_img(str(nifti_path))
        
        # Compute TR timings
        events = self.compute_tr_timings(events_df)
        
        # Create masker and extract time series
        masker = self.create_masker(mask_path)
        voxel_time_series = masker.fit_transform(img)  # (n_trs, n_voxels)
        n_trs = voxel_time_series.shape[0]
        n_voxels = voxel_time_series.shape[1]
        
        self.logger.info(f"Loaded fMRI: {n_trs} TRs, {n_voxels} voxels")
        
        # Validate TR bounds
        valid_events, stats = self.validate_tr_bounds(events, n_trs)
        
        if valid_events.empty:
            self.logger.error("No valid events after validation")
            return {}
        
        # Extract patterns for each post
        post_patterns = {}
        
        for _, row in valid_events.iterrows():
            post_id = row[self.config.post_col]
            start = int(row["shifted_onset_tr"])
            end = int(row["shifted_offset_tr"])
            
            pattern = self.extract_post_pattern(voxel_time_series, start, end)
            post_patterns[post_id] = pattern
            
            if verbose:
                self.logger.debug(
                    f"Post {post_id}: TR [{start}:{end}], "
                    f"pattern shape {pattern.shape}, "
                    f"mean={pattern.mean():.3f}, std={pattern.std():.3f}"
                )
        
        self.logger.info(
            f"Extracted {len(post_patterns)} posts "
            f"(dropped {stats['total'] - stats['valid']} invalid)"
        )
        
        return post_patterns
    
    def batch_extract(
        self,
        runs_dict: dict[str, list[Path]],
        events_df: pd.DataFrame,
        output_dir: Path,
        mask_path: Optional[Path] = None,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Batch extract patterns for multiple subjects and runs.
        
        Args:
            runs_dict: {subject_id: [nifti_path1, nifti_path2, ...]}
            events_df: Full events DataFrame with all subjects/runs
            output_dir: Directory to save NPZ files
            mask_path: Optional brain mask
            verbose: Print detailed info
            
        Returns:
            Summary DataFrame with extraction statistics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        summary_records = []
        
        for subject, nifti_paths in runs_dict.items():
            subject = str(subject).strip()
            subject_dir = output_dir / subject
            subject_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"\n{'='*60}\nProcessing subject: {subject}\n{'='*60}")
            
            for nifti_path in nifti_paths:
                nifti_path = Path(nifti_path)
                
                # Infer run type from filename
                run_type = infer_run_type_from_filename(nifti_path, self.config.valid_run_types)
                
                if run_type is None:
                    self.logger.warning(f"Could not infer run type from {nifti_path.name}")
                    summary_records.append({
                        "subject": subject,
                        "nifti": str(nifti_path),
                        "run_type": None,
                        "status": "run_type_not_found",
                        "n_posts": 0,
                        "n_voxels": 0,
                    })
                    continue
                
                # Filter events for this subject + run
                run_events = self.load_and_filter_events(
                    events_df,
                    subject=subject,
                    run_type=run_type
                )
                
                if run_events.empty:
                    self.logger.warning(f"No events found for {subject} / {run_type}")
                    summary_records.append({
                        "subject": subject,
                        "nifti": str(nifti_path),
                        "run_type": run_type,
                        "status": "no_events",
                        "n_posts": 0,
                        "n_voxels": 0,
                    })
                    continue
                
                # Extract patterns
                try:
                    post_patterns = self.extract_single_run(
                        nifti_path=nifti_path,
                        events_df=run_events,
                        mask_path=mask_path,
                        verbose=verbose
                    )
                    
                    if not post_patterns:
                        summary_records.append({
                            "subject": subject,
                            "nifti": str(nifti_path),
                            "run_type": run_type,
                            "status": "no_valid_posts",
                            "n_posts": 0,
                            "n_voxels": 0,
                        })
                        continue
                    
                    # Sort post IDs naturally and stack data
                    post_ids = sorted(post_patterns.keys(), key=natural_sort_key)
                    data = np.vstack([post_patterns[pid] for pid in post_ids])
                    
                    # Save as NPZ
                    output_file = subject_dir / f"{nifti_path.stem}_desc-roi_patterns.npz"
                    np.savez_compressed(
                        output_file,
                        post_ids=np.array(post_ids, dtype=str),
                        data=data.astype(self.config.output_precision),
                        run_type=run_type,
                        subject=subject,
                        # Save config for reproducibility
                        tr=self.config.tr,
                        shift_tr=self.config.shift_tr,
                        time_unit=self.config.time_unit,
                    )
                    
                    self.logger.info(f"Saved to {output_file}")
                    
                    summary_records.append({
                        "subject": subject,
                        "nifti": str(nifti_path),
                        "run_type": run_type,
                        "status": "success",
                        "n_posts": len(post_ids),
                        "n_voxels": data.shape[1],
                        "output_file": str(output_file),
                    })
                    
                except Exception as e:
                    self.logger.error(f"Failed to process {nifti_path.name}: {e}", exc_info=True)
                    summary_records.append({
                        "subject": subject,
                        "nifti": str(nifti_path),
                        "run_type": run_type,
                        "status": f"error: {str(e)[:100]}",
                        "n_posts": 0,
                        "n_voxels": 0,
                    })
        
        # Create and save summary
        summary_df = pd.DataFrame(summary_records)
        summary_path = output_dir / "extraction_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        self.logger.info(f"\nSaved summary to {summary_path}")
        
        # Print summary statistics
        success_count = (summary_df["status"] == "success").sum()
        total_count = len(summary_df)
        self.logger.info(
            f"\n{'='*60}\n"
            f"EXTRACTION COMPLETE: {success_count}/{total_count} runs successful\n"
            f"{'='*60}"
        )
        
        return summary_df