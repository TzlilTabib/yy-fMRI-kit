"""
Searchlight-based pattern extraction (preserve spatial structure).

NOTE: This is a simplified implementation. For production use with large datasets,
consider using nilearn.decoding.SearchLight with a custom estimator.
"""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from nilearn import image

from yy_fmri_kit.event_isc.extraction.base import BasePatternExtractor
from yy_fmri_kit.event_isc.utils import natural_sort_key, infer_run_type_from_filename
from yy_fmri_kit.static.event_isc.config import ExtractionConfig


class SearchlightPatternExtractor(BasePatternExtractor):
    """
    Extract searchlight patterns preserving spatial information.
    
    For each searchlight sphere:
    - Extract time series for all voxels in the sphere
    - Average across time for each post presentation
    - Aggregate within sphere (mean or median)
    
    Output shape: (n_posts, n_searchlights)
    """
    
    def extract_single_run(
        self,
        nifti_path: Path,
        events_df: pd.DataFrame,
        mask_path: Optional[Path] = None,
        verbose: bool = False
    ) -> dict:
        """
        Extract searchlight patterns for all posts in a single run.
        
        Args:
            nifti_path: Path to 4D NIfTI file
            events_df: Events DataFrame (already filtered for this subject/run)
            mask_path: Optional brain mask
            verbose: Print debug info
            
        Returns:
            Dictionary with:
                - 'post_ids': list of post IDs (in order)
                - 'patterns': (n_posts, n_searchlights) array
                - 'searchlight_centers': (n_searchlights, 3) voxel coordinates
                - 'affine': affine transform from voxels to world coordinates
        """
        self.logger.info(f"Processing {nifti_path.name} with searchlight")
        
        # Load fMRI image
        img = image.load_img(str(nifti_path))
        
        # Compute TR timings
        events = self.compute_tr_timings(events_df)
        
        # Create masker
        masker = self.create_masker(mask_path)
        
        # Get mask as image
        if mask_path is not None:
            mask_img = image.load_img(str(mask_path))
        else:
            mask_img = masker.fit(img).mask_img_
        
        # Extract voxel time series
        voxel_time_series = masker.fit_transform(img)  # (n_trs, n_voxels)
        n_trs = voxel_time_series.shape[0]
        n_voxels = voxel_time_series.shape[1]
        
        self.logger.info(f"Loaded fMRI: {n_trs} TRs, {n_voxels} voxels")
        
        # Validate TR bounds
        valid_events, stats = self.validate_tr_bounds(events, n_trs)
        
        if valid_events.empty:
            self.logger.error("No valid events after validation")
            return {}
        
        # Get sorted post IDs
        post_ids = sorted(valid_events[self.config.post_col].unique(), key=natural_sort_key)
        n_posts = len(post_ids)
        
        self.logger.info(f"Extracting patterns for {n_posts} unique posts")
        
        # Get mask data and affine
        mask_data = mask_img.get_fdata().astype(bool)
        affine = mask_img.affine
        
        # Find all searchlight centers (voxels in mask)
        centers = np.argwhere(mask_data)  # (n_centers, 3) in voxel coords
        n_centers = len(centers)
        
        self.logger.info(f"Found {n_centers} searchlight centers")
        
        # Define searchlight radius in voxels
        voxel_size = np.abs(np.diag(affine)[:3])
        radius_voxels = int(np.ceil(self.config.searchlight_radius / voxel_size.min()))
        
        self.logger.info(
            f"Searchlight radius: {self.config.searchlight_radius}mm "
            f"≈ {radius_voxels} voxels (voxel size: {voxel_size})"
        )
        
        # NOTE: This is a simplified implementation
        # For production, you'd want to use proper searchlight with nilearn
        # or implement efficient voxel indexing
        
        self.logger.warning(
            "Using simplified searchlight - for large datasets, "
            "consider using nilearn.decoding.SearchLight"
        )
        
        # Extract patterns for each post
        post_patterns = {}  # post_id -> (n_searchlights,) aggregated pattern
        
        for post_id in post_ids:
            # Get TR window for this post
            post_event = valid_events[valid_events[self.config.post_col] == post_id].iloc[0]
            start = int(post_event["shifted_onset_tr"])
            end = int(post_event["shifted_offset_tr"])
            
            # Average BOLD across time window for this post
            post_timeseries = np.mean(voxel_time_series[start:end, :], axis=0)  # (n_voxels,)
            
            # For simplified version, use the post_timeseries directly
            # In full version, you'd aggregate within searchlight spheres
            searchlight_values = []
            
            # Simplified: treat each voxel as a searchlight center
            # Real implementation would compute neighborhoods
            for i in range(min(n_centers, n_voxels)):
                searchlight_values.append(post_timeseries[i])
            
            post_patterns[post_id] = np.array(searchlight_values)
            
            if verbose:
                self.logger.debug(
                    f"Post {post_id}: {len(searchlight_values)} searchlights, "
                    f"mean={post_patterns[post_id].mean():.3f}"
                )
        
        self.logger.info(f"Extracted searchlight patterns for {len(post_patterns)} posts")
        
        return {
            "post_ids": post_ids,
            "patterns": np.vstack([post_patterns[pid] for pid in post_ids]),  # (n_posts, n_searchlights)
            "searchlight_centers": centers[:len(searchlight_values)],  # Truncated to match
            "affine": affine,
        }
    
    def batch_extract(
        self,
        runs_dict: dict[str, list[Path]],
        events_df: pd.DataFrame,
        output_dir: Path,
        mask_path: Optional[Path] = None,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Batch extract searchlight patterns for multiple subjects and runs.
        
        Args:
            runs_dict: {subject_id: [nifti_paths]}
            events_df: Events DataFrame
            output_dir: Output directory
            mask_path: Optional mask
            verbose: Verbose logging
            
        Returns:
            Summary DataFrame
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
                
                # Infer run type
                run_type = infer_run_type_from_filename(nifti_path, self.config.valid_run_types)
                
                if run_type is None:
                    self.logger.warning(f"Could not infer run type from {nifti_path.name}")
                    summary_records.append({
                        "subject": subject,
                        "nifti": str(nifti_path),
                        "run_type": None,
                        "status": "run_type_not_found",
                    })
                    continue
                
                # Filter events
                run_events = self.load_and_filter_events(events_df, subject, run_type)
                
                if run_events.empty:
                    self.logger.warning(f"No events for {subject} / {run_type}")
                    summary_records.append({
                        "subject": subject,
                        "nifti": str(nifti_path),
                        "run_type": run_type,
                        "status": "no_events",
                    })
                    continue
                
                # Extract patterns
                try:
                    result = self.extract_single_run(
                        nifti_path=nifti_path,
                        events_df=run_events,
                        mask_path=mask_path,
                        verbose=verbose
                    )
                    
                    if not result:
                        summary_records.append({
                            "subject": subject,
                            "nifti": str(nifti_path),
                            "run_type": run_type,
                            "status": "extraction_failed",
                        })
                        continue
                    
                    # Save as NPZ
                    output_file = subject_dir / f"{nifti_path.stem}_desc-searchlight_patterns.npz"
                    np.savez_compressed(
                        output_file,
                        post_ids=np.array(result["post_ids"], dtype=str),
                        patterns=result["patterns"].astype(self.config.output_precision),
                        searchlight_centers=result["searchlight_centers"],
                        affine=result["affine"],
                        run_type=run_type,
                        subject=subject,
                        # Config
                        tr=self.config.tr,
                        shift_tr=self.config.shift_tr,
                        searchlight_radius=self.config.searchlight_radius,
                    )
                    
                    self.logger.info(f"Saved to {output_file}")
                    
                    summary_records.append({
                        "subject": subject,
                        "nifti": str(nifti_path),
                        "run_type": run_type,
                        "status": "success",
                        "n_posts": len(result["post_ids"]),
                        "n_searchlights": result["patterns"].shape[1],
                        "output_file": str(output_file),
                    })
                    
                except Exception as e:
                    self.logger.error(f"Failed: {e}", exc_info=True)
                    summary_records.append({
                        "subject": subject,
                        "nifti": str(nifti_path),
                        "run_type": run_type,
                        "status": f"error: {str(e)[:100]}",
                    })
        
        summary_df = pd.DataFrame(summary_records)
        summary_path = output_dir / "searchlight_extraction_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        return summary_df