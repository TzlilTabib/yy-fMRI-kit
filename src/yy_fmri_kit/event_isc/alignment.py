"""
Align neural patterns across subjects to common post order.
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from yy_fmri_kit.event_isc.utils import natural_sort_key
from yy_fmri_kit.static.event_isc.config import AlignmentConfig

logging.basicConfig(level=logging.INFO)


class PatternAligner:
    """
    Align neural patterns across subjects to enable ISC analysis.
    
    Handles the fact that subjects saw posts in random order by:
    1. Finding common posts across subjects
    2. Reordering each subject's data to canonical order
    3. Validating alignment quality
    """
    
    def __init__(self, config: AlignmentConfig):
        """
        Initialize pattern aligner.
        
        Args:
            config: AlignmentConfig with alignment parameters
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load_subject_data(
        self,
        npz_path: Path,
        pattern_type: str = "roi"
    ) -> dict:
        """
        Load one subject's pattern data from NPZ file.
        
        Args:
            npz_path: Path to NPZ file
            pattern_type: "roi" or "searchlight"
            
        Returns:
            Dict with 'subject', 'post_ids', 'data', 'run_type'
        """
        npz = np.load(npz_path, allow_pickle=True)
        
        # Determine data key based on pattern type
        if pattern_type == "roi":
            data_key = "data"
        else:  # searchlight
            data_key = "patterns" if "patterns" in npz.files else "data"
        
        return {
            "subject": npz["subject"].item() if "subject" in npz.files else npz_path.parent.name,
            "post_ids": npz["post_ids"].astype(str).tolist(),
            "data": npz[data_key],
            "run_type": npz["run_type"].item() if "run_type" in npz.files else None,
            "n_features": npz[data_key].shape[1],
        }
    
    def find_common_posts(
        self,
        subjects_data: list[dict],
        strategy: Optional[str] = None
    ) -> list[str]:
        """
        Find common posts across all subjects based on alignment strategy.
        
        Args:
            subjects_data: List of dicts from load_subject_data()
            strategy: "intersection", "union", or "first_subject" (None = use config)
            
        Returns:
            List of post IDs in canonical order
        """
        strategy = strategy or self.config.strategy
        
        all_post_sets = [set(sd["post_ids"]) for sd in subjects_data]
        
        if strategy == "intersection":
            # Only posts seen by ALL subjects
            common_posts = set.intersection(*all_post_sets)
            self.logger.info(f"Intersection strategy: {len(common_posts)} common posts")
            
        elif strategy == "union":
            # All posts seen by ANY subject (will have NaNs for missing)
            common_posts = set.union(*all_post_sets)
            self.logger.info(f"Union strategy: {len(common_posts)} total posts")
            
        elif strategy == "first_subject":
            # Use first subject's posts as canonical
            common_posts = all_post_sets[0]
            self.logger.info(f"First-subject strategy: {len(common_posts)} posts from {subjects_data[0]['subject']}")
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Sort canonically
        canonical_order = sorted(common_posts, key=natural_sort_key)
        
        return canonical_order
    
    def align_subject_to_canonical(
        self,
        subject_data: dict,
        canonical_posts: list[str]
    ) -> Optional[np.ndarray]:
        """
        Reorder one subject's data to match canonical post order.
        
        Args:
            subject_data: Dict from load_subject_data()
            canonical_posts: Canonical post order
            
        Returns:
            Aligned data array (n_posts, n_features) or None if too many missing
        """
        post_to_idx = {pid: i for i, pid in enumerate(subject_data["post_ids"])}
        
        # Check for missing posts
        missing = [pid for pid in canonical_posts if pid not in post_to_idx]
        
        if missing:
            if not self.config.allow_missing_posts:
                self.logger.warning(
                    f"Subject {subject_data['subject']} missing {len(missing)}/{len(canonical_posts)} posts"
                )
                return None
            else:
                self.logger.info(
                    f"Subject {subject_data['subject']}: {len(missing)} missing posts (will use NaN)"
                )
        
        # Build aligned array
        n_features = subject_data["n_features"]
        aligned = np.full((len(canonical_posts), n_features), np.nan, dtype=np.float32)
        
        for i, pid in enumerate(canonical_posts):
            if pid in post_to_idx:
                aligned[i] = subject_data["data"][post_to_idx[pid]]
        
        return aligned
    
    def align_run(
        self,
        npz_files: list[Path],
        pattern_type: str = "roi"
    ) -> dict:
        """
        Align all subjects for one run type to common post order.
        
        Args:
            npz_files: List of NPZ file paths for this run
            pattern_type: "roi" or "searchlight"
            
        Returns:
            Dict with:
                - 'subjects': list of subject IDs (aligned)
                - 'post_ids': canonical post order
                - 'data': list of aligned arrays (n_subjects x [n_posts, n_features])
                - 'n_dropped': number of subjects dropped
                - 'dropped_subjects': list of dropped subject IDs
        """
        self.logger.info(f"\n{'='*60}\nAligning {len(npz_files)} subjects\n{'='*60}")
        
        if len(npz_files) < self.config.min_subjects:
            raise ValueError(
                f"Need at least {self.config.min_subjects} subjects, got {len(npz_files)}"
            )
        
        # Load all subjects
        subjects_data = [self.load_subject_data(f, pattern_type) for f in npz_files]
        
        # Check feature dimensions match
        if self.config.check_feature_dims:
            feature_dims = [sd["n_features"] for sd in subjects_data]
            if len(set(feature_dims)) > 1:
                raise ValueError(
                    f"Feature dimensions don't match across subjects: {set(feature_dims)}"
                )
        
        # Find canonical post order
        canonical_posts = self.find_common_posts(subjects_data)
        
        if len(canonical_posts) == 0:
            raise ValueError("No common posts found across subjects!")
        
        self.logger.info(f"Canonical order: {len(canonical_posts)} posts")
        
        # Align each subject
        aligned_data = []
        kept_subjects = []
        dropped_subjects = []
        
        for subject_data in subjects_data:
            aligned = self.align_subject_to_canonical(subject_data, canonical_posts)
            
            if aligned is None:
                dropped_subjects.append(subject_data["subject"])
                continue
            
            aligned_data.append(aligned)
            kept_subjects.append(subject_data["subject"])
        
        # Check we still have enough subjects
        if len(kept_subjects) < self.config.min_subjects:
            raise ValueError(
                f"After alignment, only {len(kept_subjects)} subjects remain "
                f"(need {self.config.min_subjects})"
            )
        
        self.logger.info(
            f"\nAlignment complete:\n"
            f"  Kept: {len(kept_subjects)} subjects\n"
            f"  Dropped: {len(dropped_subjects)} subjects\n"
            f"  Posts: {len(canonical_posts)}\n"
            f"  Shape per subject: {aligned_data[0].shape}"
        )
        
        if dropped_subjects:
            self.logger.warning(f"Dropped subjects: {dropped_subjects}")
        
        return {
            "subjects": kept_subjects,
            "post_ids": canonical_posts,
            "data": aligned_data,  # List of (n_posts, n_features) arrays
            "n_dropped": len(dropped_subjects),
            "dropped_subjects": dropped_subjects,
        }
    
    def align_all_runs(
        self,
        output_dir: Path,
        run_types: list[str],
        pattern_type: str = "roi"
    ) -> dict[str, dict]:
        """
        Align all run types.
        
        Args:
            output_dir: Directory containing subject NPZ files
            run_types: List of run types to process (e.g., ["AntiLeft", "ProLeft"])
            pattern_type: "roi" or "searchlight"
            
        Returns:
            Dict mapping run_type -> alignment result
        """
        output_dir = Path(output_dir)
        
        if pattern_type == "roi":
            pattern_str = "desc-roi_patterns"
        else:
            pattern_str = "desc-searchlight_patterns"
        
        results = {}
        
        for run_type in run_types:
            self.logger.info(f"\n{'#'*60}\nProcessing run: {run_type}\n{'#'*60}")
            
            # Find all NPZ files for this run
            # Find all NPZ files for this run (flexible matching for complex BIDS names)
            all_npz = list(output_dir.rglob(f"*{pattern_str}.npz"))
            npz_files = [f for f in all_npz if f"task-{run_type}" in f.name]

            # Keep only one file per subject (in case of multiple sessions)
            subject_files = {}
            for f in npz_files:
                subject = f.name.split('_')[0]  # Extract sub-XX
                if subject not in subject_files:
                    subject_files[subject] = f

            npz_files = sorted(subject_files.values())
            
            if len(npz_files) < self.config.min_subjects:
                self.logger.warning(
                    f"Skipping {run_type}: only {len(npz_files)} files found "
                    f"(need {self.config.min_subjects})"
                )
                continue
            
            try:
                aligned = self.align_run(npz_files, pattern_type)
                results[run_type] = aligned
                
            except Exception as e:
                self.logger.error(f"Failed to align {run_type}: {e}", exc_info=True)
        
        return results
    
    def save_aligned(
        self,
        aligned_result: dict,
        output_path: Path,
        run_type: str
    ):
        """
        Save aligned data to NPZ file.
        
        Args:
            aligned_result: Result from align_run()
            output_path: Where to save
            run_type: Run type name
        """
        np.savez_compressed(
            output_path,
            subjects=np.array(aligned_result["subjects"], dtype=str),
            post_ids=np.array(aligned_result["post_ids"], dtype=str),
            data=np.stack(aligned_result["data"], axis=0),  # (n_subjects, n_posts, n_features)
            run_type=run_type,
            n_dropped=aligned_result["n_dropped"],
            dropped_subjects=np.array(aligned_result["dropped_subjects"], dtype=str),
        )
        
        self.logger.info(f"Saved aligned data to {output_path}")