"""
ISC analysis wrapper with statistical testing.

Wraps the core compute_isc functionality with:
- Permutation testing for significance
- Multiple comparison correction
- Easy loading from aligned data files
"""
from __future__ import annotations

import logging
from typing import Literal, Optional, Tuple

import numpy as np

# Import from your existing ISC module
from yy_fmri_kit.isc.compute import compute_isc, compute_isc_brainiak

logging.basicConfig(level=logging.INFO)


class ISCAnalyzer:
    """
    High-level interface for ISC analysis with statistical testing.
    
    Wraps your existing compute_isc functionality and adds:
    - Permutation testing
    - Multiple comparison correction  
    - Easy integration with aligned data format
    
    Examples:
        >>> # From aligned NPZ file
        >>> analyzer = ISCAnalyzer()
        >>> results = analyzer.analyze_from_file(
        ...     "aligned/AntiLeft_aligned.npz",
        ...     n_permutations=1000,
        ... )
        >>> print(f"Significant voxels: {results['significant'].sum()}")
        
        >>> # From numpy array
        >>> patterns = np.load("aligned.npz")["data"]  # (n_subjects, n_posts, n_voxels)
        >>> results = analyzer.analyze(patterns, n_permutations=1000)
    """
    
    def __init__(
        self,
        backend: Literal["native", "brainiak"] = "native",
        fisher_z: bool = True,
        nan_policy: Literal["propagate", "omit"] = "omit",
    ):
        """
        Initialize ISC analyzer.
        
        Args:
            backend: "native" (uses your compute_isc) or "brainiak"
            fisher_z: Apply Fisher z-transform before averaging (recommended)
            nan_policy: "omit" (recommended) or "propagate"
        """
        self.backend = backend
        self.fisher_z = fisher_z
        self.nan_policy = nan_policy
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def compute_isc(
        self,
        data: np.ndarray,
        return_subjectwise: bool = False,
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """
        Compute ISC using configured backend.
        
        Args:
            data: (n_subjects, n_timepoints, n_features) array
            return_subjectwise: Return per-subject ISC values
            
        Returns:
            If return_subjectwise=False:
                isc_mean: (n_features,) mean ISC across subjects
            If return_subjectwise=True:
                (isc_subjectwise, isc_mean) tuple where:
                    isc_subjectwise: (n_subjects, n_features)
                    isc_mean: (n_features,)
        """
        # Convert to list format expected by your compute_isc
        data_list = [data[i] for i in range(data.shape[0])]
        
        if self.backend == "native":
            return compute_isc(
                data_list,
                method="loo",
                fisher_z=self.fisher_z,
                nan_policy=self.nan_policy,
                return_subjectwise=return_subjectwise,
            )
        elif self.backend == "brainiak":
            return compute_isc_brainiak(
                data_list,
                fisher_z=self.fisher_z,
                return_subjectwise=return_subjectwise,
            )
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def permutation_test(
        self,
        data: np.ndarray,
        n_permutations: int = 1000,
        random_seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform permutation test for ISC significance.
        
        Strategy: For each permutation, randomly shuffle timepoint order 
        independently for each subject. This destroys temporal alignment
        while preserving marginal distributions.
        
        Args:
            data: (n_subjects, n_timepoints, n_features)
            n_permutations: Number of permutations
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of:
                - isc_subjectwise: (n_subjects, n_features) observed ISC per subject
                - p_values: (n_features,) two-tailed p-values
                - null_distribution: (n_permutations, n_features) null ISC means
        """
        rng = np.random.RandomState(random_seed)
        
        n_subjects, n_timepoints, n_features = data.shape
        
        self.logger.info(
            f"Permutation test: {n_permutations} permutations, "
            f"{n_subjects} subjects, {n_timepoints} timepoints, {n_features} features"
        )
        
        # Compute observed ISC (subjectwise for later use)
        isc_subjectwise, isc_mean = self.compute_isc(data, return_subjectwise=True)
        
        self.logger.info(f"Observed mean ISC: {isc_mean.mean():.4f}")
        
        # Null distribution
        null_isc = np.zeros((n_permutations, n_features), dtype=np.float32)
        
        for perm_i in range(n_permutations):
            # Permute timepoint order for each subject independently
            # This breaks temporal alignment across subjects
            permuted_data = np.zeros_like(data)
            for subj_i in range(n_subjects):
                perm_order = rng.permutation(n_timepoints)
                permuted_data[subj_i] = data[subj_i, perm_order, :]
            
            # Compute ISC on permuted data (only need mean)
            _, perm_isc_mean = self.compute_isc(permuted_data, return_subjectwise=True)
            null_isc[perm_i] = perm_isc_mean
            
            if (perm_i + 1) % 100 == 0:
                self.logger.info(
                    f"  Permutation {perm_i + 1}/{n_permutations}: "
                    f"null ISC = {perm_isc_mean.mean():.4f}"
                )
        
        # Compute p-values (two-tailed)
        # For each feature, count how many null values are >= observed
        p_values = np.zeros(n_features, dtype=np.float32)
        for feat in range(n_features):
            # Two-tailed: count more extreme values (in absolute value)
            n_extreme = np.sum(np.abs(null_isc[:, feat]) >= np.abs(isc_mean[feat]))
            # Add 1 to numerator and denominator (observed counts as one permutation)
            p_values[feat] = (n_extreme + 1) / (n_permutations + 1)
        
        self.logger.info(
            f"Permutation test complete. "
            f"Mean null ISC: {null_isc.mean():.4f}, "
            f"Median p-value: {np.median(p_values):.4f}"
        )
        
        return isc_subjectwise, p_values, null_isc
    
    def correct_multiple_comparisons(
        self,
        p_values: np.ndarray,
        alpha: float = 0.05,
        method: Literal["fdr_bh", "bonferroni", "none"] = "fdr_bh",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Correct for multiple comparisons.
        
        Args:
            p_values: (n_features,) uncorrected p-values
            alpha: Significance threshold (default 0.05)
            method: "fdr_bh" (Benjamini-Hochberg), "bonferroni", or "none"
            
        Returns:
            Tuple of:
                - significant: (n_features,) boolean mask
                - corrected_p: (n_features,) corrected p-values
        """
        n_features = len(p_values)
        
        if method == "none":
            self.logger.info("No multiple comparison correction applied")
            significant = p_values < alpha
            return significant, p_values.copy()
        
        elif method == "bonferroni":
            corrected_alpha = alpha / n_features
            significant = p_values < corrected_alpha
            corrected_p = np.minimum(p_values * n_features, 1.0)
            
            self.logger.info(
                f"Bonferroni correction (α={alpha:.3f}): "
                f"{significant.sum()}/{n_features} significant "
                f"({100*significant.sum()/n_features:.2f}%)"
            )
            return significant, corrected_p
        
        elif method == "fdr_bh":
            from scipy.stats import false_discovery_control
            
            corrected_p = false_discovery_control(p_values, method='bh').astype(np.float32)
            significant = corrected_p < alpha
            
            self.logger.info(
                f"FDR-BH correction (α={alpha:.3f}): "
                f"{significant.sum()}/{n_features} significant "
                f"({100*significant.sum()/n_features:.2f}%)"
            )
            
            return significant, corrected_p
        
        else:
            raise ValueError(f"Unknown correction method: {method}")
    
    def analyze(
        self,
        data: np.ndarray,
        n_permutations: int = 1000,
        alpha: float = 0.05,
        correction: Literal["fdr_bh", "bonferroni", "none"] = "fdr_bh",
        random_seed: int = 42,
    ) -> dict:
        """
        Run complete ISC analysis pipeline.
        
        Args:
            data: (n_subjects, n_timepoints, n_features) array
            n_permutations: Number of permutations (1000+ recommended)
            alpha: Significance threshold (default 0.05)
            correction: Multiple comparison correction method
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary with:
                - isc_subjectwise: (n_subjects, n_features) ISC per subject
                - isc_mean: (n_features,) mean ISC across subjects
                - isc_std: (n_features,) standard deviation across subjects
                - p_values: (n_features,) uncorrected p-values
                - p_corrected: (n_features,) corrected p-values
                - significant: (n_features,) boolean significance mask
                - null_distribution: (n_permutations, n_features) null ISC
                - n_significant: int, number of significant features
                - alpha: significance threshold used
                - correction: correction method used
        """
        print(f"DEBUG: Data shape entering analyze: {data.shape}") # Should be (N, 18, Searchlights)
        self.logger.info(
            f"\n{'='*60}\n"
            f"ISC Analysis\n"
            f"{'='*60}\n"
            f"Data shape: {data.shape}\n"
            f"  Subjects: {data.shape[0]}\n"
            f"  Timepoints: {data.shape[1]}\n"
            f"  Features: {data.shape[2]}\n"
            f"Backend: {self.backend}\n"
            f"Fisher z: {self.fisher_z}\n"
            f"NaN policy: {self.nan_policy}\n"
            f"Permutations: {n_permutations}\n"
            f"Alpha: {alpha}\n"
            f"Correction: {correction}\n"
            f"{'='*60}"
        )
        
        # Permutation test
        isc_subjectwise, p_values, null_dist = self.permutation_test(
            data,
            n_permutations=n_permutations,
            random_seed=random_seed,
        )
        print(f"DEBUG: Subjectwise ISC shape: {isc_subjectwise.shape}") # Should be (N, Searchlights)
        # Summary statistics
        isc_mean = np.mean(isc_subjectwise, axis=0)
        isc_std = np.std(isc_subjectwise, axis=0, ddof=1)  # Sample std
        
        # Multiple comparison correction
        significant, p_corrected = self.correct_multiple_comparisons(
            p_values,
            alpha=alpha,
            method=correction,
        )
        
        # Package results
        results = {
            "isc_subjectwise": isc_subjectwise,
            "isc_mean": isc_mean,
            "isc_std": isc_std,
            "p_values": p_values,
            "p_corrected": p_corrected,
            "significant": significant,
            "null_distribution": null_dist,
            "n_significant": int(significant.sum()),
            "alpha": alpha,
            "correction": correction,
        }
        
        # Summary report
        if results['n_significant'] > 0:
            sig_mean = isc_mean[significant].mean()
            sig_std = isc_mean[significant].std()
        else:
            sig_mean = np.nan
            sig_std = np.nan
        
        self.logger.info(
            f"\n{'='*60}\n"
            f"Results Summary:\n"
            f"  Overall:\n"
            f"    Mean ISC: {isc_mean.mean():.4f} ± {isc_mean.std():.4f}\n"
            f"    Range: [{isc_mean.min():.4f}, {isc_mean.max():.4f}]\n"
            f"  Significance:\n"
            f"    Significant features: {results['n_significant']}/{len(significant)} "
            f"({100*results['n_significant']/len(significant):.1f}%)\n"
            f"    Mean ISC (significant only): {sig_mean:.4f} ± {sig_std:.4f}\n"
            f"  Null distribution:\n"
            f"    Mean: {null_dist.mean():.4f} ± {null_dist.std():.4f}\n"
            f"{'='*60}"
        )
        
        return results
    
    def analyze_from_file(
        self,
        npz_path: str,
        **kwargs
    ) -> dict:
        """
        Analyze ISC from aligned NPZ file.
        
        Args:
            npz_path: Path to aligned NPZ file (from PatternAligner)
            **kwargs: Additional arguments for analyze()
            
        Returns:
            Results dictionary from analyze()
        """
        import os
        from pathlib import Path
        
        npz_path = Path(npz_path)
        
        if not npz_path.exists():
            raise FileNotFoundError(f"File not found: {npz_path}")
        
        data = np.load(npz_path, allow_pickle=True)
        patterns = data["data"]  # (n_subjects, n_posts, n_features)
        
        self.logger.info(f"\nLoaded aligned data from: {npz_path.name}")
        self.logger.info(f"  Subjects: {len(data['subjects'])}")
        self.logger.info(f"  Posts: {len(data['post_ids'])}")
        self.logger.info(f"  Features: {patterns.shape[2]}")
        
        if "run_type" in data.files:
            self.logger.info(f"  Run type: {data['run_type']}")
        
        return self.analyze(patterns, **kwargs)
    
    def save_results(
        self,
        results: dict,
        output_path: str,
    ):
        """
        Save ISC analysis results to NPZ file.
        
        Args:
            results: Results dictionary from analyze()
            output_path: Where to save results
        """
        from pathlib import Path
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez_compressed(
            output_path,
            **results
        )
        
        self.logger.info(f"Results saved to: {output_path}")