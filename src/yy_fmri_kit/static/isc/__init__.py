"""
Inter-subject correlation (ISC) analysis.

This module provides tools for computing and analyzing ISC.

Core functions:
- compute_isc: Native leave-one-out ISC implementation
- compute_isc_brainiak: BrainIAK backend for ISC

High-level interface:
- ISCAnalyzer: Wrapper with permutation testing and statistics

Example:
    from yy_fmri_kit.isc import ISCAnalyzer
    
    analyzer = ISCAnalyzer(fisher_z=True)
    results = analyzer.analyze_from_file(
        "aligned/AntiLeft_aligned.npz",
        n_permutations=1000,
        correction="fdr_bh"
    )
    
    print(f"Significant voxels: {results['significant'].sum()}")
"""

# Import core functions
from yy_fmri_kit.isc.compute import (
    compute_isc,
    compute_isc_brainiak,
)

# Import high-level analyzer
from yy_fmri_kit.isc.analyzer import ISCAnalyzer

# Import from other modules if they exist
# (keep your existing imports like parcel, timeseries, voxel)
try:
    from yy_fmri_kit.isc.parcel import *
except ImportError:
    pass

try:
    from yy_fmri_kit.isc.timeseries import *
except ImportError:
    pass

try:
    from yy_fmri_kit.isc.voxel import *
except ImportError:
    pass

__all__ = [
    # Core functions
    "compute_isc",
    "compute_isc_brainiak",
    # High-level interface
    "ISCAnalyzer",
]