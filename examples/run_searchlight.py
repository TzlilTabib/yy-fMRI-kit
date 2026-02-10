"""
Full Searchlight Pipeline: All Subjects
========================================

This script runs the complete searchlight ISC analysis:
1. Extract searchlight patterns from all subjects
2. Align patterns across subjects
3. Compute ISC for each searchlight sphere
4. Statistical testing with permutations

Date: 2026-02-09
"""

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import importlib
from yy_fmri_kit import event_isc, isc
importlib.reload(event_isc)
importlib.reload(isc)
# Import from yy-fMRI-kit
from yy_fmri_kit.event_isc.alignment import PatternAligner
from yy_fmri_kit.event_isc.extraction import SearchlightPatternExtractor
from yy_fmri_kit.static.event_isc.config import (
    ExtractionConfig,
    AlignmentConfig,
)
from yy_fmri_kit.event_isc.extraction import (SearchlightPatternExtractor)
from yy_fmri_kit.event_isc.alignment import (PatternAligner)
from yy_fmri_kit.isc.analyzer import ISCAnalyzer

# ============================================================================
# CONFIGURATION - CHANGE THESE TO MATCH YOUR DATA
# ============================================================================

# Data paths
DATA_DIR = Path("/path/to/data/derivatives/denoised")
EVENTS_CSV = Path("/path/to/behavioral_analyses/data/combined_events_with_bids.csv")
# Output directory
OUTPUT_DIR = Path("/path/to/data/derivatives/searchlight")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Experiment parameters
TR = 1.0              # Your TR in seconds
SHIFT_TR = 4          # Hemodynamic delay in TRs
SEARCHLIGHT_RADIUS = 5.0  # Radius in mm (6-9mm typical)

# Run types to analyze
RUN_TYPES = ["AntiLeft", "AntiRight", "ProLeft", "ProRight"]
TEST_SUBJECTS = [
    'sub-1', 'sub-6', 'sub-20', 'sub-21', 'sub-22', 'sub-23', 
    'sub-24', 'sub-25', 'sub-26', 'sub-27', 'sub-30', 'sub-31'
]

# ============================================================================
# OPTIONAL: BUILD GROUP MASK
# ============================================================================
from nilearn import image

print("Creating group mask...")

# Find all preprocessed functional files for your test subjects
functional_files = []
for sub in TEST_SUBJECTS:
    subject_dir = DATA_DIR / sub
    # Find any task functional file
    task_files = list(subject_dir.rglob("*task-*_bold.nii.gz"))
    if task_files:
        functional_files.append(task_files[0])
        print(f"  {subject_dir.name}: {task_files[0].name}")

if len(functional_files) < 2:
    print("Not enough functional files found!")
else:
    print(f"\nCreating masks from {len(functional_files)} subjects...")
    
    # Load all images and create binary masks
    masks = []
    for i, func_file in enumerate(functional_files):
        print(f"  Processing subject {i+1}/{len(functional_files)}...", end="")
        
        # Load image
        img = image.load_img(str(func_file))
        data = img.get_fdata()
        
        # Create binary mask (voxels with signal across time)
        # Take mean across time, threshold at small positive value
        mean_data = np.mean(data, axis=-1)
        mask_data = (mean_data > 0).astype(np.float32)
        
        # Create mask image
        mask_img = image.new_img_like(img, mask_data, copy_header=True)
        masks.append(mask_img)
        
        n_voxels = int(mask_data.sum())
        print(f" {n_voxels:,} voxels")
    
    # Compute intersection manually (more reliable)
    print("\nComputing intersection...")
    
    # Stack all mask data
    mask_arrays = [m.get_fdata() for m in masks]
    mask_stack = np.stack(mask_arrays, axis=0)
    
    # Intersection = voxel is in ALL subjects (all values > 0)
    intersection = np.all(mask_stack > 0, axis=0).astype(np.float32)
    
    # Create final mask image
    group_mask = image.new_img_like(masks[0], intersection, copy_header=True)
    
    # Save group mask
    mask_file = OUTPUT_DIR / "group_mask.nii.gz"
    group_mask.to_filename(str(mask_file))
    
    # Count voxels
    n_voxels_group = int(intersection.sum())
    
    print(f"\n✓ Created group mask: {mask_file}")
    print(f"  Voxels in group mask: {n_voxels_group:,}")
    print(f"  Subjects included: {len(functional_files)}")
    
    # Show coverage per subject
    print(f"\nCoverage per subject:")
    for i, mask_arr in enumerate(mask_arrays):
        n_vox_subj = int(mask_arr.sum())
        pct_coverage = 100 * n_voxels_group / n_vox_subj if n_vox_subj > 0 else 0
        print(f"  Subject {i+1}: {n_vox_subj:,} voxels → {pct_coverage:.1f}% retained in group mask")
    
    # Verify mask is valid
    if n_voxels_group == 0:
        print("\n⚠️  WARNING: Group mask is empty!")
        print("  Possible issues:")
        print("  1. Different brain templates/spaces")
        print("  2. Very different brain coverage")
        print("  3. Check that functional files are in same space")
    else:
        print(f"\n✓ Group mask is valid with {n_voxels_group:,} voxels")
# ============================================================================

MASK_FILE = mask_file  # Path to group mask, or None for whole brain

# ISC parameters
N_PERMUTATIONS = 1000  # For statistical testing (1000+ recommended)
ALPHA = 0.05
CORRECTION = "fdr_bh"  # "fdr_bh", "bonferroni", or "none"

# Processing
N_JOBS = -1  # Number of parallel jobs (-1 = all CPUs)

print("="*80)
print("SEARCHLIGHT ISC PIPELINE")
print("="*80)
print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nConfiguration:")
print(f"  Data directory: {DATA_DIR}")
print(f"  Events file: {EVENTS_CSV}")
print(f"  Output directory: {OUTPUT_DIR}")
print(f"  TR: {TR}s")
print(f"  Shift: {SHIFT_TR} TRs ({SHIFT_TR * TR}s)")
print(f"  Searchlight radius: {SEARCHLIGHT_RADIUS}mm")
print(f"  Run types: {RUN_TYPES}")
print(f"  Permutations: {N_PERMUTATIONS}")
print(f"  Parallel jobs: {N_JOBS}")

# ============================================================================
# STEP 0: LOAD DATA AND BUILD RUNS DICTIONARY
# ============================================================================

print("\n" + "="*80)
print("STEP 0: LOADING DATA")
print("="*80)

# Load events
print(f"\nLoading events from: {EVENTS_CSV}")
events_df = pd.read_csv(EVENTS_CSV)
print(f"✓ Loaded {len(events_df)} events")
print(f"  Subjects: {events_df['subject'].nunique()}")
print(f"  Run types: {events_df['run_type'].unique().tolist() if 'run_type' in events_df.columns else 'N/A'}")

# Build runs dictionary
print("\nBuilding runs dictionary...")
runs_dict = {}

# Find subject directories
subject_dirs = sorted([d for d in DATA_DIR.glob("sub-*") if d.is_dir()])
print(f"Found {len(subject_dirs)} subject directories")

for sub in TEST_SUBJECTS:  # Limit to first 3 subjects for testing
    subject_dir = DATA_DIR / sub
    
    # Find all NIfTI files for this subject
    nifti_files = list(subject_dir.rglob("*_bold.nii.gz"))
    
    # Filter for task runs (adjust pattern to match your files)
    task_files = [
        f for f in nifti_files 
        if any(task in f.name.lower() for task in ['antileft', 'antiright', 'proleft', 'proright'])
    ]
    
    if task_files:
        runs_dict[sub] = task_files
        print(f"  {sub}: {len(task_files)} runs")

print(f"\n✓ Built runs dictionary with {len(runs_dict)} subjects")
total_runs = sum(len(files) for files in runs_dict.values())
print(f"  Total runs: {total_runs}")

# ============================================================================
# STEP 1: EXTRACT SEARCHLIGHT PATTERNS
# ============================================================================

print("\n" + "="*80)
print("STEP 1: EXTRACTING SEARCHLIGHT PATTERNS")
print("="*80)

# Configure extraction
extraction_config = ExtractionConfig(
    tr=TR,
    shift_tr=SHIFT_TR,
    time_unit="seconds",
    smoothing_fwhm=None,  # No smoothing for searchlight (done at searchlight level)
    detrend=True,
    standardize=False,    # ISC computation handles standardization
    searchlight_radius=SEARCHLIGHT_RADIUS,
    searchlight_n_jobs=N_JOBS,
    valid_run_types=RUN_TYPES,
    # IMPORTANT: Set your actual column names here
    post_col="post_id",  # ← CHANGE to your post ID column
    subject_col="bids_id",
    run_col="run",
)

print("\nExtraction configuration:")
print(f"  TR: {extraction_config.tr}s")
print(f"  Hemodynamic shift: {extraction_config.shift_tr} TRs")
print(f"  Searchlight radius: {extraction_config.searchlight_radius}mm")
print(f"  Post column: '{extraction_config.post_col}'")
print(f"  Subject column: '{extraction_config.subject_col}'")
print(f"  Run column: '{extraction_config.run_col}'")

# Create extractor
extractor = SearchlightPatternExtractor(extraction_config)

# Output directory for extracted patterns
extraction_output = OUTPUT_DIR / "searchlight_patterns"

print(f"\nExtracting patterns...")
print(f"  Output: {extraction_output}")
print(f"  This may take a while (searchlight is slower than ROI)...")

# Run batch extraction
summary = extractor.batch_extract(
    runs_dict=runs_dict,
    events_df=events_df,
    output_dir=extraction_output,
    mask_path=MASK_FILE,
    verbose=False,  # Set True for detailed logging
)

print("\n✓ Extraction complete!")
print("\nExtraction summary:")
print(summary.groupby('status').size())

# Save summary
summary_file = OUTPUT_DIR / "extraction_summary.csv"
summary.to_csv(summary_file, index=False)
print(f"\nSaved summary to: {summary_file}")

# Show successful extractions
success = summary[summary['status'] == 'success']
if len(success) > 0:
    print(f"\nSuccessful extractions:")
    print(f"  Runs: {len(success)}")
    print(f"  Mean posts/run: {success['n_posts'].mean():.1f}")
    print(f"  Mean searchlights: {success['n_searchlights'].mean():.0f}")

# Show failures
failures = summary[summary['status'] != 'success']
if len(failures) > 0:
    print(f"\n⚠️  {len(failures)} runs failed:")
    print(failures[['subject', 'run_type', 'status']])

# ============================================================================
# STEP 2: ALIGN PATTERNS ACROSS SUBJECTS
# ============================================================================

print("\n" + "="*80)
print("STEP 2: ALIGNING PATTERNS ACROSS SUBJECTS")
print("="*80)

# Configure alignment
alignment_config = AlignmentConfig(
    strategy="union",  # Only posts seen by ALL subjects
    min_subjects=2,
    allow_missing_posts=True,
    check_feature_dims=True,
)

print("\nAlignment configuration:")
print(f"  Strategy: {alignment_config.strategy}")
print(f"  Min subjects: {alignment_config.min_subjects}")

# Create aligner
aligner = PatternAligner(alignment_config)

# Align all run types
print("\nAligning run types...")
aligned_results = aligner.align_all_runs(
    output_dir=extraction_output,
    run_types=RUN_TYPES,
    pattern_type="searchlight"
)

print(f"\n✓ Aligned {len(aligned_results)} run types")

# Save aligned data
aligned_dir = OUTPUT_DIR / "aligned"
aligned_dir.mkdir(exist_ok=True, parents=True)

for run_type, result in aligned_results.items():
    output_file = aligned_dir / f"{run_type}_searchlight_aligned.npz"
    aligner.save_aligned(result, output_file, run_type)
    
    print(f"\n{run_type}:")
    print(f"  Subjects: {len(result['subjects'])}")
    print(f"  Posts: {len(result['post_ids'])}")
    print(f"  Searchlights: {result['data'][0].shape[1]}")
    print(f"  Shape per subject: {result['data'][0].shape}")
    print(f"  Saved: {output_file.name}")

# ============================================================================
# STEP 3: COMPUTE ISC FOR EACH RUN TYPE
# ============================================================================

print("\n" + "="*80)
print("STEP 3: COMPUTING ISC WITH PERMUTATION TESTING")
print("="*80)

# Configure ISC analysis
analyzer = ISCAnalyzer(
    backend="native",  # Use your compute_isc
    fisher_z=True,     # Recommended for inference
    nan_policy="omit", # Handle missing data
)

print("\nISC configuration:")
print(f"  Backend: {analyzer.backend}")
print(f"  Fisher z: {analyzer.fisher_z}")
print(f"  Permutations: {N_PERMUTATIONS}")
print(f"  Alpha: {ALPHA}")
print(f"  Correction: {CORRECTION}")

# Analyze each run type
isc_results = {}
isc_output = OUTPUT_DIR / "isc_results"
isc_output.mkdir(exist_ok=True, parents=True)

for run_type in aligned_results.keys():
    print(f"\n{'#'*80}")
    print(f"Analyzing: {run_type}")
    print(f"{'#'*80}")
    
    # Load aligned data
    aligned_file = aligned_dir / f"{run_type}_searchlight_aligned.npz"
    
    # Run ISC analysis
    results = analyzer.analyze_from_file(
        aligned_file,
        n_permutations=N_PERMUTATIONS,
        alpha=ALPHA,
        correction=CORRECTION,
        random_seed=42,
    )
    
    isc_results[run_type] = results
    
    # Save results
    results_file = isc_output / f"{run_type}_searchlight_isc.npz"
    analyzer.save_results(results, results_file)
    print(f"\nSaved results to: {results_file}")

# ============================================================================
# STEP 4: SUMMARY AND COMPARISON
# ============================================================================

print("\n" + "="*80)
print("STEP 4: SUMMARY")
print("="*80)

# Compare across run types
comparison_data = []
for run_type, results in isc_results.items():
    comparison_data.append({
        'run_type': run_type,
        'n_subjects': results['isc_subjectwise'].shape[0],
        'n_posts': results['isc_subjectwise'].shape[1] if results['isc_subjectwise'].ndim > 2 else 1,
        'n_searchlights': len(results['isc_mean']),
        'mean_isc': results['isc_mean'].mean(),
        'std_isc': results['isc_mean'].std(),
        'n_significant': results['n_significant'],
        'pct_significant': 100 * results['n_significant'] / len(results['significant']),
        'mean_isc_sig': results['isc_mean'][results['significant']].mean() if results['n_significant'] > 0 else np.nan,
    })

comparison_df = pd.DataFrame(comparison_data)

print("\n" + "="*80)
print("ISC COMPARISON ACROSS RUN TYPES")
print("="*80)
print(comparison_df.to_string(index=False))

# Save comparison
comparison_file = OUTPUT_DIR / 'isc_comparison.csv'
comparison_df.to_csv(comparison_file, index=False)
print(f"\n✓ Saved comparison to: {comparison_file}")

# ============================================================================
# STEP 5: BRAIN MAP RECONSTRUCTION
# ============================================================================
from nilearn import image

for run_type, results in isc_results.items():
    # Load original searchlight centers and affine from one of the extraction files
    # to map the 1D results back into a 3D NIfTI volume
    sample_sub_file = list((OUTPUT_DIR / "searchlight_patterns").rglob("*.npz"))[0]
    with np.load(sample_sub_file, allow_pickle=True) as meta:
        affine = meta['affine']
        centers = meta['searchlight_centers']
    
    # Initialize an empty 3D volume (zeros) using the group mask shape
    # Assuming group_mask was created in your script earlier
    isc_map_data = np.zeros(group_mask.shape)
    
    # Fill the volume with ISC values at the center of each searchlight
    # centers are [x, y, z] indices
    for i in range(len(results['isc_mean'])):
        x, y, z = centers[i].astype(int)
        isc_map_data[x, y, z] = results['isc_mean'][i]
    
    # Convert to Nifti
    isc_nii = image.new_img_like(group_mask, isc_map_data)
    isc_nii_file = OUTPUT_DIR / f"{run_type}_isc_map.nii.gz"
    isc_nii.to_filename(str(isc_nii_file))
    print(f"✓ Created 3D Brain Map: {isc_nii_file.name}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("PIPELINE COMPLETE!")
print("="*80)

print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print(f"\nOutput files:")
print(f"  Extracted patterns: {extraction_output}/")
print(f"  Aligned data: {aligned_dir}/")
print(f"  ISC results: {isc_output}/")
print(f"  Summary: {OUTPUT_DIR}/isc_comparison.csv")

print("\n" + "="*80)
print("WHAT YOU GOT:")
print("="*80)
print("""
For each run type, you now have:

1. Searchlight Patterns (extraction_output/):
   - One NPZ file per subject × run
   - Contains: post_ids, patterns (n_posts × n_searchlights)
   - Each searchlight = local pattern of voxels within radius

2. Aligned Data (aligned/):
   - One NPZ file per run type
   - Shape: (n_subjects, n_posts, n_searchlights)
   - All subjects aligned to same post order

3. ISC Results (isc_results/):
   - isc_mean: (n_searchlights,) - mean ISC per searchlight
   - isc_subjectwise: (n_subjects, n_searchlights) - ISC per subject
   - p_values: (n_searchlights,) - statistical significance
   - significant: (n_searchlights,) - boolean mask
   - null_distribution: (n_permutations, n_searchlights)

4. Comparison (isc_comparison.csv):
   - Summary statistics across run types
""")

print("\n" + "="*80)
print("NEXT STEPS:")
print("="*80)
print("""
1. Examine isc_comparison.csv to compare run types
2. Load ISC results to create brain maps:
   - Use searchlight centers to map back to 3D brain
   - Visualize with nilearn.plotting
3. Identify searchlights with highest ISC
4. Compare political conditions (Anti vs Pro, Left vs Right)
5. Correlate with behavioral measures
""")

print(f"\n{'='*80}\n")
