"""
Global configuration parameters and command templates used throughout the yy_fmri_kit package
"""

HEUDICONV_VERSION = "1.3.4"

# Heudiconv CMD template for convert_dicom_to_bids 
HEUDICONV_CMD_TEMPLATE = (
    'docker run --rm '
    '-v "{dicom}:/data:ro" '
    '-v "{bids}:/out" '
    '-v "{heur}:/tzlil_heuristic.py:ro" '
    f'nipy/heudiconv:{HEUDICONV_VERSION} '
    '--files /data '
    '-o /out -f /tzlil_heuristic.py -c dcm2niix -b --overwrite --grouping all '
    '-s {subject} -ss {session}'
)

# BIDS validator CMD template for convert_dicom_to_bids 
BIDS_VALIDATOR_CMD_TEMPLATE = (
    'docker run --rm '
    '-v "{bids_path}:/data:ro" '
    'bids/validator:latest '
    '/data --verbose --no-color --json'
)

# --- fMRIPrep (Docker) ---
FMRIPREP_VERSION = "25.2.0"

# fMRIPrep docker wrapper command
# IMPORTANT: omit '--participant-label {label}' to let fmriprep run multiple subjects
FMRIPREP_DOCKER_WRAPPER = (
    'fmriprep-docker '
    '{bids} {out} participant '
    '--participant-label {label} '
    '--fs-license-file {fs_license} '
    '-w {work} '
    '--output-spaces {spaces} '
    '--nthreads {nthreads} '
    '{extra}'
)

# plain fMRIPrep docker command
# IMPORTANT: omit '--participant-label {label}' to let fmriprep run multiple subjects
FMRIPREP_DOCKER_TEMPLATE = (
    'docker run --rm '
    '-v "{bids}:/data:ro" '
    '-v "{out}:/out" '
    '-v "{work}:/work" '
    '-v "{fs_license}:/opt/freesurfer/license.txt:ro" '
    f'nipreps/fmriprep:{FMRIPREP_VERSION} '
    '/data /out participant '
    '--participant-label {label} '
    '--fs-license-file /opt/freesurfer/license.txt '
    '-w /work '
    '--output-spaces {spaces} '
    '--nthreads {nthreads} '
    '{extra}'
)

# fmriprep defaults parameters
FMRIPREP_DEFAULT_SPACES = ["MNI152NLin2009cAsym:res-2", "fsaverage:den-10k"]
FMRIPREP_DEFAULT_NTHREADS = 6
FMRIPREP_DEFAULT_EXTRA = "--skip-bids-validation --low-mem --omp-nthreads 2"

# Denoising Parameters
SMOOTHING_FWHM = 2.0          # Spatial smoothing in mm
REPETITION_TIME = 2.0         # TR in seconds
SPIKE_CUTOFF = 3              # Cutoff for global signal and temporal derivative spikes