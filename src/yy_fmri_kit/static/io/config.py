"""
Global configuration parameters and command templates used throughout the yy_fmri_kit.io module.
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
