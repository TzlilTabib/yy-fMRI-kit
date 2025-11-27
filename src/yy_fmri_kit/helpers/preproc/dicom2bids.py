import subprocess
from pathlib import Path
import pandas as pd
from yy_fmri_kit.static.preproc.config import HEUDICONV_CMD_TEMPLATE, BIDS_VALIDATOR_CMD_TEMPLATE

# -------------------------------------- 1 --------------------------------------
def get_dicom_info(
    dicom_path: Path | str,
):
    """
    Generate a DataFrame containing DICOM file information.
    Parameters
    ----------
    dicom_path : Path | str
        Path to the root directory containing DICOM files.
    Returns
    -------
    pd.DataFrame
        DataFrame containing subject codes, DICOM paths, and session IDs.
    """
    df = pd.DataFrame(columns=["subject_code", "dicom_path", "session_id"])
    manifest_dir = Path(dicom_path).parent / "manifest"
    for i, ses in enumerate(dicom_path.iterdir()):
        session_id = "".join(ses.name.split("_")[-2:])
        subject_code = str(i + 1).zfill(3)
        df.loc[len(df)] = [subject_code, str(ses), session_id]
    df.to_csv(manifest_dir / "dicom_info.tsv", sep="\t", index=False)
    return (manifest_dir / "dicom_info.tsv")

# -------------------------------------- 2 --------------------------------------
def convert_dicom_to_bids(
    df: pd.DataFrame,
    heuristic: Path | str,
    bids_path: Path | str,
    heudiconv_template: str = HEUDICONV_CMD_TEMPLATE,
    overwrite: bool = False,
):  
    """
    Convert DICOM files to BIDS format using Heudiconv.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing DICOM file information.
    heudiconv_cmd_template : str
        Template for the Heudiconv command.
    heuristic : Path | str
        Heuristic file for the conversion.
    bids_path : Path | str
        Output directory for BIDS files.
    overwrite : bool, optional
        If True, overwrite existing BIDS files, by default False
    """
    if isinstance(heudiconv_template, (str, Path)) and Path(heudiconv_template).exists():
        template_str = Path(heudiconv_template).read_text().strip()
    else:
        template_str = str(heudiconv_template)
    if not Path(heuristic).exists():
        raise FileNotFoundError(f"Heuristic file not found: {heuristic}")
    print(df.columns)
    
    print(f"Starting HeuDiConv for {len(df)} subjects...")
    for _, row in df.iterrows():
        subject_code = row["subject_code"]
        dicom_path = Path(row["dicom_path"])
        session = row["session_id"]
        cmd = heudiconv_template.format(
            dicom=dicom_path,
            bids=bids_path,
            subject=subject_code,
            session=session,
            heur=heuristic,
        )
        if overwrite:
            cmd += " --overwrite"
        print(f"Running: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
    
# -------------------------------------- 3 --------------------------------------
def validate_bids_dataset(
    bids_path: Path | str,
    validator_template: str
) -> bool:
    """
    Runs the BIDS Validator on the specified directory using a Docker container.
    
    Parameters
    ----------
    bids_path : Path | str
        Output directory for BIDS files.
    """

    bids_path = str(Path(bids_path).resolve())
    print("\n--- Running BIDS Validator ---")

    cmd = validator_template.format(bids_path=bids_path)

    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            check=True,
            capture_output=True,
            text=True
        )
        print("BIDS Validator Output:\n", result.stdout)
        print("✅ BIDS Validation Passed!")
        return True
        
    except subprocess.CalledProcessError as e:
        print("❌ BIDS Validation FAILED! Please check the output for errors.")
        print("\nValidator STDOUT:\n", e.stdout)
        print("\nValidator STDERR:\n", e.stderr)
        print("\nValidator Command Run:\n", e.cmd)
        return False
    except FileNotFoundError:
        print("❌ ERROR: Docker command or BIDS Validator image not found. Is Docker running?")
        return False