"""Console script for yy_fmri_kit."""

import typer
from rich.console import Console
from pathlib import Path
import pandas as pd
import subprocess
import glob
import argparse

from yy_fmri_kit.helpers.dicom2bids import (
    get_dicom_info,
    convert_dicom_to_bids,
    validate_bids_dataset
)
from yy_fmri_kit.helpers.fmriprep import (
    run_fmriprep_one_subject,
)

from yy_fmri_kit.helpers.denoising import run_nltools_denoising
from yy_fmri_kit.helpers.parcellation import parcellate_subject

from yy_fmri_kit.static.preproc.config import (
    HEUDICONV_CMD_TEMPLATE,
    BIDS_VALIDATOR_CMD_TEMPLATE,
    FMRIPREP_DOCKER_WRAPPER,
    FMRIPREP_DOCKER_TEMPLATE,
    FMRIPREP_DEFAULT_SPACES,
    FMRIPREP_DEFAULT_NTHREADS,
    FMRIPREP_DEFAULT_EXTRA,
    REPETITION_TIME,
    SMOOTHING_FWHM,
    SPIKE_CUTOFF,
)

app = typer.Typer()
console = Console()

@app.command("dicom2bids")
def run_dicom_to_bids_pipeline(
    dicom_root_path: Path = typer.Argument(..., exists=True, readable=True),
    heuristic_file_path: Path = typer.Argument(..., exists=True, readable=True),
    bids_root_path: Path = typer.Argument(...),
    heudiconv_template: str = typer.Option(HEUDICONV_CMD_TEMPLATE, "--heudiconv-template"),
    validator_template: str = typer.Option(BIDS_VALIDATOR_CMD_TEMPLATE, "--validator-template"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Pass --overwrite to heudiconv"),
) -> bool:
    """
    Main pipeline to convert raw DICOMs to validated BIDS.
    """
    console.print(f"\n[bold]--- Starting DICOM-to-BIDS Pipeline ---[/bold]\n")
    
    # 1. Create Manifest
    try:
        dicom_info_tsv = get_dicom_info(dicom_root_path)
        console.print(f"[green]Manifest[/green] written to: {dicom_info_tsv}")
    except Exception as e:
        console.print(f"[red]CRITICAL ERROR during manifest creation:[/red] {e}")
        raise typer.Exit(code=1)

    # 2. Conversion
    try:
        df = pd.read_csv(dicom_info_tsv, sep="\t")
        convert_dicom_to_bids(
            df=df,
            heuristic=heuristic_file_path,
            bids_path=bids_root_path,
            heudiconv_template=heudiconv_template,
            overwrite=overwrite
        )
    except subprocess.CalledProcessError:
        console.print("\n🛑 Pipeline Halted: DICOM Conversion failed. See error log above.")
        return typer.Exit(code=2)
    except Exception as e:
        console.print(f"[red]CRITICAL ERROR during conversion step:[/red] {e}")
        return typer.Exit(code=3)

    # 3. Validation
    is_valid = validate_bids_dataset(
        bids_path=bids_root_path, 
        validator_template=validator_template
    )
    
    if not is_valid:
        console.print("\n🛑 Pipeline Halted: BIDS Validation resulted in errors.")

    console.print(f"\n[bold green]--- 🎉 DICOM-to-BIDS Pipeline Complete and Validated ---[/bold green]\n")
    return True

@app.command("fmriprep")
def run_fmriprep(
    bids_root_path: Path = typer.Argument(..., exists=True, readable=True),
    derivatives_root_path: Path = typer.Argument(...),
    work_dir: Path = typer.Argument(...),
    fs_license_file: Path = typer.Argument(..., exists=True, readable=True),
    subject_label: str = typer.Argument(..., help="e.g., '1' for sub-01"),
    use_wrapper: bool = typer.Option(True, "--use-wrapper/--no-wrapper",
                                     help="Use fmriprep-docker wrapper (True) or plain Docker template (False)"),
    nthreads: int = typer.Option(FMRIPREP_DEFAULT_NTHREADS, "--nthreads"),
    space: list[str] = typer.Option(FMRIPREP_DEFAULT_SPACES, "--space"),
    extra: str = typer.Option(FMRIPREP_DEFAULT_EXTRA, "--extra",
                              help="Extra fMRIPrep flags, e.g. '--skip-bids-validation --low-mem --omp-nthreads 2'"),
) -> bool:
    """
    Run fMRIPrep for a single subject + quick sanity check.
    """
    template = FMRIPREP_DOCKER_WRAPPER if use_wrapper else FMRIPREP_DOCKER_TEMPLATE

    try:
        run_fmriprep_one_subject(
            bids_root=bids_root_path,
            derivatives_root=derivatives_root_path,
            work_dir=work_dir,
            fs_license_file=fs_license_file,
            subject_label=subject_label,
            template=template,
            spaces=space,
            nthreads=nthreads,
            extra=extra,
        )
    except subprocess.CalledProcessError as e:
        console.print("🛑 fMRIPrep failed.")
        console.print(e.stderr or "")
        raise typer.Exit(code=e.returncode or 2)


@app.command("denoise")
def run_denoising(
    fmriprep_root: Path = typer.Argument(..., exists=True, help="Path to fMRIPrep derivatives directory."),
    subject_label: str = typer.Argument(..., help="Subject label (without 'sub-')."),
    output_root: Path = typer.Option(None, "--output-root", help="Optional denoised output root directory."),
    tr: float = typer.Option(REPETITION_TIME, "--tr", help="Repetition time in seconds."),
    fwhm: float = typer.Option(SMOOTHING_FWHM, "--fwhm", help="Smoothing kernel FWHM in mm."),
    spike_cutoff: float = typer.Option(SPIKE_CUTOFF, "--spike-cutoff", help="Spike detection cutoff."),
):
    """
    Denoise **all runs** for one subject from fMRIPrep derivatives.
    """
    func_dir = fmriprep_root / f"sub-{subject_label}" / "func"
    if not func_dir.exists():
        console.print(f"[red]No func folder found for sub-{subject_label} at {func_dir}[/red]")
        raise typer.Exit(code=1)

    func_files = sorted(glob.glob(str(func_dir / f"sub-{subject_label}_*preproc*bold.nii.gz")))
    if not func_files:
        console.print(f"[yellow]No preprocessed BOLD files found for sub-{subject_label}.[/yellow]")
        raise typer.Exit(code=0)

    console.print(f"[cyan]\nFound {len(func_files)} runs for sub-{subject_label}[/cyan]")

    for f in func_files:
        try:
            run_nltools_denoising(
                func_file=f,
                confounds_file=None,
                output_dir=output_root,
                tr=tr,
                fwhm=fwhm,
                spike_cutoff=spike_cutoff,
            )
        except Exception as e:
            console.print(f"[red]⚠️ Error processing {f}: {e}[/red]")

    console.print(f"\n[green]🎉 Denoising completed for sub-{subject_label}![/green]")

@app.command("parcellate")
def run_parcellation():
    p = argparse.ArgumentParser(
        description="Run parcellation on all cleaned BOLD runs of one subject."
    )
    p.add_argument("--derivatives", required=True, type=Path,
                   help="Path to derivatives directory (e.g. /path/to/data/derivatives)")
    p.add_argument("--sub", required=True,
                   help="Subject ID (e.g. sub-1)")
    # TemplateFlow atlas parameters
    p.add_argument("--tf-template", default="MNI152NLin2009cAsym",
                   help="TemplateFlow template (default: MNI152NLin2009cAsym)")
    p.add_argument("--tf-atlas", default="Schaefer2018",
                   help="Atlas name in TemplateFlow (default: Schaefer2018)")
    p.add_argument("--tf-desc", default="400Parcels7Networks",
                   help="Atlas descriptor (e.g. 400Parcels7Networks, 200Parcels7Networks)")
    p.add_argument("--tf-resolution", type=int, default=2,
                   help="Atlas resolution (1=res-01, 2=res-02)")
    # Optional local paths
    p.add_argument("--atlas-nii", type=Path, default=None,
                   help="Local atlas NIfTI file (overrides TemplateFlow)")
    p.add_argument("--labels-file", type=Path, default=None,
                   help="Local labels TSV (overrides TemplateFlow)")
    # Processing options
    p.add_argument("--standardize", default="zscore", choices=["zscore","psc","none"],
                   help="Standardization mode (default: zscore)")
    p.add_argument("--space", default="MNI152NLin2009cAsym",
                   help="BOLD space identifier (default: MNI152NLin2009cAsym)")
    p.add_argument("--out-root", type=Path, default=None,
                   help="Output root folder (default: derivatives/parcellation)")
    args = p.parse_args()

    print(f"\n▶ Running parcellation for {args.sub} ...\n")

    results = parcellate_subject(
        derivatives_dir=args.derivatives,
        sub=args.sub,
        atlas_nii=args.atlas_nii,
        labels_file=args.labels_file,
        tf_template=args.tf_template,
        tf_atlas=args.tf_atlas,
        tf_desc=args.tf_desc,
        tf_resolution=args.tf_resolution,
        space=args.space,
        standardize=("zscore" if args.standardize=="zscore"
                     else "psc" if args.standardize=="psc"
                     else False),
        out_root=args.out_root,
    )

    print(f"\n✅ Parcellation complete for {args.sub}")
    print(f"   → {len(results)} runs processed")
    for f in results:
        print(f"     - {f.name}")
    print()


if __name__ == "__main__":
    app()
