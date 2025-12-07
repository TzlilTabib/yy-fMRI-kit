from pathlib import Path
import re
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from yy_fmri_kit.io.find_files import build_denoised_runs_dict
from yy_fmri_kit.postproc.time_shift import find_first_hrf_peak

def plot_hrf_for_run(
    bold_path: Path,
    mask_path: Path,
    TR: float,
    onset_sec: float = 8.0,
    zscore: bool = True,
    title: str | None = None,
    save_dir: Path | None = None,
    save_name: str | None = None,
    show: bool = True,
    mark_peak: bool = True,
) -> dict | None:
    """
    Plot the mean BOLD timecourse in the auditory mask with:
      - vertical line at onset_sec
      - optional marker for the first HRF peak

    Returns peak_info dict (from find_first_hrf_peak) or None if mark_peak=False.
    """
    ts = mean_ts_in_mask(bold_path, mask_path)

    if zscore:
        ts = (ts - ts.mean()) / ts.std()

    n_tp = len(ts)
    time = np.arange(n_tp) * TR

    peak_info = None
    if mark_peak:
        peak_info = find_first_hrf_peak(
            ts=ts,
            TR=TR,
            onset_sec=onset_sec,
            search_window=(2.0, 10.0),
            min_prominence=0.3,
            zscore=False,   # already z-scored above
        )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time, ts, label="Auditory ROI BOLD")
    ax.axvline(onset_sec, linestyle="--", label=f"Stim onset ({onset_sec:.1f} s)")

    if peak_info is not None:
        ax.axvline(
            peak_info["peak_time_sec"],
            linestyle=":",
            label=f"Peak ({peak_info['peak_latency_sec']:.1f} s after onset)",
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("BOLD (z-score)" if zscore else "BOLD (a.u.)")
    if title is not None:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    # ---- Saving ----
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        if save_name is None:
            save_name = bold_path.stem + "_auditoryHRF.png"
        out_path = save_dir / save_name
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return peak_info


def plot_auditory_hrf_all_runs(
    derivatives_dir: str | Path,
    mask_path: str | Path,
    TR: float,
    denoise_folder: str = "",
    onset_sec: float = 8.0,
    subjects: list[str] | None = None,
    save_png: Path | None = None,
):
    """
    Iterate over all subjects and denoised runs in `derivatives_dir`
    and plot the auditory HRF with a vertical line at onset_sec.

    - derivatives_dir: root of your denoised derivatives (or its parent if you use denoise_folder)
    - mask_path: path to your auditory spherical mask (in same MNI space)
    - TR: repetition time in seconds
    - denoise_folder: optional subfolder under derivatives_dir (e.g., 'denoised')
    - onset_sec: when the first post starts (≈ 8 s)
    - subjects: optional list like ['sub-0002', 'sub-0003']; if None, will infer.
    - save_png: directory to save PNGs, or None to just show plots.
    """
    derivatives_dir = Path(derivatives_dir).resolve()
    mask_path = Path(mask_path).resolve()

    subject_runs = build_denoised_runs_dict(
        derivatives_dir,
        denoise_folder=denoise_folder,
        subjects=subjects,
    )

    # Make save dir once if requested
    if save_png is not None:
        save_png = Path(save_png)
        save_png.mkdir(parents=True, exist_ok=True)

    for sub, runs in subject_runs.items():
        for bold_path in runs:
            task = get_task_from_bold_path(bold_path) or "unknownTask"
            title = f"{sub} – {task}\n{bold_path.name}"
            print(f"Plotting {title}")

            # Safe filename for saving
            safe_sub = sub.replace(" ", "_")
            safe_task = task.replace(" ", "_")
            save_name = f"{safe_sub}_{safe_task}.png" if save_png is not None else None

            plot_hrf_for_run(
                bold_path=bold_path,
                mask_path=mask_path,
                TR=TR,
                onset_sec=onset_sec,
                zscore=True,
                title=title,
                save_dir=save_png,
                save_name=save_name,
                show=(save_png is None),  # if saving, don't spam windows
            )


def mean_ts_in_mask(bold_path: Path, mask_path: Path) -> np.ndarray:
    """
    Compute mean time series within a (binary) mask.
    bold_path : 4D NIfTI (X, Y, Z, T)
    mask_path : 3D NIfTI (X, Y, Z), non-zero voxels = mask
    """
    bold_img = nib.load(str(bold_path))
    bold_data = bold_img.get_fdata()          # (X, Y, Z, T)

    mask_img = nib.load(str(mask_path))
    mask_data = mask_img.get_fdata() > 0      # boolean mask

    if bold_data.shape[:3] != mask_data.shape:
        raise ValueError(
            f"Shape mismatch between BOLD {bold_data.shape[:3]} and mask {mask_data.shape}"
        )

    # Flatten spatial dims, keep time
    X, Y, Z, T = bold_data.shape
    bold_2d = bold_data.reshape(-1, T)        # (V, T)
    mask_1d = mask_data.reshape(-1)           # (V,)

    if mask_1d.sum() == 0:
        raise ValueError(f"Mask {mask_path} is empty!")

    masked_data = bold_2d[mask_1d]            # (V_mask, T)
    ts = masked_data.mean(axis=0)             # (T,)
    return ts


# ---------- Extract task name from filename ----------

_TASK_RE = re.compile(r"task-([a-zA-Z0-9]+)")

def get_task_from_bold_path(p: Path) -> str | None:
    """
    Try to infer the task label from a BOLD filename, e.g.
    'sub-01_ses-202505251228_task-ProLeft_space-..._bold.nii.gz'
    -> 'ProLeft'
    """
    m = _TASK_RE.search(p.name)
    if m:
        return m.group(1)
    return None