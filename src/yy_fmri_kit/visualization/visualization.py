# ======================================================
# TODO: add more visualization functions as needed
# ======================================================

import numpy as np
from pathlib import Path
from nilearn import image, plotting
import nibabel as nib
import matplotlib.pyplot as plt

def plot_single_voxel_timeseries(pre_bold, post_bold,
                                 ijk=None,
                                 tr=2.0,
                                 title=None):
    """
    Plot pre vs post time series at a single voxel index (i, j, k).

    ijk: tuple, voxel indices (i, j, k). If None, uses center voxel.
    """
    pre = image.load_img(pre_bold)
    post = image.load_img(post_bold)

    if pre.shape[:3] != post.shape[:3]:
        raise ValueError("Pre/Post shapes differ; use matching pair for voxel QC.")

    data_pre = pre.get_fdata()
    data_post = post.get_fdata()

    if ijk is None:
        ijk = (pre.shape[0] // 2, pre.shape[1] // 2, pre.shape[2] // 2)

    i, j, k = ijk
    pre_ts = data_pre[i, j, k, :]
    post_ts = data_post[i, j, k, :]

    n = min(len(pre_ts), len(post_ts))
    pre_ts = pre_ts[:n]
    post_ts = post_ts[:n]

    x = np.arange(n) * float(tr)
    if title is None:
        title = f"Voxel {ijk}: pre vs post"

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, pre_ts, label="Pre", alpha=0.7)
    ax.plot(x, post_ts, label="Post", alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Signal (a.u.)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig, ax

def plot_voxelwise_std_hist(pre_bold, post_bold, mask_img=None,
                            bins=100, log=False,
                            title="Voxelwise std: pre vs post"):
    """
    Plot histograms of voxelwise std before vs after denoising.
    """
    pre_std = voxelwise_std(pre_bold, mask_img)
    post_std = voxelwise_std(post_bold, mask_img)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pre_std, bins=bins, alpha=0.5, label="Pre", density=True)
    ax.hist(post_std, bins=bins, alpha=0.5, label="Post", density=True)

    if log:
        ax.set_yscale("log")

    ax.set_xlabel("Voxelwise std (a.u.)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig, ax

def plot_std_maps(pre_bold, post_bold,
                  vmax=None,
                  title_pre="Pre-denoising voxelwise std",
                  title_post="Post-denoising voxelwise std"):
    """
    Plot 3D std maps for pre and post denoising.
    """
    pre_std_img = compute_std_map(pre_bold)
    post_std_img = compute_std_map(post_bold)

    # If no vmax given, use pre's 95th percentile for a nice shared scale
    if vmax is None:
        vmax = np.percentile(pre_std_img.get_fdata(), 95)

    display_pre = plotting.plot_stat_map(
        pre_std_img,
        vmax=vmax,
        title=title_pre,
    )
    display_post = plotting.plot_stat_map(
        post_std_img,
        vmax=vmax,
        title=title_post,
    )
    return display_pre, display_post


# == PRIVATE HELPERS ==
def voxelwise_std(img, mask_img=None):
    """
    Compute voxelwise std across time.

    img: 4D NIfTI or path
    mask_img: 3D NIfTI or path or None
    Returns: 1D array of std values for voxels inside mask (or all nonzero voxels).
    """
    img = image.load_img(img)
    data = img.get_fdata()

    if data.ndim != 4:
        raise ValueError(f"Expected 4D image, got {data.ndim}D.")

    if mask_img is not None:
        mask_img = image.load_img(mask_img)
        mask = mask_img.get_fdata().astype(bool)
    else:
        # default: keep voxels with any signal
        mask = np.any(data != 0, axis=-1)

    # (X*Y*Z, T)
    flat = data[mask, :]
    stds = flat.std(axis=1)
    return stds

def compute_std_map(bold_img):
    """
    Return 3D NIfTI of voxelwise std across time.
    """
    bold_img = image.load_img(bold_img)
    data = bold_img.get_fdata()
    if data.ndim != 4:
        raise ValueError(f"Expected 4D image, got {data.ndim}D.")
    std_vol = data.std(axis=-1)
    return nib.Nifti1Image(std_vol, bold_img.affine, bold_img.header)


def get_matching_denoised(pre_bold_path: Path, denoised_root: Path) -> Path:
    """
    Given a specific preproc BOLD path, return its matching nltoolsClean file.
    """
    pre_bold_path = Path(pre_bold_path)
    subj = pre_bold_path.name.split("_")[0]  # 'sub-1'
    den_subdir = Path(denoised_root) / subj / "func"
    return den_subdir / pre_bold_path.name.replace(
        "_bold.nii.gz",
        "_desc-nltoolsClean_bold.nii.gz"
    )
