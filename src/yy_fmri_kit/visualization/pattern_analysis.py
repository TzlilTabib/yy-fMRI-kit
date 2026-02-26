"""
visuzalization.pattern_analysis.py
==================
Visualisation functions for parcellated ISC and RSA results.

Functions
---------
plot_rdm               — single RDM heatmap for one subject / condition
plot_rdm_comparison    — side-by-side RDMs across subjects or conditions
plot_rsa_bar           — per-condition RSA/ISC bar chart (group mean ± SEM)
plot_isc_parcels       — horizontal bar chart of top N significant parcels
plot_network_summary   — aggregate ISC/RSA by Schaefer network
plot_null_distribution — observed stat vs permutation null for one parcel
plot_subject_isc       — per-subject ISC in a chosen parcel (like Chen Fig 2c)

All functions return the matplotlib Figure so you can save them:
    fig = plot_rdm(...)
    fig.savefig("rdm.png", dpi=150, bbox_inches="tight")
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from scipy.spatial.distance import pdist, squareform
from scipy.stats import sem
from nilearn import datasets, surface, plotting
from nilearn import datasets as nl_datasets, surface, plotting, image
import nibabel as nib
from pathlib import Path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _post_patterns_to_rdm(post_patterns: dict, subject_idx: int) -> np.ndarray:
    """
    Build a post × post RDM (1 - Pearson r) for one subject from post_patterns.

    post_patterns : {post_id: (n_subjects, n_parcels)}
    subject_idx   : which subject row to use
    """
    post_ids = sorted(post_patterns.keys())
    mat = np.stack([post_patterns[p][subject_idx] for p in post_ids])  # (n_posts, n_parcels)
    rdm = squareform(pdist(mat, metric="correlation"))
    return rdm  # (n_posts, n_posts), values in [0, 2]


def _extract_network(parcel_name: str) -> str:
    """Extract Schaefer 7-network label from parcel name."""
    # e.g. '7Networks_LH_Default_PCC_1' → 'Default'
    parts = parcel_name.split("_")
    if len(parts) >= 3:
        return parts[2]
    return "Unknown"


# ---------------------------------------------------------------------------
# 1. Single RDM
# ---------------------------------------------------------------------------

def plot_rdm(
    post_patterns : dict,
    subject_idx   : int = 0,
    subject_label : str = "Subject 1",
    run_type      : str = "",
    post_ids      : Optional[List] = None,
    cmap          : str = "RdBu_r",
    vmin          : float = 0.0,
    vmax          : float = 2.0,
    ax            : Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot a single post × post RDM (1 - Pearson r) for one subject.

    Parameters
    ----------
    post_patterns : {post_id: (n_subjects, n_parcels)}  from extract_post_patterns()
    subject_idx   : which subject to plot (row index)
    subject_label : label shown in the title
    run_type      : condition name for the title
    post_ids      : optional list of post IDs for axis labels (auto if None)
    cmap          : colormap
    vmin, vmax    : color scale limits (0–2 for correlation distance)
    ax            : existing Axes to draw into (creates figure if None)

    Returns
    -------
    matplotlib Figure

    Example
    -------
    fig = plot_rdm(patterns['AntiLeft'], subject_idx=0, run_type='AntiLeft')
    """
    rdm      = _post_patterns_to_rdm(post_patterns, subject_idx)
    ids      = post_ids or sorted(post_patterns.keys())
    n_posts  = len(ids)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    im = ax.imshow(rdm, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label="1 − r  (dissimilarity)")

    tick_step = max(1, n_posts // 10)
    ticks = range(0, n_posts, tick_step)
    ax.set_xticks(list(ticks))
    ax.set_yticks(list(ticks))
    ax.set_xticklabels([ids[i] for i in ticks], rotation=90, fontsize=7)
    ax.set_yticklabels([ids[i] for i in ticks], fontsize=7)
    ax.set_xlabel("Post ID")
    ax.set_ylabel("Post ID")
    ax.set_title(f"RDM — {run_type}  |  {subject_label}", fontsize=10)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. RDM comparison across subjects or conditions
# ---------------------------------------------------------------------------

def plot_rdm_comparison(
    post_patterns_dict : Dict[str, dict],
    subject_idx        : int = 0,
    subject_label      : str = "Subject 1",
    cmap               : str = "RdBu_r",
    vmin               : float = 0.0,
    vmax               : float = 2.0,
) -> plt.Figure:
    """
    Plot one RDM per condition side by side for one subject.

    Parameters
    ----------
    post_patterns_dict : {run_type: post_patterns}  e.g. the full `patterns` dict
    subject_idx        : which subject to plot
    subject_label      : label for the suptitle

    Returns
    -------
    matplotlib Figure

    Example
    -------
    fig = plot_rdm_comparison(patterns, subject_idx=0)
    """
    run_types = list(post_patterns_dict.keys())
    n         = len(run_types)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, run_type in zip(axes, run_types):
        plot_rdm(
            post_patterns_dict[run_type],
            subject_idx   = subject_idx,
            subject_label = "",
            run_type      = run_type,
            cmap          = cmap,
            vmin          = vmin,
            vmax          = vmax,
            ax            = ax,
        )

    fig.suptitle(f"RDMs — {subject_label}", fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Group-mean RDM per condition
# ---------------------------------------------------------------------------

def plot_group_rdm(
    post_patterns_dict : Dict[str, dict],
    cmap               : str = "RdBu_r",
    vmin               : float = 0.0,
    vmax               : float = 2.0,
) -> plt.Figure:
    """
    Plot the group-average RDM (mean across subjects) for each condition.

    Example
    -------
    fig = plot_group_rdm(patterns)
    """
    run_types = list(post_patterns_dict.keys())
    n         = len(run_types)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, run_type in zip(axes, run_types):
        pp       = post_patterns_dict[run_type]
        post_ids = sorted(pp.keys())
        n_subs   = next(iter(pp.values())).shape[0]

        # Average RDM across subjects
        rdms = np.stack([
            _post_patterns_to_rdm(pp, s) for s in range(n_subs)
        ])
        mean_rdm = rdms.mean(axis=0)

        im = ax.imshow(mean_rdm, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax, label="1 − r")

        n_posts   = len(post_ids)
        tick_step = max(1, n_posts // 10)
        ticks     = list(range(0, n_posts, tick_step))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([post_ids[i] for i in ticks], rotation=90, fontsize=7)
        ax.set_yticklabels([post_ids[i] for i in ticks], fontsize=7)
        ax.set_title(f"Group RDM — {run_type}", fontsize=10)
        ax.set_xlabel("Post ID")
        ax.set_ylabel("Post ID")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. ISC / RSA bar chart across conditions
# ---------------------------------------------------------------------------

def plot_condition_bar(
    results_dict  : Dict[str, pd.DataFrame],
    stat_col      : str = "r",
    title         : str = "ISC across conditions",
    ylabel        : str = "Mean ISC (r)",
    color_map     : Optional[Dict[str, str]] = None,
) -> plt.Figure:
    """
    Bar chart of mean ± SEM of the stat across parcels, one bar per condition.

    Parameters
    ----------
    results_dict : {run_type: DataFrame}  output of results_to_dataframe()
    stat_col     : which column to plot (default 'r')
    title        : figure title
    ylabel       : y-axis label
    color_map    : optional {run_type: color} dict

    Returns
    -------
    matplotlib Figure

    Example
    -------
    fig = plot_condition_bar(all_isc, title="ISC across conditions")
    fig = plot_condition_bar(all_rsa, title="RSA across conditions", ylabel="RSA r")
    """
    default_colors = {
        "AntiLeft"  : "#d62728",
        "AntiRight" : "#ff7f0e",
        "ProLeft"   : "#1f77b4",
        "ProRight"  : "#2ca02c",
    }
    colors = color_map or default_colors

    run_types = list(results_dict.keys())
    means     = [results_dict[rt][stat_col].mean() for rt in run_types]
    sems      = [sem(results_dict[rt][stat_col].dropna()) for rt in run_types]
    clrs      = [colors.get(rt, "#888888") for rt in run_types]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(run_types, means, yerr=sems, color=clrs,
                  capsize=5, edgecolor="k", linewidth=0.7, alpha=0.85)
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticklabels(run_types, rotation=15, ha="right")

    # Annotate n significant parcels
    for bar, rt in zip(bars, run_types):
        n_sig = results_dict[rt]["significant"].sum()
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + sems[run_types.index(rt)] + 0.002,
                f"n={n_sig}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Top significant parcels — horizontal bar chart
# ---------------------------------------------------------------------------

def plot_isc_parcels(
    df      : pd.DataFrame,
    run_type: str = "",
    top_n   : int = 20,
    sig_only: bool = True,
    color   : str = "#1f77b4",
) -> plt.Figure:
    """
    Horizontal bar chart of top N parcels ranked by ISC/RSA r.

    Parameters
    ----------
    df       : results DataFrame from results_to_dataframe()
    run_type : label for the title
    top_n    : number of parcels to show
    sig_only : if True, show only FDR-significant parcels

    Returns
    -------
    matplotlib Figure

    Example
    -------
    fig = plot_isc_parcels(all_isc['AntiLeft'], run_type='AntiLeft', top_n=20)
    """
    plot_df = df[df["significant"] == 1].copy() if sig_only else df.copy()
    plot_df = plot_df.nlargest(top_n, "r")

    if plot_df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No significant parcels", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(f"{run_type} — no significant parcels")
        return fig

    fig, ax = plt.subplots(figsize=(7, max(3, len(plot_df) * 0.35)))
    ax.barh(plot_df["parcel"], plot_df["r"],
            color=color, edgecolor="k", linewidth=0.5, alpha=0.85)
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_xlabel("ISC r")
    ax.set_title(f"Top parcels — {run_type}"
                 + (" (FDR significant)" if sig_only else ""))
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Network-level summary
# ---------------------------------------------------------------------------

def plot_network_summary(
    results_dict : Dict[str, pd.DataFrame],
    stat_col     : str = "r",
    title        : str = "ISC by network",
    ylabel        : str = "Mean ISC (r)",
) -> plt.Figure:
    """
    Aggregate ISC/RSA by Schaefer 7-network and plot as grouped bars
    (one group per network, one bar per condition).

    Parameters
    ----------
    results_dict : {run_type: DataFrame}  must have a 'parcel' column
    stat_col     : column to aggregate
    title, ylabel: axis labels

    Returns
    -------
    matplotlib Figure

    Example
    -------
    fig = plot_network_summary(all_isc, title="ISC by network")
    """
    # Build tidy dataframe: parcel, network, run_type, r
    rows = []
    for run_type, df in results_dict.items():
        tmp = df[["parcel", stat_col]].copy()
        tmp["network"]  = tmp["parcel"].apply(_extract_network)
        tmp["run_type"] = run_type
        rows.append(tmp)
    tidy = pd.concat(rows, ignore_index=True)

    networks  = sorted(tidy["network"].unique())
    run_types = list(results_dict.keys())
    x         = np.arange(len(networks))
    width     = 0.8 / len(run_types)

    default_colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
    fig, ax = plt.subplots(figsize=(max(8, len(networks) * 1.2), 5))

    for i, run_type in enumerate(run_types):
        sub   = tidy[tidy["run_type"] == run_type]
        means = [sub[sub["network"] == net][stat_col].mean() for net in networks]
        sems_ = [sem(sub[sub["network"] == net][stat_col].dropna()) for net in networks]
        offset = (i - len(run_types) / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=sems_, label=run_type,
               color=default_colors[i % len(default_colors)],
               capsize=3, edgecolor="k", linewidth=0.5, alpha=0.85)

    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(networks, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Condition", bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. Null distribution for one parcel
# ---------------------------------------------------------------------------

def plot_null_distribution(
    null_dist    : np.ndarray,
    obs          : np.ndarray,
    parcel_idx   : int,
    parcel_name  : str = "",
    run_type     : str = "",
    p_val        : Optional[float] = None,
) -> plt.Figure:
    """
    Histogram of the permutation null distribution for one parcel,
    with the observed statistic marked.

    Parameters
    ----------
    null_dist  : (n_perms, n_parcels)  from permutation_test()
    obs        : (n_parcels,)          observed statistic
    parcel_idx : which parcel to plot
    parcel_name: label for the title
    run_type   : condition label
    p_val      : optional p-value to show in the title

    Returns
    -------
    matplotlib Figure

    Example
    -------
    # Find the index of a parcel of interest
    pcc_idx = parcel_names.index('7Networks_LH_Default_PCC_1')
    fig = plot_null_distribution(null_isc, obs_isc, pcc_idx,
                                  parcel_name='LH Default PCC 1',
                                  run_type='AntiLeft', p_val=p_isc[pcc_idx])
    """
    null = null_dist[:, parcel_idx]
    obs_val = obs[parcel_idx]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(null, bins=40, color="#aec7e8", edgecolor="white",
            linewidth=0.4, label="Null distribution")
    ax.axvline(obs_val, color="#d62728", linewidth=2,
               label=f"Observed r = {obs_val:.3f}")
    ax.set_xlabel("ISC r")
    ax.set_ylabel("Count")

    p_str = f"  p = {p_val:.3f}" if p_val is not None else ""
    ax.set_title(f"{parcel_name}  |  {run_type}{p_str}", fontsize=10)
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 8. Per-subject ISC in one parcel (Chen et al. Fig 2c style)
# ---------------------------------------------------------------------------

def plot_subject_isc(
    isc_subj    : np.ndarray,
    parcel_idx  : int,
    subjects    : List[str],
    parcel_name : str = "",
    run_type    : str = "",
) -> plt.Figure:
    """
    Bar plot of per-subject ISC values in one parcel of interest.

    Parameters
    ----------
    isc_subj   : (n_subjects, n_parcels) from compute_isc()
    parcel_idx : column index for the parcel of interest
    subjects   : list of subject labels
    parcel_name: parcel label for the title
    run_type   : condition label

    Returns
    -------
    matplotlib Figure

    Example
    -------
    pcc_idx = parcel_names.index('7Networks_LH_Default_PCC_1')
    fig = plot_subject_isc(isc_subj, pcc_idx, cfg.subjects,
                            parcel_name='LH Default PCC 1',
                            run_type='AntiLeft')
    """
    vals   = isc_subj[:, parcel_idx]
    colors = ["#d62728" if v >= 0 else "#1f77b4" for v in vals]

    fig, ax = plt.subplots(figsize=(max(5, len(subjects) * 0.6), 4))
    ax.bar(range(len(subjects)), vals, color=colors,
           edgecolor="k", linewidth=0.5, alpha=0.85)
    ax.axhline(0, color="k", linewidth=0.8)

    # Group mean ± SEM
    group_mean = np.nanmean(vals)
    group_sem  = sem(vals[~np.isnan(vals)])
    ax.errorbar(len(subjects) - 0.5, group_mean, yerr=group_sem,
                fmt="o", color="k", markersize=6, capsize=4,
                label=f"Group mean = {group_mean:.3f}")

    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels([s.replace("sub-", "S") for s in subjects],
                        rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("ISC r")
    ax.set_title(f"{parcel_name}  |  {run_type}", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 9. RSA scatter: subject i RDM vs group-average RDM
# ---------------------------------------------------------------------------

def plot_rsa_scatter(
    post_patterns : dict,
    subject_idx   : int = 0,
    subject_label : str = "Subject 1",
    run_type      : str = "",
) -> plt.Figure:
    """
    Scatter plot of one subject's RDM (upper triangle) against the
    group-average RDM of all other subjects — visualises the RSA correlation.

    Parameters
    ----------
    post_patterns : {post_id: (n_subjects, n_parcels)}
    subject_idx   : which subject to highlight
    subject_label : label for the title
    run_type      : condition label

    Returns
    -------
    matplotlib Figure

    Example
    -------
    fig = plot_rsa_scatter(patterns['AntiLeft'], subject_idx=0,
                            subject_label='sub-1', run_type='AntiLeft')
    """
    post_ids  = sorted(post_patterns.keys())
    n_subs    = next(iter(post_patterns.values())).shape[0]

    # Build all RDMs (upper triangle only)
    rdms = np.stack([
        pdist(
            np.stack([post_patterns[p][s] for p in post_ids]),
            metric="correlation"
        )
        for s in range(n_subs)
    ])  # (n_subs, n_pairs)

    subj_rdm   = rdms[subject_idx]
    others_avg = (rdms.sum(axis=0) - rdms[subject_idx]) / (n_subs - 1)

    r = np.corrcoef(subj_rdm, others_avg)[0, 1]

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(others_avg, subj_rdm, alpha=0.3, s=8, color="#1f77b4")

    # Regression line
    m, b = np.polyfit(others_avg, subj_rdm, 1)
    x_line = np.linspace(others_avg.min(), others_avg.max(), 100)
    ax.plot(x_line, m * x_line + b, color="#d62728", linewidth=1.5,
            label=f"r = {r:.3f}")

    ax.set_xlabel("Group-average RDM (others)")
    ax.set_ylabel(f"{subject_label} RDM")
    ax.set_title(f"RSA scatter — {run_type}  |  {subject_label}", fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 10. Brain-behavior RSA visualisations
# ---------------------------------------------------------------------------

def plot_similarity_matrices(
    neural_sim   : np.ndarray,
    beh_sim      : np.ndarray,
    subjects     : List[str],
    parcel_idx   : int,
    parcel_name  : str = "",
    run_type     : str = "",
) -> plt.Figure:
    """
    Side-by-side heatmaps of the neural similarity matrix (for one parcel)
    and the behavioral similarity matrix.

    Parameters
    ----------
    neural_sim  : (n_subjects, n_subjects, n_parcels) from compute_brain_behavior_rsa()
    beh_sim     : (n_subjects, n_subjects) from make_behavioral_rdm()
    subjects    : ordered list of subject IDs
    parcel_idx  : which parcel to show
    parcel_name : label for the title
    run_type    : condition label

    Example
    -------
    pcc_idx = parcel_names.index('7Networks_LH_Default_PCC_1')
    fig = plot_similarity_matrices(neural_sim, beh_sim, cfg.subjects,
                                    pcc_idx, 'LH Default PCC 1', 'AntiLeft')
    """
    labels = [s.replace("sub-", "S") for s in subjects]
    neu    = neural_sim[:, :, parcel_idx]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, mat, title, cmap in zip(
        axes,
        [neu,     beh_sim],
        [f"Neural similarity\n{parcel_name}  |  {run_type}",
         "Behavioral similarity\n(political affiliation)"],
        ["RdBu_r", "RdBu_r"],
    ):
        vabs = max(abs(mat).max(), 0.01)
        norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs) \
               if mat.min() < 0 else None
        kwargs = dict(cmap=cmap, aspect="auto")
        if norm:
            kwargs["norm"] = norm
        else:
            kwargs.update(vmin=0, vmax=1)

        im = ax.imshow(mat, **kwargs)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=10)

    fig.tight_layout()
    return fig


def plot_brain_behavior_scatter(
    neural_sim  : np.ndarray,
    beh_sim     : np.ndarray,
    subjects    : List[str],
    parcel_idx  : int,
    parcel_name : str = "",
    run_type    : str = "",
) -> plt.Figure:
    """
    Scatter plot of neural vs behavioral similarity (upper triangle pairs).
    Each dot is one subject pair.

    Example
    -------
    fig = plot_brain_behavior_scatter(neural_sim, beh_sim, cfg.subjects,
                                       pcc_idx, 'LH Default PCC 1', 'AntiLeft')
    """
    n       = beh_sim.shape[0]
    idx     = np.triu_indices(n, k=1)
    neu_vec = neural_sim[:, :, parcel_idx][idx]
    beh_vec = beh_sim[idx]

    # Subject pair labels for hover annotation
    pair_labels = [f"{subjects[i].replace('sub-','')}–{subjects[j].replace('sub-','')}"
                   for i, j in zip(*idx)]

    r = np.corrcoef(neu_vec, beh_vec)[0, 1]
    m, b = np.polyfit(beh_vec, neu_vec, 1)
    x_line = np.linspace(beh_vec.min(), beh_vec.max(), 100)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(beh_vec, neu_vec, alpha=0.6, s=40, color="#1f77b4", zorder=3)
    ax.plot(x_line, m * x_line + b, color="#d62728",
            linewidth=1.8, label=f"r = {r:.3f}")

    ax.set_xlabel("Behavioral similarity\n(political affiliation)")
    ax.set_ylabel("Neural similarity (r)")
    ax.set_title(f"Brain–behavior RSA\n{parcel_name}  |  {run_type}", fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_brain_behavior_bar(
    results_dict : Dict[str, pd.DataFrame],
    title        : str = "Brain–behavior RSA across conditions",
) -> plt.Figure:
    """
    Bar chart comparing brain-behavior RSA r across conditions,
    mean ± SEM across parcels (or just the parcel-level r if no SEM needed).

    Pass a dict of results DataFrames, one per condition.

    Example
    -------
    fig = plot_brain_behavior_bar(bb_results)
    """
    return plot_condition_bar(
        results_dict,
        stat_col = "r",
        title    = title,
        ylabel   = "Brain–behavior RSA r",
    )


# ---------------------------------------------------------------------------
# 11. Brain surface map — parcel r-values from a results DataFrame
# ---------------------------------------------------------------------------

def _df_to_surface_texture(
    df          : pd.DataFrame,
    stat_col    : str,
    n_rois      : int,
    hemi        : str,
    fsaverage   : object,
) -> np.ndarray:
    """
    Internal helper: maps a parcel-level stat column from a Schaefer-200
    results DataFrame onto an fsaverage5 surface texture (per-vertex array).

    Parameters
    ----------
    df        : results DataFrame with columns 'parcel' and stat_col
    stat_col  : which column to project ('r', 'p_raw', etc.)
    n_rois    : number of Schaefer parcels (200 by default)
    hemi      : 'left' or 'right'
    fsaverage : nilearn fsaverage5 dataset object

    Returns
    -------
    np.ndarray of shape (n_vertices,) — vertex-level texture ready to plot
    """
    from nilearn import datasets as nl_datasets, surface

    # --- load the Schaefer volumetric atlas and project to surface once ------
    atlas      = nl_datasets.fetch_atlas_schaefer_2018(n_rois=n_rois, resolution_mm=1)
    atlas_img  = atlas.maps
    labels_raw = list(atlas.labels)
    # decode bytes if needed (older nilearn versions)
    labels = [l.decode() if isinstance(l, bytes) else l for l in labels_raw]

    # project atlas parcellation labels to the surface
    mesh = fsaverage.pial_left if hemi == "left" else fsaverage.pial_right
    atlas_texture = surface.vol_to_surf(atlas_img, mesh, interpolation="nearest")

    # build label → stat value lookup from the DataFrame
    label_to_val = dict(zip(df["parcel"].values, df[stat_col].values))

    # map: for each vertex, find its atlas index → label → stat value
    texture = np.zeros(atlas_texture.shape[0], dtype=float)
    for vert_idx, parcel_idx in enumerate(atlas_texture):
        idx = int(parcel_idx)
        if 1 <= idx <= len(labels):
            lbl = labels[idx - 1]          # Schaefer labels are 1-based
            texture[vert_idx] = label_to_val.get(lbl, 0.0)

    return texture


def plot_brain_map(
    results_dict  : Dict[str, pd.DataFrame],
    stat_col      : str = "r",
    mask_nonsig   : bool = False,
    n_rois        : int = 200,
    cmap          : str = "RdBu_r",
    vmin          : Optional[float] = None,
    vmax          : Optional[float] = None,
    symmetric_cbar: bool = True,
    views         : List[str] = ("lateral", "medial"),
    title_prefix  : str = "",
) -> plt.Figure:
    """
    Project parcel-level RSA / ISC statistics onto fsaverage5 brain surfaces
    and display lateral + medial views for both hemispheres, one row per
    condition.

    Works directly with the CSV files from your RSA pipeline — just load them
    with pd.read_csv and pass as a dict.

    Parameters
    ----------
    results_dict   : {condition_name: DataFrame}
                     Each DataFrame must have columns: 'parcel', stat_col,
                     and (if mask_nonsig=True) 'significant'.
                     Parcel names must follow Schaefer-200 convention, e.g.
                     '7Networks_LH_Default_PCC_1'.
    stat_col       : column to visualise. Use 'r' for RSA/ISC correlation,
                     'p_raw' for uncorrected p-values, etc.
    mask_nonsig    : if True, set non-significant parcels (significant == 0)
                     to NaN so they render as flat grey on the surface.
    n_rois         : number of Schaefer parcels (default 200).
    cmap           : matplotlib colormap name.
    vmin, vmax     : colour scale limits. If None and symmetric_cbar=True,
                     limits are set to ±98th-percentile |r| across all
                     conditions so every subplot shares the same scale.
    symmetric_cbar : if True (default), forces vmin = -vmax so that 0 is
                     always the midpoint — appropriate for correlation values.
    views          : surface views to render per hemisphere.
                     Default ('lateral', 'medial') gives 4 panels per row.
    title_prefix   : optional string prepended to each subplot title.

    Returns
    -------
    matplotlib Figure

    Notes
    -----
    Requires nilearn ≥ 0.10:
        pip install nilearn

    Examples
    --------
    # Load your 4 conditions
    import pandas as pd
    conditions = ['AntiLeft', 'AntiRight', 'ProLeft', 'ProRight']
    bb_results = {c: pd.read_csv(f'{c}.csv') for c in conditions}

    # Brain-behavior RSA r-values, uncorrected, symmetric colour bar
    fig = plot_brain_map(bb_results, stat_col='r')
    fig.savefig('brain_behavior_rsa.png', dpi=150, bbox_inches='tight')

    # Same but mask non-significant parcels to grey
    fig = plot_brain_map(bb_results, stat_col='r', mask_nonsig=True)

    # Neural similarity mean r (load from the npy-derived CSVs)
    neural_results = {c: pd.read_csv(f'{c}_neural_mean.csv') for c in conditions}
    fig = plot_brain_map(neural_results, stat_col='r',
                         title_prefix='Neural similarity — ')
    """
    from nilearn import datasets as nl_datasets, surface, plotting

    conditions = list(results_dict.keys())
    n_conds    = len(conditions)
    n_views    = len(views)
    hemis      = ["left", "right"]
    n_cols     = len(hemis) * n_views          # e.g. 4 for 2 hemis × 2 views

    # ── colour scale: shared across all conditions ──────────────────────────
    if vmin is None or vmax is None:
        all_vals = np.concatenate([
            df[stat_col].dropna().values for df in results_dict.values()
        ])
        abs_max = np.percentile(np.abs(all_vals[np.isfinite(all_vals)]), 98)
        if symmetric_cbar:
            vmin_use = -abs_max
            vmax_use =  abs_max
        else:
            vmin_use = np.percentile(all_vals[np.isfinite(all_vals)],  2)
            vmax_use = abs_max
    else:
        vmin_use, vmax_use = vmin, vmax

    # ── fsaverage5 surface meshes ────────────────────────────────────────────
    fsaverage = nl_datasets.fetch_surf_fsaverage("fsaverage5")

    # ── figure layout ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        n_conds, n_cols,
        figsize=(4.0 * n_cols, 3.2 * n_conds),
        subplot_kw={"projection": "3d"},   # nilearn needs 3-D axes
    )
    # normalise axes to 2-D array for uniform indexing
    if n_conds == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    col_labels = [f"{h[0].upper()}H {v}" for h in hemis for v in views]

    for row_i, cond in enumerate(conditions):
        df = results_dict[cond].copy()

        # optionally mask non-significant parcels
        if mask_nonsig and "significant" in df.columns:
            df.loc[df["significant"] == 0, stat_col] = np.nan

        col_i = 0
        for hemi in hemis:
            bg_map = fsaverage.sulc_left if hemi == "left" else fsaverage.sulc_right
            mesh   = fsaverage.pial_left if hemi == "left" else fsaverage.pial_right

            texture = _df_to_surface_texture(df, stat_col, n_rois, hemi, fsaverage)

            for view in views:
                ax = axes[row_i, col_i]

                plotting.plot_surf_stat_map(
                    mesh,
                    texture,
                    hemi        = hemi,
                    view        = view,
                    bg_map      = bg_map,
                    cmap        = cmap,
                    vmin        = vmin_use,
                    vmax        = vmax_use,
                    colorbar    = (col_i == n_cols - 1),  # only on last column
                    darkness    = 0.5,
                    axes        = ax,
                )

                # column header on first row only
                if row_i == 0:
                    ax.set_title(col_labels[col_i], fontsize=9, pad=2)

                col_i += 1

        # row label (condition name) on the leftmost axis
        axes[row_i, 0].text2D(
            -0.08, 0.5,
            f"{title_prefix}{cond}",
            transform   = axes[row_i, 0].transAxes,
            fontsize    = 10,
            fontweight  = "bold",
            va          = "center",
            rotation    = 90,
        )

    fig.suptitle(
        f"{title_prefix}{'(masked: FDR sig only)' if mask_nonsig else '(uncorrected)'}",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 12. Quick single-condition brain map (useful for inline notebook inspection)
# ---------------------------------------------------------------------------

def plot_brain_map_single(
    df            : pd.DataFrame,
    run_type      : str = "",
    stat_col      : str = "r",
    mask_nonsig   : bool = False,
    n_rois        : int = 200,
    cmap          : str = "RdBu_r",
    vmin          : Optional[float] = None,
    vmax          : Optional[float] = None,
    symmetric_cbar: bool = True,
) -> plt.Figure:
    """
    Single-condition wrapper around plot_brain_map.
    Shows lateral + medial views for both hemispheres in one row.

    Parameters
    ----------
    df          : results DataFrame (parcel, r, p_raw, p_fdr, significant)
    run_type    : condition label used in the title
    stat_col    : column to plot (default 'r')
    mask_nonsig : grey-out parcels where significant == 0
    n_rois      : Schaefer atlas size (default 200)
    cmap        : colormap (default 'RdBu_r')
    vmin, vmax  : colour limits (auto if None)
    symmetric_cbar : centre colour bar on 0 (default True — good for r values)

    Returns
    -------
    matplotlib Figure

    Example
    -------
    df = pd.read_csv('AntiLeft.csv')
    fig = plot_brain_map_single(df, run_type='AntiLeft')
    fig.savefig('AntiLeft_brain.png', dpi=150, bbox_inches='tight')
    """
    return plot_brain_map(
        results_dict   = {run_type: df},
        stat_col       = stat_col,
        mask_nonsig    = mask_nonsig,
        n_rois         = n_rois,
        cmap           = cmap,
        vmin           = vmin,
        vmax           = vmax,
        symmetric_cbar = symmetric_cbar,
        title_prefix   = "",
    )

"""
Interactive brain map functions
=============================================================

    plot_brain_map_interactive(results_dict, ...)
        → one nilearn interactive HTML viewer per condition, saved to disk.
          Shows ALL parcels coloured by r-value (uncorrected view) or only
          p-threshold-surviving parcels (thresholded view), controlled by
          the `p_threshold` argument.

    save_brain_map_html(df, run_type, output_dir, ...)
        → thin single-condition wrapper, mirrors the style of the rest of
          the module.

"""

# ── internal: build a whole-brain NIfTI from parcel-level stats ─────────────

def _parcels_to_nifti(
    df          : pd.DataFrame,
    stat_col    : str,
    n_rois      : int = 400,
    p_threshold : Optional[float] = None,
    p_col       : str = "p_raw",
    tian_nii    : Optional[str] = None,
) :
    """
    Map a parcel-level DataFrame onto a combined Schaefer + Tian NIfTI image.

    Schaefer parcels are matched by label name (e.g. 7Networks_LH_Vis_1).
    Tian parcels are matched by label name against the Tian atlas integers,
    and placed in the same volume by filling voxels that Schaefer leaves empty.

    Parameters
    ----------
    df          : results DataFrame with columns parcel, stat_col, p_col
    stat_col    : column to map onto the brain
    n_rois      : number of Schaefer parcels (default 400)
    p_threshold : if set, zero out parcels with p_col >= p_threshold
    p_col       : p-value column for thresholding
    tian_nii    : path to the Tian subcortical atlas NIfTI (same MNI space).
                  Required to visualise Tian parcels. If None, Tian parcels
                  are silently skipped (cortex-only view).
                  e.g. 'Tian_Subcortex_S2_MNI152_2mm.nii.gz'

    Returns
    -------
    stat_img, thresh_img : NIfTI images on the Schaefer grid
    """
    from nilearn import datasets as nl_datasets
    from nilearn.image import resample_to_img
    import nibabel as nib

    # ── Schaefer atlas ───────────────────────────────────────────────────────
    atlas     = nl_datasets.fetch_atlas_schaefer_2018(n_rois=n_rois, resolution_mm=2)
    atlas_img = nib.load(atlas.maps) if isinstance(atlas.maps, str) else atlas.maps
    sch_labels = [
        l.decode() if isinstance(l, bytes) else l
        for l in atlas.labels
    ]
    combined_data = atlas_img.get_fdata().copy()   # will hold Schaefer + Tian IDs

    # ── Tian atlas (optional) ────────────────────────────────────────────────
    # Detect which parcels in the DataFrame are Tian (not in Schaefer labels)
    sch_label_set = set(sch_labels)
    tian_parcels  = df[~df["parcel"].isin(sch_label_set)]["parcel"].tolist()

    if tian_parcels and tian_nii is not None:
        ti_img = nib.load(str(tian_nii))
        # Resample Tian to Schaefer grid if needed
        if (ti_img.shape[:3] != atlas_img.shape[:3] or
                not np.allclose(ti_img.affine, atlas_img.affine, atol=1e-3)):
            ti_img = resample_to_img(ti_img, atlas_img, interpolation="nearest")
        ti_data = ti_img.get_fdata()

        # Tian integer IDs present in the volume (sorted)
        ti_ids = sorted(np.unique(ti_data[ti_data > 0]).astype(int))

        if len(ti_ids) != len(tian_parcels):
            print(f"  ⚠  Tian: {len(ti_ids)} IDs in volume but "
                  f"{len(tian_parcels)} non-Schaefer parcels in CSV — "
                  f"mapping by position (check label order)")

        # Map Tian integer → parcel name by position
        ti_id_to_label = {tid: lbl
                          for tid, lbl in zip(ti_ids, tian_parcels)}

        # Offset Tian IDs so they don't clash with Schaefer (1-400)
        offset = n_rois
        for tid, lbl in ti_id_to_label.items():
            mask = (ti_data == tid) & (combined_data == 0)   # only empty cortex voxels
            combined_data[mask] = offset + tid

    elif tian_parcels and tian_nii is None:
        print(f"  ℹ  {len(tian_parcels)} Tian parcels in CSV but tian_nii not "
              f"provided — subcortical parcels will be invisible. "
              f"Pass tian_nii='path/to/Tian_atlas.nii.gz' to include them.")

    # ── build label → value lookup ───────────────────────────────────────────
    label_to_stat = dict(zip(df["parcel"].values, df[stat_col].values))
    if p_threshold is not None and p_col in df.columns:
        label_to_p = dict(zip(df["parcel"].values, df[p_col].values))
    else:
        label_to_p = {}

    # ── fill stat volume ─────────────────────────────────────────────────────
    stat_vol   = np.zeros_like(combined_data, dtype=np.float32)
    thresh_vol = np.zeros_like(combined_data, dtype=np.float32)

    # Schaefer parcels: integer 1..n_rois → sch_labels[idx-1]
    for idx, lbl in enumerate(sch_labels, start=1):
        mask = combined_data == idx
        val  = label_to_stat.get(lbl, 0.0)
        stat_vol[mask] = val
        if p_threshold is not None:
            thresh_vol[mask] = val if label_to_p.get(lbl, 1.0) < p_threshold else 0.0
        else:
            thresh_vol[mask] = val

    # Tian parcels: integer (offset + tid) → label name
    if tian_parcels and tian_nii is not None:
        for tid, lbl in ti_id_to_label.items():
            mask = combined_data == (offset + tid)
            val  = label_to_stat.get(lbl, 0.0)
            stat_vol[mask] = val
            if p_threshold is not None:
                thresh_vol[mask] = val if label_to_p.get(lbl, 1.0) < p_threshold else 0.0
            else:
                thresh_vol[mask] = val

    n_mapped = (stat_vol != 0).any()
    print(f"  [_parcels_to_nifti] Schaefer={len(sch_labels)} | "
          f"Tian={len(tian_parcels)} | any nonzero={n_mapped}")

    stat_img   = nib.Nifti1Image(stat_vol,   atlas_img.affine, atlas_img.header)
    thresh_img = nib.Nifti1Image(thresh_vol, atlas_img.affine, atlas_img.header)
    return stat_img, thresh_img


# ── public API ───────────────────────────────────────────────────────────────

def save_brain_map_html(
    df          : pd.DataFrame,
    run_type    : str,
    output_dir  : str | Path = ".",
    stat_col    : str = "r",
    p_threshold : Optional[float] = None,
    p_col       : str = "p_raw",
    n_rois      : int = 400,
    cmap        : str = "RdBu_r",
    vmin        : Optional[float] = None,
    vmax        : Optional[float] = None,
    symmetric_cbar : bool = True,
    open_browser: bool = False,
    tian_nii    : Optional[str] = None,
) -> Path:
    """
    Build a self-contained interactive nilearn HTML viewer for one condition
    and save it to disk.

    The HTML contains a zoomable, rotatable 3-D glass brain + slice viewer
    coloured by `stat_col` (default: RSA r-value).  All parcels are rendered
    by default; pass `p_threshold` to show only surviving parcels.

    Parameters
    ----------
    df           : results DataFrame — must have columns 'parcel' and stat_col.
                   Expects Schaefer-200 parcel names such as
                   '7Networks_LH_Default_PCC_1'.
    run_type     : condition name used in the HTML title and output filename.
    output_dir   : folder where the .html file is written (created if absent).
    stat_col     : column to visualise (default 'r').
    p_threshold  : if given, zero-out parcels where p_col >= p_threshold before
                   plotting, so only sub-threshold parcels are visible.
                   E.g. p_threshold=0.05 → show only p < 0.05 (uncorrected).
                       p_threshold=None  → show all parcels (uncorrected r map).
    p_col        : which p-value column to threshold on (default 'p_raw').
                   Use 'p_fdr' to threshold on FDR-corrected values.
    n_rois       : Schaefer atlas size (default 200).
    cmap         : diverging colormap — 'RdBu_r' keeps 0 = white/neutral.
    vmin, vmax   : colour scale limits.  If None and symmetric_cbar=True,
                   auto-set to ±max(|r|) across all parcels in this condition.
    symmetric_cbar : centre the colour bar on 0 (appropriate for r-values).
    open_browser : if True, open the saved HTML in your default browser.

    Returns
    -------
    Path to the saved HTML file.

    Examples
    --------
    import pandas as pd
    from parcellated_viz import save_brain_map_html

    df = pd.read_csv('AntiLeft.csv')

    # Uncorrected: all parcels visible, coloured by r
    html_path = save_brain_map_html(df, 'AntiLeft', output_dir='figures/')

    # Thresholded: only parcels with p_raw < 0.05 visible
    html_path = save_brain_map_html(df, 'AntiLeft', output_dir='figures/',
                                     p_threshold=0.05, p_col='p_raw')

    # FDR-corrected view
    html_path = save_brain_map_html(df, 'AntiLeft', output_dir='figures/',
                                     p_threshold=0.05, p_col='p_fdr')

    # Display inline in a Jupyter notebook
    from IPython.display import IFrame
    IFrame(str(html_path), width=900, height=500)
    """
    from nilearn import plotting

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── colour limits ────────────────────────────────────────────────────────
    if vmin is None or vmax is None:
        vals    = df[stat_col].dropna().values
        abs_max = float(np.percentile(np.abs(vals[np.isfinite(vals)]), 98))
        if symmetric_cbar:
            vmin_use, vmax_use = -abs_max, abs_max
        else:
            vmin_use = float(np.percentile(vals[np.isfinite(vals)], 2))
            vmax_use = abs_max
    else:
        vmin_use, vmax_use = vmin, vmax

    # ── build NIfTI images ───────────────────────────────────────────────────
    stat_img, thresh_img = _parcels_to_nifti(
        df, stat_col, n_rois, p_threshold, p_col, tian_nii=tian_nii
    )

    # ── display threshold ────────────────────────────────────────────────────
    # nilearn requires threshold < max(|img|), and threshold > 0 so that
    # zero-filled voxels (inter-parcel gaps) are transparent and MNI shows
    # through. We use the minimum non-zero absolute r-value in the data,
    # divided by 2 — this is always smaller than any real parcel value while
    # still clearing the empty voxels between parcels.
    nonzero_vals = df[stat_col].dropna().values
    nonzero_vals = nonzero_vals[np.abs(nonzero_vals) > 0]
    if len(nonzero_vals) > 0:
        eps = float(np.abs(nonzero_vals).min()) / 2.0
    else:
        eps = 1e-6   # fallback: should never happen with real data

    if p_threshold is not None:
        display_img = thresh_img
        thresh_kw   = eps
        p_label     = f"p<{p_threshold} ({p_col})"
    else:
        display_img = stat_img
        thresh_kw   = eps           # all parcels visible; gaps transparent
        p_label     = "uncorrected (all parcels)"

    title = f"RSA {stat_col} — {run_type}  [{p_label}]"

    # ── nilearn interactive viewer ───────────────────────────────────────────
    html_view = plotting.view_img(
        display_img,
        bg_img      = "MNI152",
        cmap        = cmap,
        threshold   = thresh_kw,
        vmin        = vmin_use,
        vmax        = vmax_use,
        title       = title,
        symmetric_cmap = symmetric_cbar,
    )

    # ── save ─────────────────────────────────────────────────────────────────
    suffix    = f"_p{p_threshold}_{p_col}" if p_threshold is not None else "_uncorrected"
    out_path  = output_dir / f"{run_type}_rsa_{stat_col}{suffix}.html"
    html_view.save_as_html(str(out_path))
    print(f"  Saved → {out_path}")

    if open_browser:
        import webbrowser
        webbrowser.open(out_path.as_uri())

    return out_path


def plot_brain_map_interactive(
    results_dict : Dict[str, pd.DataFrame],
    output_dir   : str | Path = ".",
    stat_col     : str = "r",
    p_thresholds : List[Optional[float]] = (None, 0.05),
    p_col        : str = "p_raw",
    n_rois       : int = 400,
    cmap         : str = "RdBu_r",
    vmin         : Optional[float] = None,
    vmax         : Optional[float] = None,
    symmetric_cbar : bool = True,
    tian_nii       : Optional[str] = None,
) -> Dict[str, Dict[str, Path]]:
    """
    Batch-generate interactive HTML brain maps for all conditions and all
    requested p-thresholds.

    This is the multi-condition driver. It calls save_brain_map_html for
    every (condition × p_threshold) combination and returns a nested dict
    of output paths so you can embed them in a notebook or report.

    Parameters
    ----------
    results_dict  : {condition: DataFrame}  — load with pd.read_csv()
    output_dir    : folder for all HTML files
    stat_col      : column to map (default 'r')
    p_thresholds  : list of thresholds to generate.
                    None     → all parcels shown (uncorrected colour map)
                    0.05     → only p_col < 0.05 parcels visible
                    [None, 0.05, 0.1] → three HTMLs per condition
    p_col         : 'p_raw' or 'p_fdr'
    n_rois        : Schaefer atlas size
    cmap          : colormap
    vmin, vmax    : colour limits (shared across all conditions if None)
    symmetric_cbar: centre on 0

    Returns
    -------
    {condition: {threshold_label: Path}}

    Examples
    --------
    import pandas as pd
    from parcellated_viz import plot_brain_map_interactive

    conditions = ['AntiLeft', 'AntiRight', 'ProLeft', 'ProRight']
    bb = {c: pd.read_csv(f'{c}.csv') for c in conditions}

    # Generate: uncorrected + p<0.05 (uncorrected) for all conditions
    paths = plot_brain_map_interactive(
        bb,
        output_dir   = 'figures/interactive/',
        p_thresholds = [None, 0.05],
        p_col        = 'p_raw',
    )

    # FDR-corrected view only
    paths = plot_brain_map_interactive(
        bb,
        output_dir   = 'figures/interactive/',
        p_thresholds = [0.05],
        p_col        = 'p_fdr',
    )

    # Display one inline in Jupyter
    from IPython.display import IFrame
    IFrame(str(paths['AntiLeft'][None]), width=900, height=500)

    # --- Neural similarity maps (same workflow) ---
    neural = {c: pd.read_csv(f'{c}_neural_mean.csv') for c in conditions}
    paths = plot_brain_map_interactive(
        neural,
        output_dir = 'figures/interactive/neural/',
        p_thresholds = [None, 0.05],
    )
    """
    # ── shared colour limits across all conditions ───────────────────────────
    if vmin is None or vmax is None:
        all_vals = np.concatenate([
            df[stat_col].dropna().values for df in results_dict.values()
        ])
        finite   = all_vals[np.isfinite(all_vals)]
        abs_max  = float(np.percentile(np.abs(finite), 98))
        if symmetric_cbar:
            vmin_shared, vmax_shared = -abs_max, abs_max
        else:
            vmin_shared = float(np.percentile(finite, 2))
            vmax_shared = abs_max
    else:
        vmin_shared, vmax_shared = vmin, vmax

    output_paths: Dict[str, Dict] = {}

    for cond, df in results_dict.items():
        output_paths[cond] = {}
        print(f"\n── {cond} ──")
        for thresh in p_thresholds:
            path = save_brain_map_html(
                df            = df,
                run_type      = cond,
                output_dir    = output_dir,
                stat_col      = stat_col,
                p_threshold   = thresh,
                p_col         = p_col,
                n_rois        = n_rois,
                cmap          = cmap,
                vmin          = vmin_shared,
                vmax          = vmax_shared,
                symmetric_cbar= symmetric_cbar,
                tian_nii      = tian_nii,
            )
            output_paths[cond][thresh] = path

    return output_paths