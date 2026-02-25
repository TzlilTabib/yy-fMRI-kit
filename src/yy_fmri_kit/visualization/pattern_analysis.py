"""
parcellated_viz.py
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