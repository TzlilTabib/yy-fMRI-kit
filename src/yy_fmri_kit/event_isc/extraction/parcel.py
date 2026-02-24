"""
parcel.py
======================
Parcellated ISC and RSA for event-based fMRI data.

Designed for studies where:
- Subjects watched N items (posts) per condition in randomised order
- Data is pre-extracted as parcellated timeseries TSVs (rows=TRs, cols=parcels)
- Event timing is in a combined CSV with columns: subject, run_type, post_id,
  onset (seconds), duration (seconds)

The unit of analysis is a **post** (item). For each post we average the BOLD
signal over its TRs (with an HRF shift) to get one spatial pattern per parcel.
We then measure how similar those patterns are *across subjects* for the same
post — this is event-level ISC.

For RSA we additionally ask whether the *structure* of the post × post
dissimilarity matrix is shared across subjects.

Usage (from a notebook)
------------------------
    from parcellated_isc_rsa import Config, load_data, extract_post_patterns
    from parcellated_isc_rsa import compute_isc, compute_rsa, permutation_test
    from parcellated_isc_rsa import fdr_correct, results_to_dataframe

    cfg = Config(
        data_dir   = Path("derivatives/denoised"),
        events_csv = Path("behavioral/combined_events_with_bids.csv"),
        subjects   = ['sub-1', 'sub-6', ...],
        run_types  = ['AntiLeft', 'AntiRight', 'ProLeft', 'ProRight'],
        tr         = 1.0,
        shift_tr   = 4,
    )

    events_df          = load_events(cfg)
    ts_dict            = load_timeseries(cfg)          # {(sub, run_type): DataFrame}
    patterns           = extract_post_patterns(ts_dict, events_df, cfg)
    # patterns: {run_type: {post_id: array (n_subjects, n_parcels)}}

    isc_r, isc_p       = compute_isc(patterns['AntiLeft'], cfg)
    rsa_r, rsa_p       = compute_rsa(patterns['AntiLeft'], cfg)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, ttest_1samp
from statsmodels.stats.multitest import fdrcorrection

__all__ = [
    "Config",
    "load_events",
    "load_timeseries",
    "extract_post_patterns",
    "compute_isc",
    "compute_rsa",
    "permutation_test",
    "fdr_correct",
    "results_to_dataframe",
]

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# For one run type:
#   post_id  →  array of shape (n_subjects, n_parcels)
#               rows are subjects that watched that post, in a consistent order
PostPatterns = Dict[int, np.ndarray]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """
    All study-level parameters.

    Parameters
    ----------
    data_dir      : root folder containing per-subject sub-directories with TSV files
    events_csv    : path to the combined events CSV
    subjects      : list of subject IDs (e.g. ['sub-1', 'sub-6', ...])
    run_types     : list of condition names (e.g. ['AntiLeft', 'ProLeft', ...])
    tr            : TR in seconds
    shift_tr      : HRF shift in TRs (typically 4–6 for TR=1s)
    tsv_glob      : glob pattern to find a subject's TSV for a given run_type.
                    Use {subject} and {run_type} as placeholders.
    subject_col   : column name for subject ID in events CSV
    run_col       : column name for run/condition in events CSV
    post_col      : column name for post/item ID in events CSV
    onset_col     : column name for event onset (seconds)
    duration_col  : column name for event duration (seconds)
    n_perms       : number of permutations for null distribution
    fdr_q         : FDR threshold
    seed          : random seed

    Example
    -------
    cfg = Config(
        data_dir   = Path("derivatives/denoised"),
        events_csv = Path("behavioral/combined_events.csv"),
        subjects   = ['sub-1', 'sub-6', 'sub-20'],
        run_types  = ['AntiLeft', 'AntiRight', 'ProLeft', 'ProRight'],
        tr         = 1.0,
        shift_tr   = 4,
    )
    """
    data_dir    : Path
    events_csv  : Path
    subjects    : List[str]
    run_types   : List[str]
    tr          : float = 1.0
    shift_tr    : int   = 4

    # TSV filename pattern — adjust to match your BIDS naming
    tsv_glob    : str = (
        "{subject}/ses-*/func/*task-{run_type}*atlas-Schaefer2018*timeseries.tsv"
    )

    # Column names in events CSV
    subject_col  : str = "bids_id"
    run_col      : str = "run_type"
    post_col     : str = "post_id"
    onset_col    : str = "onset"
    duration_col : str = "duration"

    n_perms : int   = 1000
    fdr_q   : float = 0.05
    seed    : int   = 42

    def __post_init__(self):
        self.data_dir  = Path(self.data_dir)
        self.events_csv = Path(self.events_csv)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_events(cfg: Config) -> pd.DataFrame:
    """
    Load the combined events CSV and keep only the configured subjects /
    run types.

    Returns
    -------
    pd.DataFrame with at least columns:
        subject, run_type, post_id, onset, duration

    Example
    -------
    events_df = load_events(cfg)
    print(events_df.head())
    """
    df = pd.read_csv(cfg.events_csv)

    # Rename to standard internal names
    df = df.rename(columns={
        cfg.subject_col  : "subject",
        cfg.run_col      : "run_type",
        cfg.post_col     : "post_id",
        cfg.onset_col    : "onset",
        cfg.duration_col : "duration",
    })

    df = df[df["subject"].isin(cfg.subjects)]
    df = df[df["run_type"].isin(cfg.run_types)]
    df = df.reset_index(drop=True)

    print(f"[load_events] {len(df)} events | "
          f"{df['subject'].nunique()} subjects | "
          f"{df['run_type'].nunique()} run types | "
          f"{df['post_id'].nunique()} unique post IDs")
    return df


def load_timeseries(cfg: Config) -> Dict[Tuple[str, str], pd.DataFrame]:
    """
    Load parcellated timeseries TSVs for all subjects × run_types.

    Returns
    -------
    dict mapping (subject, run_type) → pd.DataFrame
        shape: (n_trs, n_parcels), columns are parcel names

    Example
    -------
    ts_dict = load_timeseries(cfg)
    ts = ts_dict[('sub-1', 'AntiLeft')]   # DataFrame (n_trs, n_parcels)
    parcel_names = ts.columns.tolist()
    """
    ts_dict: Dict[Tuple[str, str], pd.DataFrame] = {}
    missing = []

    for sub in cfg.subjects:
        for run_type in cfg.run_types:
            pattern = cfg.tsv_glob.format(subject=sub, run_type=run_type)
            matches = list(cfg.data_dir.glob(pattern))

            if not matches:
                missing.append((sub, run_type))
                continue
            if len(matches) > 1:
                print(f"  ⚠  Multiple TSVs for {sub}/{run_type}, using first: "
                      f"{matches[0].name}")

            ts_dict[(sub, run_type)] = pd.read_csv(matches[0], sep="\t")

    print(f"[load_timeseries] Loaded {len(ts_dict)} timeseries")
    if missing:
        print(f"  ⚠  Missing ({len(missing)}): "
              + ", ".join(f"{s}/{r}" for s, r in missing[:5])
              + (" ..." if len(missing) > 5 else ""))

    return ts_dict


# ---------------------------------------------------------------------------
# Pattern extraction
# ---------------------------------------------------------------------------

def _trs_for_event(onset: float, duration: float,
                   shift_tr: int, tr: float, n_trs: int) -> np.ndarray:
    """
    Convert onset + duration (seconds) to TR indices with HRF shift.
    Clips to valid range [0, n_trs).
    """
    start = int(round(onset / tr)) + shift_tr
    end   = int(round((onset + duration) / tr)) + shift_tr
    trs   = np.arange(start, end)
    return trs[(trs >= 0) & (trs < n_trs)]


def extract_post_patterns(
    ts_dict  : Dict[Tuple[str, str], pd.DataFrame],
    events_df: pd.DataFrame,
    cfg      : Config,
) -> Dict[str, PostPatterns]:
    """
    For each condition (run_type), extract one mean spatial pattern per post
    per subject by averaging BOLD over the post's TRs (with HRF shift).

    Parameters
    ----------
    ts_dict   : output of load_timeseries()
    events_df : output of load_events()
    cfg       : Config

    Returns
    -------
    patterns : {run_type: {post_id: array (n_subjects, n_parcels)}}
        Only posts seen by ALL subjects that have a valid timeseries are
        included. Subjects are in a fixed, consistent order (cfg.subjects).

    Example
    -------
    patterns = extract_post_patterns(ts_dict, events_df, cfg)

    # shape for one post in one condition
    print(patterns['AntiLeft'][101].shape)   # (n_subjects, n_parcels)

    # parcel names — same order as axis-1
    parcel_names = ts_dict[(cfg.subjects[0], cfg.run_types[0])].columns.tolist()
    """
    patterns: Dict[str, PostPatterns] = {}

    for run_type in cfg.run_types:
        run_events = events_df[events_df["run_type"] == run_type]
        post_ids   = sorted(run_events["post_id"].unique())

        # Which subjects have a TSV for this run?
        valid_subs = [s for s in cfg.subjects if (s, run_type) in ts_dict]
        if len(valid_subs) < 2:
            print(f"[extract] {run_type}: skipping — only {len(valid_subs)} subjects")
            continue

        run_patterns: PostPatterns = {}

        for post_id in post_ids:
            sub_vectors = []

            for sub in valid_subs:
                ts  = ts_dict[(sub, run_type)].values  # (n_trs, n_parcels)
                evt = run_events[
                    (run_events["subject"] == sub) &
                    (run_events["post_id"] == post_id)
                ]
                if evt.empty:
                    sub_vectors.append(None)
                    continue

                row  = evt.iloc[0]
                trs  = _trs_for_event(
                    row["onset"], row["duration"],
                    cfg.shift_tr, cfg.tr, n_trs=len(ts)
                )
                if len(trs) == 0:
                    sub_vectors.append(None)
                    continue

                sub_vectors.append(ts[trs].mean(axis=0))  # (n_parcels,)

            # Keep post only if ALL valid subjects have a pattern
            if all(v is not None for v in sub_vectors):
                run_patterns[post_id] = np.stack(sub_vectors)  # (n_subs, n_parcels)

        patterns[run_type] = run_patterns
        n_posts   = len(run_patterns)
        n_parcels = next(iter(run_patterns.values())).shape[1] if n_posts else 0
        print(f"[extract] {run_type}: {n_posts} posts × {len(valid_subs)} subjects "
              f"× {n_parcels} parcels")

    return patterns


# ---------------------------------------------------------------------------
# ISC
# ---------------------------------------------------------------------------

def compute_isc(
    post_patterns: PostPatterns,
    method       : str = "leave_one_out",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute inter-subject correlation (ISC) per parcel using the leave-one-out
    approach: for each subject, correlate their pattern with the mean of all
    others, averaged across posts. Then average across subjects.

    Parameters
    ----------
    post_patterns : {post_id: (n_subjects, n_parcels)}
    method        : 'leave_one_out' (only method currently supported)

    Returns
    -------
    isc_mean  : (n_parcels,)  group-average ISC per parcel
    isc_subj  : (n_subjects, n_parcels) per-subject ISC

    Example
    -------
    isc_mean, isc_subj = compute_isc(patterns['AntiLeft'])
    # isc_mean[p] = average correlation between subjects in parcel p
    """
    post_ids  = sorted(post_patterns.keys())
    n_posts   = len(post_ids)
    n_subjects, n_parcels = next(iter(post_patterns.values())).shape

    # Stack to (n_posts, n_subjects, n_parcels)
    data = np.stack([post_patterns[p] for p in post_ids])

    isc_subj = np.zeros((n_subjects, n_parcels))

    for s in range(n_subjects):
        # Leave-one-out: mean of all other subjects, shape (n_posts, n_parcels)
        others_mean = (data[:, :, :].sum(axis=1) - data[:, s, :]) / (n_subjects - 1)

        for p in range(n_parcels):
            r = pearsonr(data[:, s, p], others_mean[:, p])[0]
            isc_subj[s, p] = r

    isc_mean = isc_subj.mean(axis=0)
    return isc_mean, isc_subj


def permutation_test(
    post_patterns: PostPatterns,
    cfg          : Config,
    analysis     : str = "isc",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Permutation test for ISC or RSA.

    For ISC: shuffles post labels within each subject independently to break
    stimulus-locked structure while preserving each subject's spatial patterns.

    For RSA: shuffles post labels of one subject's RDM before correlating
    with the group-average RDM.

    Parameters
    ----------
    post_patterns : {post_id: (n_subjects, n_parcels)}
    cfg           : Config (uses cfg.n_perms, cfg.seed)
    analysis      : 'isc' or 'rsa'

    Returns
    -------
    obs           : (n_parcels,)  observed statistic
    p_vals        : (n_parcels,)  permutation p-value (one-tailed, > 0)
    null_dist     : (n_perms, n_parcels) full null distribution

    Example
    -------
    obs_isc, p_isc, null = permutation_test(patterns['AntiLeft'], cfg, 'isc')
    rejected, p_fdr = fdr_correct(p_isc, cfg.fdr_q)
    """
    rng      = np.random.default_rng(cfg.seed)
    post_ids = sorted(post_patterns.keys())
    n_posts  = len(post_ids)
    n_subjects, n_parcels = next(iter(post_patterns.values())).shape

    # (n_posts, n_subjects, n_parcels)
    data = np.stack([post_patterns[p] for p in post_ids])

    # Observed statistic
    if analysis == "isc":
        obs, _ = compute_isc(post_patterns)
    elif analysis == "rsa":
        obs = _compute_rsa_from_array(data)
    else:
        raise ValueError(f"analysis must be 'isc' or 'rsa', got '{analysis}'")

    null_dist = np.zeros((cfg.n_perms, n_parcels))

    for i in range(cfg.n_perms):
        # Shuffle post order independently per subject
        perm_data = data.copy()
        for s in range(n_subjects):
            idx = rng.permutation(n_posts)
            perm_data[:, s, :] = data[idx, s, :]

        perm_patterns = {post_ids[j]: perm_data[j] for j in range(n_posts)}

        if analysis == "isc":
            null_dist[i], _ = compute_isc(perm_patterns)
        else:
            null_dist[i] = _compute_rsa_from_array(perm_data)

    p_vals = np.mean(null_dist >= obs, axis=0)
    return obs, p_vals, null_dist


# ---------------------------------------------------------------------------
# RSA
# ---------------------------------------------------------------------------

def _make_rdm(patterns: np.ndarray) -> np.ndarray:
    """
    Build RDM for one subject's patterns.

    Parameters
    ----------
    patterns : (n_posts, n_parcels)

    Returns
    -------
    rdm : (n_posts, n_posts) symmetric dissimilarity matrix (1 - Pearson r)
    """
    return squareform(pdist(patterns, metric="correlation"))


def _compute_rsa_from_array(data: np.ndarray) -> np.ndarray:
    """
    For each parcel, build each subject's RDM from their post patterns,
    then correlate each subject's RDM with the mean RDM of all others
    (leave-one-out), and return the group mean RSA r.

    Parameters
    ----------
    data : (n_posts, n_subjects, n_parcels)

    Returns
    -------
    rsa_mean : (n_parcels,)
    """
    n_posts, n_subjects, n_parcels = data.shape
    rsa_subj = np.zeros((n_subjects, n_parcels))

    for p in range(n_parcels):
        # Build RDM per subject for this parcel: (n_subjects, n_posts*(n_posts-1)/2)
        rdms = np.stack([
            pdist(data[:, s, p].reshape(-1, 1), metric="cityblock")
            if n_posts == 1
            else pdist(data[:, s, [p]], metric="correlation")
            for s in range(n_subjects)
        ])  # (n_subjects, n_pairs)

        for s in range(n_subjects):
            others_mean_rdm = (rdms.sum(axis=0) - rdms[s]) / (n_subjects - 1)
            if rdms[s].std() < 1e-10 or others_mean_rdm.std() < 1e-10:
                rsa_subj[s, p] = np.nan
            else:
                rsa_subj[s, p] = pearsonr(rdms[s], others_mean_rdm)[0]

    return np.nanmean(rsa_subj, axis=0)


def compute_rsa(
    post_patterns: PostPatterns,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute RSA per parcel using leave-one-out cross-subject RDM correlation.

    For each parcel, builds a post × post RDM per subject (using 1 - Pearson r
    as distance across parcels... wait — here each "pattern" is a scalar for
    one parcel, so distance is just abs difference). Then correlates each
    subject's RDM with the mean of all others.

    Note: For a richer RSA where the RDM reflects distances in the full
    parcel space (all parcels), use compute_rsa_multivariate() below.

    Parameters
    ----------
    post_patterns : {post_id: (n_subjects, n_parcels)}

    Returns
    -------
    rsa_mean : (n_parcels,)  group-average RSA per parcel
    rsa_subj : (n_subjects, n_parcels)

    Example
    -------
    rsa_mean, rsa_subj = compute_rsa(patterns['AntiLeft'])
    """
    post_ids = sorted(post_patterns.keys())
    data = np.stack([post_patterns[p] for p in post_ids])  # (n_posts, n_subs, n_parcels)
    n_posts, n_subjects, n_parcels = data.shape

    rsa_subj = np.zeros((n_subjects, n_parcels))

    for p in range(n_parcels):
        # Each subject's "pattern" for this parcel is a vector of length n_posts
        # RDM = pairwise absolute difference across posts
        parcel_data = data[:, :, p]  # (n_posts, n_subjects)
        rdms = np.stack([
            pdist(parcel_data[:, s].reshape(-1, 1), metric="cityblock")
            for s in range(n_subjects)
        ])  # (n_subjects, n_pairs)

        for s in range(n_subjects):
            others = (rdms.sum(axis=0) - rdms[s]) / (n_subjects - 1)
            if rdms[s].std() < 1e-10 or others.std() < 1e-10:
                rsa_subj[s, p] = np.nan
            else:
                rsa_subj[s, p] = pearsonr(rdms[s], others)[0]

    rsa_mean = np.nanmean(rsa_subj, axis=0)
    return rsa_mean, rsa_subj


def compute_rsa_multivariate(
    post_patterns: PostPatterns,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multivariate RSA: the RDM is built from the *full parcel vector* across
    all parcels (1 - Pearson r between whole-brain parcel patterns for each
    pair of posts). Returns one RSA value per subject — not per parcel.

    This is the second-order similarity analysis equivalent to Chen et al.
    Fig 8, but across subjects rather than within.

    Parameters
    ----------
    post_patterns : {post_id: (n_subjects, n_parcels)}

    Returns
    -------
    rsa_mean : scalar  group-average RSA (leave-one-out)
    rsa_subj : (n_subjects,)

    Example
    -------
    rsa_r, rsa_subj = compute_rsa_multivariate(patterns['AntiLeft'])
    print(f"Whole-brain RSA r = {rsa_r:.3f}")
    """
    post_ids   = sorted(post_patterns.keys())
    n_subjects = next(iter(post_patterns.values())).shape[0]
    data = np.stack([post_patterns[p] for p in post_ids])  # (n_posts, n_subs, n_parcels)

    # RDM per subject: (n_subjects, n_pairs)
    rdms = np.stack([
        pdist(data[:, s, :], metric="correlation")
        for s in range(n_subjects)
    ])

    rsa_subj = np.zeros(n_subjects)
    for s in range(n_subjects):
        others = (rdms.sum(axis=0) - rdms[s]) / (n_subjects - 1)
        if rdms[s].std() < 1e-10 or others.std() < 1e-10:
            rsa_subj[s] = np.nan
        else:
            rsa_subj[s] = pearsonr(rdms[s], others)[0]

    return float(np.nanmean(rsa_subj)), rsa_subj


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def fdr_correct(
    p_vals: np.ndarray,
    q     : float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Benjamini-Hochberg FDR correction, NaN-safe.

    Parameters
    ----------
    p_vals : (n_parcels,)
    q      : FDR threshold

    Returns
    -------
    rejected : bool (n_parcels,)
    p_fdr    : corrected p-values (n_parcels,)

    Example
    -------
    rejected, p_fdr = fdr_correct(p_isc, q=0.05)
    sig_parcels = parcel_names[rejected]
    """
    valid    = ~np.isnan(p_vals)
    rejected = np.zeros(len(p_vals), dtype=bool)
    p_fdr    = np.ones(len(p_vals))

    if valid.sum() > 0:
        rej_v, pfdr_v   = fdrcorrection(p_vals[valid], alpha=q)
        rejected[valid] = rej_v
        p_fdr[valid]    = pfdr_v

    return rejected, p_fdr


# ---------------------------------------------------------------------------
# Results packaging
# ---------------------------------------------------------------------------

def results_to_dataframe(
    parcel_names: List[str],
    obs         : np.ndarray,
    p_vals      : np.ndarray,
    rejected    : np.ndarray,
    p_fdr       : np.ndarray,
    subj_vals   : Optional[np.ndarray] = None,
    subjects    : Optional[List[str]]  = None,
) -> pd.DataFrame:
    """
    Package parcellated results into a tidy DataFrame.

    Parameters
    ----------
    parcel_names : list of parcel label strings
    obs          : (n_parcels,) observed statistic (ISC r or RSA r)
    p_vals       : (n_parcels,) raw p-values
    rejected     : (n_parcels,) bool significance mask
    p_fdr        : (n_parcels,) FDR-corrected p-values
    subj_vals    : optional (n_subjects, n_parcels) per-subject values
    subjects     : optional list of subject IDs (for subj_vals columns)

    Returns
    -------
    pd.DataFrame

    Example
    -------
    df = results_to_dataframe(parcel_names, isc_mean, p_isc, rejected, p_fdr,
                               subj_vals=isc_subj, subjects=cfg.subjects)
    df.sort_values('r').tail(10)
    """
    df = pd.DataFrame({
        "parcel"     : parcel_names,
        "r"          : obs,
        "p_raw"      : p_vals,
        "p_fdr"      : p_fdr,
        "significant": rejected.astype(int),
    })

    if subj_vals is not None and subjects is not None:
        for i, sub in enumerate(subjects):
            df[f"r_{sub}"] = subj_vals[i]

    return df