"""
parcellated_isc_rsa.py
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
    "load_data",
    "extract_post_patterns",
    "compute_isc",
    "compute_rsa",
    "permutation_test",
    "load_affiliation",
    "make_behavioral_rdm",
    "compute_brain_behavior_rsa",
    "permutation_test_brain_behavior",
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

    # Drop fully duplicate columns (same name appearing more than once)
    df = df.loc[:, ~df.columns.duplicated()]

    # Validate required columns exist
    needed = [cfg.subject_col, cfg.run_col, cfg.post_col,
              cfg.onset_col, cfg.duration_col]
    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        raise KeyError(
            f"[load_events] Columns not found in events CSV: {missing_cols}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # Keep all columns (no renaming) — downstream code uses cfg.*_col to access
    df = df[df[cfg.subject_col].isin(cfg.subjects)]
    df = df[df[cfg.run_col].isin(cfg.run_types)]

    # Drop rows with missing or non-post post_id (fixation/black leftovers)
    df = df[df[cfg.post_col].notna()]
    df = df[df[cfg.post_col].astype(str) != "n/a"]
    df = df[df[cfg.post_col].astype(str) != "nan"]
    df[cfg.post_col] = df[cfg.post_col].astype(str)

    df = df.reset_index(drop=True)

    print(f"[load_events] {len(df)} events | "
          f"{df[cfg.subject_col].nunique()} subjects | "
          f"{df[cfg.run_col].nunique()} run types | "
          f"{df[cfg.post_col].nunique()} unique post IDs")
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


def load_data(
    cfg: Config,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], pd.DataFrame]]:
    """
    Convenience wrapper: load events CSV and all timeseries TSVs in one call.

    Parameters
    ----------
    cfg : Config

    Returns
    -------
    events_df : pd.DataFrame  (output of load_events)
    ts_dict   : dict          (output of load_timeseries)

    Example
    -------
    events_df, ts_dict = load_data(cfg)
    parcel_names = ts_dict[(cfg.subjects[0], cfg.run_types[0])].columns.tolist()
    """
    events_df = load_events(cfg)
    ts_dict   = load_timeseries(cfg)
    return events_df, ts_dict


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
        run_events = events_df[events_df[cfg.run_col] == run_type]
        post_ids   = sorted(run_events[cfg.post_col].unique())

        # Which subjects have BOTH a TSV and events for this run?
        subs_with_events = set(run_events[cfg.subject_col].unique())
        valid_subs = [s for s in cfg.subjects
                      if (s, run_type) in ts_dict and s in subs_with_events]
        if len(valid_subs) < 2:
            print(f"[extract] {run_type}: skipping — only {len(valid_subs)} subjects")
            continue

        run_patterns: PostPatterns = {}

        for post_id in post_ids:
            sub_vectors = []

            for sub in valid_subs:
                ts  = ts_dict[(sub, run_type)].values  # (n_trs, n_parcels)
                evt = run_events[
                    (run_events[cfg.subject_col] == sub) &
                    (run_events[cfg.post_col]    == post_id)
                ]
                if evt.empty:
                    sub_vectors.append(None)
                    continue

                row  = evt.iloc[0]
                trs  = _trs_for_event(
                    row[cfg.onset_col], row[cfg.duration_col],
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
    total    = data.sum(axis=1)  # (n_posts, n_parcels)

    for s in range(n_subjects):
        target = data[:, s, :]                               # (n_posts, n_parcels)
        others = (total - target) / (n_subjects - 1)        # (n_posts, n_parcels)

        # Vectorised Pearson r across the posts dimension for all parcels at once
        t_c = target - target.mean(axis=0)
        o_c = others - others.mean(axis=0)
        num = (t_c * o_c).sum(axis=0)
        den = np.sqrt((t_c ** 2).sum(axis=0) * (o_c ** 2).sum(axis=0))
        isc_subj[s] = np.where(den > 1e-10, num / den, 0.0)

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
    For each parcel, build each subject's RDM from their post-level activation
    profile, then correlate each subject's RDM with the leave-one-out mean RDM
    of all other subjects.

    For a single parcel, each subject has a vector of length n_posts (their
    activation to each post). The RDM is pairwise absolute differences between
    posts — i.e. how differently did this parcel respond to each pair of posts.

    Parameters
    ----------
    data : (n_posts, n_subjects, n_parcels)

    Returns
    -------
    rsa_mean : (n_parcels,)
    """
    n_posts, n_subjects, n_parcels = data.shape

    # Build all RDMs at once:
    # For each parcel p and subject s, rdms[s, p] is the upper-triangle vector
    # of pairwise absolute differences across posts.
    # data[:, s, p] is shape (n_posts,) — activation per post for subject s, parcel p
    # We want pairwise |x_i - x_j| for all post pairs.

    # Efficient: use broadcasting for absolute differences
    # data shape: (n_posts, n_subjects, n_parcels)
    # diff[i,j,s,p] = |data[i,s,p] - data[j,s,p]|
    d = data[:, np.newaxis, :, :] - data[np.newaxis, :, :, :]  # (n_posts, n_posts, n_subs, n_parcels)
    d = np.abs(d)

    # Extract upper triangle indices
    tri_i, tri_j = np.triu_indices(n_posts, k=1)
    # rdms shape: (n_pairs, n_subjects, n_parcels)
    rdms = d[tri_i, tri_j, :, :]
    # Transpose to (n_subjects, n_parcels, n_pairs) for easier indexing
    rdms = rdms.transpose(1, 2, 0)   # (n_subjects, n_parcels, n_pairs)

    rsa_subj = np.zeros((n_subjects, n_parcels))
    rdms_sum = rdms.sum(axis=0)       # (n_parcels, n_pairs)

    for s in range(n_subjects):
        others = (rdms_sum - rdms[s]) / (n_subjects - 1)  # (n_parcels, n_pairs)
        subj   = rdms[s]                                    # (n_parcels, n_pairs)

        # Vectorised Pearson r between subj[p] and others[p] for all parcels
        s_c   = subj   - subj.mean(axis=1, keepdims=True)
        o_c   = others - others.mean(axis=1, keepdims=True)
        num   = (s_c * o_c).sum(axis=1)
        den   = np.sqrt((s_c**2).sum(axis=1) * (o_c**2).sum(axis=1))
        rsa_subj[s] = np.where(den > 1e-10, num / den, np.nan)

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

    # Reuse the fast vectorised implementation from _compute_rsa_from_array
    rsa_mean = _compute_rsa_from_array(data)

    # Also compute per-subject values for reporting
    d        = data[:, np.newaxis, :, :] - data[np.newaxis, :, :, :]
    d        = np.abs(d)
    tri_i, tri_j = np.triu_indices(n_posts, k=1)
    rdms     = d[tri_i, tri_j, :, :].transpose(1, 2, 0)  # (n_subs, n_parcels, n_pairs)
    rdms_sum = rdms.sum(axis=0)

    rsa_subj = np.zeros((n_subjects, n_parcels))
    for s in range(n_subjects):
        others  = (rdms_sum - rdms[s]) / (n_subjects - 1)
        subj    = rdms[s]
        s_c     = subj   - subj.mean(axis=1, keepdims=True)
        o_c     = others - others.mean(axis=1, keepdims=True)
        num     = (s_c * o_c).sum(axis=1)
        den     = np.sqrt((s_c**2).sum(axis=1) * (o_c**2).sum(axis=1))
        rsa_subj[s] = np.where(den > 1e-10, num / den, np.nan)

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


# ---------------------------------------------------------------------------
# Brain-behavior RSA (subject × subject)
# ---------------------------------------------------------------------------

def make_behavioral_rdm(
    affiliation  : Dict[str, float],
    subjects     : List[str],
    scale_max    : float = 100.0,
) -> np.ndarray:
    """
    Build a subject × subject behavioral similarity matrix from a continuous
    political affiliation score.

    Similarity = 1 - |score_i - score_j| / scale_max
    So subjects with identical scores get similarity = 1,
    and subjects at opposite ends get similarity = 0.

    Parameters
    ----------
    affiliation : {subject_id: score}  scores on a 0–100 scale
                  (0 = far right, 100 = far left)
    subjects    : ordered list of subject IDs (must match neural data order)
    scale_max   : maximum possible difference (default 100)

    Returns
    -------
    beh_sim : (n_subjects, n_subjects) symmetric similarity matrix

    Example
    -------
    affiliation = {
        'sub-1': 20, 'sub-6': 75, 'sub-20': 45, ...
    }
    beh_sim = make_behavioral_rdm(affiliation, cfg.subjects)
    """
    n    = len(subjects)
    sims = np.zeros((n, n))
    for i, si in enumerate(subjects):
        for j, sj in enumerate(subjects):
            sims[i, j] = 1 - abs(affiliation[si] - affiliation[sj]) / scale_max
    return sims


def _upper_triangle(mat: np.ndarray) -> np.ndarray:
    """Return the upper triangle (excluding diagonal) as a 1-D vector."""
    idx = np.triu_indices(mat.shape[0], k=1)
    return mat[idx]


def compute_brain_behavior_rsa(
    post_patterns : PostPatterns,
    beh_sim       : np.ndarray,
    subjects      : List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each parcel, correlate the neural subject × subject similarity matrix
    with the behavioral similarity matrix.

    Neural similarity for a parcel: average each subject's pattern across all
    posts → one vector per subject → Pearson r between every subject pair.

    Parameters
    ----------
    post_patterns : {post_id: (n_subjects, n_parcels)}
    beh_sim       : (n_subjects, n_subjects) from make_behavioral_rdm()
    subjects      : ordered subject list (must match axis-0 of post_patterns arrays)

    Returns
    -------
    rsa_r    : (n_parcels,)  Pearson r between neural and behavioral similarity
    neural_sim : (n_subjects, n_subjects, n_parcels) neural similarity matrices
                 (useful for visualisation)

    Example
    -------
    beh_sim  = make_behavioral_rdm(affiliation, cfg.subjects)
    rsa_r, neural_sim = compute_brain_behavior_rsa(
        patterns['AntiLeft'], beh_sim, cfg.subjects)
    """
    post_ids   = sorted(post_patterns.keys())
    n_subjects, n_parcels = next(iter(post_patterns.values())).shape

    # data: (n_posts, n_subjects, n_parcels)
    data = np.stack([post_patterns[p] for p in post_ids])

    # Each subject's profile per parcel = activation across n_posts
    # parcel_profiles: (n_subjects, n_posts, n_parcels)
    parcel_profiles = data.transpose(1, 0, 2)  # (n_subjects, n_posts, n_parcels)

    # Build neural similarity matrix vectorised:
    # For each parcel, correlate every subject pair's activation profile (length n_posts)
    # neural_sim: (n_subjects, n_subjects, n_parcels)
    neural_sim = np.zeros((n_subjects, n_subjects, n_parcels))
    np.fill_diagonal(neural_sim[:, :, 0], 1.0)  # placeholder, filled below

    for i in range(n_subjects):
        for j in range(i, n_subjects):
            x = parcel_profiles[i]  # (n_posts, n_parcels)
            y = parcel_profiles[j]  # (n_posts, n_parcels)
            # Vectorised Pearson r across posts for all parcels at once
            x_c = x - x.mean(axis=0)
            y_c = y - y.mean(axis=0)
            num = (x_c * y_c).sum(axis=0)
            den = np.sqrt((x_c**2).sum(axis=0) * (y_c**2).sum(axis=0))
            r   = np.where(den > 1e-10, num / den, 0.0)
            neural_sim[i, j, :] = r
            neural_sim[j, i, :] = r  # symmetric

    # Correlate upper triangle of neural sim with behavioral sim per parcel
    beh_vec = _upper_triangle(beh_sim)  # (n_pairs,)
    rsa_r   = np.zeros(n_parcels)

    for p in range(n_parcels):
        neu_vec = _upper_triangle(neural_sim[:, :, p])
        if neu_vec.std() < 1e-10 or beh_vec.std() < 1e-10:
            rsa_r[p] = np.nan
        else:
            rsa_r[p] = pearsonr(neu_vec, beh_vec)[0]

    return rsa_r, neural_sim


def permutation_test_brain_behavior(
    post_patterns : PostPatterns,
    beh_sim       : np.ndarray,
    subjects      : List[str],
    cfg           : "Config",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Permutation test for brain-behavior RSA.

    Shuffles subject labels on the behavioral similarity matrix to break
    the brain-behavior correspondence while preserving the structure of each.

    Parameters
    ----------
    post_patterns : {post_id: (n_subjects, n_parcels)}
    beh_sim       : (n_subjects, n_subjects) behavioral similarity matrix
    subjects      : ordered subject list
    cfg           : Config (uses cfg.n_perms, cfg.seed)

    Returns
    -------
    obs_r     : (n_parcels,)           observed brain-behavior RSA r
    p_vals    : (n_parcels,)           permutation p-value
    null_dist : (n_perms, n_parcels)   full null distribution

    Example
    -------
    obs_r, p_vals, null = permutation_test_brain_behavior(
        patterns['AntiLeft'], beh_sim, cfg.subjects, cfg)
    rejected, p_fdr = fdr_correct(p_vals, cfg.fdr_q)
    """
    rng = np.random.default_rng(cfg.seed)

    obs_r, _ = compute_brain_behavior_rsa(post_patterns, beh_sim, subjects)
    n_parcels  = len(obs_r)
    n_subjects = len(subjects)
    null_dist  = np.zeros((cfg.n_perms, n_parcels))

    for i in range(cfg.n_perms):
        # Shuffle subject labels on the behavioral matrix
        perm_idx   = rng.permutation(n_subjects)
        beh_perm   = beh_sim[np.ix_(perm_idx, perm_idx)]
        null_r, _  = compute_brain_behavior_rsa(post_patterns, beh_perm, subjects)
        null_dist[i] = null_r

    p_vals = np.mean(null_dist >= obs_r, axis=0)
    return obs_r, p_vals, null_dist


# ---------------------------------------------------------------------------
# Load behavioral data
# ---------------------------------------------------------------------------

def load_affiliation(
    csv_path  : Path,
    subjects  : List[str],
    score_col : str = "camp_support",
    id_col    : str = "subject_num",
) -> Dict[str, float]:
    """
    Load political affiliation scores from the behavioral CSV and map them
    to BIDS subject IDs.

    Handles duplicate subject entries by averaging their scores.

    Parameters
    ----------
    csv_path  : path to the CSV (e.g. political_attitude_q_08122025.csv)
    subjects  : list of BIDS subject IDs (e.g. ['sub-1', 'sub-6', ...])
                subject numbers are extracted from these ('sub-6' → 6)
    score_col : column containing the affiliation score (default 'camp_support')
                0 = far right, 100 = far left
    id_col    : column containing the numeric subject ID (default 'subject_num')

    Returns
    -------
    affiliation : {'sub-N': score}  only for subjects present in both
                  the CSV and the subjects list

    Also prints a warning for any fMRI subjects missing from the CSV.

    Example
    -------
    affiliation = load_affiliation(
        Path("political_attitude_q_08122025.csv"),
        subjects = cfg.subjects,
    )
    beh_sim = make_behavioral_rdm(affiliation, cfg.subjects)
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # Average duplicates
    df_agg = (
        df.groupby(id_col)[score_col]
        .mean()
        .reset_index()
    )

    # Support two id_col formats:
    #   - numeric (e.g. subject_num=6)  → match against int extracted from 'sub-6'
    #   - string  (e.g. bids_id='sub-6') → match directly
    sample_val = df_agg[id_col].iloc[0]
    use_numeric = isinstance(sample_val, (int, float)) and not isinstance(sample_val, bool)

    affiliation = {}
    missing     = []

    for bids_id in subjects:
        if use_numeric:
            key = int(bids_id.split("-")[1])
        else:
            key = bids_id
        row = df_agg[df_agg[id_col] == key]
        if row.empty:
            missing.append(bids_id)
        else:
            affiliation[bids_id] = float(row[score_col].values[0])

    if missing:
        print(f"[load_affiliation] ⚠  No behavioral data for: {missing}")

    print(f"[load_affiliation] Loaded scores for {len(affiliation)} subjects")
    for sub, score in sorted(affiliation.items(),
                              key=lambda x: int(x[0].split('-')[1])):
        print(f"  {sub}: {score_col} = {score:.1f}")

    return affiliation