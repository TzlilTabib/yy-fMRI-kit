"""
Audio envelope extraction (Hilbert) aligned to fMRI run time using events CSV.

This module is designed for your events file structure:
    post_id, onset_s, duration_s, run, subject, scan_name, bids_id

Core idea:
- Compute Hilbert amplitude envelope for each post WAV.
- Place each post envelope into a *continuous run timeline* using onset_s/duration_s.
  (This preserves small silent gaps, e.g. ~50 ms, and prevents cumulative drift.)
- Optionally downsample to TR and z-score for fMRI comparison (e.g., left Heschl's gyrus).

Typical usage (one run):
    cfg = EnvelopeConfig(speech_band=None, env_lowpass_hz=10.0, target_sr=16000)
    sr_used, env_run, df = build_run_envelope(events_csv, wav_dir, cfg)
    env_tr = zscore(downsample_envelope_to_tr(env_run, sr_used, tr=TR))

Assumptions:
- Your WAVs are named "{post_id}.wav" (e.g., "anti_left_12.wav") inside wav_dir.
  If not, edit `wav_from_post_id()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, hilbert, resample_poly


# =========================
# Configuration
# =========================

@dataclass(frozen=True)
class EnvelopeConfig:
    speech_band: Optional[tuple[float, float]] = None  # e.g. (200, 2000)
    env_lowpass_hz: Optional[float] = 10.0             # smooth envelope (<~10 Hz)
    target_sr: Optional[int] = None                    # e.g. 16000; if None keep native
    dtype: str = "float32"


# =========================
# WAV I/O + preprocessing
# =========================

def _to_mono(wave: np.ndarray) -> np.ndarray:
    if wave.ndim == 1:
        return wave
    if wave.ndim == 2:
        return wave.mean(axis=1)
    raise ValueError(f"Unexpected WAV shape: {wave.shape}")


def _butter_bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    if not (0 < lo < hi < nyq):
        raise ValueError(f"Invalid bandpass: lo={lo}, hi={hi}, nyq={nyq}")
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, x)


def _butter_lowpass(x: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    if not (0 < cutoff < nyq):
        raise ValueError(f"Invalid lowpass cutoff={cutoff}, nyq={nyq}")
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, x)


def _resample_to_sr(x: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    """Rational resampling using polyphase filtering (good anti-aliasing)."""
    if fs_in == fs_out:
        return x
    from fractions import Fraction
    frac = Fraction(fs_out, fs_in).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    return resample_poly(x, up, down)


def load_wav(path: str | Path) -> tuple[int, np.ndarray]:
    """
    Returns (sample_rate, mono_wave_float64) with DC removed.
    """
    sr, w = wavfile.read(str(path))
    w = _to_mono(w)

    # Convert to float64 roughly in [-1, 1]
    if np.issubdtype(w.dtype, np.integer):
        maxv = np.iinfo(w.dtype).max
        w = w.astype(np.float64) / maxv
    else:
        w = w.astype(np.float64)

    # Remove DC offset
    w = w - np.mean(w)
    return int(sr), w


# =========================
# Envelope computation
# =========================

def compute_envelope(
    wave: np.ndarray,
    sr: int,
    cfg: EnvelopeConfig,
    *,
    return_raw: bool = False,
):
    x = wave
    sr_used = sr

    if cfg.target_sr is not None:
        x = _resample_to_sr(x, sr, cfg.target_sr)
        sr_used = cfg.target_sr

    if cfg.speech_band is not None:
        x = _butter_bandpass(x, sr_used, *cfg.speech_band)

    env_raw = np.abs(hilbert(x))

    env = env_raw
    if cfg.env_lowpass_hz is not None:
        env = _butter_lowpass(env, sr_used, cfg.env_lowpass_hz)

    if return_raw:
        return sr_used, env.astype(cfg.dtype), env_raw.astype(cfg.dtype)
    else:
        return sr_used, env.astype(cfg.dtype)


# =========================
# Events → run-aligned envelope
# =========================

def wav_from_post_id(post_id: str, wav_dir: str | Path) -> Path:
    """
    Map events 'post_id' → WAV filename.
    Default: "{post_id}.wav" (e.g., anti_left_12.wav).
    Edit this if your naming differs.
    """
    wav_dir = Path(wav_dir)
    p = wav_dir / f"{post_id}.wav"
    if not p.exists():
        raise FileNotFoundError(f"Missing WAV for post_id='{post_id}': {p}")
    return p


def load_events_csv(events_csv: str | Path) -> pd.DataFrame:
    """
    Load and validate your events CSV structure.
    """
    df = pd.read_csv(events_csv)
    required = {"post_id", "onset_s", "duration_s"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Events CSV missing columns: {missing}")

    df = df.sort_values("onset_s").reset_index(drop=True).copy()

    # Ensure numeric
    df["onset_s"] = pd.to_numeric(df["onset_s"], errors="raise")
    df["duration_s"] = pd.to_numeric(df["duration_s"], errors="raise")
    return df


def compute_gap_stats(df: pd.DataFrame) -> dict:
    """
    Compute gap-to-next in seconds (positive values = silence/gap).
    """
    end = df["onset_s"] + df["duration_s"]
    gaps = df["onset_s"].shift(-1) - end
    gaps = gaps.dropna().astype(float)

    return {
        "n_posts": int(len(df)),
        "gap_mean_s": float(gaps.mean()) if len(gaps) else float("nan"),
        "gap_std_s": float(gaps.std(ddof=1)) if len(gaps) > 1 else float("nan"),
        "gap_min_s": float(gaps.min()) if len(gaps) else float("nan"),
        "gap_max_s": float(gaps.max()) if len(gaps) else float("nan"),
    }

def infer_post_type_from_run_label(run_label: str) -> str:
    """
    By your example, run contains the type already (AntiLeft, ProRight, etc.).
    If you later change naming, edit this function.
    """
    return str(run_label)

# =========================
# Single-run builder
# =========================

def build_run_envelope(
    df_run: pd.DataFrame,
    wav_dir: str | Path,
    cfg: EnvelopeConfig,
    *,
    run_duration_s: float | None = None,
) -> tuple[int, np.ndarray, pd.DataFrame, dict]:
    """
    Build continuous run-aligned envelope for ONE run.
    Assumes df_run contains ONLY that run's rows, with onset_s starting at 0-ish.
    """
    df_run = df_run.sort_values("onset_s").reset_index(drop=True).copy()

    # Resolve wav paths
    wav_paths = [wav_from_post_id(pid, wav_dir) for pid in df_run["post_id"]]
    df_run["wav_path"] = [str(p) for p in wav_paths]

    clip_env = []
    clip_env_raw = []
    sr_used: Optional[int] = None

    for p in wav_paths:
        sr, w = load_wav(p)
        sr_e, env_smooth, env_raw = compute_envelope(
            w, sr, cfg, return_raw=True
        )

        if sr_used is None:
            sr_used = sr_e
        elif sr_e != sr_used:
            raise ValueError("Sample-rate mismatch; set cfg.target_sr to a fixed value.")

        clip_env.append(env_smooth.astype(np.float32, copy=False))
        clip_env_raw.append(env_raw.astype(np.float32, copy=False))

    assert sr_used is not None

    last_end = float((df_run["onset_s"] + df_run["duration_s"]).max())
    if run_duration_s is None:
        run_duration_s = last_end

    n_total = int(np.ceil(run_duration_s * sr_used))
    env_run = np.zeros(n_total, dtype=np.float32)
    env_run_raw = np.zeros(n_total, dtype=np.float32)

    # Place each clip using onset/duration (preserves gaps)
    # Place each clip using onset/duration (preserves gaps)
    for i, row in df_run.iterrows():
        onset = float(row["onset_s"])
        dur = float(row["duration_s"])

        start = int(round(onset * sr_used))
        expected_len = int(round(dur * sr_used))

        e = clip_env[i]
        r = clip_env_raw[i]

        # enforce duration window for BOTH smooth and raw
        if len(e) > expected_len:
            e = e[:expected_len]
            r = r[:expected_len]
        elif len(e) < expected_len:
            pad = expected_len - len(e)
            e = np.pad(e, (0, pad), mode="constant")
            r = np.pad(r, (0, pad), mode="constant")
        else:
            # lengths match; still ensure raw matches too
            if len(r) > expected_len:
                r = r[:expected_len]
            elif len(r) < expected_len:
                r = np.pad(r, (0, expected_len - len(r)), mode="constant")

        end = min(start + expected_len, n_total)
        if end > start:
            n = end - start
            env_run[start:end] = e[:n]
            env_run_raw[start:end] = r[:n]

    info = compute_gap_stats(df_run)
    info.update({"sr_used": int(sr_used), "run_duration_s": float(run_duration_s)})
    return sr_used, env_run, env_run_raw, df_run, info

# =========================
# Multi-run orchestration (your need)
# =========================

def split_events_by_run(
    events_csv: str | Path,
    *,
    expected_posts_per_run: int = 18,
    enforce_one_type_per_run: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Returns dict: run_label -> df_run (only that run).

    Validations:
    - each run has expected_posts_per_run rows (default 18)
    - optionally, each run label defines a single type (by definition here)
    """
    df = load_events_csv(events_csv)

    runs = {}
    problems = []

    for run_label, df_run in df.groupby("run", sort=False):
        df_run = df_run.copy()

        # Validate count
        if expected_posts_per_run is not None and len(df_run) != expected_posts_per_run:
            problems.append(f"Run '{run_label}' has {len(df_run)} rows (expected {expected_posts_per_run}).")

        # Validate "one post type per run"
        if enforce_one_type_per_run:
            # simplest: post type IS the run label
            inferred_type = infer_post_type_from_run_label(run_label)
            df_run["post_type"] = inferred_type

            # If you ever encode type somewhere else, enforce uniqueness here:
            if df_run["post_type"].nunique() != 1:
                problems.append(f"Run '{run_label}' contains multiple post types.")

        runs[str(run_label)] = df_run

    if problems:
        raise ValueError("Events file failed run validation:\n- " + "\n- ".join(problems))

    return runs


# =========================
# TR alignment helpers
# =========================

def downsample_envelope_to_tr(env: np.ndarray, sr: int, tr: float) -> np.ndarray:
    """
    Convert envelope sampled at `sr` to one value per TR (mean within each TR).
    """
    if tr <= 0:
        raise ValueError("TR must be > 0")

    samples_per_tr = int(round(sr * tr))
    if samples_per_tr <= 0:
        raise ValueError("TR too small for this sample rate")

    n_tr = len(env) // samples_per_tr
    if n_tr < 1:
        raise ValueError("Envelope shorter than one TR")

    trimmed = env[: n_tr * samples_per_tr]
    env_tr = trimmed.reshape(n_tr, samples_per_tr).mean(axis=1)
    return env_tr.astype(np.float32, copy=False)


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return ((x - x.mean()) / (x.std() + 1e-12)).astype(np.float32, copy=False)


def lag_corr(x: np.ndarray, y: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Correlate x and y over lags in [-max_lag, +max_lag] on the TR grid.
    Positive lag means x is shifted forward relative to y.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have same shape. got {x.shape} vs {y.shape}")

    lags = np.arange(-max_lag, max_lag + 1)
    r = np.zeros_like(lags, dtype=np.float64)

    for k, lag in enumerate(lags):
        if lag < 0:
            xs, ys = x[:lag], y[-lag:]
        elif lag > 0:
            xs, ys = x[lag:], y[:-lag]
        else:
            xs, ys = x, y

        if xs.size < 5 or xs.std() == 0 or ys.std() == 0:
            r[k] = np.nan
        else:
            r[k] = np.corrcoef(xs, ys)[0, 1]

    return lags, r

# =========================
# One-subject, all runs → env_tr per run
# =========================

def envelopes_per_run_to_tr(
    events_csv: str | Path,
    wav_dir: str | Path,
    cfg: EnvelopeConfig,
    *,
    tr: float,
    expected_posts_per_run: int = 18,
) -> dict[str, dict]:
    """
    Returns dict keyed by run label, each with:
        env_tr, env_run (smoothed), env_run_raw, df_run, info
    """
    runs = split_events_by_run(
        events_csv,
        expected_posts_per_run=expected_posts_per_run,
        enforce_one_type_per_run=True,
    )

    out: dict[str, dict] = {}
    for run_label, df_run in runs.items():
        sr_used, env_run, env_run_raw, df_run_used, info = build_run_envelope(df_run, wav_dir, cfg)
        env_tr = zscore(downsample_envelope_to_tr(env_run, sr_used, tr=tr))

        out[run_label] = {
            "env_tr": env_tr,
            "env_run": env_run,
            "env_run_raw": env_run_raw,
            "df_run": df_run_used,
            "info": info,
        }

    return out