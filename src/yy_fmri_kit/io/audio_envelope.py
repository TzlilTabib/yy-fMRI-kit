"""
Audio envelope extraction using Hilbert transform.

Main Functions:
- load_wav(): Loads audio, converts to mono float format, removes DC offset
- compute_envelope(): Applies the Hilbert transform to extract amplitude envelope
- concat_envelopes_in_order(): Processes multiple WAV files and concatenates their envelopes in sequence
- downsample_envelope_to_tr(): Downsamples envelope to match fMRI TR (repetition time) by averaging samples within each TR window
- zscore(): Z-score normalization for statistical analysis

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import hilbert, butter, filtfilt, resample_poly


@dataclass(frozen=True)
class EnvelopeConfig:
    speech_band: Optional[tuple[float, float]] = None  # e.g. (200, 2000)
    env_lowpass_hz: Optional[float] = 10.0             # smooth envelope (<10 Hz)
    target_sr: Optional[int] = None                    # e.g. 16000; if None keep native
    dtype: str = "float32"


def _to_mono(wave: np.ndarray) -> np.ndarray:
    # wave shape can be (n,) or (n, ch)
    if wave.ndim == 1:
        return wave
    if wave.ndim == 2:
        return wave.mean(axis=1)
    raise ValueError(f"Unexpected WAV shape: {wave.shape}")


def _butter_bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, x)


def _butter_lowpass(x: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, x)


def _resample_to_sr(x: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    """
    Rational resampling using polyphase filtering (good anti-aliasing).
    """
    if fs_in == fs_out:
        return x
    # Find a rational approximation to fs_out/fs_in
    from fractions import Fraction
    frac = Fraction(fs_out, fs_in).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    return resample_poly(x, up, down)


def load_wav(path: str | Path) -> tuple[int, np.ndarray]:
    """
    Returns (sample_rate, mono_wave_float64)
    """
    sr, w = wavfile.read(str(path))
    w = _to_mono(w)

    # Convert to float64 in [-1, 1] approximately
    if np.issubdtype(w.dtype, np.integer):
        maxv = np.iinfo(w.dtype).max
        w = w.astype(np.float64) / maxv
    else:
        w = w.astype(np.float64)

    # Remove DC offset
    w = w - np.mean(w)
    return int(sr), w


def compute_envelope(wave: np.ndarray, sr: int, cfg: EnvelopeConfig) -> tuple[int, np.ndarray]:
    """
    Hilbert amplitude envelope with optional bandpass and lowpass smoothing.
    Returns (sr_used, envelope)
    """
    x = wave

    # optional resample first
    sr_used = sr
    if cfg.target_sr is not None:
        x = _resample_to_sr(x, sr, cfg.target_sr).astype(np.float64, copy=False)
        sr_used = cfg.target_sr

    # optional bandpass (e.g., speech energy)
    if cfg.speech_band is not None:
        lo, hi = cfg.speech_band
        x = _butter_bandpass(x, sr_used, lo, hi)

    env = np.abs(hilbert(x))

    # optional smoothing of envelope
    if cfg.env_lowpass_hz is not None:
        env = _butter_lowpass(env, sr_used, cfg.env_lowpass_hz)

    return sr_used, env.astype(cfg.dtype, copy=False)


def wav_paths_from_events_tsv(events_tsv: str | Path, wav_dir: str | Path) -> list[Path]:
    """
    Returns WAV paths in stimulus presentation order based on the events.tsv.
    """
    events_tsv = Path(events_tsv)
    wav_dir = Path(wav_dir)

    df = pd.read_csv(events_tsv, sep="\t")
    if "onset" not in df.columns or "stim_file" not in df.columns:
        raise ValueError(f"events.tsv missing required columns: {events_tsv}")

    df = df.sort_values("onset").reset_index(drop=True)

    # Convert "Stim1" filename (could be mp4/avi/etc) -> wav stem
    stim_files = df["stim_file"].astype(str).tolist()
    wav_paths = []
    for s in stim_files:
        stem = Path(s).stem  # removes extension
        wav_paths.append(wav_dir / f"{stem}.wav")

    missing = [p for p in wav_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Some WAV files are missing for stim_file entries. First few missing:\n"
            + "\n".join(str(p) for p in missing[:10])
        )

    return wav_paths


def concat_envelopes_in_order(
    wav_paths_in_order: Sequence[str | Path],
    cfg: EnvelopeConfig,
) -> tuple[int, np.ndarray, list[Path]]:
    """
    Loads WAVs in the provided order, computes envelope per clip, concatenates.
    Returns (sr_used, env_full, resolved_paths)
    """
    envs = []
    sr_used: Optional[int] = None
    resolved = [Path(p) for p in wav_paths_in_order]

    for p in resolved:
        sr, w = load_wav(p)
        sr_e, env = compute_envelope(w, sr, cfg)
        if sr_used is None:
            sr_used = sr_e
        elif sr_e != sr_used:
            raise ValueError(
                f"Sample rate mismatch after resampling: got {sr_e}, expected {sr_used}. "
                f"Check cfg.target_sr."
            )
        envs.append(env)

    assert sr_used is not None
    env_full = np.concatenate(envs, axis=0)
    return sr_used, env_full, resolved


def downsample_envelope_to_tr(env: np.ndarray, sr: int, tr: float) -> np.ndarray:
    """
    Convert envelope sampled at `sr` to one value per TR (mean within each TR).
    This is robust and easy to interpret.
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