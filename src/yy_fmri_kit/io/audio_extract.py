"""
Audio extraction from video files using ffmpeg.
"""

from __future__ import annotations

import subprocess
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence


@dataclass(frozen=True)
class AudioExtractResult:
    video_path: Path
    wav_path: Path
    created: bool  # False if skipped because cached


class FFmpegNotFoundError(RuntimeError):
    pass


def _ensure_ffmpeg(ffmpeg_path: Optional[str] = None) -> str:
    """
    Return a usable ffmpeg executable path or raise.
    """
    exe = ffmpeg_path or shutil.which("ffmpeg")
    if not exe:
        raise FFmpegNotFoundError(
            "ffmpeg was not found. Install ffmpeg and ensure it's on PATH, "
            "or pass ffmpeg_path='/full/path/to/ffmpeg'."
        )
    return exe


def extract_wav_from_video(
    video_path: str | Path,
    out_wav_path: str | Path,
    *,
    sample_rate: int = 16000,
    mono: bool = True,
    overwrite: bool = False,
    ffmpeg_path: Optional[str] = None,
    timeout_s: int = 300,
) -> AudioExtractResult:
    """
    Extract audio from a video file into WAV using ffmpeg via subprocess.

    Parameters
    ----------
    video_path : path to .mp4/.mov/etc
    out_wav_path : path to .wav output
    sample_rate : target sampling rate (Hz)
    mono : whether to downmix to 1 channel
    overwrite : overwrite existing wav
    ffmpeg_path : optional explicit ffmpeg path
    timeout_s : subprocess timeout

    Returns
    -------
    AudioExtractResult
    """
    video_path = Path(video_path)
    out_wav_path = Path(out_wav_path)
    out_wav_path.parent.mkdir(parents=True, exist_ok=True)

    if out_wav_path.exists() and not overwrite:
        return AudioExtractResult(video_path=video_path, wav_path=out_wav_path, created=False)

    ffmpeg = _ensure_ffmpeg(ffmpeg_path)

    cmd: list[str] = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",  # keep output clean; errors still show
        "-y" if overwrite else "-n",
        "-i", str(video_path),
        "-vn",  # no video
        "-ar", str(sample_rate),
    ]
    if mono:
        cmd += ["-ac", "1"]
    cmd += [str(out_wav_path)]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or e.stdout or "").strip()
        raise RuntimeError(
            f"ffmpeg failed extracting WAV.\n"
            f"Video: {video_path}\nOut: {out_wav_path}\nError: {msg}"
        ) from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"ffmpeg timed out after {timeout_s}s.\nVideo: {video_path}\nOut: {out_wav_path}"
        ) from e

    if not out_wav_path.exists():
        raise RuntimeError(f"ffmpeg reported success but output file not found: {out_wav_path}")

    return AudioExtractResult(video_path=video_path, wav_path=out_wav_path, created=True)


def batch_extract_wav(
    video_paths: Sequence[str | Path],
    out_dir: str | Path,
    *,
    sample_rate: int = 16000,
    mono: bool = True,
    overwrite: bool = False,
    ffmpeg_path: Optional[str] = None,
) -> list[AudioExtractResult]:
    """
    Extract WAVs for a list of videos into out_dir, using the same basenames.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[AudioExtractResult] = []
    for vp in video_paths:
        vp = Path(vp)
        out_wav = out_dir / (vp.stem + ".wav")
        res = extract_wav_from_video(
            vp, out_wav,
            sample_rate=sample_rate,
            mono=mono,
            overwrite=overwrite,
            ffmpeg_path=ffmpeg_path,
        )
        results.append(res)
    return results
