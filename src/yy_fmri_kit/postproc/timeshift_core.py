from scipy.signal import find_peaks
import numpy as np
import re
from pathlib import Path

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



def find_first_hrf_peak(
    ts: np.ndarray,
    TR: float,
    onset_sec: float,
    search_window: tuple[float, float] = (0.0, 10.0),
    min_prominence: float = 0.3,
    zscore: bool = True,
) -> dict:
    """
    Find the first HRF peak after a known stimulus onset in an ROI timecourse.

    Returns dict with:
      - peak_time_sec    : time of first peak (sec, from scanner trigger)
      - peak_latency_sec : time after onset (sec)
      - peak_value       : value at the peak (z-score if zscore=True)
      - peak_index       : index in the original ts
    """
    ts = np.asarray(ts, dtype=float)

    if zscore:
        ts = (ts - ts.mean()) / ts.std()

    T = len(ts)
    time = np.arange(T) * TR

    # Absolute search window [onset+2, onset+10] by default
    t_start = onset_sec + search_window[0]
    t_end   = onset_sec + search_window[1]

    idx_start = np.searchsorted(time, t_start)
    idx_end   = np.searchsorted(time, t_end)

    if idx_start >= T or idx_start >= idx_end:
        raise ValueError("Search window is outside the timecourse.")

    ts_window = ts[idx_start:idx_end]
    time_window = time[idx_start:idx_end]

    # detect peaks
    peaks, props = find_peaks(ts_window, prominence=min_prominence)

    if len(peaks) == 0:
        # fall back to max in window
        local_idx = int(np.argmax(ts_window))
        peak_idx = idx_start + local_idx
    else:
        local_idx = int(peaks[0])       # earliest peak
        peak_idx = idx_start + local_idx

    peak_time_sec = float(time[peak_idx])
    peak_latency = float(peak_time_sec - onset_sec)
    peak_value = float(ts[peak_idx])

    return {
        "peak_time_sec": peak_time_sec,
        "peak_latency_sec": peak_latency,
        "peak_value": peak_value,
        "peak_index": int(peak_idx),
    }
