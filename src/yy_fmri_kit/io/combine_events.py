from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import pandas as pd


@dataclass(frozen=True)
class Event:
    """One post (event) within a run."""
    post_id: str                 # e.g., "pro_left_19"
    onset_s: float               # seconds from run start
    duration_s: float            # seconds
    run: str                     # e.g., "ProLeft" (post type) or "ProLeft_run1"
    subject: str                 # e.g., "0027" or "sub-0027"
    scan_name: str | None = None # e.g., "Potilical_views"


_FILENAME_RE = re.compile(
    r"^(?P<scan>.+)-YY_PL-(?P<sub>\d{4})-(?P<run>[A-Za-z]+)-(?P<runnum>\d+)_events\.(?P<ext>tsv|txt)$"
)


def parse_event_filename(path: Path) -> dict:
    """
    Parse filenames like:
      Potilical_views-YY_PL-0027-ProLeft-1_events.tsv
    Returns dict with scan_name, subject, run, run_num.
    """
    m = _FILENAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Unrecognized event filename format: {path.name}")
    d = m.groupdict()
    return {
        "scan_name": d["scan"],
        "subject": d["sub"],          # "0027"
        "run": d["run"],              # "ProLeft"
        "run_num": int(d["runnum"]),  # 1
        "ext": d["ext"],
    }


def read_events_file(path: Path, *, run_label: str, subject: str, scan_name: str | None) -> list[Event]:
    """
    Reads the TSV/TXT events file and returns a list[Event].

    Expects columns (based on your example):
      onset, duration, trial_type, stim_file
    """
    df = pd.read_csv(path, sep="\t")

    required = {"onset", "duration", "stim_file"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}. Found: {list(df.columns)}")
    
    df = df[df["stim_file"] != "n/a"].copy()

    events: list[Event] = []
    for _, row in df.iterrows():
        stim = str(row["stim_file"])
        # robust: handle windows backslashes in stim_file
        post_id = Path(stim.replace("\\", "/")).stem  # "pro_left_19"

        events.append(
            Event(
                post_id=post_id,
                onset_s=float(row["onset"]),
                duration_s=float(row["duration"]),
                run=run_label,
                subject=subject,
                scan_name=scan_name,
            )
        )
    return events


def build_events(event_folder: Path, subjects: list[str]) -> pd.DataFrame:
    """
    Build a dataframe of events from e-prime event files.

    Args:
        event_folder: folder containing event files (tsv/txt).
        subjects: list of subject ids to include. Accepts ["0027", ...] or ["sub-0027", ...].
    Returns:
        DataFrame with columns: post_id, onset_s, duration_s, run, subject, scan_name
    """
    # normalize subject filters: allow passing "sub-0027" or "0027"
    subjects_norm = {s.replace("sub-", "") for s in subjects}

    all_events: list[Event] = []
    for path in sorted(event_folder.glob("*_events.tsv")) + sorted(event_folder.glob("*_events.txt")):
        try:
            meta = parse_event_filename(path)
        except ValueError:
            continue  # ignore unrelated files

        if meta["subject"] not in subjects_norm:
            continue

        # choose how you want to encode run:
        # Option A: just post-type "ProLeft"
        run_label = meta["run"]
        # Option B (if you ever have multiple of same type): f"{meta['run']}_run{meta['run_num']}"

        all_events.extend(
            read_events_file(
                path,
                run_label=run_label,
                subject=meta["subject"],
                scan_name=meta["scan_name"],
            )
        )

    events_df = pd.DataFrame([e.__dict__ for e in all_events])

    if not events_df.empty:
        events_df = events_df[["post_id", "onset_s", "duration_s", "run", "subject", "scan_name"]]

    return events_df
