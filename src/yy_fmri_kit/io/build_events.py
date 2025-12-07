"""
Convert E-Prime .txt log files to BIDS-like events.tsv files.
Consider using this script before running time-shift (prior to ISC), 
to have clean event onsets and durations - interpreting HRF peaks is tricky otherwise.
"""

from pathlib import Path
import csv

def parse_eprime_to_events(txt_path):
    """
    Convert one E-Prime .txt log file into a clean BIDS-like events.tsv
    containing:
    - onset (sec from scanner trigger)
    - duration (sec)
    - trial_type (AntiLeft, AntiRight, ProLeft, ProRight)
    - stim_file (video filename)
    """

    txt_path = Path(txt_path)
    out_path = txt_path.with_suffix("").with_name(txt_path.stem + "_events.tsv")

    # Try UTF-16 first (E-Prime default), fallback to utf-8
    def open_text(path):
        for enc in ("utf-16", "utf-8"):
            try:
                return open(path, "r", encoding=enc, errors="replace")
            except UnicodeError:
                continue
        return open(path, "r", encoding="latin-1", errors="replace")

    # ------------------------------------------------------------
    # 1. Get TaskSelect + scanner trigger time
    # ------------------------------------------------------------

    task_select = None
    run_start_ms = None
    inside_frame = False
    current_proc = None

    with open_text(txt_path) as f:
        for line in f:
            line = line.strip()

            if line.startswith("*** LogFrame Start ***"):
                inside_frame = True
                current_proc = None
                continue

            if line.startswith("*** LogFrame End ***"):
                inside_frame = False
                current_proc = None
                continue

            if not inside_frame and "TaskSelect:" in line:
                _, val = line.split(":", 1)
                task_select = val.strip()

            if inside_frame and ":" in line:
                key, val = [p.strip() for p in line.split(":", 1)]

                if key == "Procedure":
                    current_proc = val

                if current_proc == "Introduction" and key == "Wait4Scanner.OffsetTime":
                    run_start_ms = float(val)

    if run_start_ms is None:
        raise ValueError(f"No scanner trigger found in {txt_path}")

    # ------------------------------------------------------------
    # 2. Extract video trial onsets
    # ------------------------------------------------------------

    events = []
    inside_frame = False
    current_proc = None
    current_frame = {}

    with open_text(txt_path) as f:
        for line in f:
            line = line.strip()

            if line.startswith("*** LogFrame Start ***"):
                inside_frame = True
                current_proc = None
                current_frame = {}
                continue

            if line.startswith("*** LogFrame End ***"):

                if (
                    current_proc == "TrialProcStimRec"
                    and "MovieDisplay2.OnsetTime" in current_frame
                    and "Stim1" in current_frame
                    and "Duration1" in current_frame
                ):
                    onset_ms = float(current_frame["MovieDisplay2.OnsetTime"])
                    onset_sec = (onset_ms - run_start_ms) / 1000.0

                    duration_ms = float(current_frame["Duration1"])
                    duration_sec = duration_ms / 1000.0

                    events.append(
                        dict(
                            onset=onset_sec,
                            duration=duration_sec,
                            trial_type=task_select,
                            stim_file=current_frame["Stim1"]
                        )
                    )

                inside_frame = False
                current_proc = None
                current_frame = {}
                continue

            if inside_frame and ":" in line:
                key, val = [p.strip() for p in line.split(":", 1)]
                if key == "Procedure":
                    current_proc = val
                current_frame[key] = val

    # ------------------------------------------------------------
    # 3. Write events.tsv
    # ------------------------------------------------------------

    out_path = txt_path.with_suffix("").with_name(txt_path.stem + "_events.tsv")

    with open(out_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(
            f_out,
            fieldnames=["onset", "duration", "trial_type", "stim_file"],
            delimiter="\t"
        )
        writer.writeheader()
        for ev in events:
            writer.writerow(ev)

    print(f"Saved events to {out_path} (n={len(events)})")

    return out_path
