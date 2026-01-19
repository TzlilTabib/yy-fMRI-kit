"""
Convert E-Prime .txt log files to BIDS-like events.tsv files.
Adds baseline events:
- beginning: Fix4sec, BlackScreen (inferred if needed)
- end: Fix10sec, BlackScreenEnd (gap until Goodbye, optional/inferred)
"""

from pathlib import Path
import csv


def parse_eprime_to_events(txt_path):
    """
    Convert one E-Prime .txt log file into a clean BIDS-like events.tsv
    containing:
    - onset (sec from scanner trigger)
    - duration (sec)
    - trial_type (task name for videos; "fixation"/"black" for baselines)
    - stim_file (video filename, or "n/a")
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

    def _first_key_ending_with(frame: dict, suffix: str):
        """Return the first value in frame whose key ends with suffix (e.g., '.OnsetTime')."""
        for k, v in frame.items():
            if k.endswith(suffix):
                return v
        return None

    # ------------------------------------------------------------
    # 1) Pass 1: TaskSelect + scanner trigger + baseline onsets
    # ------------------------------------------------------------
    task_select = None
    run_start_ms = None

    fix4_onset_ms = None
    fix4_dur_ms = 4000.0  # default, but we’ll override if we can read it
    fix10_onset_ms = None
    fix10_dur_ms = 10000.0

    goodbye_onset_ms = None

    inside_frame = False
    current_proc = None
    current_frame = {}

    with open_text(txt_path) as f:
        for raw in f:
            line = raw.strip()

            if line.startswith("*** LogFrame Start ***"):
                inside_frame = True
                current_proc = None
                current_frame = {}
                continue

            if line.startswith("*** LogFrame End ***"):
                # harvest info from the finished frame
                if current_proc == "Fix4sec":
                    onset_val = _first_key_ending_with(current_frame, ".OnsetTime")
                    if onset_val is not None:
                        fix4_onset_ms = float(onset_val)
                    # If Duration is logged for this proc, prefer it (some designs include it)
                    if "Fix4000ms.Duration" in current_frame:
                        fix4_dur_ms = float(current_frame["Fix4000ms.Duration"])

                elif current_proc == "Fix10sec":
                    onset_val = _first_key_ending_with(current_frame, ".OnsetTime")
                    if onset_val is not None:
                        fix10_onset_ms = float(onset_val)
                    if "Fix10000ms.Duration" in current_frame:
                        fix10_dur_ms = float(current_frame["Fix10000ms.Duration"])

                elif current_proc == "EndProc":
                    onset_val = _first_key_ending_with(current_frame, ".OnsetTime")
                    if onset_val is not None:
                        goodbye_onset_ms = float(onset_val)

                inside_frame = False
                current_proc = None
                current_frame = {}
                continue

            # outside frames: TaskSelect is in header area in your file
            if (not inside_frame) and ("TaskSelect:" in line):
                _, val = line.split(":", 1)
                task_select = val.strip()

            if inside_frame and ":" in line:
                key, val = [p.strip() for p in line.split(":", 1)]
                if key == "Procedure":
                    current_proc = val
                if current_proc == "Introduction" and key == "Wait4Scanner.OffsetTime":
                    run_start_ms = float(val)

                current_frame[key] = val

    if run_start_ms is None:
        raise ValueError(f"No scanner trigger found in {txt_path}")

    # ------------------------------------------------------------
    # 2) Pass 2: Extract video trial events (+ track first/last movie timing)
    # ------------------------------------------------------------
    events = []
    inside_frame = False
    current_proc = None
    current_frame = {}

    first_movie_onset_ms = None
    last_movie_end_ms = None

    with open_text(txt_path) as f:
        for raw in f:
            line = raw.strip()

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
                    duration_ms = float(current_frame["Duration1"])

                    # track first/last for black-screen inference
                    if first_movie_onset_ms is None or onset_ms < first_movie_onset_ms:
                        first_movie_onset_ms = onset_ms

                    end_ms = onset_ms + duration_ms
                    if last_movie_end_ms is None or end_ms > last_movie_end_ms:
                        last_movie_end_ms = end_ms

                    events.append(
                        dict(
                            onset=(onset_ms - run_start_ms) / 1000.0,
                            duration=duration_ms / 1000.0,
                            trial_type=task_select,
                            stim_file=current_frame["Stim1"],
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
    # 3) Add baseline events (Fixation + Black screens)
    # ------------------------------------------------------------
    def add_event(onset_ms, dur_ms, trial_type, stim_file="n/a"):
        events.append(
            dict(
                onset=(onset_ms - run_start_ms) / 1000.0,
                duration=dur_ms / 1000.0,
                trial_type=trial_type,
                stim_file=stim_file,
            )
        )

    # --- Beginning: Fix4sec
    if fix4_onset_ms is not None:
        add_event(fix4_onset_ms, fix4_dur_ms, "fixation")

        # Beginning: BlackScreen (often missing onset in log, infer from end of Fix4sec)
        if first_movie_onset_ms is not None:
            black_start_ms = fix4_onset_ms + fix4_dur_ms
            black_dur_ms = first_movie_onset_ms - black_start_ms
            # guard against tiny negatives due to rounding / clock jitter
            if black_dur_ms > 0:
                add_event(black_start_ms, black_dur_ms, "black")

    # --- End: Fix10sec
    if fix10_onset_ms is not None:
        add_event(fix10_onset_ms, fix10_dur_ms, "fixation_end")

        # Optional end black: gap until Goodbye (if any)
        if goodbye_onset_ms is not None:
            end_black_start_ms = fix10_onset_ms + fix10_dur_ms
            end_black_dur_ms = goodbye_onset_ms - end_black_start_ms
            if end_black_dur_ms > 0:
                add_event(end_black_start_ms, end_black_dur_ms, "black_end")

    # Sort by onset (important since we appended baselines after movies)
    events.sort(key=lambda x: x["onset"])

    # ------------------------------------------------------------
    # 4) Write events.tsv
    # ------------------------------------------------------------
    with open(out_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(
            f_out,
            fieldnames=["onset", "duration", "trial_type", "stim_file"],
            delimiter="\t",
        )
        writer.writeheader()
        for ev in events:
            writer.writerow(ev)

    print(f"Saved events to {out_path} (n={len(events)})")
    return out_path
