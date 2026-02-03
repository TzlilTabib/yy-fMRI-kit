from unicodedata import name
import pandas as pd
import re
from pathlib import Path
import numpy as np
from nilearn import image, input_data

# ==============================================================
# Function to extract post-level neural patterns from one nii scan
# ==============================================================

def extract_post_patterns(
    nifti_path,
    eprime_csv=None,
    events_df=None,
    mask_path=None,
    onset_col=None,
    duration_col=None,
    time_unit="seconds",
    tr=2,
    shift_tr=4,
    *,
    return_df=False,   
    verbose=False):            
    """
    Slices NIfTI scans into post-level patterns based on E-Prime timings.
    Returns dict {post_id: pattern_vector} by default, or a DataFrame if return_df=True.
    """

    # 1. Load the E-Prime timing data and filter valid posts only
    if events_df is not None:
        df = events_df.copy()
    else:
        if eprime_csv is None:
            raise ValueError("Provide either eprime_csv or events_df.")
        df = pd.read_csv(eprime_csv, sep='\t', header=0)
    
    # filtering for stim_file (NaN + "nan" + whitespace)
    post_col = "stim_file" if "stim_file" in df.columns else "post_id"

    post_vals = df[post_col].astype(str).str.strip()
    df = df[df[post_col].notna() & (post_vals != "") & (post_vals.str.lower() != "nan")].copy()

    # 2. Load the fMRI image
    img = image.load_img(nifti_path)

    # 3) compute onset/offset in TR indices
    if onset_col is None:
        onset_col = "onset_tr" if "onset_tr" in df.columns else ("onset" if "onset" in df.columns else "onset_s")
    if duration_col is None:
        duration_col = "duration_tr" if "duration_tr" in df.columns else ("duration" if "duration" in df.columns else "duration_s")

    if time_unit == "seconds":                         
        onset_tr = (df[onset_col] / tr).round().astype(int)
        dur_tr = (df[duration_col] / tr).round().astype(int)
    else:
        onset_tr = df[onset_col].astype(int)
        dur_tr = df[duration_col].astype(int)

    df["onset_tr"] = onset_tr                            
    df["duration_tr"] = dur_tr                           

    df["offset_tr"] = df["onset_tr"] + df["duration_tr"]   
    df["shifted_onset_tr"] = df["onset_tr"] + int(shift_tr) 
    df["shifted_offset_tr"] = df["offset_tr"] + int(shift_tr)

    # 4. Prepare masker
    # None check
    if mask_path is not None:
        masker = input_data.NiftiMasker(mask_img=mask_path, smoothing_fwhm=None, detrend=True)
    else:
        masker = input_data.NiftiMasker(smoothing_fwhm=None, detrend=True)

    voxel_time_series = masker.fit_transform(img)  # (n_TRs, n_features)
    n_trs = voxel_time_series.shape[0]         

    post_patterns = {}
    rows_for_df = []                               

    kept = 0
    skipped = 0

    # 5. Loop posts
    for _, row in df.iterrows():
        post_id = row[post_col]
        start = int(row['shifted_onset_tr'])
        end = int(row['shifted_offset_tr'])

        # validity checks (prevents silent NaNs / crashes)
        if end <= start:
            skipped += 1
            continue
        if start < 0 or end > n_trs:
            skipped += 1
            continue

        pattern_vector = np.mean(voxel_time_series[start:end, :], axis=0)
        post_patterns[post_id] = pattern_vector
        kept += 1

        # optional per-post debug
        if verbose:
            print(post_id, start, end, pattern_vector.shape)

        if return_df:
            rows_for_df.append({
                post_col: post_id,
                "start_tr": start,
                "end_tr": end,
            })

    # summary print (won’t crash if nothing kept)
    if verbose and not return_df:
        print(f"[extract_post_patterns] kept={kept}, skipped={skipped}, features={voxel_time_series.shape[1]}")


    if return_df:
        meta = pd.DataFrame(rows_for_df)

        post_ids = meta[post_col].tolist()
        data = np.vstack([post_patterns[k] for k in post_ids]) if len(meta) else np.empty((0, voxel_time_series.shape[1]))

        feat_cols = [f"f{i:06d}" for i in range(data.shape[1])]
        return pd.concat([meta.reset_index(drop=True), pd.DataFrame(data, columns=feat_cols)], axis=1)

    return post_patterns 

# ==============================================================
# low level helpers
# ==============================================================
RUN_TYPES = ["AntiLeft", "AntiRight", "ProLeft", "ProRight"]

def infer_run_type_from_filename(p: Path) -> str | None:
    name = p.name
    for rt in RUN_TYPES:
        if rt.lower() in name.lower():
            return rt
    return None

def natural_sort_key(s: str):
    # "anti_left_12" < "anti_left_15" and "anti_left_3" < "anti_left_10"
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]

# ==============================================================
# Batch function to extract post patterns for multiple subjects/runs
# ==============================================================

def run_all_subjects(
    runs_dict: dict[str, list[Path]],
    events_path: str | Path,
    *,
    subject_col: str,
    run_col: str,              
    out_dir: str | Path = "roi_outputs",
    mask_path: str | Path | None = None,
    shift_tr: int = 4,
    time_unit: str = "seconds",    
    tr: float = 1.5,              
    onset_col: str | None = None,   
    duration_col: str | None = None 
):
    """
    Batch ROI extraction using one big events CSV including all subjects/runs.
    filtered by (subject, run).
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_events = pd.read_csv(events_path, sep=",")
    all_events[subject_col] = all_events[subject_col].astype(str).str.strip()
    all_events[run_col] = all_events[run_col].astype(str).str.strip()

    summary = []

    for sub, run_paths in runs_dict.items():
        sub = str(sub).strip()
        sub_out = out_dir / sub
        sub_out.mkdir(parents=True, exist_ok=True)

        for nii_path in run_paths:
            nii_path = Path(nii_path)

            run_type = infer_run_type_from_filename(nii_path)
            if run_type is None:
                summary.append({"sub": sub, "nii": str(nii_path), "status": "run_type_not_found"})
                continue
            if run_type not in RUN_TYPES:
                summary.append({"sub": sub, "nii": str(nii_path), "run": run_type, "status": "skipped_nonpolitical"})
                continue

            # filter events for THIS subject + THIS run TYPE
            ev = all_events[
                (all_events[subject_col] == sub) &
                (all_events[run_col].astype(str).str.lower() == run_type.lower())
            ].copy()

            if ev.empty:
                summary.append({"sub": sub, "nii": str(nii_path), "run": run_type, "status": "no_events"})
                continue

            # run extraction
            patterns_dict = extract_post_patterns(
                nifti_path=str(nii_path),
                events_df=ev,                          
                mask_path=str(mask_path) if mask_path else None,
                shift_tr=shift_tr,
                time_unit=time_unit,
                tr=tr,
                onset_col=onset_col,
                duration_col=duration_col,
            )

            post_ids = sorted(patterns_dict.keys(), key=natural_sort_key)
            data = np.vstack([patterns_dict[pid] for pid in post_ids])

            # save NPZ with explicit order + matrix (ISC-ready)
            out_file = sub_out / f"{nii_path.stem}_desc-roi_patterns.npz"
            
            np.savez_compressed(
                out_file,
                post_ids=np.array(post_ids, dtype=str),
                data=data.astype(np.float32),   
                run_type=run_type,
                subject=sub
            )

            summary.append({
                "sub": sub,
                "nii": str(nii_path),
                "run": run_type,
                "status": "ok",
                "n_posts": int(data.shape[0]),
                "n_features": int(data.shape[1]),
                "out": str(out_file),
            })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    return summary_df

