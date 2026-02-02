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
    stim = df['stim_file'].astype(str).str.strip()
    df = df[df['stim_file'].notna()].copy()
    df = df[stim.str.lower() != "nan"].copy()
    df = df[stim != ""].copy()

    # 2. Load the fMRI image
    img = image.load_img(nifti_path)

    # 3. Compute offset from duration + apply HRF shift
    df['offset'] = df['onset'] + df['duration']
    df['shifted_onset'] = df['onset'] + shift_tr
    df['shifted_offset'] = df['offset'] + shift_tr

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
        post_id = row['stim_file']
        start = int(row['shifted_onset'])
        end = int(row['shifted_offset'])

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
                "stim_file": post_id,
                "start_tr": start,
                "end_tr": end,
            })

    # summary print (won’t crash if nothing kept)
    if verbose and not return_df:
        print(f"[extract_post_patterns] kept={kept}, skipped={skipped}, features={voxel_time_series.shape[1]}")

    if return_df:
        # DataFrame output = metadata + feature columns
        meta = pd.DataFrame(rows_for_df)
        X = np.vstack([post_patterns[k] for k in meta["stim_file"]]) if len(meta) else np.empty((0, voxel_time_series.shape[1]))
        feat_cols = [f"f{i:06d}" for i in range(X.shape[1])]
        return pd.concat([meta.reset_index(drop=True), pd.DataFrame(X, columns=feat_cols)], axis=1)

    return post_patterns

# ==============================================================
# low level helper
# ==============================================================
RUN_TYPES = ["AntiLeft", "AntiRight", "ProLeft", "ProRight"]

def infer_run_type_from_filename(p: Path) -> str | None:
    name = p.name
    for rt in RUN_TYPES:
        if rt.lower() in name.lower():
            return rt
    return None

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
):
    """
    Batch ROI extraction using one big events CSV including all subjects/runs.
    filtered by (subject, run).
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_events = pd.read_csv(events_path, sep=",")

    summary = []

    for sub, run_paths in runs_dict.items():
        sub_out = out_dir / sub
        sub_out.mkdir(parents=True, exist_ok=True)

        for nii_path in run_paths:
            nii_path = Path(nii_path)

            run_type = infer_run_type_from_filename(nii_path)
            if run_type is None:
                summary.append({"sub": sub, "nii": str(nii_path), "status": "run_type_not_found"})
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
            patterns_df = extract_post_patterns(
                nifti_path=str(nii_path),
                events_df=ev,                          
                mask_path=str(mask_path) if mask_path else None,
                shift_tr=shift_tr,
            )

            # save (for now, save dict with npz to keep your current return type)
            out_file = sub_out / f"{nii_path.stem}_desc-roi_patterns.npz"
            np.savez_compressed(out_file, **patterns_df)

            summary.append({
                "sub": sub,
                "nii": str(nii_path),
                "run": run_type,
                "status": "ok",
                "n_posts": len(patterns_df),
                "out": str(out_file),
            })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    return summary_df

