import re
import numpy as np
from pathlib import Path

def natural_sort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]

def load_run_npzs(out_dir: Path, run_type: str):
    files = sorted(out_dir.rglob(f"*task-{run_type}*_desc-roi_patterns.npz"))
    if len(files) < 2:
        raise ValueError(f"Need ≥2 subjects for {run_type}. Found {len(files)}.")
    return files

def build_data_list_for_run(out_dir: Path, run_type: str):
    files = load_run_npzs(out_dir, run_type)

    # load all
    subs = []
    post_ids_list = []
    data_list = []

    for f in files:
        npz = np.load(f, allow_pickle=True)
        post_ids = npz["post_ids"].astype(str).tolist()
        data = npz["data"]  # (posts, features)

        sub = npz["subject"].item() if "subject" in npz.files else f.parent.name
        subs.append(sub)
        post_ids_list.append(post_ids)
        data_list.append(data)

    # define canonical order = sorted post_ids (stable) from first subject
    canonical = sorted(post_ids_list[0], key=natural_sort_key)

    # reindex everyone to canonical order
    aligned = []
    kept_subs = []

    for sub, post_ids, data in zip(subs, post_ids_list, data_list):
        idx = {pid: i for i, pid in enumerate(post_ids)}

        # require all canonical posts exist
        missing = [pid for pid in canonical if pid not in idx]
        if missing:
            print(f"⚠️ Skipping {sub} (missing {len(missing)} posts)")
            continue

        aligned_data = np.vstack([data[idx[pid]] for pid in canonical])
        aligned.append(aligned_data)
        kept_subs.append(sub)

    if len(aligned) < 2:
        raise ValueError("After alignment, fewer than 2 subjects remain.")

    return kept_subs, canonical, aligned
