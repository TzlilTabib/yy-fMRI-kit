"""
parcel_pattern.py
=================
Whole-brain voxel-pattern extraction within a parcellation atlas.

Option A: one NiftiMasker call per run extracts the full brain at once.
The atlas label for every voxel column is stored in the NPZ so downstream
analysis can slice per-parcel without re-loading NIfTIs.

How it differs from roi.py
---------------------------
- roi.py applies a binary ROI mask  →  (T, n_roi_voxels)
- parcel_pattern.py applies the full atlas as a brain mask  →  (T, n_brain_voxels)
  and also stores voxel_parcel_labels (n_brain_voxels,) so the analysis module
  can do:  parcel_data = data[:, voxel_parcel_labels == parcel_id]

How it differs from parcel.py (TSV-based)
------------------------------------------
- parcel.py loads pre-extracted TSV files (T, n_parcels) — one scalar per parcel
  per TR; Pearson r is computed across posts (timecourse similarity).
- parcel_pattern.py extracts voxel vectors within each parcel; Pearson r is
  computed across voxels (spatial pattern similarity).

NPZ output per subject/run
---------------------------
  data                : (n_posts, n_brain_voxels)  float32
  post_ids            : (n_posts,)                 str
  voxel_parcel_labels : (n_brain_voxels,)           int32   voxel → parcel int ID
  parcel_ids          : (n_parcels,)                int32   ordered unique IDs
  parcel_names        : (n_parcels,)                str     matching parcel_ids
  run_type            : str  (0-d array)
  subject             : str
  tr, shift_tr        : float / int  (reproducibility)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image
from nilearn.input_data import NiftiMasker

from yy_fmri_kit.event_isc.extraction.base import BasePatternExtractor
from yy_fmri_kit.event_isc.utils import natural_sort_key, infer_run_type_from_filename
from yy_fmri_kit.static.event_isc.config import ExtractionConfig


__all__ = ["ParcelPatternExtractor"]

log = logging.getLogger(__name__)


class ParcelPatternExtractor(BasePatternExtractor):
    """
    Extract whole-brain voxel patterns using a parcellation atlas.

    One NiftiMasker call per run → (T, n_brain_voxels).  The atlas label
    for each voxel column is aligned to the extracted array at fit time and
    stored in every NPZ for downstream per-parcel slicing.

    Parameters
    ----------
    config      : ExtractionConfig  (tr, shift_tr, detrend, standardize, …)
    atlas_nii   : integer-labelled parcellation NIfTI (dseg)
    labels_tsv  : TSV with columns ``id`` (int) and ``name`` (str)
    min_voxels  : parcels with fewer voxels (after resampling) are skipped in
                  the analysis, but their voxels are still stored in the NPZ.
                  Default 10 — all Schaefer 400 + Tian S3 parcels clear this.
    """

    def __init__(
        self,
        config    : ExtractionConfig,
        atlas_nii : Path,
        labels_tsv: Path,
        min_voxels: int = 10,
    ) -> None:
        super().__init__(config)
        self._atlas_nii  = Path(atlas_nii)
        self._labels_tsv = Path(labels_tsv)
        self.min_voxels  = min_voxels

        # Loaded once at init
        self._atlas_img     = image.load_img(str(self._atlas_nii))
        self._brain_mask_img = image.math_img("img > 0", img=self._atlas_img)

        labels_df           = pd.read_csv(str(self._labels_tsv), sep="\t")
        self._parcel_ids    = labels_df["id"].to_numpy(dtype=np.int32)
        self._parcel_names  = labels_df["name"].to_numpy(dtype=object)

        # Warn if atlas contains integer labels not in the TSV (common in mixed atlases)
        atlas_data     = self._atlas_img.get_fdata(dtype=np.float32).astype(np.int32)
        atlas_labels   = set(np.unique(atlas_data).tolist()) - {0}
        tsv_labels     = set(self._parcel_ids.tolist())
        unlabelled     = atlas_labels - tsv_labels
        if unlabelled:
            log.warning(
                f"Atlas has {len(unlabelled)} integer label(s) with no entry in "
                f"labels TSV: {sorted(unlabelled)}.  Their voxels will be extracted "
                f"(stored in NPZ) but excluded from per-parcel analysis."
            )

        # Cached after the first run is processed (all runs share the same space)
        self._voxel_parcel_labels: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_voxel_labels(self, fitted_masker: NiftiMasker) -> np.ndarray:
        """
        Align atlas integer labels to the voxel column order produced by
        *fitted_masker*.

        The masker may have resampled the mask to match the BOLD image.
        We resample the full atlas with the same target affine/shape so the
        voxel ordering is identical, then apply the fitted binary mask to
        extract the per-column label.

        Returns
        -------
        voxel_parcel_labels : (n_brain_voxels,) int32
        """
        fitted_mask_img = fitted_masker.mask_img_          # binary, in BOLD space

        # Resample atlas (integer labels) to BOLD space with nearest-neighbour
        atlas_resampled = image.resample_to_img(
            self._atlas_img,
            fitted_mask_img,
            interpolation  = "nearest",
            force_resample = True,
            copy_header    = True,
        )
        atlas_data = np.round(atlas_resampled.get_fdata()).astype(np.int32)
        mask_data  = fitted_mask_img.get_fdata().astype(bool)

        voxel_labels = atlas_data[mask_data]               # (n_brain_voxels,)

        log.info(
            f"  Atlas resampled to BOLD space: "
            f"{atlas_data.shape}, {mask_data.sum()} brain voxels"
        )
        return voxel_labels

    def _make_masker(self) -> NiftiMasker:
        return NiftiMasker(
            mask_img      = self._brain_mask_img,
            smoothing_fwhm= self.config.smoothing_fwhm,
            detrend       = self.config.detrend,
            standardize   = self.config.standardize,
            resampling_target="data",   # resample mask → BOLD space
        )

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------

    def extract_single_run(
        self,
        nifti_path: Path,
        events_df : pd.DataFrame,
        mask_path : Optional[Path] = None,   # unused; kept for API compat
        verbose   : bool = False,
    ) -> dict[str, np.ndarray]:
        """
        Extract whole-brain voxel patterns for all posts in one run.

        Parameters
        ----------
        nifti_path : path to 4D denoised NIfTI
        events_df  : events already filtered for this subject / run

        Returns
        -------
        post_patterns : {post_id: (n_brain_voxels,) float32}
        """
        log.info(f"Processing {nifti_path.name}")

        bold_img = image.load_img(str(nifti_path))

        # Compute TR timing windows
        events = self.compute_tr_timings(events_df)

        # Fit masker + extract (T, n_brain_voxels)
        masker = self._make_masker()
        brain_ts = masker.fit_transform(bold_img)          # (n_trs, n_brain_voxels)
        n_trs, n_voxels = brain_ts.shape
        log.info(f"  Loaded BOLD: {n_trs} TRs, {n_voxels} brain voxels")

        # Build voxel→parcel label map (cached after first run)
        if self._voxel_parcel_labels is None:
            self._voxel_parcel_labels = self._build_voxel_labels(masker)

        # Validate TR windows
        valid_events, stats = self.validate_tr_bounds(events, n_trs)
        if valid_events.empty:
            log.error("  No valid events after TR bound validation")
            return {}

        # Extract one pattern vector per post
        post_patterns: dict[str, np.ndarray] = {}
        for _, row in valid_events.iterrows():
            post_id = row[self.config.post_col]
            start   = int(row["shifted_onset_tr"])
            end     = int(row["shifted_offset_tr"])
            pattern = self.extract_post_pattern(brain_ts, start, end)  # (n_brain_voxels,)
            post_patterns[post_id] = pattern.astype(np.float32)

            if verbose:
                log.debug(
                    f"  Post {post_id}: TR [{start}:{end}], "
                    f"mean={pattern.mean():.4f}, std={pattern.std():.4f}"
                )

        n_dropped = stats["total"] - stats["valid"]
        log.info(
            f"  Extracted {len(post_patterns)} posts "
            f"({n_dropped} dropped out-of-bounds)"
        )
        return post_patterns

    # ------------------------------------------------------------------
    # Batch extraction
    # ------------------------------------------------------------------

    def batch_extract(
        self,
        runs_dict : dict[str, list[Path]],
        events_df : pd.DataFrame,
        output_dir: Path,
        verbose   : bool = False,
    ) -> pd.DataFrame:
        """
        Extract and save parcel-pattern NPZs for multiple subjects/runs.

        NPZ filename pattern (matches ROI extractor convention):
            {nifti_stem}_desc-parcel_patterns.npz

        Parameters
        ----------
        runs_dict  : {subject_id: [nifti_path, …]}
        events_df  : full events DataFrame (all subjects / conditions)
        output_dir : root output directory; subject sub-dirs are created
        verbose    : print per-post debug info

        Returns
        -------
        summary_df : DataFrame with one row per run (status, n_posts, n_voxels)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        records = []

        for subject, nifti_paths in runs_dict.items():
            subject    = str(subject).strip()
            subject_dir = output_dir / subject
            subject_dir.mkdir(parents=True, exist_ok=True)

            log.info(f"\n{'='*60}\nSubject: {subject}\n{'='*60}")

            for nifti_path in nifti_paths:
                nifti_path = Path(nifti_path)

                run_type = infer_run_type_from_filename(
                    nifti_path, self.config.valid_run_types
                )
                if run_type is None:
                    log.warning(f"  Cannot infer run type from {nifti_path.name}")
                    records.append(_record(subject, nifti_path, None, "run_type_not_found"))
                    continue

                run_events = self.load_and_filter_events(
                    events_df, subject=subject, run_type=run_type
                )
                if run_events.empty:
                    log.warning(f"  No events for {subject} / {run_type}")
                    records.append(_record(subject, nifti_path, run_type, "no_events"))
                    continue

                try:
                    post_patterns = self.extract_single_run(
                        nifti_path, run_events, verbose=verbose
                    )
                    if not post_patterns:
                        records.append(
                            _record(subject, nifti_path, run_type, "no_valid_posts")
                        )
                        continue

                    post_ids = sorted(post_patterns.keys(), key=natural_sort_key)
                    data     = np.vstack(
                        [post_patterns[pid] for pid in post_ids]
                    )  # (n_posts, n_brain_voxels)

                    out_path = subject_dir / f"{nifti_path.stem}_desc-parcel_patterns.npz"
                    np.savez_compressed(
                        out_path,
                        data                = data,
                        post_ids            = np.array(post_ids, dtype=str),
                        voxel_parcel_labels = self._voxel_parcel_labels,
                        parcel_ids          = self._parcel_ids,
                        parcel_names        = self._parcel_names,
                        run_type            = run_type,
                        subject             = subject,
                        tr                  = self.config.tr,
                        shift_tr            = self.config.shift_tr,
                    )
                    log.info(f"  Saved {out_path.name}  shape={data.shape}")
                    records.append(
                        _record(subject, nifti_path, run_type, "success",
                                n_posts=len(post_ids), n_voxels=data.shape[1],
                                output_file=str(out_path))
                    )

                except Exception as exc:
                    log.error(f"  FAILED: {exc}", exc_info=True)
                    records.append(
                        _record(subject, nifti_path, run_type,
                                f"error: {str(exc)[:120]}")
                    )

        summary = pd.DataFrame(records)
        summary_path = output_dir / "extraction_summary.csv"
        summary.to_csv(summary_path, index=False)

        n_ok  = (summary["status"] == "success").sum()
        n_all = len(summary)
        log.info(
            f"\n{'='*60}\n"
            f"DONE: {n_ok}/{n_all} runs successful\n"
            f"{'='*60}"
        )
        return summary


# ------------------------------------------------------------------
# Internal helper
# ------------------------------------------------------------------

def _record(
    subject   : str,
    nifti_path: Path,
    run_type  : Optional[str],
    status    : str,
    n_posts   : int = 0,
    n_voxels  : int = 0,
    output_file: Optional[str] = None,
) -> dict:
    r = {
        "subject"    : subject,
        "nifti"      : str(nifti_path),
        "run_type"   : run_type,
        "status"     : status,
        "n_posts"    : n_posts,
        "n_voxels"   : n_voxels,
    }
    if output_file is not None:
        r["output_file"] = output_file
    return r
