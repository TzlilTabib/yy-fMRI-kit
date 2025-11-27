"""
Heuristic file describing associations between DICOM series and BIDS output files.
This file is used to map DICOM series related to Tzlil's thesis project to BIDS-compatible output files.
"""

from __future__ import annotations

from heudiconv.utils import SeqInfo

def _safe(s):
    # return lowercase strings, tolerate None
    return (s or "").lower()

def _has(s: SeqInfo, needle: str) -> bool:
    n = needle.lower()
    return n in _safe(s.protocol_name) or n in _safe(s.series_description)

def _log(tag: str, s: SeqInfo):
    print(f"[heur] {tag}: series_id={s.series_id}  "
          f"prot='{s.protocol_name}'  desc='{s.series_description}'  "
          f"imgtype={getattr(s, 'image_type', None)}")

def create_key(
    template: str | None,
    outtype: tuple[str, ...] = ("nii.gz", "json"),
    annotation_classes: None = None,
) -> tuple[str, tuple[str, ...], None]:
    if template is None or not template:
        raise ValueError("Template must be a valid format string")
    return (template, outtype, annotation_classes)


def infotodict(
    seqinfo: list[SeqInfo],
) -> dict[tuple[str, tuple[str, ...], None], list]:
    """Heuristic evaluator for determining which runs belong where

    allowed template fields - follow python string module:

    item: index within category
    subject: participant id
    seqitem: run number during scanning
    subindex: sub index within group
    session: scan index for longitudinal acq
    """
    t1_corrected = create_key(
        "sub-{subject}/{session}/anat/sub-{subject}_{session}_ce-corrected_T1w"
    )
    t1_uncorrected = create_key(
        "sub-{subject}/{session}/anat/sub-{subject}_{session}_ce-uncorrected_T1w"
    )
    t2_corrected = create_key(
        "sub-{subject}/{session}/anat/sub-{subject}_{session}_ce-corrected_T2w"
    )
    t2_uncorrected = create_key(
        "sub-{subject}/{session}/anat/sub-{subject}_{session}_ce-uncorrected_T2w"
    )
    flair = create_key(
        "sub-{subject}/{session}/anat/sub-{subject}_{session}_FLAIR"
    )
    fmap_ap = create_key(
        "sub-{subject}/{session}/fmap/sub-{subject}_{session}_dir-AP_epi"
    )
    fmap_pa = create_key(
        "sub-{subject}/{session}/fmap/sub-{subject}_{session}_dir-PA_epi"
    )
    fmap_task_ap = create_key(
        "sub-{subject}/{session}/fmap/sub-{subject}_{session}_acq-task_dir-AP_epi"
    )
    fmap_task_pa = create_key(
        "sub-{subject}/{session}/fmap/sub-{subject}_{session}_acq-task_dir-PA_epi"
    )
    rest = create_key(
        "sub-{subject}/{session}/func/sub-{subject}_{session}_task-rest_bold"
    )
    rest_sbref = create_key(
        "sub-{subject}/{session}/func/sub-{subject}_{session}_task-rest_sbref"
    )
    # Functional tasks
    anti_right = create_key(
        "sub-{subject}/{session}/func/sub-{subject}_{session}_task-AntiRight_bold"
    )
    anti_right_sbref = create_key(
        "sub-{subject}/{session}/func/sub-{subject}_{session}_task-AntiRight_sbref"
    )
    anti_left = create_key(
        "sub-{subject}/{session}/func/sub-{subject}_{session}_task-AntiLeft_bold"
    )
    anti_left_sbref = create_key(
        "sub-{subject}/{session}/func/sub-{subject}_{session}_task-AntiLeft_sbref"
    )
    pro_right = create_key(
        "sub-{subject}/{session}/func/sub-{subject}_{session}_task-ProRight_bold"
    )
    pro_right_sbref = create_key(
        "sub-{subject}/{session}/func/sub-{subject}_{session}_task-ProRight_sbref"
    )
    pro_left = create_key(
        "sub-{subject}/{session}/func/sub-{subject}_{session}_task-ProLeft_bold"
    )
    pro_left_sbref = create_key(
        "sub-{subject}/{session}/func/sub-{subject}_{session}_task-ProLeft_sbref"
    )
    emotional_pain = create_key(
        "sub-{subject}/{session}/func/sub-{subject}_{session}_task-EmotionalPain_bold"
    )
    emotional_pain_sbref = create_key(
        "sub-{subject}/{session}/func/sub-{subject}_{session}_task-EmotionalPain_sbref"
    )

    info: dict[tuple[str, tuple[str, ...], None], list] = {
        t1_corrected: [],
        t1_uncorrected: [],
        t2_corrected: [],
        t2_uncorrected: [],
        flair: [],
        fmap_ap: [],
        fmap_pa: [],
        fmap_task_ap: [],
        fmap_task_pa: [],
        rest: [],
        rest_sbref: [],
        anti_right: [],
        anti_right_sbref: [],
        anti_left: [],
        anti_left_sbref: [],
        pro_right: [],
        pro_right_sbref: [],
        pro_left: [],
        pro_left_sbref: [],
        emotional_pain: [],
        emotional_pain_sbref: [],
    }

    for s in seqinfo:
        # T1w / T2w / FLAIR
        if _has(s, "T1w_MPRAGE"):
            if getattr(s, "image_type", None) and "NORM" in s.image_type:
                _log("T1w_corrected", s)
                info[t1_corrected].append(s.series_id)
            else:
                _log("T1w_uncorrected", s)
                info[t1_uncorrected].append(s.series_id)

        elif _has(s, "T2w_SPC"):
            if getattr(s, "image_type", None) and "NORM" in s.image_type:
                _log("T2w_corrected", s)
                info[t2_corrected].append(s.series_id)
            else:
                _log("T2w_uncorrected", s)
                info[t2_uncorrected].append(s.series_id)

        elif _has(s, "t2_tirm_tra_dark-fluid_FLAIR"):
            _log("FLAIR", s)
            info[flair].append(s.series_id)

        # Fieldmaps (task and non-task)
        elif _has(s, "SpinEchoFieldMap_TASK_AP") or _has(s, "SE_rsfMRI_FieldMap_AP") or _has(s, "SpinEchoFieldMap_AP"):
            _log("fmap_ap", s)
            info[fmap_ap].append(s.series_id)
        elif _has(s, "SpinEchoFieldMap_TASK_PA") or _has(s, "SE_rsfMRI_FieldMap_PA") or _has(s, "SpinEchoFieldMap_PA"):
            _log("fmap_pa", s)
            info[fmap_pa].append(s.series_id)

        # Rest
        elif _has(s, "rsfMRI_AP") and not _has(s, "SBRef"):
            _log("rest", s)
            info[rest].append(s.series_id)
        elif _has(s, "rsfMRI_AP_SBRef"):
            _log("rest_sbref", s)
            info[rest_sbref].append(s.series_id)

        # Tasks (examples—apply same pattern to all your existing ones)
        elif _has(s, "Task_Anti_Right") and not _has(s, "SBRef"):
            _log("anti_right", s)
            info[anti_right].append(s.series_id)
        elif _has(s, "Task_Anti_Right_SBRef"):
            _log("anti_right_sbref", s)
            info[anti_right_sbref].append(s.series_id)

        elif _has(s, "Task_Anti_Left") and not _has(s, "SBRef"):
            _log("anti_left", s)
            info[anti_left].append(s.series_id)
        elif _has(s, "Task_Anti_Left_SBRef"):
            _log("anti_left_sbref", s)
            info[anti_left_sbref].append(s.series_id)

        elif _has(s, "Task_Pro_Right") and not _has(s, "SBRef"):
            _log("pro_right", s)
            info[pro_right].append(s.series_id)
        elif _has(s, "Task_Pro_Right_SBRef"):
            _log("pro_right_sbref", s)
            info[pro_right_sbref].append(s.series_id)

        elif _has(s, "Task_Pro_Left") and not _has(s, "SBRef"):
            _log("pro_left", s)
            info[pro_left].append(s.series_id)
        elif _has(s, "Task_Pro_Left_SBRef"):
            _log("pro_left_sbref", s)
            info[pro_left_sbref].append(s.series_id)

        elif _has(s, "Task_Emotional_Pain") and not _has(s, "SBRef"):
            _log("emotional_pain", s)
            info[emotional_pain].append(s.series_id)
        elif _has(s, "Task_Emotional_Pain_SBRef"):
            _log("emotional_pain_sbref", s)
            info[emotional_pain_sbref].append(s.series_id)
    return info
