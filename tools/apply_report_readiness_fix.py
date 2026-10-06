from pathlib import Path


WRITER = Path(__file__).resolve().parents[1] / "src" / "flag_recognition" / "report_writer.py"
source = WRITER.read_text(encoding="utf-8")

old_state_helper = '''def _brief_missing_dimension_from_state(
    brief_state: dict[str, Any],
    existing_request: str,
    latest_message: str,
) -> str:
    topics = [str(item).strip().casefold() for item in brief_state.get("topics", [])]
    period = str(brief_state.get("period") or "").strip()
    angles = [str(item).strip() for item in brief_state.get("angles", []) if str(item).strip()]
    depth = str(brief_state.get("depth") or "").strip()

    if not topics:
        return "topic"
    if "history" in topics and not period:
        return "period"
    if "history" in topics and period and not angles:
        return "angle"
    if not depth:
        return "depth"
    return _brief_missing_dimension(existing_request, latest_message)
'''

new_state_helper = '''def _brief_missing_dimension_from_state(
    brief_state: dict[str, Any],
    existing_request: str,
    latest_message: str,
) -> str:
    topics = [str(item).strip().casefold() for item in brief_state.get("topics", [])]
    period = str(brief_state.get("period") or "").strip()

    if not topics:
        return "topic"
    if "history" in topics and not period:
        return "period"
    return ""
'''

old_missing = '''def _brief_missing_dimension(existing_request: str, latest_message: str) -> str:
    brief = _canonicalize_report_brief(existing_request, latest_message)
    groups = _requested_topic_groups(brief)
    latest_groups = _requested_topic_groups(latest_message)
    has_period = _has_explicit_time_range(brief)
    has_depth = bool(_extract_depth_label(brief))

    if not groups:
        return "topic"
    if "history" in groups and not has_period:
        return "period"
    if (
        "history" in groups
        and has_period
        and not latest_groups
        and not has_depth
    ):
        return "angle"
    if not has_depth:
        return "depth"
    return ""
'''

new_missing = '''def _brief_missing_dimension(existing_request: str, latest_message: str) -> str:
    brief = _canonicalize_report_brief(existing_request, latest_message)
    groups = _requested_topic_groups(brief)
    has_period = _has_explicit_time_range(brief)

    if not groups:
        return "topic"
    if "history" in groups and not has_period:
        return "period"
    return ""
'''

old_ready = '''def _brief_is_sufficiently_specific(
    existing_request: str,
    latest_message: str,
) -> bool:
    brief = _canonicalize_report_brief(existing_request, latest_message)
    groups = _requested_topic_groups(brief)
    has_period = _has_explicit_time_range(brief)
    has_depth = bool(_extract_depth_label(brief))

    if not groups or not has_depth:
        return False
    if "history" in groups and not has_period:
        return False
    return True
'''

new_ready = '''def _brief_is_sufficiently_specific(
    existing_request: str,
    latest_message: str,
) -> bool:
    brief = _canonicalize_report_brief(existing_request, latest_message)
    groups = _requested_topic_groups(brief)
    has_period = _has_explicit_time_range(brief)

    if not groups:
        return False
    if "history" in groups and not has_period:
        return False
    return True
'''

for old, new, label in (
    (old_state_helper, new_state_helper, "state missing-dimension helper"),
    (old_missing, new_missing, "missing-dimension helper"),
    (old_ready, new_ready, "readiness helper"),
):
    if old not in source:
        raise RuntimeError(f"{label} anchor not found")
    source = source.replace(old, new, 1)

WRITER.write_text(source, encoding="utf-8")
