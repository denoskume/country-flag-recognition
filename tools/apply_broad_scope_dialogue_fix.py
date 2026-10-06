from pathlib import Path


WRITER = Path("src/flag_recognition/report_writer.py")
source = WRITER.read_text(encoding="utf-8")

helper_anchor = '''def _is_report_generation_confirmation(
    latest_message: str,
    recent_history: list[dict[str, str]],
) -> bool:
'''
helper = '''def _user_accepts_current_report_scope(message: str) -> bool:
    """Return True when the user explicitly declines further optional narrowing."""
    normalized = _normalized_request_text(message)
    accepted = {
        "anything is fine",
        "anything works",
        "any is fine",
        "whatever is fine",
        "whatever works",
        "no preference",
        "no preferences",
        "you choose",
        "your choice",
        "all are fine",
        "all of them",
    }
    return normalized in accepted


def _is_report_generation_confirmation(
    latest_message: str,
    recent_history: list[dict[str, str]],
) -> bool:
'''
if helper_anchor not in source:
    raise RuntimeError("report confirmation helper anchor not found")
source = source.replace(helper_anchor, helper, 1)

confirmation_anchor = '''    explicit_report_request = _is_explicit_report_request(latest)
    report_confirmation = _is_report_generation_confirmation(latest, recent_history)

    scope_fields = (
'''
confirmation_replacement = '''    explicit_report_request = _is_explicit_report_request(latest)
    report_confirmation = _is_report_generation_confirmation(latest, recent_history)
    accepts_current_scope = _user_accepts_current_report_scope(latest)

    scope_fields = (
'''
if confirmation_anchor not in source:
    raise RuntimeError("dialogue confirmation anchor not found")
source = source.replace(confirmation_anchor, confirmation_replacement, 1)

ready_anchor = '''    deterministic_ready = _brief_is_sufficiently_specific(
        existing or normalized_request,
        latest,
    )
    brief_ready = ready or prior_ready or deterministic_ready

    # Flag Intelligence is report-first: before a report exists, substantive
'''
ready_replacement = '''    deterministic_ready = _brief_is_sufficiently_specific(
        existing or normalized_request,
        latest,
    )
    if accepts_current_scope and has_report_scope:
        brief_state["ready"] = True
        brief_state["missing"] = []
    brief_ready = ready or prior_ready or deterministic_ready or (
        accepts_current_scope and has_report_scope
    )

    # Flag Intelligence is report-first: before a report exists, substantive
'''
if ready_anchor not in source:
    raise RuntimeError("dialogue readiness anchor not found")
source = source.replace(ready_anchor, ready_replacement, 1)

WRITER.write_text(source, encoding="utf-8")
