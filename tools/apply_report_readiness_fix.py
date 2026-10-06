from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRITER = ROOT / "src" / "flag_recognition" / "report_writer.py"
GENERATION_TESTS = ROOT / "tests" / "test_report_generation_intent.py"
DIALOGUE_TESTS = ROOT / "tests" / "test_dialogue_latency.py"

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


generation_tests = GENERATION_TESTS.read_text(encoding="utf-8")
generation_tests = generation_tests.replace(
    'def test_follow_up_angle_does_not_reask_known_period(monkeypatch):',
    'def test_follow_up_angle_refines_complete_brief_and_generates(monkeypatch):',
    1,
)
old_generation_assertions = '''    assert result["action"] == "ask"\n    assert result["brief_state"]["period"] == "1950 to 2020"\n    assert "policy impact" in result["brief_state"]["angles"]\n    assert "time span" not in result["reply"].casefold()\n    assert "period" not in result["reply"].casefold()\n    assert "detailed" in result["reply"].casefold()\n'''
new_generation_assertions = '''    assert result["action"] == "generate"\n    assert result["brief_state"]["period"] == "1950 to 2020"\n    assert "policy impact" in result["brief_state"]["angles"]\n'''
if old_generation_assertions not in generation_tests:
    raise RuntimeError("generation test assertion anchor not found")
generation_tests = generation_tests.replace(
    old_generation_assertions,
    new_generation_assertions,
    1,
)
GENERATION_TESTS.write_text(generation_tests, encoding="utf-8")


dialogue_tests = DIALOGUE_TESTS.read_text(encoding="utf-8")
dialogue_tests = dialogue_tests.replace(
    'def test_time_range_does_not_force_generation_when_scope_can_still_be_refined(monkeypatch):',
    'def test_time_range_generates_when_topic_and_period_are_clear(monkeypatch):',
    1,
)
old_time_assertions = '''    assert result["action"] == "ask"\n    assert "political" in result["reply"].lower()\n'''
new_time_assertions = '''    assert result["action"] == "generate"\n'''
if old_time_assertions not in dialogue_tests:
    raise RuntimeError("time-range test assertion anchor not found")
dialogue_tests = dialogue_tests.replace(old_time_assertions, new_time_assertions, 1)

dialogue_tests = dialogue_tests.replace(
    'def test_brief_without_angle_still_needs_clarification():',
    'def test_brief_without_angle_is_ready_when_topic_and_period_are_clear():',
    1,
)
old_ready_assertion = '''    assert report_writer._brief_is_sufficiently_specific(\n        "France history from 1950 to 2020",\n        "",\n    ) is False\n'''
new_ready_assertion = '''    assert report_writer._brief_is_sufficiently_specific(\n        "France history from 1950 to 2020",\n        "",\n    ) is True\n'''
if old_ready_assertion not in dialogue_tests:
    raise RuntimeError("brief readiness test assertion anchor not found")
dialogue_tests = dialogue_tests.replace(old_ready_assertion, new_ready_assertion, 1)
DIALOGUE_TESTS.write_text(dialogue_tests, encoding="utf-8")
