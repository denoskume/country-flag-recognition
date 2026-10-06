from pathlib import Path


def _app_source() -> str:
    root = Path(__file__).resolve().parents[1]
    return (root / "app.py").read_text(encoding="utf-8")


def test_flag_only_upload_is_not_sent_through_report_readiness_controller():
    app = _app_source()

    assert "The country was identified from an uploaded flag." not in app
    assert "opening_question" in app
    assert "assistant_text = str(opening_reply or opening_question(country)).strip()" in app


def test_flag_only_opening_can_never_be_generation_message():
    app = _app_source()

    flag_entry_start = app.index("if user_request is None:")
    flag_entry_end = app.index("    report = {", flag_entry_start)
    flag_entry = app[flag_entry_start:flag_entry_end]

    assert "continue_report_conversation(" not in flag_entry
    assert "Generating your report now." not in flag_entry
    assert "fi_stage = \"awaiting_interest\"" in flag_entry
    assert "fi_report_request = \"\"" in flag_entry
    assert "fi_report_state = {}" in flag_entry


def test_flag_with_text_defers_opening_and_reuses_text_as_first_semantic_turn():
    app = _app_source()

    assert "defer_opening: bool = False" in app
    assert "defer_opening=bool(prompt_text)" in app
    assert "conversation_text = prompt_text" in app
    assert "conversation_process = True" in app


def test_flag_preview_dimensions_remain_fixed_after_entry_change():
    app = _app_source()

    assert "width:2cm" in app
    assert "height:1cm" in app
    assert "object-fit:contain" in app
