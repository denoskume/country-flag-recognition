from pathlib import Path


APP_SOURCE = Path("app.py").read_text(encoding="utf-8")


def test_new_flag_upload_resets_stale_conversation_before_current_message():
    assert "def _reset_flag_intelligence_session_for_new_upload()" in APP_SOURCE
    reset_call = "_reset_flag_intelligence_session_for_new_upload()"
    upload_branch = APP_SOURCE.split("if prompt_files:", 1)[1].split("if prompt_text or prompt_files:", 1)[0]
    assert reset_call in upload_branch


def test_new_flag_upload_reset_clears_previous_report_and_messages():
    assert "st.session_state.fi_messages = []" in APP_SOURCE
    assert "st.session_state.fi_report_pdf = None" in APP_SOURCE
    assert "st.session_state.fi_report_available = False" in APP_SOURCE
