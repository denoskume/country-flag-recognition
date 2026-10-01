from pathlib import Path


def test_app_persists_report_and_live_context():
    root = Path(__file__).resolve().parents[1]
    app = (root / "app.py").read_text(encoding="utf-8")
    assert '"report_ready"' in app
    assert 'persistent_report_download' in app
    assert 'conversation_history=st.session_state.fi_messages[:-1]' in app
    assert 'report_available=bool(st.session_state.fi_report_available)' in app
    assert 'turn.get("country")' in app
    assert 'if st.session_state.fi_report_available' in app

def test_country_knowledge_module_has_valid_nobel_motivation_code():
    root = Path(__file__).resolve().parents[1]
    text = (root / "src/flag_recognition/country_knowledge.py").read_text(encoding="utf-8")
    assert "clean_motivation = motivation.strip" in text
