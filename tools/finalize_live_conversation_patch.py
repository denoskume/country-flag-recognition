from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"
KNOWLEDGE = ROOT / "src/flag_recognition/country_knowledge.py"
TEST = ROOT / "tests/test_live_conversation_state.py"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"Missing patch anchor: {label}")
    return text.replace(old, new, 1)


app = APP.read_text(encoding="utf-8")

app = replace_once(
    app,
    '''for message in st.session_state.fi_messages:\n    _render_chat_message(message["role"], message["content"])\n\nprompt_submission = st.chat_input(\n''',
    '''for message in st.session_state.fi_messages:\n    _render_chat_message(message["role"], message["content"])\n\n# Keep the generated artifact visible during later conversational turns.\nif st.session_state.fi_report_available and st.session_state.fi_report_pdf:\n    st.download_button(\n        "Download PDF Report",\n        data=st.session_state.fi_report_pdf,\n        file_name=(\n            st.session_state.fi_report_filename\n            or "flag_intelligence_report.pdf"\n        ),\n        mime="application/pdf",\n        use_container_width=True,\n        type="primary",\n        key="persistent_report_download",\n    )\n\nprompt_submission = st.chat_input(\n''',
    "persistent report button",
)

app = replace_once(
    app,
    '''        action = str(turn.get("action") or "").strip()\n\n        st.session_state.fi_report_request = normalized_request\n''',
    '''        action = str(turn.get("action") or "").strip()\n        turn_country = str(turn.get("country") or country_name).strip()\n\n        # The model may resolve a natural country switch mid-conversation.\n        if turn_country and turn_country.casefold() != country_name.casefold():\n            switched_code = (\n                country_code_from_text(turn_country)\n                or _country_code_from_free_text(turn_country)\n            )\n            if switched_code is not None:\n                country_code = switched_code\n                country_name = display_country_name(switched_code)\n                st.session_state.fi_country_code = switched_code\n                st.session_state.fi_country_name = country_name\n                st.session_state.fi_report_pdf = None\n                st.session_state.fi_report_filename = ""\n                st.session_state.fi_report_available = False\n                st.session_state.fi_image_bytes = None\n\n        st.session_state.fi_report_request = normalized_request\n''',
    "model-driven country switch",
)

app = replace_once(
    app,
    '''            st.session_state.fi_stage = "report_ready"\n        else:\n            st.session_state.fi_stage = "clarifying"\n''',
    '''            st.session_state.fi_stage = "report_ready"\n        elif action == "ask":\n            st.session_state.fi_stage = "clarifying"\n        else:\n            st.session_state.fi_stage = (\n                "report_ready"\n                if st.session_state.fi_report_available\n                else "clarifying"\n            )\n''',
    "reply stage preservation",
)

APP.write_text(app, encoding="utf-8")

knowledge = KNOWLEDGE.read_text(encoding="utf-8")
knowledge = replace_once(
    knowledge,
    '''            if motivation:\n                detail += f" for {motivation.strip('"')}"\n''',
    '''            if motivation:\n                clean_motivation = motivation.strip('"')\n                detail += f" for {clean_motivation}"\n''',
    "Nobel motivation quoting",
)
KNOWLEDGE.write_text(knowledge, encoding="utf-8")

TEST.write_text('''from pathlib import Path\n\n\ndef test_app_persists_report_and_live_context():\n    root = Path(__file__).resolve().parents[1]\n    app = (root / "app.py").read_text(encoding="utf-8")\n    assert '"report_ready"' in app\n    assert 'persistent_report_download' in app\n    assert 'conversation_history=st.session_state.fi_messages[:-1]' in app\n    assert 'report_available=bool(st.session_state.fi_report_available)' in app\n    assert 'turn.get("country")' in app\n    assert 'if st.session_state.fi_report_available' in app\n\ndef test_country_knowledge_module_has_valid_nobel_motivation_code():\n    root = Path(__file__).resolve().parents[1]\n    text = (root / "src/flag_recognition/country_knowledge.py").read_text(encoding="utf-8")\n    assert "clean_motivation = motivation.strip" in text\n''', encoding="utf-8")
