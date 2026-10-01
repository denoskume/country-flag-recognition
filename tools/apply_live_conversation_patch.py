from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WRITER = ROOT / "src/flag_recognition/report_writer.py"
APP = ROOT / "app.py"
TESTS = ROOT / "tests/test_dialogue_latency.py"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"Missing patch anchor: {label}")
    return text.replace(old, new, 1)


writer = WRITER.read_text(encoding="utf-8")
writer = replace_once(
    writer,
    '''def continue_report_conversation(\n    country_name: str,\n    existing_request: str,\n    latest_message: str,\n    turn_number: int = 1,\n    existing_state: dict[str, Any] | None = None,\n) -> dict[str, Any]:\n    """Update the report brief semantically with one real-time model turn."""\n''',
    '''def continue_report_conversation(\n    country_name: str,\n    existing_request: str,\n    latest_message: str,\n    turn_number: int = 1,\n    existing_state: dict[str, Any] | None = None,\n    conversation_history: list[dict[str, str]] | None = None,\n    report_available: bool = False,\n) -> dict[str, Any]:\n    """Run one model-driven live conversation turn with preserved context."""\n''',
    "conversation signature",
)
writer = replace_once(
    writer,
    '''    state = existing_state if isinstance(existing_state, dict) else {}\n\n    brief_state_schema = {\n''',
    '''    state = existing_state if isinstance(existing_state, dict) else {}\n    history = conversation_history if isinstance(conversation_history, list) else []\n    recent_history = [\n        {"role": str(item.get("role") or ""), "content": str(item.get("content") or "")}\n        for item in history[-12:]\n        if isinstance(item, dict)\n    ]\n\n    brief_state_schema = {\n''',
    "history normalization",
)
writer = replace_once(
    writer,
    '''            "action": {"type": "string", "enum": ["ask", "generate"]},\n            "normalized_request": {"type": "string"},\n''',
    '''            "action": {"type": "string", "enum": ["reply", "ask", "generate"]},\n            "country": {"type": "string"},\n            "normalized_request": {"type": "string"},\n''',
    "action enum",
)
writer = replace_once(
    writer,
    '''            "action",\n            "normalized_request",\n''',
    '''            "action",\n            "country",\n            "normalized_request",\n''',
    "required country",
)
start_marker = '    prompt = (\n        "You are Flag Intelligence managing a live report-planning conversation. "'
end_marker = '\n\n    client = OpenAI('
if "You are Flag Intelligence conducting a live, natural conversation" not in writer:
    start = writer.index(start_marker)
    end = writer.index(end_marker, start)
    prompt = '''    prompt = (\n        "You are Flag Intelligence conducting a live, natural conversation about countries. "\n        "Interpret the user's latest message using the recent dialogue, preserved semantic brief, "\n        "current country, and report availability. User wording is unpredictable: resolve pronouns, "\n        "ellipsis, fragments, corrections, short reactions, and follow-ups from context. Do not use "\n        "canned wording and do not behave like a questionnaire.\\n\\n"\n        f"CURRENT COUNTRY: {country or '(none)'}\\n"\n        f"TURN: {int(turn_number)}\\n"\n        f"REPORT AVAILABLE: {bool(report_available)}\\n"\n        f"PREVIOUS NORMALIZED REQUEST: {existing or '(none)'}\\n"\n        "PREVIOUS BRIEF STATE:\\n"\n        + json.dumps(state, ensure_ascii=False, sort_keys=True)\n        + "\\nRECENT CONVERSATION:\\n"\n        + json.dumps(recent_history, ensure_ascii=False)\n        + "\\nLATEST USER MESSAGE:\\n"\n        + latest\n        + "\\n\\nRULES:\\n"\n        "- Preserve resolved country and brief facts unless the user explicitly changes them.\\n"\n        "- If the user switches country, return the new canonical English country name.\\n"\n        "- action='reply' for ordinary conversation, acknowledgements, questions about an existing "\n        "report, requests to explain previously discussed material, or any turn that does not need a "\n        "new or updated PDF. Never regenerate merely because brief_state.ready is true.\\n"\n        "- action='ask' only when a material ambiguity prevents a requested new/updated report. "\n        "Ask one concise non-repetitive question.\\n"\n        "- action='generate' only when the user requests a new or materially updated report and the "\n        "brief is sufficiently clear.\\n"\n        "- If REPORT AVAILABLE is true, references such as 'where?', 'I am waiting', 'is it ready?', "\n        "or 'show me more' should be resolved against the existing report when context supports it.\\n"\n        "- normalized_request and brief_state are persistent semantic memory. Keep them unchanged on "\n        "ordinary conversational follow-ups unless the user changes scope.\\n"\n        "- reply naturally in the same language as the latest user message unless another language is requested.\\n"\n        "Return JSON only."\n    )'''
    writer = writer[:start] + prompt + writer[end:]
writer = replace_once(
    writer,
    '''                    "keys: action, normalized_request, reply, brief_state."\n''',
    '''                    "keys: action, country, normalized_request, reply, brief_state."\n''',
    "retry schema keys",
)
writer = replace_once(
    writer,
    '''    normalized_request = str(parsed.get("normalized_request") or "").strip()\n    reply = str(parsed.get("reply") or "").strip()\n    if not normalized_request:\n        raise RuntimeError("LLM dialogue request failed: empty normalized request")\n    if not reply:\n        raise RuntimeError("LLM dialogue request failed: empty reply")\n\n    ready = bool(brief_state.get("ready"))\n    action = "generate" if ready else "ask"\n''',
    '''    turn_country = str(parsed.get("country") or country).strip() or country\n    normalized_request = str(parsed.get("normalized_request") or existing or latest).strip()\n    reply = str(parsed.get("reply") or "").strip()\n    if not normalized_request:\n        raise RuntimeError("LLM dialogue request failed: empty normalized request")\n    if not reply:\n        raise RuntimeError("LLM dialogue request failed: empty reply")\n\n    requested_action = str(parsed.get("action") or "reply").strip().lower()\n    ready = bool(brief_state.get("ready"))\n    if requested_action == "generate" and not ready:\n        action = "ask"\n    elif requested_action in {"reply", "ask", "generate"}:\n        action = requested_action\n    else:\n        action = "reply"\n''',
    "action logic",
)
writer = replace_once(
    writer,
    '''    return {\n        "action": action,\n        "normalized_request": normalized_request,\n''',
    '''    return {\n        "action": action,\n        "country": turn_country,\n        "normalized_request": normalized_request,\n''',
    "return country",
)
WRITER.write_text(writer, encoding="utf-8")

app = APP.read_text(encoding="utf-8")
app = replace_once(
    app,
    '''if "fi_pending_input" not in st.session_state:\n    st.session_state.fi_pending_input = None\n\nfor message in st.session_state.fi_messages:\n''',
    '''if "fi_pending_input" not in st.session_state:\n    st.session_state.fi_pending_input = None\nif "fi_report_pdf" not in st.session_state:\n    st.session_state.fi_report_pdf = None\nif "fi_report_filename" not in st.session_state:\n    st.session_state.fi_report_filename = ""\nif "fi_report_available" not in st.session_state:\n    st.session_state.fi_report_available = False\n\nfor message in st.session_state.fi_messages:\n''',
    "report artifact state",
)
app = replace_once(
    app,
    '''        if st.session_state.fi_stage in {\n            "awaiting_interest",\n            "clarifying",\n        }:\n''',
    '''        if st.session_state.fi_stage in {\n            "awaiting_interest",\n            "clarifying",\n            "report_ready",\n        }:\n''',
    "report-ready conversation stage",
)
app = replace_once(
    app,
    '''        st.session_state.fi_report_state = {}\n        st.session_state.fi_clarification_turn = 0\n\n        if image is not None:\n''',
    '''        st.session_state.fi_report_state = {}\n        st.session_state.fi_clarification_turn = 0\n        st.session_state.fi_report_pdf = None\n        st.session_state.fi_report_filename = ""\n        st.session_state.fi_report_available = False\n\n        if image is not None:\n''',
    "clear old artifact on new country",
)
app = replace_once(
    app,
    '''        st.download_button(\n            "Download PDF Report",\n            data=pdf_bytes,\n            file_name=(\n                f"{_report_filename_country(country)}_report.pdf"\n            ),\n            mime="application/pdf",\n            use_container_width=True,\n            type="primary",\n        )\n''',
    '''        report_filename = f"{_report_filename_country(country)}_report.pdf"\n        st.session_state.fi_report_pdf = pdf_bytes\n        st.session_state.fi_report_filename = report_filename\n        st.session_state.fi_report_available = True\n        st.download_button(\n            "Download PDF Report",\n            data=pdf_bytes,\n            file_name=report_filename,\n            mime="application/pdf",\n            use_container_width=True,\n            type="primary",\n            key="fresh_report_download",\n        )\n''',
    "persist generated PDF",
)
# Never wipe semantic context immediately after successful generation.
reset1 = '''                st.session_state.fi_stage = "idle"\n                st.session_state.fi_country_code = None\n                st.session_state.fi_country_name = None\n                st.session_state.fi_report_request = ""\n                st.session_state.fi_report_state = {}\n                st.session_state.fi_clarification_turn = 0\n'''
if reset1 in app:
    app = app.replace(reset1, '''                st.session_state.fi_stage = "report_ready"\n''', 1)
reset2 = '''            st.session_state.fi_stage = "idle"\n            st.session_state.fi_country_code = None\n            st.session_state.fi_country_name = None\n            st.session_state.fi_report_request = ""\n            st.session_state.fi_report_state = {}\n            st.session_state.fi_image_bytes = None\n            st.session_state.fi_clarification_turn = 0\n'''
if reset2 in app:
    app = app.replace(reset2, '''            st.session_state.fi_stage = "report_ready"\n''', 1)
# Pass real conversation history/report state to the model controller.
needle = '''                existing_state=st.session_state.fi_report_state,\n            )\n'''
replacement = '''                existing_state=st.session_state.fi_report_state,\n                conversation_history=st.session_state.fi_messages[:-1],\n                report_available=bool(st.session_state.fi_report_available),\n            )\n'''
# Apply to all controller calls where session state exists.
app = app.replace(needle, replacement)
APP.write_text(app, encoding="utf-8")

tests = TESTS.read_text(encoding="utf-8")
if "test_post_report_followup_can_reply_without_regeneration" not in tests:
    tests += '''\n\ndef test_post_report_followup_can_reply_without_regeneration(monkeypatch):\n    _FakeClient.payload = {\n        "action": "reply",\n        "country": "France",\n        "normalized_request": "France history and public figures",\n        "reply": "The report is ready and remains available in this conversation.",\n        "brief_state": {\n            "subject": "history and public figures",\n            "topics": ["history", "public figures"],\n            "period": "",\n            "angles": [],\n            "depth": "",\n            "exclusions": [],\n            "current_events": False,\n            "other_constraints": [],\n            "ready": True,\n            "missing": [],\n        },\n    }\n    _FakeClient.instances = []\n    monkeypatch.setattr(report_writer, "OpenAI", _FakeClient)\n    result = report_writer.continue_report_conversation(\n        "France",\n        "France history and public figures",\n        "where...?",\n        turn_number=4,\n        existing_state=_FakeClient.payload["brief_state"],\n        conversation_history=[\n            {"role": "user", "content": "Its history and public figures"},\n            {"role": "assistant", "content": "I will prepare that report."},\n            {"role": "user", "content": "I am waiting..."},\n        ],\n        report_available=True,\n    )\n    assert result["action"] == "reply"\n    assert result["country"] == "France"\n    assert result["normalized_request"] == "France history and public figures"\n'''
TESTS.write_text(tests, encoding="utf-8")
