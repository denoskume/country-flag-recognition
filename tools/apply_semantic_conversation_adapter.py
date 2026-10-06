from pathlib import Path

path = Path("src/flag_recognition/report_writer.py")
text = path.read_text(encoding="utf-8")
marker = "# SEMANTIC_CONVERSATION_ADAPTER_V1"

if marker not in text:
    text += '''\n\n# SEMANTIC_CONVERSATION_ADAPTER_V1\n# Conversation meaning lives in report_conversation. Keep this public adapter\n# so the Streamlit app and existing callers do not need a simultaneous API migration.\nfrom .report_conversation import continue_semantic_conversation\n\n\ndef continue_report_conversation(\n    country_name: str,\n    existing_request: str,\n    latest_message: str,\n    turn_number: int = 1,\n    existing_state: dict[str, Any] | None = None,\n    conversation_history: list[dict[str, str]] | None = None,\n    report_available: bool = False,\n) -> dict[str, Any]:\n    """Adapt semantic conversation actions to the app's legacy action names."""\n    result = continue_semantic_conversation(\n        country_name,\n        existing_request,\n        latest_message,\n        turn_number=turn_number,\n        existing_state=existing_state,\n        conversation_history=conversation_history,\n        report_available=report_available,\n    )\n    semantic_action = str(result.get("action") or "converse").strip().casefold()\n    legacy_action = {\n        "clarify": "ask",\n        "generate": "generate",\n        "status": "reply",\n        "converse": "reply",\n    }.get(semantic_action, "reply")\n\n    adapted = dict(result)\n    adapted["semantic_action"] = semantic_action\n    adapted["action"] = legacy_action\n    return adapted\n'''
    path.write_text(text, encoding="utf-8")
