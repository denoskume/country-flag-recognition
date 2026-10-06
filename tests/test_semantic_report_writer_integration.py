from flag_recognition import report_writer


def test_report_writer_delegates_to_semantic_conversation(monkeypatch):
    captured = {}

    def fake_semantic(
        country_name,
        existing_request,
        latest_message,
        turn_number=1,
        existing_state=None,
        conversation_history=None,
        report_available=False,
    ):
        captured.update(
            country=country_name,
            existing=existing_request,
            latest=latest_message,
            turn=turn_number,
            state=existing_state,
            history=conversation_history,
            report_available=report_available,
        )
        return {
            "action": "clarify",
            "country": country_name,
            "normalized_request": "history of French Republics",
            "reply": "Which Republics should the report cover?",
            "brief_state": {
                "subject": "French Republics",
                "scope": "",
                "period": "",
                "entity_range": "",
                "topics": ["history of French Republics"],
                "angles": [],
                "depth": "",
                "exclusions": [],
                "current_events": False,
                "other_constraints": [],
                "confidence": 0.8,
                "ready": False,
                "ambiguities": ["Republic range"],
                "changed_fields": [],
            },
        }

    monkeypatch.setattr(report_writer, "continue_semantic_conversation", fake_semantic)

    result = report_writer.continue_report_conversation(
        "France",
        "history of French Republics",
        "1 to 5",
        turn_number=2,
        existing_state={"subject": "French Republics"},
        conversation_history=[{"role": "assistant", "content": "Which Republics?"}],
        report_available=False,
    )

    assert captured["latest"] == "1 to 5"
    assert captured["turn"] == 2
    assert result["action"] == "ask"
    assert result["semantic_action"] == "clarify"


def test_semantic_generate_maps_to_legacy_generate(monkeypatch):
    monkeypatch.setattr(
        report_writer,
        "continue_semantic_conversation",
        lambda *args, **kwargs: {
            "action": "generate",
            "country": "France",
            "normalized_request": "French Republics I through V",
            "reply": "Generating your report now.",
            "brief_state": {
                "subject": "French Republics",
                "scope": "",
                "period": "",
                "entity_range": "First through Fifth Republic",
                "topics": ["history"],
                "angles": [],
                "depth": "",
                "exclusions": [],
                "current_events": False,
                "other_constraints": [],
                "confidence": 0.98,
                "ready": True,
                "ambiguities": [],
                "changed_fields": ["entity_range"],
            },
        },
    )

    result = report_writer.continue_report_conversation(
        "France", "history of French Republics", "1 to 5"
    )

    assert result["action"] == "generate"
    assert result["semantic_action"] == "generate"
    assert result["brief_state"]["entity_range"] == "First through Fifth Republic"


def test_semantic_status_and_converse_map_to_reply(monkeypatch):
    for semantic_action in ("status", "converse"):
        monkeypatch.setattr(
            report_writer,
            "continue_semantic_conversation",
            lambda *args, _action=semantic_action, **kwargs: {
                "action": _action,
                "country": "France",
                "normalized_request": "",
                "reply": "The report is ready above." if _action == "status" else "Hello!",
                "brief_state": {},
            },
        )
        result = report_writer.continue_report_conversation("France", "", "hello")
        assert result["action"] == "reply"
        assert result["semantic_action"] == semantic_action
