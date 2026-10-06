import importlib
import importlib.util
import json
from types import SimpleNamespace

import pytest


MODULE_NAME = "flag_recognition.report_conversation"


def _conversation_module():
    return importlib.import_module(MODULE_NAME)


def test_semantic_conversation_component_exists():
    assert importlib.util.find_spec(MODULE_NAME) is not None


class _FakeResponses:
    payload = {}

    def create(self, **kwargs):
        return SimpleNamespace(output_text=json.dumps(type(self).payload))


class _FakeClient:
    def __init__(self, *args, **kwargs):
        self.responses = _FakeResponses()


def _configure_model(monkeypatch, payload):
    conversation = _conversation_module()
    _FakeResponses.payload = payload
    monkeypatch.setattr(conversation, "OpenAI", _FakeClient)
    monkeypatch.setattr(conversation, "llm_auth_token", lambda: "test-key")
    monkeypatch.setattr(conversation, "llm_model_name", lambda: "test-model")
    return conversation


def _brief(**overrides):
    brief = {
        "subject": "",
        "scope": "",
        "period": "",
        "entity_range": "",
        "topics": [],
        "angles": [],
        "depth": "",
        "exclusions": [],
        "current_events": False,
        "other_constraints": [],
        "confidence": 0.9,
        "ready": False,
        "ambiguities": [],
        "changed_fields": [],
    }
    brief.update(overrides)
    return brief


def _payload(*, action, request, reply, brief, country="France"):
    return {
        "action": action,
        "country": country,
        "normalized_request": request,
        "reply": reply,
        "brief_state": brief,
    }


def test_french_republic_range_is_semantic_scope_not_calendar_period(monkeypatch):
    conversation = _configure_model(
        monkeypatch,
        _payload(
            action="generate",
            request="History of the French Republics from the First through the Fifth Republic",
            reply="Generating your report now.",
            brief=_brief(
                subject="French Republics",
                scope="political and constitutional history",
                entity_range="First through Fifth Republic",
                topics=["history of the French Republics"],
                ready=True,
                changed_fields=["entity_range"],
            ),
        ),
    )

    result = conversation.continue_semantic_conversation(
        "France",
        "history about French republics",
        "1 to 5",
        existing_state=_brief(
            subject="French Republics",
            topics=["history of the French Republics"],
        ),
        conversation_history=[
            {"role": "user", "content": "history about French republics"},
            {"role": "assistant", "content": "Which Republics should the report cover?"},
        ],
    )

    assert result["action"] == "generate"
    assert result["brief_state"]["period"] == ""
    assert result["brief_state"]["entity_range"] == "First through Fifth Republic"


def test_unknown_topic_can_generate_without_keyword_registration(monkeypatch):
    conversation = _configure_model(
        monkeypatch,
        _payload(
            action="generate",
            request="France semiconductor sovereignty with emphasis on industrial policy",
            reply="Generating your report now.",
            brief=_brief(
                subject="French semiconductor sovereignty",
                scope="semiconductor supply chain and strategic autonomy",
                topics=["semiconductor sovereignty"],
                angles=["industrial policy"],
                ready=True,
            ),
        ),
    )

    result = conversation.continue_semantic_conversation(
        "France", "", "semiconductor sovereignty and industrial policy"
    )

    assert result["action"] == "generate"
    assert result["brief_state"]["topics"] == ["semiconductor sovereignty"]


def test_contextual_refinement_preserves_previous_scope(monkeypatch):
    previous = _brief(
        subject="Senegalese music",
        scope="traditional and modern music",
        topics=["music"],
    )
    conversation = _configure_model(
        monkeypatch,
        _payload(
            action="generate",
            country="Senegal",
            request="Senegalese music after independence, focusing on women",
            reply="Generating your report now.",
            brief=_brief(
                subject="Senegalese music",
                scope="traditional and modern music",
                period="after independence",
                topics=["music"],
                angles=["women"],
                ready=True,
                changed_fields=["period", "angles"],
            ),
        ),
    )

    result = conversation.continue_semantic_conversation(
        "Senegal",
        "Senegalese music",
        "after independence, focus on women",
        existing_state=previous,
    )

    assert result["brief_state"]["subject"] == "Senegalese music"
    assert result["brief_state"]["period"] == "after independence"
    assert result["brief_state"]["angles"] == ["women"]


def test_explicit_correction_clears_stale_constraint(monkeypatch):
    previous = _brief(
        subject="French Republics",
        topics=["history"],
        angles=["constitutional changes"],
    )
    conversation = _configure_model(
        monkeypatch,
        _payload(
            action="generate",
            request="General history of the French Republics",
            reply="Generating your report now.",
            brief=_brief(
                subject="French Republics",
                topics=["history"],
                angles=[],
                ready=True,
                changed_fields=["angles"],
            ),
        ),
    )

    result = conversation.continue_semantic_conversation(
        "France",
        "French Republics with emphasis on constitutional changes",
        "remove that emphasis; cover them generally",
        existing_state=previous,
    )

    assert result["brief_state"]["angles"] == []


def test_true_ambiguity_produces_one_targeted_clarification(monkeypatch):
    conversation = _configure_model(
        monkeypatch,
        _payload(
            action="clarify",
            request="",
            reply="What does ‘1 to 5’ refer to?",
            brief=_brief(
                confidence=0.35,
                ambiguities=["The numeric range has no clear referent."],
            ),
        ),
    )

    result = conversation.continue_semantic_conversation("France", "", "1 to 5")

    assert result["action"] == "clarify"
    assert result["reply"].endswith("?")
    assert len(result["brief_state"]["ambiguities"]) == 1


def test_anything_is_fine_accepts_existing_scope(monkeypatch):
    previous = _brief(
        subject="Japanese cuisine",
        scope="regional specialties",
        topics=["cuisine"],
        ambiguities=["specific region"],
    )
    conversation = _configure_model(
        monkeypatch,
        _payload(
            action="generate",
            country="Japan",
            request="Japanese cuisine with representative regional specialties",
            reply="Generating your report now.",
            brief=_brief(
                subject="Japanese cuisine",
                scope="representative regional specialties",
                topics=["cuisine"],
                ready=True,
                ambiguities=[],
                changed_fields=["scope"],
            ),
        ),
    )

    result = conversation.continue_semantic_conversation(
        "Japan",
        "Japanese cuisine; regional specialties",
        "anything is fine",
        existing_state=previous,
    )

    assert result["action"] == "generate"
    assert result["brief_state"]["ready"] is True
    assert result["brief_state"]["ambiguities"] == []


def test_model_failure_preserves_previous_brief_and_degrades_to_clarification(monkeypatch):
    conversation = _conversation_module()

    class _FailingResponses:
        def create(self, **kwargs):
            raise RuntimeError("backend unavailable")

    class _FailingClient:
        def __init__(self, *args, **kwargs):
            self.responses = _FailingResponses()

    monkeypatch.setattr(conversation, "OpenAI", _FailingClient)
    monkeypatch.setattr(conversation, "llm_auth_token", lambda: "test-key")
    monkeypatch.setattr(conversation, "llm_model_name", lambda: "test-model")

    previous = _brief(
        subject="Senegalese music",
        topics=["music"],
        scope="traditional music",
    )
    result = conversation.continue_semantic_conversation(
        "Senegal",
        "Senegalese traditional music",
        "after independence",
        existing_state=previous,
    )

    assert result["action"] == "clarify"
    assert result["brief_state"]["subject"] == "Senegalese music"
    assert result["brief_state"]["topics"] == ["music"]
    assert result["reply"].endswith("?")


def test_opening_question_never_claims_generation():
    conversation = _conversation_module()
    reply = conversation.opening_question("Senegal")

    assert "Senegal" in reply
    assert "?" in reply
    assert "generat" not in reply.casefold()
    assert "report is ready" not in reply.casefold()
