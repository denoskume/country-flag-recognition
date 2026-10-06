import json
from types import SimpleNamespace

import pytest

from flag_recognition import report_writer


READY_BRIEF = {
    "subject": "Côte d’Ivoire historical and cultural developments",
    "topics": ["history", "culture"],
    "period": "1950 to 2020",
    "angles": ["major historical developments", "major cultural developments"],
    "depth": "balanced overview",
    "exclusions": [],
    "current_events": False,
    "other_constraints": [],
    "ready": True,
    "missing": [],
}


class _FakeResponses:
    payload = {}

    def create(self, **kwargs):
        return SimpleNamespace(output_text=json.dumps(type(self).payload))


class _FakeClient:
    def __init__(self, *args, **kwargs):
        self.responses = _FakeResponses()


def _configure_model(monkeypatch, *, latest_payload):
    _FakeResponses.payload = latest_payload
    monkeypatch.setattr(report_writer, "OpenAI", _FakeClient)
    monkeypatch.setattr(report_writer, "llm_auth_token", lambda: "test-key")
    monkeypatch.setattr(report_writer, "llm_model_name", lambda: "test-model")


def _payload(reply: str, action: str = "reply"):
    return {
        "action": action,
        "country": "Côte d’Ivoire",
        "normalized_request": (
            "history and culture from 1950 to 2020; balanced overview"
        ),
        "reply": reply,
        "brief_state": dict(READY_BRIEF),
    }


def test_explicit_report_request_forces_generation_even_if_model_only_replies(monkeypatch):
    _configure_model(
        monkeypatch,
        latest_payload=_payload("Here is a concise report covering that period."),
    )

    result = report_writer.continue_report_conversation(
        "Côte d’Ivoire",
        "history and culture from 1950 to 2020; balanced overview",
        "give me a report",
        existing_state=dict(READY_BRIEF),
        report_available=False,
    )

    assert result["action"] == "generate"


def test_explicit_pdf_request_forces_generation_even_if_model_only_replies(monkeypatch):
    _configure_model(
        monkeypatch,
        latest_payload=_payload("Here is a concise PDF report."),
    )

    result = report_writer.continue_report_conversation(
        "Côte d’Ivoire",
        "history and culture from 1950 to 2020; balanced overview",
        "pdf report",
        existing_state=dict(READY_BRIEF),
        report_available=False,
    )

    assert result["action"] == "generate"


def test_clear_country_scope_generates_report_without_requiring_report_keyword(monkeypatch):
    _configure_model(
        monkeypatch,
        latest_payload=_payload(
            "From the 1950s onward, Côte d’Ivoire underwent major changes."
        ),
    )

    result = report_writer.continue_report_conversation(
        "Côte d’Ivoire",
        "",
        (
            "Explore Côte d’Ivoire from 1950 to 2020, focusing on major "
            "historical and cultural developments"
        ),
        existing_state={},
        report_available=False,
    )

    assert result["action"] == "generate"


def test_incomplete_country_need_is_clarified_instead_of_answered_in_chat(monkeypatch):
    incomplete_brief = {
        "subject": "Côte d’Ivoire economic history",
        "topics": ["economy", "history"],
        "period": "",
        "angles": [],
        "depth": "",
        "exclusions": [],
        "current_events": False,
        "other_constraints": [],
        "ready": False,
        "missing": ["period", "depth"],
    }
    _configure_model(
        monkeypatch,
        latest_payload={
            "action": "reply",
            "country": "Côte d’Ivoire",
            "normalized_request": "economic history",
            "reply": (
                "Côte d’Ivoire’s economic history is a tale of boom, bust, and resilience. "
                "It began in the early twentieth century as a colonial cash-crop economy."
            ),
            "brief_state": incomplete_brief,
        },
    )

    result = report_writer.continue_report_conversation(
        "Côte d’Ivoire",
        "",
        "Its economical history",
        existing_state={},
        report_available=False,
    )

    assert result["action"] == "ask"
    assert result["reply"].endswith("?")
    assert len(result["reply"]) < 220
    assert "economic history is" not in result["reply"].casefold()


def test_follow_up_angle_refines_complete_brief_and_generates(monkeypatch):
    _configure_model(
        monkeypatch,
        latest_payload={
            "action": "ask",
            "country": "Côte d’Ivoire",
            "normalized_request": "economic history; policy impact",
            "reply": (
                "Could you specify the time span or particular policy areas "
                "you want emphasized within that period?"
            ),
            "brief_state": {
                "subject": "Côte d’Ivoire economic history",
                "topics": ["economy", "history"],
                "period": "",
                "angles": ["policy impact"],
                "depth": "",
                "exclusions": [],
                "current_events": False,
                "other_constraints": [],
                "ready": False,
                "missing": ["period", "depth"],
            },
        },
    )

    result = report_writer.continue_report_conversation(
        "Côte d’Ivoire",
        "history and economy; from 1950 to 2020",
        "policy impact",
        turn_number=3,
        existing_state={
            "subject": "Côte d’Ivoire economic history",
            "topics": ["economy", "history"],
            "period": "1950 to 2020",
            "angles": [],
            "depth": "",
            "exclusions": [],
            "current_events": False,
            "other_constraints": [],
            "ready": False,
            "missing": ["depth", "angles"],
        },
        report_available=False,
    )

    assert result["action"] == "generate"
    assert result["brief_state"]["period"] == "1950 to 2020"
    assert "policy impact" in result["brief_state"]["angles"]
    assert not result["reply"].endswith("?")
    assert "specify" not in result["reply"].casefold()


def test_ready_brief_generates_automatically_after_requirement_gathering(monkeypatch):
    _configure_model(
        monkeypatch,
        latest_payload=_payload(
            "That gives me everything I need.",
            action="reply",
        ),
    )

    result = report_writer.continue_report_conversation(
        "Côte d’Ivoire",
        "history and culture from 1950 to 2020; balanced overview",
        "balanced overview",
        turn_number=3,
        existing_state=dict(READY_BRIEF),
        report_available=False,
    )

    assert result["action"] == "generate"


def test_yes_after_pdf_offer_generates_when_no_report_exists(monkeypatch):
    _configure_model(
        monkeypatch,
        latest_payload=_payload("Sure! I’m generating the PDF report now."),
    )

    result = report_writer.continue_report_conversation(
        "Côte d’Ivoire",
        "history and culture from 1950 to 2020; balanced overview",
        "yes",
        existing_state=dict(READY_BRIEF),
        conversation_history=[
            {
                "role": "assistant",
                "content": "Would you like me to create the PDF report now?",
            }
        ],
        report_available=False,
    )

    assert result["action"] == "generate"


def test_existing_report_status_question_does_not_regenerate(monkeypatch):
    _configure_model(
        monkeypatch,
        latest_payload=_payload("The report is ready above.", action="reply"),
    )

    result = report_writer.continue_report_conversation(
        "Côte d’Ivoire",
        "history and culture from 1950 to 2020; balanced overview",
        "where is it?",
        existing_state=dict(READY_BRIEF),
        report_available=True,
    )

    assert result["action"] == "reply"
