import json
from types import SimpleNamespace

from flag_recognition import report_writer


class _FakeResponses:
    payload = {}

    def create(self, **kwargs):
        return SimpleNamespace(output_text=json.dumps(type(self).payload))


class _FakeClient:
    def __init__(self, *args, **kwargs):
        self.responses = _FakeResponses()


def _configure_model(monkeypatch, payload):
    _FakeResponses.payload = payload
    monkeypatch.setattr(report_writer, "OpenAI", _FakeClient)
    monkeypatch.setattr(report_writer, "llm_auth_token", lambda: "test-key")
    monkeypatch.setattr(report_writer, "llm_model_name", lambda: "test-model")


def test_follow_up_angle_does_not_reask_known_period(monkeypatch):
    _configure_model(
        monkeypatch,
        {
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

    assert result["action"] == "ask"
    assert result["brief_state"]["period"] == "1950 to 2020"
    assert "policy impact" in result["brief_state"]["angles"]
    assert "time span" not in result["reply"].casefold()
    assert "period" not in result["reply"].casefold()
    assert "detailed" in result["reply"].casefold()
