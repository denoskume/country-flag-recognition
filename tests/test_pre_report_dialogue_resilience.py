import json
from types import SimpleNamespace

from flag_recognition import report_conversation, report_writer

# Regression coverage for the Côte d'Ivoire live-chat failure reproduced on 6 Oct 2026.


class _CountryOnlyResponses:
    def create(self, **kwargs):
        return SimpleNamespace(
            output_text=json.dumps(
                {
                    "intent": "country_only",
                    "country": "Côte d'Ivoire",
                    "request": "",
                    "reply": (
                        "Côte d'Ivoire est un pays d'Afrique de l'Ouest, riche en culture "
                        "et en histoire."
                    ),
                }
            )
        )


class _CountryOnlyClient:
    def __init__(self, *args, **kwargs):
        self.responses = _CountryOnlyResponses()


class _BusyResponses:
    def create(self, **kwargs):
        raise RuntimeError(
            "The language service is temporarily busy after retrying both Groq models."
        )


class _BusyClient:
    def __init__(self, *args, **kwargs):
        self.responses = _BusyResponses()


def test_country_only_turn_never_passes_model_country_exposition_to_chat(monkeypatch):
    monkeypatch.setattr(report_writer, "OpenAI", _CountryOnlyClient)
    monkeypatch.setattr(report_writer, "llm_auth_token", lambda: "test-key")
    monkeypatch.setattr(report_writer, "llm_model_name", lambda: "test-model")

    result = report_writer.interpret_country_request(
        "I would like to learn about cote d'ivoire"
    )

    assert result["intent"] == "country_only"
    assert result["country"] == "Côte d'Ivoire"
    assert result["request"] == ""
    assert result["reply"] == ""


def test_busy_semantic_backend_still_generates_when_explicit_period_completes_scope():
    previous_state = {
        "subject": "political history",
        "scope": "political history of Côte d'Ivoire",
        "period": "",
        "entity_range": "",
        "topics": ["political history"],
        "angles": [],
        "depth": "",
        "exclusions": [],
        "current_events": False,
        "other_constraints": [],
        "confidence": 0.9,
        "ready": False,
        "ambiguities": ["period"],
        "changed_fields": [],
    }

    result = report_conversation.continue_semantic_conversation(
        "Côte d'Ivoire",
        "political history of Côte d'Ivoire",
        "period 1930 to 2000",
        existing_state=previous_state,
        client_factory=_BusyClient,
        auth_token_fn=lambda: "test-key",
        model_name_fn=lambda: "test-model",
    )

    assert result["action"] == "generate"
    assert result["brief_state"]["period"] == "1930 to 2000"
    assert result["brief_state"]["ready"] is True
    assert result["brief_state"]["ambiguities"] == []
    assert "1930 to 2000" in result["normalized_request"]
