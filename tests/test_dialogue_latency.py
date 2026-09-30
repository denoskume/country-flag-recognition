import json
from types import SimpleNamespace

import pytest

from flag_recognition import llm_backend
from flag_recognition import report_writer


class _FakeResponses:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(output_text=json.dumps(self.payload))


class _FakeClient:
    payload = {}
    instances = []

    def __init__(self, *args, **kwargs):
        self.responses = _FakeResponses(type(self).payload)
        type(self).instances.append(self)


def test_greeting_is_generated_by_llm(monkeypatch):
    _FakeClient.payload = {
        "intent": "greeting",
        "country": "",
        "request": "",
        "reply": "Hi! What would you like to explore today?",
    }
    _FakeClient.instances = []
    monkeypatch.setattr(report_writer, "OpenAI", _FakeClient)

    result = report_writer.interpret_country_request("Hey there!")

    assert result["_source"] == "model"
    assert result["intent"] == "greeting"
    assert result["reply"]
    assert _FakeClient.instances[0].responses.calls == 1


def test_country_only_reply_is_generated_by_llm(monkeypatch):
    _FakeClient.payload = {
        "intent": "country_only",
        "country": "France",
        "request": "",
        "reply": "What would you like to explore about France?",
    }
    _FakeClient.instances = []
    monkeypatch.setattr(report_writer, "OpenAI", _FakeClient)

    result = report_writer.interpret_country_request("France")

    assert result["_source"] == "model"
    assert result["country"] == "France"
    assert result["reply"]
    assert _FakeClient.instances[0].responses.calls == 1


def test_follow_up_is_generated_by_llm(monkeypatch):
    _FakeClient.payload = {
        "action": "ask",
        "normalized_request": "history",
        "reply": "Which period of French history interests you most?",
    }
    _FakeClient.instances = []
    monkeypatch.setattr(report_writer, "OpenAI", _FakeClient)

    result = report_writer.continue_report_conversation(
        "France",
        "",
        "Its history",
        turn_number=1,
    )

    assert result["action"] == "ask"
    assert result["reply"]
    assert _FakeClient.instances[0].responses.calls == 1


def test_dialogue_failure_does_not_fallback(monkeypatch):
    class _FailResponses:
        def create(self, **kwargs):
            raise RuntimeError("ollama unavailable")

    class _FailClient:
        def __init__(self, *args, **kwargs):
            self.responses = _FailResponses()

    monkeypatch.setattr(report_writer, "OpenAI", _FailClient)

    with pytest.raises(RuntimeError, match="LLM dialogue request failed"):
        report_writer.interpret_country_request("Hello")


def test_groq_is_default_backend(monkeypatch):
    monkeypatch.delenv("FLAG_INTELLIGENCE_LLM_BACKEND", raising=False)
    assert llm_backend.llm_backend_name() == "groq"


def test_groq_default_model(monkeypatch):
    monkeypatch.setenv("FLAG_INTELLIGENCE_LLM_BACKEND", "groq")
    monkeypatch.delenv("FLAG_INTELLIGENCE_GROQ_MODEL", raising=False)
    assert llm_backend.llm_model_name() == "openai/gpt-oss-20b"


def test_groq_uses_groq_api_key(monkeypatch):
    monkeypatch.setenv("FLAG_INTELLIGENCE_LLM_BACKEND", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    assert llm_backend.llm_auth_token() == "test-groq-key"


def test_history_only_scope_selects_history_sections():
    keys = report_writer._requested_report_sections("only its history")
    assert "historical_journey" in keys
    assert "origins_early_history" in keys
    assert "key_historical_timeline" in keys
    assert "economy_trade_industries" not in keys


def test_broad_scope_keeps_full_report():
    keys = report_writer._requested_report_sections("complete country report")
    assert keys == report_writer.REPORT_SECTION_KEYS


def test_plain_history_request_is_scoped_to_history():
    keys = report_writer._requested_report_sections("Its history")
    assert keys == report_writer.HISTORY_REPORT_SECTION_KEYS


def test_history_request_with_country_name_is_scoped_to_history():
    keys = report_writer._requested_report_sections("Tell me about the history of France")
    assert keys == report_writer.HISTORY_REPORT_SECTION_KEYS


def test_explicit_full_history_request_keeps_full_report():
    keys = report_writer._requested_report_sections(
        "Give me a complete report about France including its history"
    )
    assert keys == report_writer.REPORT_SECTION_KEYS


def test_legacy_ollama_backend_is_normalized_to_groq(monkeypatch):
    monkeypatch.setenv("FLAG_INTELLIGENCE_LLM_BACKEND", "ollama")
    assert llm_backend.llm_backend_name() == "groq"


def test_probe_groq_uses_direct_responses_endpoint(monkeypatch):
    monkeypatch.setenv("FLAG_INTELLIGENCE_LLM_BACKEND", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")

    class _Response:
        status_code = 200
        text = '{"status":"completed","output":[{"type":"message"}]}'

        def raise_for_status(self):
            return None

        def json(self):
            return {"status": "completed", "output": [{"type": "message"}]}

    calls = {}

    def _post(url, **kwargs):
        calls["url"] = url
        calls["headers"] = kwargs.get("headers")
        calls["json"] = kwargs.get("json")
        return _Response()

    monkeypatch.setattr(llm_backend.requests, "post", _post)

    ok, detail = llm_backend.probe_llm_backend()

    assert ok is True
    assert "groq ok" in detail
    assert calls["url"].endswith("/responses")
    assert calls["json"]["model"] == "openai/gpt-oss-20b"


def test_llm_credentials_available_for_groq(monkeypatch):
    monkeypatch.setenv("FLAG_INTELLIGENCE_LLM_BACKEND", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    assert llm_backend.llm_credentials_available() is True


def test_llm_credentials_missing_for_groq(monkeypatch):
    monkeypatch.setenv("FLAG_INTELLIGENCE_LLM_BACKEND", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert llm_backend.llm_credentials_available() is False


def test_period_history_request_uses_period_history_sections():
    keys = report_writer._requested_report_sections("History of France from 1950 to 2026")
    assert keys == report_writer.HISTORY_PERIOD_REPORT_SECTION_KEYS
    assert "origins_early_history" not in keys


def test_explicit_time_range_is_detected():
    assert report_writer._has_explicit_time_range("From 1950 to 2026") is True
    assert report_writer._has_explicit_time_range("Its history") is False


def test_time_range_does_not_force_generation_when_scope_can_still_be_refined(monkeypatch):
    _FakeClient.payload = {
        "action": "ask",
        "normalized_request": "History of France from 1950 to 2026",
        "reply": "Would you like political, economic, cultural developments, or all of them?",
    }
    _FakeClient.instances = []
    monkeypatch.setattr(report_writer, "OpenAI", _FakeClient)

    result = report_writer.continue_report_conversation(
        "France",
        "history",
        "From 1950 to 2026",
        turn_number=2,
    )

    assert result["action"] == "ask"
    assert "political" in result["reply"].lower()


def test_economic_and_cultural_scope_is_not_full_report():
    keys = report_writer._requested_report_sections(
        "In-depth economic and cultural developments in France from 1950 to 2020"
    )
    assert "economy_trade_industries" in keys
    assert "culture_cuisine_music_sport" in keys
    assert "government_structure" not in keys
    assert "cost_of_living" not in keys
    assert keys != report_writer.REPORT_SECTION_KEYS


def test_complete_brief_detects_period_angles_and_depth():
    assert report_writer._brief_is_sufficiently_specific(
        "France history and culture from 1950 to 2020; focus on economic and cultural developments",
        "in-depth analysis",
    ) is True


def test_brief_without_angle_still_needs_clarification():
    assert report_writer._brief_is_sufficiently_specific(
        "France history from 1950 to 2020",
        "",
    ) is False


def test_canonical_brief_preserves_latest_explicit_topics():
    brief = report_writer._canonicalize_report_brief(
        "history and culture from 1950 to 2020",
        "Economic and cultural",
        "culture and history of France 1950-2020",
    )
    assert "economy" in brief
    assert "culture" in brief
    assert "history" not in brief


def test_canonical_brief_preserves_topics_when_depth_arrives_later():
    brief = report_writer._canonicalize_report_brief(
        "economy and culture from 1950 to 2020",
        "balanced overview",
        "culture and history of France 1950-2020 balanced overview",
    )
    assert "economy" in brief
    assert "culture" in brief
    assert "balanced overview" in brief
    assert "1950 to 2020" in brief


def test_scoped_review_retries_with_json_object_when_strict_schema_fails(monkeypatch):
    class _Responses:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise RuntimeError("json_validate_failed")
            return SimpleNamespace(
                output_text=json.dumps({
                    "introduction": "Reviewed introduction.",
                    "conclusion": "Reviewed conclusion.",
                })
            )

    class _Client:
        instances = []

        def __init__(self, *args, **kwargs):
            self.responses = _Responses()
            type(self).instances.append(self)

    monkeypatch.setattr(report_writer, "OpenAI", _Client)

    result = report_writer._review_scoped_report(
        api_key="test-key",
        model="openai/gpt-oss-20b",
        country_name="France",
        user_request="economy and culture from 1950 to 2020; detailed analysis",
        section_keys=("introduction", "conclusion"),
        draft={
            "introduction": "Draft introduction.",
            "conclusion": "Draft conclusion.",
        },
    )

    assert result["introduction"] == "Reviewed introduction."
    assert result["conclusion"] == "Reviewed conclusion."
    calls = _Client.instances[0].responses.calls
    assert len(calls) == 2
    assert calls[0]["text"]["format"]["type"] == "json_schema"
    assert calls[1]["text"]["format"]["type"] == "json_object"
