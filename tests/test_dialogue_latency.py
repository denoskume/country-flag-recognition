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
