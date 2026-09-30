import os

from flag_recognition import llm_backend
from flag_recognition import report_writer


class _ForbiddenClient:
    def __init__(self, *args, **kwargs):
        raise AssertionError("trivial dialogue must not call the LLM backend")


def test_greeting_uses_instant_local_fast_path(monkeypatch):
    monkeypatch.setattr(report_writer, "OpenAI", _ForbiddenClient)
    result = report_writer.interpret_country_request("Hey there!")
    assert result["intent"] == "greeting"
    assert result["_source"] == "fast_path"
    assert result["reply"]


def test_help_uses_instant_local_fast_path(monkeypatch):
    monkeypatch.setattr(report_writer, "OpenAI", _ForbiddenClient)
    result = report_writer.interpret_country_request("What can you do?")
    assert result["intent"] == "general"
    assert result["_source"] == "fast_path"
    assert result["reply"]


def test_ollama_respects_interactive_timeout(monkeypatch):
    monkeypatch.setenv("FLAG_INTELLIGENCE_OLLAMA_TIMEOUT", "180")
    responses = llm_backend._OllamaResponses(timeout=20.0)
    assert responses.timeout == 20.0
