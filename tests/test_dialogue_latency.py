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


def test_report_conversation_never_calls_llm(monkeypatch):
    monkeypatch.setattr(report_writer, "OpenAI", _ForbiddenClient)
    result = report_writer.continue_report_conversation(
        "France",
        "",
        "I want to know about its history",
        turn_number=1,
    )
    assert result["action"] in {"ask", "generate"}
    assert result["reply"]


def test_initial_country_request_can_skip_dialogue_llm(monkeypatch):
    monkeypatch.setattr(report_writer, "OpenAI", _ForbiddenClient)
    monkeypatch.setenv("FLAG_INTELLIGENCE_DIALOGUE_LLM", "false")
    result = report_writer.interpret_country_request(
        "Tell me about France history"
    )
    assert result["_source"] == "fast_path"
    assert result["request"] == "Tell me about France history"


def test_dialogue_stays_local_even_if_legacy_flag_is_enabled(monkeypatch):
    monkeypatch.setattr(report_writer, "OpenAI", _ForbiddenClient)
    monkeypatch.setenv("FLAG_INTELLIGENCE_DIALOGUE_LLM", "true")
    first = report_writer.interpret_country_request("Tell me about France")
    follow = report_writer.continue_report_conversation(
        "France", "", "history", turn_number=1
    )
    assert first["_source"] == "fast_path"
    assert follow["reply"]


def test_history_only_scope_selects_history_sections():
    keys = report_writer._requested_report_sections("only its history")
    assert "historical_journey" in keys
    assert "origins_early_history" in keys
    assert "key_historical_timeline" in keys
    assert "economy_trade_industries" not in keys


def test_broad_scope_keeps_full_report():
    keys = report_writer._requested_report_sections("complete country report")
    assert keys == report_writer.REPORT_SECTION_KEYS


def test_prune_report_to_history_scope_removes_unrelated_sections():
    report = {
        "introduction": "Intro",
        "historical_journey": "History",
        "economy_trade_industries": "Economy",
        "conclusion": "Conclusion",
        "__qa_passed": True,
    }
    scoped = report_writer.prune_report_to_request(report, "only its history")
    assert scoped["historical_journey"] == "History"
    assert "economy_trade_industries" not in scoped
    assert scoped["__scope_sections"] == report_writer.HISTORY_REPORT_SECTION_KEYS
