import json
from types import SimpleNamespace

import pytest

from flag_recognition import llm_backend
from flag_recognition import report_writer


@pytest.fixture(autouse=True)
def _mock_report_writer_auth(monkeypatch):
    monkeypatch.setattr(report_writer, "llm_auth_token", lambda: "test-key")


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
        "brief_state": {
            "subject": "history",
            "topics": ["history"],
            "period": "",
            "angles": [],
            "depth": "",
            "exclusions": [],
            "current_events": False,
            "other_constraints": [],
            "ready": False,
            "missing": ["period"],
        },
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


def test_time_range_generates_when_topic_and_period_are_clear(monkeypatch):
    _FakeClient.payload = {
        "action": "ask",
        "normalized_request": "History of France from 1950 to 2026",
        "reply": "Would you like political, economic, cultural developments, or all of them?",
        "brief_state": {
            "subject": "history",
            "topics": ["history"],
            "period": "1950 to 2026",
            "angles": [],
            "depth": "",
            "exclusions": [],
            "current_events": False,
            "other_constraints": [],
            "ready": False,
            "missing": ["preferred emphasis"],
        },
    }
    _FakeClient.instances = []
    monkeypatch.setattr(report_writer, "OpenAI", _FakeClient)

    result = report_writer.continue_report_conversation(
        "France",
        "history",
        "From 1950 to 2026",
        turn_number=2,
    )

    assert result["action"] == "generate"


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


def test_brief_without_angle_is_ready_when_topic_and_period_are_clear():
    assert report_writer._brief_is_sufficiently_specific(
        "France history from 1950 to 2020",
        "",
    ) is True


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


def test_relative_end_date_today_is_understood():
    start, end = report_writer._extract_requested_year_range(
        "from 1950 till today"
    )
    assert start == 1950
    assert end >= 2026


def test_relative_end_date_present_is_understood():
    start, end = report_writer._extract_requested_year_range(
        "1950 to present"
    )
    assert start == 1950
    assert end >= 2026


def test_unpredictable_wording_is_carried_by_model_brief_state(monkeypatch):
    _FakeClient.payload = {
        "action": "generate",
        "normalized_request": (
            "France: how everyday life changed from the post-war boom "
            "to the smartphone era, with emphasis on work and family life"
        ),
        "reply": "Got it — I’ll prepare that focused report.",
        "brief_state": {
            "subject": "changes in everyday life",
            "topics": ["work", "family life"],
            "period": "post-war boom to smartphone era",
            "angles": ["social change"],
            "depth": "detailed",
            "exclusions": [],
            "current_events": False,
            "other_constraints": [],
            "ready": True,
            "missing": [],
        },
    }
    _FakeClient.instances = []
    monkeypatch.setattr(report_writer, "OpenAI", _FakeClient)

    result = report_writer.continue_report_conversation(
        "France",
        "",
        "Show me how ordinary life shifted from the boom years to smartphones, mostly work and family.",
        turn_number=1,
        existing_state={},
    )

    assert result["action"] == "generate"
    assert result["brief_state"]["subject"] == "changes in everyday life"
    assert result["brief_state"]["ready"] is True


def test_unknown_topic_defaults_to_generic_focused_report():
    keys = report_writer._requested_report_sections(
        "How street typography shaped public identity in Marseille"
    )
    assert keys == report_writer.GENERIC_FOCUSED_REPORT_SECTION_KEYS
    assert keys != report_writer.REPORT_SECTION_KEYS


def test_scoped_parser_preserves_dynamic_section_keys():
    keys = report_writer.GENERIC_FOCUSED_REPORT_SECTION_KEYS
    payload = {
        "introduction": "Intro",
        "focused_analysis": "Concrete analysis",
        "key_developments": "Key developments",
        "context_and_implications": "Context and implications",
        "conclusion": "Conclusion",
    }
    parsed = report_writer._parse_writer_response(
        json.dumps(payload),
        allowed_keys=keys,
    )
    assert parsed["focused_analysis"] == "Concrete analysis"
    assert parsed["key_developments"] == "Key developments"
    assert parsed["context_and_implications"] == "Context and implications"


def test_scoped_report_requires_all_requested_sections(monkeypatch):
    _FakeClient.payload = {
        "introduction": "Intro",
        "focused_analysis": "",
        "key_developments": "",
        "context_and_implications": "",
        "conclusion": "Conclusion",
    }
    _FakeClient.instances = []
    monkeypatch.setattr(report_writer, "OpenAI", _FakeClient)

    with pytest.raises(RuntimeError, match="missing required sections"):
        report_writer._generate_scoped_report(
            api_key="test",
            model="openai/gpt-oss-20b",
            country_name="France",
            country_code="FR",
            user_request="French philosophy in 1960",
            section_keys=report_writer.GENERIC_FOCUSED_REPORT_SECTION_KEYS,
        )


def test_scoped_generation_receives_semantic_brief(monkeypatch):
    payload = {
        "introduction": "Concrete introduction.",
        "focused_analysis": "Sartre, Beauvoir, structuralism, and major debates.",
        "key_developments": "Specific 1960 developments and works.",
        "context_and_implications": "Intellectual context and influence.",
        "conclusion": "Evidence-based conclusion.",
    }

    class _Responses:
        def __init__(self):
            self.calls = []
        def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(output_text=json.dumps(payload))

    class _Client:
        instances = []
        def __init__(self, *args, **kwargs):
            self.responses = _Responses()
            type(self).instances.append(self)

    monkeypatch.setattr(report_writer, "OpenAI", _Client)

    report_writer._generate_scoped_report(
        api_key="test",
        model="openai/gpt-oss-20b",
        country_name="France",
        country_code="FR",
        user_request="France philosophy 1960",
        section_keys=report_writer.GENERIC_FOCUSED_REPORT_SECTION_KEYS,
        brief_state={
            "subject": "French philosophy",
            "topics": ["philosophy", "writers"],
            "period": "1960",
            "angles": ["major thinkers", "works", "debates"],
            "depth": "detailed",
            "exclusions": [],
            "current_events": False,
            "other_constraints": [],
            "ready": True,
            "missing": [],
        },
    )

    first_prompt = _Client.instances[0].responses.calls[0]["input"][1]["content"]
    assert "French philosophy" in first_prompt
    assert '"period": "1960"' in first_prompt
    assert "concrete named evidence" in first_prompt.lower()


def test_country_request_intent_is_normalized_from_model_country_and_request(monkeypatch):
    _FakeClient.payload = {
        "intent": "general",
        "country": "France",
        "request": "French philosophers",
        "reply": "Quels philosophes français souhaitez-vous explorer ?",
    }
    _FakeClient.instances = []
    monkeypatch.setattr(report_writer, "OpenAI", _FakeClient)

    result = report_writer.interpret_country_request(
        "Je voudrais explorer les philosophes français"
    )

    assert result["intent"] == "country_request"
    assert result["country"] == "France"
    assert result["request"] == "French philosophers"


def test_scoped_review_uses_plain_prose_per_section(monkeypatch):
    outputs = {
        "Introduction draft.": "Introduction reviewed.",
        "Analysis draft.": "Analysis reviewed with concrete facts.",
        "Conclusion draft.": "Conclusion reviewed.",
    }

    class _Responses:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            prompt = kwargs["input"][1]["content"]
            for draft_text, output in outputs.items():
                if draft_text in prompt:
                    return SimpleNamespace(output_text=output)
            raise AssertionError("draft section not present in review prompt")

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
        user_request="French philosophy in 1960",
        section_keys=("introduction", "focused_analysis", "conclusion"),
        draft={
            "introduction": "Introduction draft.",
            "focused_analysis": "Analysis draft.",
            "conclusion": "Conclusion draft.",
        },
        brief_state={
            "subject": "French philosophy",
            "period": "1960",
            "topics": ["philosophy"],
        },
    )

    assert result["introduction"] == "Introduction reviewed."
    assert result["focused_analysis"] == "Analysis reviewed with concrete facts."
    assert result["conclusion"] == "Conclusion reviewed."
    calls = _Client.instances[0].responses.calls
    assert len(calls) == 3
    assert all("text" not in call for call in calls)
    assert all(call["tools"] == [{"type": "browser_search"}] for call in calls)
    assert all(call["tool_choice"] == "auto" for call in calls)


def test_post_report_followup_can_reply_without_regeneration(monkeypatch):
    _FakeClient.payload = {
        "action": "reply",
        "country": "France",
        "normalized_request": "France history and public figures",
        "reply": "The report is ready and remains available in this conversation.",
        "brief_state": {
            "subject": "history and public figures",
            "topics": ["history", "public figures"],
            "period": "",
            "angles": [],
            "depth": "",
            "exclusions": [],
            "current_events": False,
            "other_constraints": [],
            "ready": True,
            "missing": [],
        },
    }
    _FakeClient.instances = []
    monkeypatch.setattr(report_writer, "OpenAI", _FakeClient)
    result = report_writer.continue_report_conversation(
        "France",
        "France history and public figures",
        "where...?",
        turn_number=4,
        existing_state=_FakeClient.payload["brief_state"],
        conversation_history=[
            {"role": "user", "content": "Its history and public figures"},
            {"role": "assistant", "content": "I will prepare that report."},
            {"role": "user", "content": "I am waiting..."},
        ],
        report_available=True,
    )
    assert result["action"] == "reply"
    assert result["country"] == "France"
    assert result["normalized_request"] == "France history and public figures"
