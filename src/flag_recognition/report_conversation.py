"""Context-adaptive requirements gathering for Flag Intelligence reports.

This module owns conversation meaning. The language model resolves arbitrary user
wording against recent dialogue and a persistent semantic brief. Deterministic
logic validates state, preserves explicit constraints, and provides safe fallback;
it does not decide whether an arbitrary topic is understandable.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from .llm_backend import (
    FlagIntelligenceClient as OpenAI,
    llm_auth_token,
    llm_model_name,
)


SCALAR_FIELDS = ("subject", "scope", "period", "entity_range", "depth")
LIST_FIELDS = ("topics", "angles", "exclusions", "other_constraints")
MUTABLE_FIELDS = SCALAR_FIELDS + LIST_FIELDS + ("current_events",)
SEMANTIC_ACTIONS = {"clarify", "generate", "status", "converse"}


def _empty_brief() -> dict[str, Any]:
    return {
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
        "confidence": 0.0,
        "ready": False,
        "ambiguities": [],
        "changed_fields": [],
    }


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _clean_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        text = _clean_text(item)
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def _strip_code_fence(text: str) -> str:
    value = str(text or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
    return value.strip()


def _explicit_year_range(text: str) -> tuple[int, int] | None:
    match = re.search(
        r"\b(1[5-9]\d{2}|20\d{2})\s*(?:-|–|—|to|until|through|till|au|à)\s*"
        r"(1[5-9]\d{2}|20\d{2})\b",
        str(text or ""),
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    start, end = int(match.group(1)), int(match.group(2))
    return (start, end) if start <= end else (end, start)


def semantic_brief_has_scope(brief_state: dict[str, Any] | None) -> bool:
    """Return whether a semantic brief contains a meaningful report need."""
    if not isinstance(brief_state, dict):
        return False
    if any(_clean_text(brief_state.get(key)) for key in SCALAR_FIELDS):
        return True
    if any(_clean_list(brief_state.get(key)) for key in LIST_FIELDS):
        return True
    return bool(brief_state.get("current_events"))


def reconcile_semantic_brief(
    previous_state: dict[str, Any] | None,
    model_state: dict[str, Any] | None,
    *,
    latest_message: str,
    existing_request: str,
) -> dict[str, Any]:
    """Merge one semantic turn without reviving constraints the user cleared."""
    previous = previous_state if isinstance(previous_state, dict) else {}
    model = model_state if isinstance(model_state, dict) else {}
    result = _empty_brief()

    changed_fields = {
        _clean_text(value).casefold()
        for value in model.get("changed_fields", [])
        if _clean_text(value).casefold() in MUTABLE_FIELDS
    } if isinstance(model.get("changed_fields"), list) else set()

    for key in SCALAR_FIELDS:
        model_value = _clean_text(model.get(key))
        previous_value = _clean_text(previous.get(key))
        result[key] = (
            model_value
            if key in changed_fields
            else (model_value or previous_value)
        )

    for key in LIST_FIELDS:
        model_items = _clean_list(model.get(key))
        previous_items = _clean_list(previous.get(key))
        result[key] = (
            model_items
            if key in changed_fields
            else (model_items or previous_items)
        )

    model_current = bool(model.get("current_events"))
    previous_current = bool(previous.get("current_events"))
    result["current_events"] = (
        model_current
        if "current_events" in changed_fields
        else (model_current or previous_current)
    )

    # Calendar-year extraction is a high-confidence fallback only. Numeric or
    # ordinal fragments such as "1 to 5" are intentionally left to semantics.
    if not result["period"]:
        year_range = _explicit_year_range(latest_message) or _explicit_year_range(existing_request)
        if year_range is not None:
            result["period"] = f"{year_range[0]} to {year_range[1]}"

    try:
        confidence = float(model.get("confidence", previous.get("confidence", 0.0)))
    except (TypeError, ValueError):
        confidence = 0.0
    result["confidence"] = max(0.0, min(1.0, confidence))

    result["ambiguities"] = _clean_list(model.get("ambiguities"))
    result["ready"] = bool(model.get("ready")) and not result["ambiguities"]
    result["changed_fields"] = sorted(changed_fields)
    return result


def _request_from_brief(model_request: str, brief: dict[str, Any]) -> str:
    """Keep model wording while ensuring preserved constraints reach the writer."""
    request = _clean_text(model_request)
    pieces: list[str] = []

    def add_piece(label: str, value: str) -> None:
        nonlocal request
        text = _clean_text(value)
        if not text:
            return
        if text.casefold() in request.casefold():
            return
        pieces.append(f"{label}: {text}")

    add_piece("subject", brief.get("subject", ""))
    add_piece("scope", brief.get("scope", ""))
    add_piece("period", brief.get("period", ""))
    add_piece("range", brief.get("entity_range", ""))
    add_piece("depth", brief.get("depth", ""))

    for label, key in (
        ("topics", "topics"),
        ("focus", "angles"),
        ("exclude", "exclusions"),
        ("constraints", "other_constraints"),
    ):
        values = _clean_list(brief.get(key))
        if values:
            add_piece(label, ", ".join(values))

    if brief.get("current_events") and "current" not in request.casefold():
        pieces.append("include current developments")

    if not request:
        request = _clean_text(brief.get("subject")) or _clean_text(brief.get("scope"))
    if pieces:
        request = "; ".join([part for part in (request, *pieces) if part])
    return request.strip(" ;")


def opening_question(country_name: str) -> str:
    """Country recognition establishes context only; it never starts a report."""
    country = _clean_text(country_name) or "this country"
    return f"I recognized {country}. What would you like to learn about it?"


def _schema() -> dict[str, Any]:
    brief = {
        "type": "object",
        "properties": {
            "subject": {"type": "string"},
            "scope": {"type": "string"},
            "period": {"type": "string"},
            "entity_range": {"type": "string"},
            "topics": {"type": "array", "items": {"type": "string"}},
            "angles": {"type": "array", "items": {"type": "string"}},
            "depth": {"type": "string"},
            "exclusions": {"type": "array", "items": {"type": "string"}},
            "current_events": {"type": "boolean"},
            "other_constraints": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "ready": {"type": "boolean"},
            "ambiguities": {"type": "array", "items": {"type": "string"}},
            "changed_fields": {
                "type": "array",
                "items": {"type": "string", "enum": list(MUTABLE_FIELDS)},
            },
        },
        "required": [
            "subject", "scope", "period", "entity_range", "topics", "angles",
            "depth", "exclusions", "current_events", "other_constraints",
            "confidence", "ready", "ambiguities", "changed_fields",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": sorted(SEMANTIC_ACTIONS)},
            "country": {"type": "string"},
            "normalized_request": {"type": "string"},
            "reply": {"type": "string"},
            "brief_state": brief,
        },
        "required": ["action", "country", "normalized_request", "reply", "brief_state"],
        "additionalProperties": False,
    }


def _prompt(
    *,
    country: str,
    existing_request: str,
    latest_message: str,
    turn_number: int,
    existing_state: dict[str, Any],
    recent_history: list[dict[str, str]],
    report_available: bool,
) -> str:
    return (
        "You are the semantic requirements interpreter for Flag Intelligence. "
        "The chat exists to understand what report the user needs; it is not a general country chatbot. "
        "Interpret the latest message from the whole conversation, not in isolation. Resolve pronouns, "
        "ellipsis, short fragments, ordinal references, numeric ranges, corrections, comparisons, and "
        "follow-ups when the recent context makes their meaning clear. Topics are open-ended: never require "
        "a topic to belong to a predefined taxonomy.\n\n"
        f"CURRENT COUNTRY: {country or '(none)'}\n"
        f"TURN: {turn_number}\n"
        f"REPORT AVAILABLE: {bool(report_available)}\n"
        f"PREVIOUS NORMALIZED REQUEST: {existing_request or '(none)'}\n"
        "PREVIOUS SEMANTIC BRIEF:\n"
        + json.dumps(existing_state, ensure_ascii=False, sort_keys=True)
        + "\nRECENT CONVERSATION:\n"
        + json.dumps(recent_history, ensure_ascii=False)
        + "\nLATEST USER MESSAGE:\n"
        + latest_message
        + "\n\nRULES:\n"
        "- Treat the current country as context from flag recognition unless the user explicitly switches country.\n"
        "- Never answer substantive country knowledge in pre-report chat. Put that content in the report.\n"
        "- `generate` when a coherent report need can be stated without materially guessing the user's intent.\n"
        "- Optional angle, depth, exclusions, region, and subtopic preferences must not block generation.\n"
        "- `clarify` only when multiple plausible interpretations would materially change the report; ask exactly one short targeted question.\n"
        "- `status` only for questions about an already generated report or generation state.\n"
        "- `converse` only for greetings/acknowledgements that do not yet express a report need.\n"
        "- A historical request does not require calendar years when another bounded scope is clear (for example an era, named republics, reign, dynasty, or post-independence period).\n"
        "- Interpret `1 to 5`, `the first three`, `both`, `traditional only`, `after independence`, etc. from recent dialogue when a referent is clear.\n"
        "- If the user says `anything is fine`, `whatever`, or `you choose`, accept a reasonable default and generate instead of repeating the question.\n"
        "- Preserve previous fields unless the user changes them. Put only explicitly added/replaced/cleared fields in changed_fields. A cleared field must be returned empty and listed in changed_fields.\n"
        "- `period` is for temporal scope. `entity_range` is for non-calendar ranges such as First through Fifth Republic.\n"
        "- confidence reflects confidence in the resolved semantic brief. ambiguities lists only unresolved interpretations that materially block generation.\n"
        "- normalized_request must be a concise executable brief for the report writer and include all explicit user constraints.\n"
        "- reply in the user's language unless they request another language.\n"
        "Return JSON only."
    )


def _fallback_turn(
    *,
    country: str,
    existing_request: str,
    latest_message: str,
    existing_state: dict[str, Any],
) -> dict[str, Any]:
    brief = reconcile_semantic_brief(
        existing_state,
        {},
        latest_message=latest_message,
        existing_request=existing_request,
    )
    # Backend failure must not pretend the request is ready. Preserve the known
    # brief and ask one neutral question so the user can continue safely.
    brief["ready"] = False
    brief["ambiguities"] = ["Latest refinement could not be interpreted reliably."]
    reply = (
        "Could you rephrase the latest part of what you want included?"
        if semantic_brief_has_scope(brief)
        else "What would you like the report to focus on?"
    )
    return {
        "action": "clarify",
        "country": country,
        "normalized_request": _request_from_brief(existing_request, brief),
        "reply": reply,
        "brief_state": brief,
    }


def continue_semantic_conversation(
    country_name: str,
    existing_request: str,
    latest_message: str,
    turn_number: int = 1,
    existing_state: dict[str, Any] | None = None,
    conversation_history: list[dict[str, str]] | None = None,
    report_available: bool = False,
) -> dict[str, Any]:
    """Interpret one requirements-gathering turn from full conversational context."""
    country = _clean_text(country_name)
    existing = _clean_text(existing_request)
    latest = _clean_text(latest_message)
    state = existing_state if isinstance(existing_state, dict) else {}
    history = conversation_history if isinstance(conversation_history, list) else []
    recent_history = [
        {
            "role": _clean_text(item.get("role")),
            "content": _clean_text(item.get("content")),
        }
        for item in history[-12:]
        if isinstance(item, dict)
    ]

    api_key = llm_auth_token()
    if not api_key:
        return _fallback_turn(
            country=country,
            existing_request=existing,
            latest_message=latest,
            existing_state=state,
        )

    prompt = _prompt(
        country=country,
        existing_request=existing,
        latest_message=latest,
        turn_number=int(turn_number),
        existing_state=state,
        recent_history=recent_history,
        report_available=bool(report_available),
    )
    client = OpenAI(
        api_key=api_key,
        timeout=float(os.getenv("FLAG_INTELLIGENCE_DIALOGUE_TIMEOUT", "12")),
        max_retries=0,
    )

    parsed: dict[str, Any] | None = None
    try:
        response = client.responses.create(
            model=llm_model_name(),
            reasoning={"effort": "low"},
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "flag_intelligence_semantic_report_turn",
                    "strict": True,
                    "schema": _schema(),
                }
            },
            max_output_tokens=900,
        )
        candidate = json.loads(_strip_code_fence(response.output_text or ""))
        if isinstance(candidate, dict):
            parsed = candidate
    except Exception:
        parsed = None

    if parsed is None:
        try:
            response = client.responses.create(
                model=llm_model_name(),
                reasoning={"effort": "low"},
                input=prompt + "\nReturn one valid JSON object only.",
                text={"format": {"type": "json_object"}},
                max_output_tokens=1000,
            )
            candidate = json.loads(_strip_code_fence(response.output_text or ""))
            if isinstance(candidate, dict):
                parsed = candidate
        except Exception:
            parsed = None

    if not isinstance(parsed, dict) or not isinstance(parsed.get("brief_state"), dict):
        return _fallback_turn(
            country=country,
            existing_request=existing,
            latest_message=latest,
            existing_state=state,
        )

    brief = reconcile_semantic_brief(
        state,
        parsed["brief_state"],
        latest_message=latest,
        existing_request=existing,
    )
    action = _clean_text(parsed.get("action")).casefold()
    if action not in SEMANTIC_ACTIONS:
        action = "clarify"

    normalized_request = _request_from_brief(
        _clean_text(parsed.get("normalized_request")),
        brief,
    )
    reply = _clean_text(parsed.get("reply"))
    turn_country = _clean_text(parsed.get("country")) or country

    has_scope = semantic_brief_has_scope(brief)
    if action == "generate" and (not brief.get("ready") or not has_scope or not normalized_request):
        action = "clarify"
    elif (
        action == "clarify"
        and brief.get("ready")
        and has_scope
        and not brief.get("ambiguities")
        and normalized_request
    ):
        action = "generate"

    if action == "generate":
        reply = "Generating your report now."
    elif action == "clarify":
        if not reply.endswith("?") or len(reply) > 220:
            ambiguity = _clean_list(brief.get("ambiguities"))
            reply = (
                f"Could you clarify {ambiguity[0].rstrip('.')}?"
                if ambiguity
                else "What specific part of your request should I use for the report?"
            )
        # Never emit country exposition in a clarification turn.
        reply = reply[:220]
    elif action == "status" and not report_available:
        action = "clarify"
        reply = "What would you like the report to focus on?"
    elif not reply:
        reply = opening_question(turn_country) if action == "converse" else "What would you like the report to focus on?"

    return {
        "action": action,
        "country": turn_country,
        "normalized_request": normalized_request,
        "reply": reply,
        "brief_state": brief,
    }
