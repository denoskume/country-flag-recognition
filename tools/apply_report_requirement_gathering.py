from pathlib import Path


WRITER = Path(__file__).resolve().parents[1] / "src" / "flag_recognition" / "report_writer.py"

source = WRITER.read_text(encoding="utf-8")

replacements = [
    (
        '"economy": ("economy", "economic", "trade", "industry", "industrial"),',
        '"economy": ("economy", "economic", "economical", "trade", "industry", "industrial"),',
    ),
    (
        '"You are Flag Intelligence conducting a live, natural conversation about countries. "',
        '"You are Flag Intelligence gathering requirements for a tailored country report. "',
    ),
    (
        '''        "- action='reply' for ordinary conversation, acknowledgements, questions about an existing "\n        "report, requests to explain previously discussed material, or any turn that does not need a "\n        "new or updated PDF. Never regenerate merely because brief_state.ready is true.\\n"\n        "- action='ask' only when a material ambiguity prevents a requested new/updated report. "\n        "Ask one concise non-repetitive question.\\n"\n        "- action='generate' only when the user requests a new or materially updated report and the "\n        "brief is sufficiently clear.\\n"''',
        '''        "- When REPORT AVAILABLE is false, do not answer substantive country knowledge in chat. "\n        "Treat country-topic input as requirements for the report.\\n"\n        "- action='reply' only for greetings, acknowledgements, or non-substantive interaction before "\n        "a report scope exists, and for questions or status about an existing report.\\n"\n        "- action='ask' when the report need is not yet sufficiently specific. Ask exactly one concise "\n        "question about the most important missing requirement and include no explanatory country facts.\\n"\n        "- action='generate' as soon as the report brief is sufficiently clear; the user does not need "\n        "to say the word report or PDF.\\n"''',
    ),
]

for old, new in replacements:
    if old not in source:
        raise RuntimeError(f"Required replacement anchor not found: {old[:90]!r}")
    source = source.replace(old, new, 1)

old_logic = '''    requested_action = str(parsed.get("action") or "reply").strip().lower()\n    ready = bool(brief_state.get("ready"))\n    prior_ready = bool(state.get("ready"))\n    known_scope = bool(existing) or bool(str(brief_state.get("subject") or "").strip()) or bool(\n        brief_state.get("topics")\n    )\n    explicit_report_request = _is_explicit_report_request(latest)\n    report_confirmation = _is_report_generation_confirmation(latest, recent_history)\n\n    # The PDF is a real application artifact, not conversational prose. When a\n    # user explicitly asks for it (or confirms a direct offer), never allow an\n    # LLM reply to impersonate generation. A sufficiently specific first brief\n    # should also go straight to the report pipeline, which is Flag\n    # Intelligence's primary product behavior.\n    force_generation = (\n        not report_available\n        and (\n            (explicit_report_request and (known_scope or ready or prior_ready))\n            or (report_confirmation and (known_scope or ready or prior_ready))\n            or (ready and int(turn_number) <= 1)\n        )\n    )\n\n    if force_generation:\n        action = "generate"\n    elif requested_action == "generate" and not ready:\n        action = "ask"\n    elif requested_action in {"reply", "ask", "generate"}:\n        action = requested_action\n    else:\n        action = "reply"\n'''

new_logic = '''    requested_action = str(parsed.get("action") or "reply").strip().lower()\n    ready = bool(brief_state.get("ready"))\n    prior_ready = bool(state.get("ready"))\n    explicit_report_request = _is_explicit_report_request(latest)\n    report_confirmation = _is_report_generation_confirmation(latest, recent_history)\n\n    scope_fields = (\n        brief_state.get("topics"),\n        str(brief_state.get("period") or "").strip(),\n        brief_state.get("angles"),\n        str(brief_state.get("depth") or "").strip(),\n        brief_state.get("exclusions"),\n        bool(brief_state.get("current_events")),\n        brief_state.get("other_constraints"),\n    )\n    has_report_scope = (\n        bool(existing)\n        or any(bool(value) for value in scope_fields)\n        or bool(_requested_topic_groups(latest))\n    )\n    deterministic_ready = _brief_is_sufficiently_specific(\n        existing or normalized_request,\n        latest,\n    )\n    brief_ready = ready or prior_ready or deterministic_ready\n\n    # Flag Intelligence is report-first: before a report exists, substantive\n    # country requests are requirements to gather, not prose questions to answer\n    # in chat. Once the brief is ready, generation begins automatically.\n    if report_available:\n        if requested_action in {"reply", "ask", "generate"}:\n            action = requested_action\n        else:\n            action = "reply"\n    elif brief_ready and has_report_scope:\n        action = "generate"\n    elif has_report_scope or explicit_report_request or report_confirmation:\n        action = "ask"\n    elif requested_action == "generate":\n        action = "ask"\n    elif requested_action in {"reply", "ask"}:\n        action = requested_action\n    else:\n        action = "reply"\n\n    if action == "ask":\n        is_concise_question = reply.endswith("?") and len(reply) < 220\n        if not is_concise_question:\n            raw_missing = brief_state.get("missing")\n            missing_items = (\n                [str(item).strip().casefold() for item in raw_missing if str(item).strip()]\n                if isinstance(raw_missing, list)\n                else []\n            )\n            missing_dimension = (\n                missing_items[0]\n                if missing_items\n                else _brief_missing_dimension(existing or normalized_request, latest)\n            )\n            questions = {\n                "period": "What time period should the report cover?",\n                "depth": "How detailed should the report be: brief, balanced, or in-depth?",\n                "angle": "Which aspects should the report emphasize?",\n                "angles": "Which aspects should the report emphasize?",\n                "topic": "What should the report focus on?",\n                "topics": "What should the report focus on?",\n                "exclusions": "Is there anything you want the report to leave out?",\n            }\n            reply = questions.get(\n                missing_dimension,\n                "What specific scope should the report follow?",\n            )\n'''

if old_logic not in source:
    raise RuntimeError("Report conversation action block anchor not found")
source = source.replace(old_logic, new_logic, 1)

WRITER.write_text(source, encoding="utf-8")
