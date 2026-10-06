from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WRITER = ROOT / "src/flag_recognition/report_writer.py"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"Missing patch anchor: {label}")
    return text.replace(old, new, 1)


writer = WRITER.read_text(encoding="utf-8")

helper_anchor = '''def continue_report_conversation(\n'''
helpers = r'''def _is_explicit_report_request(message: str) -> bool:
    """Return True when the user explicitly asks for a report artifact."""
    normalized = _normalized_request_text(message)
    if not normalized:
        return False
    return bool(re.search(r"\b(?:report|pdf)\b", normalized))


def _is_report_command_only(message: str) -> bool:
    """Detect short artifact commands that should keep the existing report scope."""
    normalized = _normalized_request_text(message)
    if not normalized:
        return False

    removable = {
        "a", "an", "the", "me", "my", "please", "now", "it",
        "report", "pdf", "document", "file",
        "give", "create", "generate", "make", "prepare", "build",
        "download", "export", "produce", "send", "show", "want",
        "i", "would", "like", "get",
    }
    remaining = [word for word in normalized.split() if word not in removable]
    return not remaining


def _is_report_generation_confirmation(
    latest_message: str,
    recent_history: list[dict[str, str]],
) -> bool:
    """Resolve a short yes/okay against the assistant's immediately prior report offer."""
    normalized = _normalized_request_text(latest_message)
    confirmations = {
        "yes", "yes please", "yeah", "yep", "ok", "okay", "sure",
        "please do", "go ahead", "do it", "generate it", "create it",
    }
    if normalized not in confirmations:
        return False

    last_assistant = ""
    for item in reversed(recent_history):
        if str(item.get("role") or "").strip().lower() == "assistant":
            last_assistant = _normalized_request_text(item.get("content") or "")
            break

    if not last_assistant:
        return False

    mentions_artifact = bool(re.search(r"\b(?:report|pdf)\b", last_assistant))
    offers_generation = bool(
        re.search(
            r"\b(?:create|generate|prepare|build|make|produce|export)\w*\b",
            last_assistant,
        )
    )
    return mentions_artifact and offers_generation


'''
if "def _is_explicit_report_request(" not in writer:
    if helper_anchor not in writer:
        raise RuntimeError("Missing patch anchor: helper insertion")
    writer = writer.replace(helper_anchor, helpers + helper_anchor, 1)

old_logic = '''    requested_action = str(parsed.get("action") or "reply").strip().lower()\n    ready = bool(brief_state.get("ready"))\n    if requested_action == "generate" and not ready:\n        action = "ask"\n    elif requested_action in {"reply", "ask", "generate"}:\n        action = requested_action\n    else:\n        action = "reply"\n'''
new_logic = '''    requested_action = str(parsed.get("action") or "reply").strip().lower()\n    ready = bool(brief_state.get("ready"))\n    prior_ready = bool(state.get("ready"))\n    known_scope = bool(existing) or bool(str(brief_state.get("subject") or "").strip()) or bool(\n        brief_state.get("topics")\n    )\n    explicit_report_request = _is_explicit_report_request(latest)\n    report_confirmation = _is_report_generation_confirmation(latest, recent_history)\n\n    # The PDF is a real application artifact, not conversational prose. When a\n    # user explicitly asks for it (or confirms a direct offer), never allow an\n    # LLM reply to impersonate generation. A sufficiently specific first brief\n    # should also go straight to the report pipeline, which is Flag\n    # Intelligence's primary product behavior.\n    force_generation = (\n        not report_available\n        and (\n            (explicit_report_request and (known_scope or ready or prior_ready))\n            or (report_confirmation and (known_scope or ready or prior_ready))\n            or (ready and int(turn_number) <= 1)\n        )\n    )\n\n    if force_generation:\n        action = "generate"\n    elif requested_action == "generate" and not ready:\n        action = "ask"\n    elif requested_action in {"reply", "ask", "generate"}:\n        action = requested_action\n    else:\n        action = "reply"\n\n    # Short commands such as "give me a report" or "pdf report" are artifact\n    # requests, not a replacement for the already-resolved research brief.\n    if (\n        action == "generate"\n        and existing\n        and (report_confirmation or _is_report_command_only(latest))\n    ):\n        normalized_request = existing\n'''
writer = replace_once(writer, old_logic, new_logic, "deterministic generation policy")

WRITER.write_text(writer, encoding="utf-8")
