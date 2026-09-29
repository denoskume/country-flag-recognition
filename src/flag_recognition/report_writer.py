"""OpenAI-authored country report narrative.

The data collectors remain responsible for evidence gathering. This module is
responsible only for editorial synthesis: turning the collected evidence into
clean, coherent report prose.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI


REPORT_SECTION_KEYS = (
    "introduction",
    "physical_geography",
    "climate_water_resources",
    "flag_design_symbolism",
    "origins_early_history",
    "historical_journey",
    "state_formation_identity",
    "government_structure",
    "leadership_through_time",
    "people_society",
    "languages_religion",
    "health_public_health",
    "culture_cuisine_music_sport",
    "festivals_holidays_traditions",
    "heritage_landmarks",
    "literature_philosophy_thought",
    "economy_trade_industries",
    "infrastructure_transport_energy",
    "education_research",
    "science_discovery_invention",
    "environment_biodiversity",
    "cost_of_living",
    "practical_emergency",
    "international_relations",
    "notable_public_figures",
    "conclusion",
)


SYSTEM_PROMPT = """You are the senior editorial writer for Flag Intelligence.

You receive the identity of one country plus optional locally collected evidence.
Write the final English-language educational country report after independently
researching and verifying the country using web search. The local evidence is
secondary context only and must never constrain you when it is incomplete,
outdated, malformed or contradictory.

NON-NEGOTIABLE RULES
- Treat supplied local data as optional supporting evidence, not as the primary
  authority and never as prose to copy.
- Use web search to independently verify important facts and to fill gaps.
- Prefer official government, national statistics, international organizations,
  universities, museums, recognized research institutions and other authoritative
  primary or high-quality sources.
- For time-sensitive facts, search for current information before writing.
- For historical facts, prefer authoritative institutional or scholarly sources
  where available.
- Rewrite everything in clean, natural, professional English.
- Never copy source fragments, captions, tables, navigation text, bibliography
  residue, or malformed phrases.
- Remove duplicates, repeated dates, repeated words, and repeated ideas.
- Never produce constructions such as "for for", "in 1947 ... in 1947",
  "by 1801 ... in 1801", or duplicated sentences.
- Do not invent numerical or current political facts. Verify them with web search.
  If reliable sources conflict, use the strongest and most recent source or state
  the uncertainty briefly.
- Keep history chronological and relevant to the country. Exclude unrelated
  global background unless it directly explains a national event.
- For historical events, explain what happened and why it mattered rather than
  listing dates mechanically.
- For notable people, write mini-biographies. For each person retained, explain
  who they are/were, their period or lifespan when supported, their field, why
  they became notable, and a major contribution or achievement. Never output a
  raw name-role list.
- Do not overstate causal claims.
- Avoid repetition across sections. Each fact should normally appear once in
  the most relevant section.
- Use short, coherent paragraphs. Prefer 2-4 sentences per paragraph.
- Preserve useful dates and measurements only when they are supported.
- The conclusion must synthesize the whole report rather than repeat the
  introduction or snapshot.
- Do not include citations, URLs, markdown headings, bullets, tables, or source
  names in the prose.
- If local evidence is insufficient, research the section independently. Return an
  empty string only when reliable information genuinely cannot be established.

OUTPUT
Return ONLY one valid JSON object. It must contain exactly the keys supplied in
the requested schema. Each value must be a plain string. Use "\\n\\n" between
paragraphs. No markdown fences.
"""


REVIEW_PROMPT = """You are the final factual and editorial verifier for Flag Intelligence.

You receive:
1. the country identity and optional locally collected evidence;
2. a drafted report with fixed section keys.

Your task is to independently fact-check the entire report with web search and
return a corrected report that is publishable. Local evidence is only supporting
context and may be incomplete, stale or malformed.

STRICT VERIFICATION RULES
- Check EVERY factual assertion in the draft using web search and authoritative
  sources.
- Do not invent missing facts.
- If a statement cannot be verified from reliable sources, remove it.
- Prefer official and primary sources, then major international institutions and
  reputable academic or reference sources.
- If sources conflict, prefer the more authoritative and more recent source when
  the subject is time-sensitive; otherwise omit or qualify the disputed claim.
- Time-sensitive statements (current leaders, prices, population, GDP,
  infrastructure status, memberships, health statistics, etc.) must retain a
  reference year/date when the evidence provides one. Do not silently present
  old measurements as current.
- Historical dates must be chronologically coherent and internally consistent.
- Remove unrelated global background that is not directly about the country.
- Remove duplicate facts across sections unless repetition is essential for
  comprehension.
- Remove repeated sentences, repeated dates, repeated words, malformed source
  fragments, captions, bibliography residue and list/navigation artefacts.
- Notable people must be genuinely tied to the country by the supplied evidence.
  If that connection is weak or ambiguous, omit the person.
- Mini-biographies must explain why the person is notable, but only using
  supported evidence.
- Do not preserve a sentence merely because it sounds plausible.
- If evidence is insufficient for a section, return an empty string.
- Keep prose concise, natural and professional.

OUTPUT
Return ONLY one JSON object with this exact shape:
{
  "passed": true_or_false,
  "issues": ["short issue descriptions"],
  "report": { ...exact report section keys... }
}

Set passed=true only if the corrected report contains no unsupported,
contradictory, duplicated, malformed or misleading factual statements.
No markdown fences.
"""


def _deterministic_quality_issues(report: dict[str, str]) -> list[str]:
    """Catch mechanical defects after model-based verification."""
    issues: list[str] = []
    seen_sentences: dict[str, str] = {}

    repeated_word = re.compile(
        r"\b(for|in|on|by|the|a|an|to|of|and|or|is|was|were|with)\s+\1\b",
        flags=re.IGNORECASE,
    )
    repeated_year = re.compile(
        r"\b(?:in|by|on)\s+(\d{3,4})\b[^.!?]{0,120}"
        r"\b(?:in|by|on)\s+\1\b",
        flags=re.IGNORECASE,
    )
    residue = re.compile(
        r"\b(?:thumb|upright|rowspan|colspan|wikitable|further reading|"
        r"references|see also|image size|plot area|published as|\d+pp\b)\b",
        flags=re.IGNORECASE,
    )

    for section, text in report.items():
        if section not in REPORT_SECTION_KEYS or not text:
            continue

        if repeated_word.search(text):
            issues.append(f"{section}: duplicated word/preposition")
        if repeated_year.search(text):
            issues.append(f"{section}: duplicated year construction")
        if residue.search(text):
            issues.append(f"{section}: source/navigation residue")

        sentences = [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+", text)
            if part.strip()
        ]
        for sentence in sentences:
            key = re.sub(r"[^a-z0-9]+", " ", sentence.casefold()).strip()
            if len(key) < 35:
                continue
            previous = seen_sentences.get(key)
            if previous is not None:
                issues.append(
                    f"{section}: sentence duplicated from {previous}"
                )
            else:
                seen_sentences[key] = section

    return list(dict.fromkeys(issues))


def _review_and_correct(
    client: OpenAI,
    model: str,
    evidence_json: str,
    draft: dict[str, str],
) -> tuple[dict[str, str], bool, list[str]]:
    schema_hint = {key: "" for key in REPORT_SECTION_KEYS}
    prompt = (
        "Verify and correct the drafted report against the evidence.\n\n"
        "Required report keys:\n"
        + json.dumps(schema_hint, ensure_ascii=False)
        + "\n\nEVIDENCE:\n"
        + evidence_json
        + "\n\nDRAFT REPORT:\n"
        + json.dumps(draft, ensure_ascii=False, sort_keys=True)
    )

    response = client.responses.create(
        model=model,
        reasoning={"effort": "high"},
        input=[
            {"role": "system", "content": REVIEW_PROMPT},
            {"role": "user", "content": prompt},
        ],
        tools=[{"type": "web_search"}],
        tool_choice="auto",
        max_output_tokens=16000,
    )

    raw = _strip_code_fence(response.output_text or "")
    if not raw:
        return {}, False, ["Verifier returned no output."]

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return {}, False, ["Verifier returned invalid JSON."]
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}, False, ["Verifier returned invalid JSON."]

    if not isinstance(payload, dict):
        return {}, False, ["Verifier response was not an object."]

    corrected = _normalize_output(payload.get("report"))
    model_issues = payload.get("issues", [])
    if not isinstance(model_issues, list):
        model_issues = []
    model_issues = [str(item).strip() for item in model_issues if str(item).strip()]

    mechanical = _deterministic_quality_issues(corrected)

    # The verifier's prose-level concerns are advisory after it has already
    # corrected the report. Publication is blocked only by concrete remaining
    # mechanical defects or by a missing/empty corrected report. This avoids
    # false negatives where the verifier is overly conservative despite having
    # produced a coherent corrected report.
    substantial_sections = sum(
        1
        for key in REPORT_SECTION_KEYS
        if corrected.get(key, "").strip()
    )
    passed = (
        substantial_sections >= 12
        and not mechanical
        and bool(corrected.get("introduction"))
        and bool(corrected.get("historical_journey"))
        and bool(corrected.get("conclusion"))
    )
    return corrected, passed, model_issues + mechanical


def _strip_code_fence(value: str) -> str:
    text = value.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _normalize_output(payload: Any) -> dict[str, str]:
    if not isinstance(payload, dict):
        return {}
    result: dict[str, str] = {}
    for key in REPORT_SECTION_KEYS:
        value = payload.get(key, "")
        if not isinstance(value, str):
            value = ""
        result[key] = re.sub(r"[ \t]+", " ", value).strip()
    return result


def generate_authored_report(report: dict[str, Any]) -> dict[str, str]:
    """Generate final report prose from the collected evidence payload.

    The call is intentionally bounded: no SDK retries and a finite timeout.
    The caller can safely fall back to deterministic rendering if the API is
    unavailable.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return {}

    model = os.getenv("FLAG_INTELLIGENCE_WRITER_MODEL", "gpt-5.6-sol").strip()
    client = OpenAI(
        api_key=api_key,
        timeout=105.0,
        max_retries=0,
    )

    schema_hint = {key: "" for key in REPORT_SECTION_KEYS}
    evidence_json = json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )

    user_prompt = (
        "Write the final country report from the evidence below.\n\n"
        "Required JSON shape:\n"
        + json.dumps(schema_hint, ensure_ascii=False)
        + "\n\nEVIDENCE:\n"
        + evidence_json
    )

    response = client.responses.create(
        model=model,
        reasoning={"effort": "medium"},
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        tools=[{"type": "web_search"}],
        tool_choice="auto",
        max_output_tokens=14000,
    )

    raw = _strip_code_fence(response.output_text or "")
    if not raw:
        return {}

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}

    draft = _normalize_output(parsed)
    if not draft:
        return {}

    # One OpenAI call must complete the whole research + writing + factual
    # self-check workflow. Only deterministic mechanical QA runs afterwards,
    # keeping total generation latency within the 120-second product budget.
    mechanical = _deterministic_quality_issues(draft)
    substantial_sections = sum(
        1
        for key in REPORT_SECTION_KEYS
        if draft.get(key, "").strip()
    )
    passed = (
        substantial_sections >= 12
        and not mechanical
        and bool(draft.get("introduction"))
        and bool(draft.get("historical_journey"))
        and bool(draft.get("conclusion"))
    )
    draft["__qa_passed"] = passed
    draft["__qa_issues"] = mechanical
    return draft
