"""LLM-authored country report narrative.

The data collectors remain responsible for evidence gathering. This module is
responsible only for editorial synthesis: turning the collected evidence into
clean, coherent report prose.
"""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from typing import Any

from .llm_backend import (
    FlagIntelligenceClient as OpenAI,
    llm_auth_token,
    llm_backend_name,
    llm_model_name,
    probe_llm_backend,
)


REPORT_GENERATION_BUDGET_SECONDS = min(
    300.0,
    max(
        120.0,
        float(os.getenv("FLAG_INTELLIGENCE_REPORT_BUDGET_SECONDS", "300")),
    ),
)


def _remaining_budget(deadline: float, *, cap: float) -> float:
    """Return a safe per-call timeout inside the shared report budget."""
    remaining = deadline - time.monotonic()
    if remaining <= 4.0:
        return 0.0
    return min(cap, max(4.0, remaining - 2.0))


REPORT_SECTION_KEYS = (
    "introduction",
    "physical_geography",
    "climate_water_resources",
    "seasons_climate_calendar",
    "flag_design_symbolism",
    "origins_early_history",
    "historical_journey",
    "key_historical_timeline",
    "state_formation_identity",
    "government_structure",
    "legal_constitutional_system",
    "leadership_through_time",
    "people_society",
    "demographics_population_structure",
    "languages_religion",
    "health_public_health",
    "culture_cuisine_music_sport",
    "festivals_holidays_traditions",
    "heritage_landmarks",
    "major_cities_regional_profiles",
    "national_symbols_identity",
    "literature_philosophy_thought",
    "economy_trade_industries",
    "infrastructure_transport_energy",
    "education_research",
    "universities_higher_education",
    "science_discovery_invention",
    "environment_biodiversity",
    "cost_of_living",
    "practical_emergency",
    "international_relations",
    "notable_figures_philosophy",
    "notable_figures_literature_poetry",
    "notable_figures_mathematics",
    "notable_figures_physics",
    "notable_figures_science_medicine",
    "notable_figures_invention_engineering",
    "notable_figures_arts_architecture",
    "notable_figures_music_cinema",
    "notable_figures_public_life",
    "notable_figures_sport",
    "notable_public_figures",
    "conclusion",
)


HISTORY_REPORT_SECTION_KEYS = (
    "introduction",
    "origins_early_history",
    "historical_journey",
    "key_historical_timeline",
    "state_formation_identity",
    "conclusion",
)


def _requested_report_sections(user_request: str) -> tuple[str, ...]:
    """Return only the sections explicitly needed for a narrow report request."""
    normalized = re.sub(
        r"[^a-z0-9 ]+",
        " ",
        str(user_request or "").casefold(),
    )
    normalized = re.sub(r"\s+", " ", normalized).strip()

    broad_markers = {
        "complete report",
        "full report",
        "everything",
        "all topics",
        "whole country",
        "all about",
    }
    if any(marker in normalized for marker in broad_markers):
        return REPORT_SECTION_KEYS

    history_requested = any(
        token in normalized
        for token in ("history", "historical", "histoire")
    )

    other_topic_markers = (
        "geography",
        "climate",
        "economy",
        "culture",
        "government",
        "politics",
        "education",
        "science",
        "universities",
        "infrastructure",
        "environment",
        "cost of living",
        "international relations",
        "society",
        "health",
        "sport",
        "sports",
        "flag",
    )

    if history_requested and not any(
        marker in normalized for marker in other_topic_markers
    ):
        return HISTORY_REPORT_SECTION_KEYS

    return REPORT_SECTION_KEYS



def prune_report_to_request(
    report: dict[str, Any],
    user_request: str,
) -> dict[str, Any]:
    """Remove unrelated authored sections for narrow report requests."""
    selected = _requested_report_sections(user_request)
    if selected == REPORT_SECTION_KEYS:
        return dict(report)

    selected_set = set(selected)
    scoped = {
        key: value
        for key, value in report.items()
        if key in selected_set or str(key).startswith("__")
    }
    scoped["__scope_sections"] = selected
    scoped["__generation_mode"] = str(
        report.get("__generation_mode") or "scoped"
    )
    return scoped


SYSTEM_PROMPT = """You are the senior editorial writer for Flag Intelligence.

You receive the identity of one country plus optional locally collected evidence.
Write the final English-language educational country report yourself from your
knowledge and the supplied context. The local evidence is supporting context only
and must never be copied blindly when it is incomplete, outdated, malformed or
contradictory.

NON-NEGOTIABLE RULES
- Treat supplied local data as optional supporting evidence, not as the primary
  authority and never as prose to copy.
- Use your own trained knowledge and reasoning to fill gaps carefully.
- For time-sensitive facts, avoid unsupported precision when the supplied context
  does not establish a current value.
- For historical facts, prioritize well-established facts and coherent chronology.
- seasons_climate_calendar should explain the country's meaningful seasonal cycle.
  Where four temperate seasons apply, identify spring, summer, autumn/fall and
  winter with their typical months, while noting important regional differences.
  In tropical, equatorial, monsoon, desert or southern-hemisphere climates, use
  the locally relevant seasonal model instead: wet/dry seasons, monsoon periods,
  cyclone seasons, hot/cool seasons or reversed southern-hemisphere months.
  Avoid forcing a four-season template where it is climatologically inappropriate.
- Rewrite everything in clean, natural, professional English.
- Never copy source fragments, captions, tables, navigation text, bibliography
  residue, or malformed phrases.
- Remove duplicates, repeated dates, repeated words, and repeated ideas.
- Never produce constructions such as "for for", "in 1947 ... in 1947",
  "by 1801 ... in 1801", or duplicated sentences.
- Do not invent numerical or current political facts. When current precision is
  uncertain, use cautious wording or omit the unsupported detail.
- Keep history chronological and relevant to the country. Exclude unrelated
  global background unless it directly explains a national event.
- key_historical_timeline must provide a concise chronological sequence of major
  national milestones using supported dates or periods. It should summarize, not
  duplicate, the longer historical_journey prose.
- For historical events, explain what happened and why it mattered rather than
  listing dates mechanically.
- Notable people must be organized by field rather than reduced to a short
  generic list. Use the dedicated notable_figures_* sections for applicable
  categories: philosophy; literature/poetry; mathematics; physics; science and
  medicine; invention/engineering; arts/architecture; music/cinema; public life;
  and sport.
- For countries with a rich documented intellectual or cultural history, aim for
  several representative figures per applicable category rather than only three
  or four people overall. A typical broad report may contain roughly 15-30
  figures across all categories, but relevance and factual reliability are more
  important than reaching a quota.
- For each retained person, write a compact mini-biography explaining who they
  are/were, their period or lifespan when well established, their precise field,
  why they became notable, and at least one major contribution, work, discovery,
  achievement or institutional role.
- Do not duplicate the same person across multiple categories unless their work
  genuinely spans fields and the repetition adds distinct educational value.
- Do not invent figures to fill a category. Leave a category empty when the
  country has no reliably established representative for it.
- notable_public_figures is reserved only for important figures who do not fit
  naturally into the dedicated categories, or for a concise cross-field synthesis.
  Never output a raw name-role list.
- The universities_higher_education section is mandatory whenever reliable
  knowledge exists. Distinguish historically significant institutions from
  currently prominent or internationally recognized universities. For each
  important institution retained, explain its city/location, founding period or
  historical origin when well established, institutional role, major academic
  strengths or contributions, and why it matters nationally or internationally.
- Do not invent ranking positions. Exact contemporary rankings are time-sensitive:
  include a rank only when the supplied context establishes the ranking body,
  edition/year and position. Otherwise use neutral wording such as "widely
  recognized", "a major national university", or "internationally prominent".
- For countries without medieval or early universities, identify the earliest
  major modern higher-education institutions instead of forcing an ancient
  university narrative. Mention historical closures, mergers, renamings or
  predecessor institutions when they are important to understanding continuity.
- Do not overstate causal claims.
- Avoid repetition across sections. Each fact should normally appear once in
  the most relevant section.
- major_cities_regional_profiles should explain the capital, largest or most
  influential cities, their economic/cultural roles, and meaningful regional
  differences without becoming a travel guide.
- demographics_population_structure should cover population distribution,
  urbanization, age structure when reliably known, migration/diaspora context,
  and major demographic patterns. Do not invent ethnic or religious percentages.
- national_symbols_identity should cover official or widely established national
  symbols such as the coat of arms, motto, anthem and other formally recognized
  symbols where relevant, while distinguishing official status from common usage.
- legal_constitutional_system should summarize the constitution, legal tradition,
  court structure and the interaction of civil, common, religious or customary
  law where applicable, without giving legal advice.
- Use short, coherent paragraphs. Prefer 2-4 sentences per paragraph.
- Preserve useful dates and measurements only when they are supported.
- The conclusion must synthesize the whole report rather than repeat the
  introduction or snapshot.
- Do not include citations, URLs, markdown headings, bullets, tables, or source
  names in the prose.
- If local evidence is insufficient, use your own knowledge conservatively.
  Return an empty string only when you cannot provide a reliable section.

OUTPUT
Return ONLY one valid JSON object. It must contain exactly the keys supplied in
the requested schema. Each value must be a plain string. Use "\\n\\n" between
paragraphs. No markdown fences.
"""


REVIEW_PROMPT = """You are the final factual and editorial verifier for Flag Intelligence.

You receive:
1. the country identity and optional locally collected evidence;
2. a drafted report with fixed section keys.

Your task is to independently fact-check the entire report using your knowledge and
return a corrected report that is publishable. Local evidence is only supporting
context and may be incomplete, stale or malformed.

STRICT VERIFICATION RULES
- Check EVERY factual assertion in the draft using your knowledge and the supplied
  country context.
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


def _sanitize_report_text(value: str) -> str:
    """Repair harmless editorial artefacts before the quality gate."""
    text = str(value or "")
    text = text.replace(" | ", " · ").replace("|", " · ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_output(payload: Any) -> dict[str, str]:
    if not isinstance(payload, dict):
        return {}
    result: dict[str, str] = {}
    for key in REPORT_SECTION_KEYS:
        value = payload.get(key, "")
        if not isinstance(value, str):
            value = ""
        result[key] = _sanitize_report_text(value)
    return result


def _cross_section_duplicate_issues(report: dict[str, str]) -> list[str]:
    """Reject reports that recycle the same prose across different sections."""
    issues: list[str] = []
    normalized: dict[str, str] = {}
    for key in REPORT_SECTION_KEYS:
        text = str(report.get(key, "") or "").strip()
        if not text:
            continue
        compact = re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()
        if len(compact) < 120:
            continue
        for previous_key, previous_text in normalized.items():
            shorter = min(len(compact), len(previous_text))
            if shorter < 120:
                continue
            # Exact/near-prefix reuse catches the stitched fallback pattern.
            common = 0
            for a, b in zip(compact, previous_text):
                if a != b:
                    break
                common += 1
            if common / shorter >= 0.72:
                issues.append(
                    f"{key}: excessive prose reuse from {previous_key}"
                )
        normalized[key] = compact
    return list(dict.fromkeys(issues))


def _report_completeness(draft: dict[str, str]) -> tuple[int, list[str]]:
    """Return substantial-section count and missing core sections."""
    substantial = [
        key for key in REPORT_SECTION_KEYS
        if str(draft.get(key, "") or "").strip()
    ]
    core_sections = (
        "introduction",
        "physical_geography",
        "climate_water_resources",
        "seasons_climate_calendar",
        "flag_design_symbolism",
        "origins_early_history",
        "historical_journey",
        "key_historical_timeline",
        "state_formation_identity",
        "government_structure",
        "legal_constitutional_system",
        "people_society",
        "demographics_population_structure",
        "culture_cuisine_music_sport",
        "heritage_landmarks",
        "major_cities_regional_profiles",
        "economy_trade_industries",
        "infrastructure_transport_energy",
        "education_research",
        "universities_higher_education",
        "science_discovery_invention",
        "practical_emergency",
        "international_relations",
        "conclusion",
    )
    missing_core = [
        key for key in core_sections
        if not str(draft.get(key, "") or "").strip()
    ]
    return len(substantial), missing_core


def _report_json_schema(keys: tuple[str, ...]) -> dict[str, object]:
    """Strict Responses API schema for authored report sections."""
    return {
        "type": "object",
        "properties": {
            key: {"type": "string"}
            for key in keys
        },
        "required": list(keys),
        "additionalProperties": False,
    }


def _structured_text_format(
    keys: tuple[str, ...],
    *,
    name: str,
) -> dict[str, object]:
    return {
        "format": {
            "type": "json_schema",
            "name": name,
            "strict": True,
            "schema": _report_json_schema(keys),
        }
    }


def _parse_writer_response(raw: str) -> dict[str, str]:
    raw = _strip_code_fence(raw or "")
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
    return _normalize_output(parsed)


SECTION_RECOVERY_GROUPS = (
    (
        "geography_climate",
        (
            "introduction",
            "physical_geography",
            "climate_water_resources",
            "seasons_climate_calendar",
            "major_cities_regional_profiles",
            "environment_biodiversity",
        ),
    ),
    (
        "flag_history",
        (
            "flag_design_symbolism",
            "origins_early_history",
            "historical_journey",
            "key_historical_timeline",
            "state_formation_identity",
            "national_symbols_identity",
        ),
    ),
    (
        "institutions",
        (
            "government_structure",
            "legal_constitutional_system",
            "leadership_through_time",
            "international_relations",
        ),
    ),
    (
        "society_culture",
        (
            "people_society",
            "demographics_population_structure",
            "languages_religion",
            "health_public_health",
            "culture_cuisine_music_sport",
            "festivals_holidays_traditions",
            "heritage_landmarks",
            "literature_philosophy_thought",
        ),
    ),
    (
        "economy_education_science",
        (
            "economy_trade_industries",
            "infrastructure_transport_energy",
            "education_research",
            "universities_higher_education",
            "science_discovery_invention",
            "cost_of_living",
            "practical_emergency",
        ),
    ),
    (
        "notable_figures_intellectual_scientific",
        (
            "notable_figures_philosophy",
            "notable_figures_literature_poetry",
            "notable_figures_mathematics",
            "notable_figures_physics",
            "notable_figures_science_medicine",
            "notable_figures_invention_engineering",
        ),
    ),
    (
        "notable_figures_cultural_public",
        (
            "notable_figures_arts_architecture",
            "notable_figures_music_cinema",
            "notable_figures_public_life",
            "notable_figures_sport",
            "notable_public_figures",
        ),
    ),
    (
        "conclusion",
        (
            "conclusion",
        ),
    ),
)


def _model_candidates(configured_model: str) -> tuple[str, ...]:
    """Ordered writer models, reduced to the local model in Ollama mode."""
    if llm_backend_name() == "ollama":
        return (llm_model_name(),)

    candidates = [
        configured_model.strip(),
        "gpt-5.6-sol",
        "gpt-5.6",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.2",
        "gpt-5",
    ]
    return tuple(dict.fromkeys(model for model in candidates if model))

def _generate_section_group(
    *,
    api_key: str,
    model: str,
    evidence_json: str,
    group_name: str,
    keys: tuple[str, ...],
    existing: dict[str, str],
    deadline: float,
) -> dict[str, str]:
    """Generate one small thematic report block with retry and model failover."""
    requested = {
        key: ""
        for key in keys
        if not str(existing.get(key, "") or "").strip()
    }
    if not requested:
        return {}

    prompt = (
        "Generate ONLY the requested Flag Intelligence sections for this country. "
        "Return one JSON object containing exactly the requested keys. WRITE each "
        "section as original, coherent, publication-ready prose for that section's "
        "specific subject. Do not copy evidence fragments verbatim, do not repeat the "
        "same paragraph under different keys, do not preserve source labels such as "
        "'Politics:', 'Art:', 'Economy:', or MediaWiki headings such as '=== ... ==='. "
        "Use the evidence only as factual support and rewrite it completely. "
        "Different keys must contain meaningfully different content. The COUNTRY "
        "EVIDENCE may contain a user_request: treat it as the report brief. Prioritize "
        "and deepen the topics the user explicitly requested, keep unrelated material "
        "concise, and do not ignore requested angles. Do not output markdown, citations, "
        "URLs or source names.\n\n"
        f"THEMATIC BLOCK: {group_name}\n"
        "REQUESTED KEYS:\n"
        + json.dumps(requested, ensure_ascii=False)
        + "\n\nCOUNTRY EVIDENCE:\n"
        + evidence_json
    )

    best: dict[str, str] = {}
    for candidate_model in _model_candidates(model):
        for _attempt in range(3):
            timeout = _remaining_budget(deadline, cap=140.0)
            if timeout <= 0:
                return best
            client = OpenAI(
                api_key=api_key,
                timeout=timeout,
                max_retries=0,
            )
            try:
                response = client.responses.create(
                    model=candidate_model,
                    reasoning={"effort": "low"},
                    input=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    text=_structured_text_format(
                        tuple(requested.keys()),
                        name="flag_intelligence_report_block",
                    ),
                    max_output_tokens=16000,
                )
                parsed = _parse_writer_response(response.output_text or "")
                block = {
                    key: parsed.get(key, "")
                    for key in requested
                    if str(parsed.get(key, "") or "").strip()
                }
                if len(block) > len(best):
                    best = block
                if len(best) == len(requested):
                    return best
            except Exception as exc:
                print(
                    f"[Flag Intelligence dialogue API] model={candidate_model} "
                    f"plain_error={_api_error_detail(exc)}"
                )
                continue
    return best


def _generate_single_section(
    *,
    api_key: str,
    model: str,
    evidence_json: str,
    key: str,
    deadline: float,
) -> str:
    """Write one section independently as a last-resort editorial pass."""
    prompt = (
        "Write ONLY the Flag Intelligence section identified below. "
        "Return one JSON object with exactly that single key. "
        "Write original, coherent, publication-ready prose specific to the section. "
        "Use the supplied country evidence only as factual support. "
        "Do not copy evidence fragments verbatim, do not include source labels, "
        "MediaWiki headings, citations, URLs, markdown, or boilerplate. "
        "If the section concerns notable figures, include several relevant people "
        "with compact mini-biographies where reliable. "
        "If the section concerns universities, name and explain historically important "
        "and currently prominent institutions where reliable. The COUNTRY EVIDENCE may "
        "contain a user_request; align depth and emphasis with that brief.\n\n"
        f"SECTION KEY: {key}\n"
        "REQUIRED JSON:\n"
        + json.dumps({key: ""}, ensure_ascii=False)
        + "\n\nCOUNTRY EVIDENCE:\n"
        + evidence_json
    )

    for candidate_model in _model_candidates(model):
        for _attempt in range(4):
            timeout = _remaining_budget(deadline, cap=100.0)
            if timeout <= 0:
                return ""
            client = OpenAI(
                api_key=api_key,
                timeout=timeout,
                max_retries=0,
            )
            try:
                response = client.responses.create(
                    model=candidate_model,
                    reasoning={"effort": "low"},
                    input=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    text=_structured_text_format(
                        (key,),
                        name="flag_intelligence_single_section",
                    ),
                    max_output_tokens=4000,
                )
                parsed = _parse_writer_response(response.output_text or "")
                value = str(parsed.get(key, "") or "").strip()
                if value:
                    return value
            except Exception as exc:
                print(
                    f"[Flag Intelligence dialogue API] model={candidate_model} "
                    f"plain_error={_api_error_detail(exc)}"
                )
                continue
    return ""


def _recover_report_in_chunks(
    *,
    api_key: str,
    model: str,
    evidence_json: str,
    draft: dict[str, str],
    deadline: float,
) -> dict[str, str]:
    """Repair an incomplete report through smaller parallel thematic calls."""
    merged = {key: str(draft.get(key, "") or "") for key in REPORT_SECTION_KEYS}

    jobs = []
    for group_name, keys in SECTION_RECOVERY_GROUPS:
        missing = [
            key for key in keys
            if not str(merged.get(key, "") or "").strip()
        ]
        if missing:
            jobs.append((group_name, keys))

    if not jobs:
        return merged

    # Small independent calls are much less timeout-prone than one very long
    # response. Reserve part of the global budget for targeted rescue.
    group_phase_deadline = min(
        deadline - 60.0,
        time.monotonic() + 220.0,
    )
    if group_phase_deadline <= time.monotonic() + 5.0:
        group_phase_deadline = min(deadline - 4.0, time.monotonic() + 100.0)

    executor = ThreadPoolExecutor(max_workers=min(2, len(jobs)))
    futures = {
        executor.submit(
            _generate_section_group,
            api_key=api_key,
            model=model,
            evidence_json=evidence_json,
            group_name=group_name,
            keys=keys,
            existing=merged,
            deadline=group_phase_deadline,
        ): group_name
        for group_name, keys in jobs
    }
    try:
        completed = as_completed(
            futures,
            timeout=max(1.0, group_phase_deadline - time.monotonic()),
        )
        for future in completed:
            try:
                block = future.result()
            except Exception:
                block = {}
            for key, value in block.items():
                if value and not str(merged.get(key, "") or "").strip():
                    merged[key] = value
    except TimeoutError:
        pass
    finally:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)

    # Final targeted rescue: retry only the groups that still contain missing
    # core content. This keeps failure isolation narrow and prevents one bad
    # thematic call from invalidating the whole report.
    _count, missing_core = _report_completeness(merged)
    if missing_core:
        missing_set = set(missing_core)
        for group_name, keys in SECTION_RECOVERY_GROUPS:
            if not missing_set.intersection(keys):
                continue
            block = _generate_section_group(
                api_key=api_key,
                model=model,
                evidence_json=evidence_json,
                group_name=f"{group_name}_final_rescue",
                keys=keys,
                existing=merged,
                deadline=deadline,
            )
            for key, value in block.items():
                if value and not str(merged.get(key, "") or "").strip():
                    merged[key] = value
            _count, missing_core = _report_completeness(merged)
            missing_set = set(missing_core)
            if not missing_core:
                break

    # Final editorial completion: write every still-empty section
    # independently. This keeps the final report authored rather than stitched.
    missing_keys = [
        key for key in REPORT_SECTION_KEYS
        if not str(merged.get(key, "") or "").strip()
    ]
    if missing_keys and deadline - time.monotonic() > 8.0:
        executor = ThreadPoolExecutor(max_workers=min(5, len(missing_keys)))
        futures = {
            executor.submit(
                _generate_single_section,
                api_key=api_key,
                model=model,
                evidence_json=evidence_json,
                key=key,
                deadline=deadline,
            ): key
            for key in missing_keys
        }
        try:
            completed = as_completed(
                futures,
                timeout=max(1.0, deadline - time.monotonic()),
            )
            for future in completed:
                key = futures[future]
                try:
                    value = future.result()
                except Exception:
                    value = ""
                if value:
                    merged[key] = value
        except TimeoutError:
            pass
        finally:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)


    return merged


def _repair_quality_issues(
    *,
    api_key: str,
    model: str,
    evidence_json: str,
    draft: dict[str, str],
    issues: list[str],
    deadline: float,
) -> dict[str, str]:
    """Rewrite sections implicated by deterministic QA while budget remains."""
    keys = []
    for issue in issues:
        key = issue.split(":", 1)[0].strip()
        if key in REPORT_SECTION_KEYS and key not in keys:
            keys.append(key)

    if not keys or deadline - time.monotonic() <= 8.0:
        return draft

    repaired = dict(draft)
    executor = ThreadPoolExecutor(max_workers=min(4, len(keys)))
    futures = {
        executor.submit(
            _generate_single_section,
            api_key=api_key,
            model=model,
            evidence_json=evidence_json,
            key=key,
            deadline=deadline,
        ): key
        for key in keys
    }
    try:
        completed = as_completed(
            futures,
            timeout=max(1.0, deadline - time.monotonic()),
        )
        for future in completed:
            key = futures[future]
            try:
                value = future.result()
            except Exception:
                value = ""
            if value:
                repaired[key] = value
    except TimeoutError:
        pass
    finally:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)

    return repaired


def _emergency_full_report_pass(
    *,
    api_key: str,
    model: str,
    country_name: str,
    deadline: float,
) -> dict[str, str]:
    """Last model-authored pass reserved for the end of the five-minute budget."""
    timeout = _remaining_budget(deadline, cap=40.0)
    if timeout <= 0:
        return {}

    schema = {key: "" for key in REPORT_SECTION_KEYS}
    prompt = (
        f"Write a complete, concise Flag Intelligence report for {country_name}. "
        "Populate every required section with distinct, factual, publication-ready "
        "English prose. Keep each section compact enough to finish reliably. "
        "Do not paste source fragments, do not repeat paragraphs, and do not include "
        "citations, URLs, markdown, source labels, or MediaWiki syntax. "
        "For notable-figure categories, include representative people where reliable. "
        "Return exactly the required JSON object.\n\n"
        "REQUIRED KEYS:\n"
        + json.dumps(schema, ensure_ascii=False)
    )

    for candidate_model in _model_candidates(model):
        timeout = _remaining_budget(deadline, cap=40.0)
        if timeout <= 0:
            break
        client = OpenAI(
            api_key=api_key,
            timeout=timeout,
            max_retries=0,
        )
        try:
            response = client.responses.create(
                model=candidate_model,
                reasoning={"effort": "low"},
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                text=_structured_text_format(
                    REPORT_SECTION_KEYS,
                    name="flag_intelligence_emergency_report",
                ),
                max_output_tokens=24000,
            )
            parsed = _parse_writer_response(response.output_text or "")
            if parsed:
                return parsed
        except Exception:
            continue
    return {}


def _api_error_detail(exc: Exception) -> str:
    return (
        f"{type(exc).__name__}"
        f" status={getattr(exc, 'status_code', None)!r}"
        f" code={getattr(exc, 'code', None)!r}"
        f" body={getattr(exc, 'body', None)!r}"
        f" message={str(exc)[:500]}"
    )


def probe_openai_api() -> tuple[bool, str]:
    """Backward-compatible probe name; checks the configured LLM backend."""
    return probe_llm_backend()


def _conversation_model_candidates(configured_model: str) -> tuple[str, ...]:
    """Ordered failover models for low-latency dialogue turns."""
    if llm_backend_name() == "ollama":
        return (llm_model_name(),)

    candidates = [
        configured_model.strip(),
        "gpt-5.6-sol",
        "gpt-5.6",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.2",
        "gpt-5",
    ]
    return tuple(dict.fromkeys(model for model in candidates if model))


def interpret_country_request(message: str) -> dict[str, str]:
    """Interpret the user's opening message with one real-time LLM call."""
    text = str(message or "").strip()
    if not text:
        raise RuntimeError("LLM dialogue request failed: empty user message")

    api_key = llm_auth_token()
    if not api_key:
        raise RuntimeError("LLM dialogue request failed: LLM backend unavailable")

    schema = {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": [
                    "greeting",
                    "country_only",
                    "country_request",
                    "general",
                    "unknown",
                ],
            },
            "country": {"type": "string"},
            "request": {"type": "string"},
            "reply": {"type": "string"},
        },
        "required": ["intent", "country", "request", "reply"],
        "additionalProperties": False,
    }
    prompt = (
        "You are Flag Intelligence, a fast conversational country assistant. "
        "Respond naturally to the user's message and interpret it at the same time. "
        "Return compact JSON only. If a country is named, use its canonical English "
        "name. If the user names only a country, ask naturally what they want to know. "
        "If they ask about a country topic, preserve that scope in request. "
        "For greetings or general questions, reply naturally and briefly. "
        "Do not use canned wording and do not invent a country.\n\n"
        f"USER MESSAGE: {text}"
    )

    client = OpenAI(
        api_key=api_key,
        timeout=float(os.getenv("FLAG_INTELLIGENCE_DIALOGUE_TIMEOUT", "12")),
        max_retries=0,
    )
    try:
        response = client.responses.create(
            model=llm_model_name(),
            reasoning={"effort": "low"},
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "flag_intelligence_dialogue_turn",
                    "strict": True,
                    "schema": schema,
                }
            },
            max_output_tokens=220,
        )
        payload = json.loads(_strip_code_fence(response.output_text or ""))
        if not isinstance(payload, dict):
            raise ValueError("dialogue output is not a JSON object")
        result = {
            "intent": str(payload.get("intent") or "unknown").strip(),
            "country": str(payload.get("country") or "").strip(),
            "request": str(payload.get("request") or "").strip(),
            "reply": str(payload.get("reply") or "").strip(),
            "_source": "model",
        }
        if not result["reply"]:
            raise ValueError("dialogue output contains no reply")
        return result
    except Exception as exc:
        raise RuntimeError(
            "LLM dialogue request failed: " + _api_error_detail(exc)
        ) from exc


def continue_report_conversation(
    country_name: str,
    existing_request: str,
    latest_message: str,
    turn_number: int = 1,
) -> dict[str, str]:
    """Handle one clarification turn with one real-time LLM call."""
    api_key = llm_auth_token()
    if not api_key:
        raise RuntimeError("LLM dialogue request failed: LLM backend unavailable")

    country = str(country_name or "").strip()
    latest = str(latest_message or "").strip()
    existing = str(existing_request or "").strip()

    schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["ask", "generate"],
            },
            "normalized_request": {"type": "string"},
            "reply": {"type": "string"},
        },
        "required": ["action", "normalized_request", "reply"],
        "additionalProperties": False,
    }
    prompt = (
        "You are Flag Intelligence in a live conversation before generating a report. "
        f"The selected country is {country}. "
        f"Previously understood request: {existing!r}. "
        f"Latest user message: {latest!r}. "
        f"Clarification turn: {int(turn_number)}. "
        "Understand the user's meaning naturally. Merge the latest instruction into "
        "normalized_request. Ask one short clarification only if it materially improves "
        "the requested report; otherwise set action='generate'. After two clarification "
        "turns, prefer generate unless the request is genuinely ambiguous. "
        "Your reply must be natural, concise, and freshly written for this turn. "
        "Return compact JSON only."
    )

    client = OpenAI(
        api_key=api_key,
        timeout=float(os.getenv("FLAG_INTELLIGENCE_DIALOGUE_TIMEOUT", "12")),
        max_retries=0,
    )
    try:
        response = client.responses.create(
            model=llm_model_name(),
            reasoning={"effort": "low"},
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "flag_intelligence_clarification_turn",
                    "strict": True,
                    "schema": schema,
                }
            },
            max_output_tokens=260,
        )
        payload = json.loads(_strip_code_fence(response.output_text or ""))
        if not isinstance(payload, dict):
            raise ValueError("clarification output is not a JSON object")
        action = str(payload.get("action") or "").strip()
        normalized_request = str(
            payload.get("normalized_request") or ""
        ).strip()
        reply = str(payload.get("reply") or "").strip()
        if action not in {"ask", "generate"} or not reply:
            raise ValueError("invalid clarification output")
        return {
            "action": action,
            "normalized_request": normalized_request,
            "reply": reply,
        }
    except Exception as exc:
        raise RuntimeError(
            "LLM dialogue request failed: " + _api_error_detail(exc)
        ) from exc

def _generate_scoped_report(
    *,
    api_key: str,
    model: str,
    country_name: str,
    country_code: str,
    user_request: str,
    section_keys: tuple[str, ...],
) -> dict[str, str]:
    """Generate a narrow report in one bounded call instead of the full report pipeline."""
    prompt = (
        f"Write a focused Flag Intelligence report about {country_name}. "
        f"The user request is: {user_request!r}. "
        "Respect that scope strictly. Do not add unrelated country chapters. "
        "Return exactly the requested JSON keys with concise, factual, "
        "publication-ready English prose. For history, use a clear chronology "
        "from early origins through major turning points to the modern state. "
        "Do not include markdown, citations, URLs, or source labels.\n\n"
        "REQUESTED KEYS:\n"
        + json.dumps({key: "" for key in section_keys}, ensure_ascii=False)
        + f"\n\nCOUNTRY CODE: {country_code}"
    )
    client = OpenAI(
        api_key=api_key,
        timeout=float(os.getenv("FLAG_INTELLIGENCE_SCOPED_REPORT_TIMEOUT", "45")),
        max_retries=0,
    )
    try:
        response = client.responses.create(
            model=model,
            reasoning={"effort": "low"},
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            text=_structured_text_format(
                section_keys,
                name="flag_intelligence_scoped_report",
            ),
            max_output_tokens=4000,
        )
        parsed = _parse_writer_response(response.output_text or "")
    except Exception as exc:
        raise RuntimeError(
            "LLM scoped report generation failed: "
            + _api_error_detail(exc)
        ) from exc

    result = {
        key: str(parsed.get(key, "") or "").strip()
        for key in section_keys
        if str(parsed.get(key, "") or "").strip()
    }
    if result:
        result["__qa_passed"] = True
        result["__qa_issues"] = []
        result["__substantial_sections"] = len(result)
        result["__generation_mode"] = "scoped_single_pass"
        result["__scope_sections"] = section_keys
    return result


def generate_authored_report(report: dict[str, Any]) -> dict[str, str]:
    """Author the complete report through independent model-written chapters."""
    api_key = llm_auth_token()
    if not api_key:
        raise RuntimeError("LLM report generation failed: LLM backend unavailable")

    model = llm_model_name()
    country_name = str(
        report.get("decision")
        or report.get("country")
        or report.get("country_code")
        or ""
    ).strip()
    if not country_name:
        raise RuntimeError("LLM report generation failed: missing country")

    requested_sections = _requested_report_sections(
        str(report.get("user_request") or "")
    )
    if requested_sections != REPORT_SECTION_KEYS:
        return _generate_scoped_report(
            api_key=api_key,
            model=model,
            country_name=country_name,
            country_code=str(report.get("country_code") or ""),
            user_request=str(report.get("user_request") or ""),
            section_keys=requested_sections,
        )

    deadline = time.monotonic() + REPORT_GENERATION_BUDGET_SECONDS
    recovery_deadline = deadline - 45.0
    evidence_json = json.dumps(
        {
            "decision": country_name,
            "country_code": report.get("country_code"),
            "input_mode": report.get("input_mode"),
            "user_request": report.get("user_request"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )

    # Primary path: every chapter is authored directly by the model.
    # No local prose or encyclopedia fragments are inserted into the report.
    draft = _recover_report_in_chunks(
        api_key=api_key,
        model=model,
        evidence_json=evidence_json,
        draft={},
        deadline=recovery_deadline,
    )

    substantial, missing_core = _report_completeness(draft)
    if (
        not draft
        or substantial < 30
        or missing_core
    ) and deadline - time.monotonic() > 4.0:
        emergency = _emergency_full_report_pass(
            api_key=api_key,
            model=model,
            country_name=country_name,
            deadline=deadline,
        )
        if emergency:
            for key in REPORT_SECTION_KEYS:
                value = str(emergency.get(key, "") or "").strip()
                if value:
                    draft[key] = value

    if not draft:
        raise RuntimeError("LLM report generation failed: no report content returned")

    # If QA detects duplicated or malformed prose, rewrite only those sections.
    mechanical = _deterministic_quality_issues(draft)
    duplicate_issues = _cross_section_duplicate_issues(draft)
    mechanical = list(dict.fromkeys(mechanical + duplicate_issues))

    if mechanical and deadline - time.monotonic() > 8.0:
        draft = _repair_quality_issues(
            api_key=api_key,
            model=model,
            evidence_json=evidence_json,
            draft=draft,
            issues=mechanical,
            deadline=deadline,
        )
        mechanical = _deterministic_quality_issues(draft)
        duplicate_issues = _cross_section_duplicate_issues(draft)
        mechanical = list(dict.fromkeys(mechanical + duplicate_issues))

    substantial, missing_core = _report_completeness(draft)

    # Complete any remaining core section individually while budget remains.
    if missing_core and deadline - time.monotonic() > 8.0:
        for key in list(missing_core):
            if deadline - time.monotonic() <= 8.0:
                break
            value = _generate_single_section(
                api_key=api_key,
                model=model,
                evidence_json=evidence_json,
                key=key,
                deadline=deadline,
            )
            if value:
                draft[key] = value
        substantial, missing_core = _report_completeness(draft)

    passed = (
        substantial >= 30
        and not missing_core
        and bool(draft.get("introduction"))
        and bool(draft.get("historical_journey"))
        and bool(draft.get("universities_higher_education"))
        and bool(draft.get("conclusion"))
    )

    issues = list(mechanical)
    if substantial < 30:
        issues.append(
            f"report incomplete: only {substantial} substantive sections"
        )
    if missing_core:
        issues.append(
            "missing core sections: " + ", ".join(missing_core)
        )

    draft["__qa_passed"] = passed
    draft["__qa_issues"] = issues
    draft["__substantial_sections"] = substantial
    draft["__generation_mode"] = "authored_chapters_with_targeted_completion"
    return draft
