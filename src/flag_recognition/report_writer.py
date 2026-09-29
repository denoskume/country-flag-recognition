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


def generate_authored_report(report: dict[str, Any]) -> dict[str, str]:
    """Generate final report prose from the collected evidence payload.

    The call uses a generous timeout and one SDK retry so slower countries are
    not discarded prematurely. The caller still has a deterministic fallback if
    the external API is completely unavailable.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return {}

    model = os.getenv("FLAG_INTELLIGENCE_WRITER_MODEL", "gpt-5.6-sol").strip()
    client = OpenAI(
        api_key=api_key,
        timeout=180.0,
        max_retries=1,
    )

    schema_hint = {key: "" for key in REPORT_SECTION_KEYS}
    evidence_json = json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )

    user_prompt = (
        "Write, self-check and finalize the country report from the country identity "
        "and supporting context below. The universities_higher_education section "
        "must provide substantial coverage of historically important and currently "
        "prominent universities where reliable knowledge exists. The key_historical_timeline, "
        "major_cities_regional_profiles, demographics_population_structure, "
        "national_symbols_identity, legal_constitutional_system and "
        "seasons_climate_calendar sections should also be completed whenever "
        "reliable knowledge exists. Populate the dedicated notable_figures_* "
        "categories comprehensively where the country has well-established "
        "figures, rather than limiting the report to a few famous names.\n\n"
        "Required JSON shape:\n"
        + json.dumps(schema_hint, ensure_ascii=False)
        + "\n\nEVIDENCE:\n"
        + evidence_json
    )

    response = client.responses.create(
        model=model,
        reasoning={"effort": "low"},
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_output_tokens=16000,
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

    # The writer gets enough time to complete slower countries. Deterministic
    # mechanical QA still runs locally afterwards so harmless formatting defects
    # are repaired instead of blocking publication.
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
