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

You receive a structured evidence payload for one country. Write the final
English-language educational country report from that evidence.

NON-NEGOTIABLE RULES
- Treat supplied data as evidence, not prose to copy.
- Rewrite everything in clean, natural, professional English.
- Never copy source fragments, captions, tables, navigation text, bibliography
  residue, or malformed phrases.
- Remove duplicates, repeated dates, repeated words, and repeated ideas.
- Never produce constructions such as "for for", "in 1947 ... in 1947",
  "by 1801 ... in 1801", or duplicated sentences.
- Do not invent numerical or current political facts. If evidence conflicts,
  omit the disputed detail or state the uncertainty briefly.
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
- If evidence for a section is insufficient, return an empty string for it.

OUTPUT
Return ONLY one valid JSON object. It must contain exactly the keys supplied in
the requested schema. Each value must be a plain string. Use "\\n\\n" between
paragraphs. No markdown fences.
"""


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
        timeout=90.0,
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

    return _normalize_output(parsed)
