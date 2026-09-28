"""Grounded learning utilities for Flag Intelligence.

The first Q&A layer is intentionally retrieval-only. It answers from the
Country Intelligence payload and never invents a fact outside that record.
"""

from __future__ import annotations

import re
from typing import Any


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "de", "des", "du",
    "est", "et", "for", "from", "how", "in", "is", "la", "le", "les",
    "of", "on", "or", "que", "quel", "quelle", "the", "to", "un", "une",
    "what", "when", "where", "which", "who", "why", "with",
}


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9À-ÖØ-öø-ÿ'-]+", text.lower())
        if len(token) > 2 and token not in STOP_WORDS
    }


def _fact_rows(intelligence: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for section_name in (
        "identity",
        "geography",
        "sovereignty",
        "national_identity",
        "government",
        "people_society",
        "culture",
        "economy",
        "infrastructure",
        "education_science",
        "environment",
        "practical",
        "international_relations",
    ):
        section = intelligence.get(section_name)
        if not isinstance(section, dict):
            continue

        for key, item in section.items():
            if not isinstance(item, dict):
                continue
            value = item.get("value")
            if value in (None, "", "Not available"):
                continue
            rows.append(
                {
                    "section": section_name,
                    "label": key.replace("_", " ").title(),
                    "value": str(value),
                    "source": str(item.get("source") or ""),
                    "source_url": str(item.get("source_url") or ""),
                    "reference_year": str(item.get("reference_year") or ""),
                }
            )

    for collection_name in ("origins", "historical_timeline"):
        collection = intelligence.get(collection_name)
        if not isinstance(collection, list):
            continue
        for item in collection:
            if not isinstance(item, dict):
                continue
            summary = item.get("summary")
            if not summary:
                continue
            sources = item.get("sources") or []
            urls = item.get("source_urls") or []
            rows.append(
                {
                    "section": collection_name,
                    "label": str(item.get("label") or item.get("period") or "History"),
                    "value": str(summary),
                    "source": ", ".join(str(value) for value in sources),
                    "source_url": str(urls[0]) if urls else "",
                    "reference_year": str(item.get("period") or ""),
                }
            )

    flag = intelligence.get("flag")
    if isinstance(flag, dict):
        for field, label in (
            ("design_origin", "Flag Design"),
            ("symbolism", "Flag Symbolism"),
            ("historical_flags", "Flag History"),
        ):
            items = flag.get(field)
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                value = item.get("value") or item.get("summary")
                if not value:
                    continue
                source = item.get("source")
                if not source:
                    sources = item.get("sources") or []
                    source = ", ".join(str(x) for x in sources)
                source_url = item.get("source_url")
                if not source_url:
                    urls = item.get("source_urls") or []
                    source_url = urls[0] if urls else ""
                rows.append(
                    {
                        "section": "flag",
                        "label": label,
                        "value": str(value),
                        "source": str(source or ""),
                        "source_url": str(source_url or ""),
                        "reference_year": str(item.get("period") or ""),
                    }
                )

    emergency = intelligence.get("emergency")
    if isinstance(emergency, list):
        for item in emergency:
            if not isinstance(item, dict):
                continue
            service = str(item.get("service") or "Emergency")
            number = str(item.get("number") or "")
            if number:
                rows.append(
                    {
                        "section": "emergency",
                        "label": service,
                        "value": number,
                        "source": str(item.get("source") or ""),
                        "source_url": str(item.get("source_url") or ""),
                        "reference_year": "",
                    }
                )

    return rows


def answer_country_question(
    question: str,
    intelligence: dict[str, Any],
    *,
    max_results: int = 3,
) -> list[dict[str, str]]:
    """Return the best grounded passages for a natural-language question."""
    query_tokens = _tokens(question)
    if not query_tokens:
        return []

    scored: list[tuple[float, dict[str, str]]] = []
    for row in _fact_rows(intelligence):
        haystack = " ".join(
            (row["section"], row["label"], row["value"])
        )
        fact_tokens = _tokens(haystack)
        overlap = query_tokens & fact_tokens
        if not overlap:
            continue

        score = len(overlap) / max(len(query_tokens), 1)
        label_tokens = _tokens(row["label"])
        score += 0.5 * len(query_tokens & label_tokens)

        # High-value intent hints improve simple questions such as
        # "capital?", "president?", "emergency number?".
        lowered = question.lower()
        if "capital" in lowered and "capital" in row["label"].lower():
            score += 2.0
        if (
            ("president" in lowered or "head of state" in lowered)
            and "state" in row["label"].lower()
        ):
            score += 2.0
        if (
            ("emergency" in lowered or "urgence" in lowered)
            and row["section"] == "emergency"
        ):
            score += 2.0
        if (
            ("history" in lowered or "histoire" in lowered)
            and row["section"] in {"origins", "historical_timeline"}
        ):
            score += 1.0
        if (
            ("flag" in lowered or "drapeau" in lowered)
            and row["section"] == "flag"
        ):
            score += 1.5

        scored.append((score, row))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [row for score, row in scored[:max_results] if score > 0]
