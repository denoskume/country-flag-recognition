"""Generic encyclopedic enrichment for Country Intelligence V2.

Structured sources remain authoritative for compact facts. This module adds
broad educational context from a country encyclopedia article without
country-specific code. Missing sections remain missing.
"""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Iterable
from urllib.parse import quote

import requests

from .country_intelligence import (
    CountryIntelligence,
    Evidence,
    PARTIAL,
    TimelineEvent,
    evidence,
)

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_PAGE = "https://en.wikipedia.org/wiki/"
USER_AGENT = (
    "country-flag-recognition/0.2 "
    "(Flag Intelligence educational portfolio project)"
)

SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "history": (
        "history", "prehistory", "early history", "ancient history",
        "medieval history", "modern history", "colonial history",
        "colonial period", "independence",
    ),
    "geography": ("geography", "climate", "biodiversity", "environment"),
    "people_society": (
        "demographics", "population", "ethnic groups", "languages",
        "religion", "society", "health",
    ),
    "culture": (
        "culture", "arts", "music", "literature", "cuisine",
        "sport", "sports", "media", "festivals",
    ),
    "economy": ("economy", "agriculture", "industry", "trade", "tourism"),
    "infrastructure": (
        "infrastructure", "transport", "transportation", "energy",
        "communications",
    ),
    "education_science": (
        "education", "science and technology", "science", "technology",
        "research",
    ),
    "government": ("government", "politics", "law", "administrative divisions"),
    "international_relations": ("foreign relations", "international relations"),
}

EARLY_HISTORY_HEADINGS = (
    "prehistory", "early history", "ancient history", "pre-colonial",
    "precolonial", "origins", "antiquity",
)


def _clean_heading(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip().lower()
    return re.sub(r"[^a-z0-9 -]", "", value)


def fetch_country_article(
    title: str,
    timeout: float = 15.0,
) -> tuple[str, str]:
    """Fetch a full plaintext encyclopedia article and canonical title."""
    response = requests.get(
        WIKIPEDIA_API,
        params={
            "action": "query",
            "prop": "extracts|info",
            "explaintext": 1,
            "redirects": 1,
            "inprop": "url",
            "titles": title,
            "format": "json",
            "formatversion": 2,
        },
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=timeout,
    )
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        raise LookupError(f"No encyclopedia article found for {title!r}")
    page = pages[0]
    extract = str(page.get("extract") or "").strip()
    canonical_title = str(page.get("title") or title).strip()
    if not extract:
        raise LookupError(f"Empty encyclopedia article for {title!r}")
    return extract, canonical_title


def split_article_sections(text: str) -> list[tuple[str, str]]:
    """Split MediaWiki plaintext into ordered heading/body pairs."""
    heading_re = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.MULTILINE)
    matches = list(heading_re.finditer(text))
    if not matches:
        return [("overview", text.strip())] if text.strip() else []

    sections: list[tuple[str, str]] = []
    lead = text[: matches[0].start()].strip()
    if lead:
        sections.append(("overview", lead))

    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        heading = match.group(2).strip()
        body = text[start:end].strip()
        if body:
            sections.append((heading, body))
    return sections


def _matches_alias(heading: str, aliases: Iterable[str]) -> bool:
    normalized = _clean_heading(heading)
    for alias in aliases:
        candidate = _clean_heading(alias)
        if normalized == candidate or normalized.startswith(candidate + " "):
            return True
        if candidate in normalized and len(candidate) >= 7:
            return True
    return False


def collect_domain_text(
    sections: list[tuple[str, str]],
    domain: str,
    *,
    max_chars: int = 4200,
) -> str:
    """Collect matching section text for one knowledge domain."""
    aliases = SECTION_ALIASES.get(domain, ())
    blocks: list[str] = []
    for heading, body in sections:
        if _matches_alias(heading, aliases):
            compact = re.sub(r"\n{3,}", "\n\n", body).strip()
            if compact:
                blocks.append(f"{heading}: {compact}")
    combined = "\n\n".join(blocks)
    if len(combined) <= max_chars:
        return combined
    clipped = combined[:max_chars].rsplit(" ", 1)[0].rstrip()
    return clipped + "…"


def collect_origins(
    sections: list[tuple[str, str]],
    source_url: str,
) -> tuple[TimelineEvent, ...]:
    """Capture explicitly early/origin sections without inventing eras."""
    events: list[TimelineEvent] = []
    for heading, body in sections:
        normalized = _clean_heading(heading)
        if not any(alias in normalized for alias in EARLY_HISTORY_HEADINGS):
            continue
        summary = " ".join(body.split())
        if len(summary) > 1200:
            summary = summary[:1197].rsplit(" ", 1)[0] + "…"
        events.append(
            TimelineEvent(
                label=heading,
                period=heading,
                summary=summary,
                sources=("Wikipedia",),
                source_urls=(source_url,),
                confidence=0.75,
            )
        )
        if len(events) >= 6:
            break
    return tuple(events)


_DATE_TOKEN = re.compile(r"(?<!\d)(?P<year>(?:1[0-9]{3}|20[0-9]{2}))(?!\d)")
_HISTORY_KEYWORDS = (
    "independ", "colon", "kingdom", "empire", "republic", "constitution",
    "president", "war", "coup", "election", "annex", "occupation",
    "federation", "state", "sovereign", "settlement", "founded", "established",
)


def extract_timeline(
    history_text: str,
    source_url: str,
    *,
    max_events: int = 18,
) -> tuple[TimelineEvent, ...]:
    """Extract explicit dated historical statements conservatively."""
    if not history_text:
        return ()

    candidates: list[tuple[int, str]] = []
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(history_text.split()))
    for sentence in sentences:
        year_match = _DATE_TOKEN.search(sentence)
        if not year_match:
            continue
        lower = sentence.lower()
        if not any(keyword in lower for keyword in _HISTORY_KEYWORDS):
            continue
        sentence = sentence.strip()
        if len(sentence) < 35:
            continue
        if len(sentence) > 700:
            sentence = sentence[:697].rsplit(" ", 1)[0] + "…"
        candidates.append((int(year_match.group("year")), sentence))

    unique: dict[tuple[int, str], tuple[int, str]] = {}
    for year, sentence in candidates:
        key = (year, re.sub(r"\W+", " ", sentence.lower())[:120])
        unique.setdefault(key, (year, sentence))

    ordered = sorted(unique.values(), key=lambda item: item[0])
    if len(ordered) > max_events:
        step = (len(ordered) - 1) / (max_events - 1)
        indices = sorted({round(i * step) for i in range(max_events)})
        ordered = [ordered[i] for i in indices]

    return tuple(
        TimelineEvent(
            label=f"Historical event — {year}",
            period=str(year),
            summary=sentence,
            sources=("Wikipedia",),
            source_urls=(source_url,),
            confidence=0.72,
        )
        for year, sentence in ordered
    )


def _domain_evidence(value: str, source_url: str) -> Evidence | None:
    if not value:
        return None
    return evidence(
        value,
        "Wikipedia",
        retrieved_at=datetime.now(timezone.utc).date().isoformat(),
        confidence=0.75,
        status=PARTIAL,
        source_url=source_url,
    )


def enrich_from_encyclopedia(
    record: CountryIntelligence,
    *,
    title: str | None = None,
    timeout: float = 15.0,
) -> CountryIntelligence:
    """Enrich a record only with sections actually present in the article."""
    article_text, canonical_title = fetch_country_article(
        title or record.name,
        timeout=timeout,
    )
    source_url = WIKIPEDIA_PAGE + quote(
        canonical_title.replace(" ", "_"),
        safe="()_-",
    )
    sections = split_article_sections(article_text)
    history_text = collect_domain_text(sections, "history", max_chars=16000)

    origins = record.origins or collect_origins(sections, source_url)
    timeline = record.historical_timeline
    extracted_timeline = extract_timeline(history_text, source_url)
    if extracted_timeline:
        legacy_only = (
            len(timeline) == 1
            and timeline[0].label == "Legacy historical context"
        )
        if not timeline or legacy_only:
            timeline = extracted_timeline

    mappings = {
        "geography": record.geography,
        "people_society": record.people_society,
        "culture": record.culture,
        "economy": record.economy,
        "infrastructure": record.infrastructure,
        "education_science": record.education_science,
        "government": record.government,
        "international_relations": record.international_relations,
    }

    for domain, target in mappings.items():
        domain_text = collect_domain_text(sections, domain)
        item = _domain_evidence(domain_text, source_url)
        if item is not None:
            target.setdefault("context", item)

    environment_text = collect_domain_text(sections, "geography")
    environment_item = _domain_evidence(environment_text, source_url)
    if environment_item is not None:
        record.environment.setdefault("context", environment_item)

    if history_text:
        history_item = _domain_evidence(history_text[:4200], source_url)
        if history_item is not None:
            record.sovereignty.setdefault("historical_context", history_item)

    record.origins = origins
    record.historical_timeline = timeline
    return record
