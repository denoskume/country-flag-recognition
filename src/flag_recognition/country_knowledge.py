"""Generic encyclopedic enrichment for Country Intelligence.

Structured sources remain authoritative for compact facts. This module turns
encyclopedic country articles into concise educational context while preserving
source provenance. Missing sections remain missing; unsupported facts are never
invented.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    "country-flag-recognition/0.3 "
    "(Flag Intelligence educational portfolio project)"
)

SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "history": (
        "history", "prehistory", "early history", "ancient history",
        "medieval history", "modern history", "colonial history",
        "colonial period", "independence",
    ),
    "geography": (
        "geography", "physical geography", "location", "terrain",
    ),
    "climate_seasons": (
        "climate", "seasons", "weather", "precipitation", "temperature",
    ),
    "rivers_lakes": (
        "rivers", "lakes", "hydrography", "drainage", "waterways",
        "water resources",
    ),
    "mountains_relief": (
        "mountains", "mountain ranges", "topography", "relief", "terrain",
        "geology", "landforms",
    ),
    "natural_resources": (
        "natural resources", "minerals", "mining", "forestry", "energy",
        "petroleum", "oil and gas", "fisheries",
    ),
    "people_society": (
        "demographics", "population", "ethnic groups", "society",
    ),
    "languages_religion": (
        "languages", "language", "religion", "religions",
    ),
    "health_system": (
        "health", "healthcare", "health care", "public health",
    ),
    "culture": (
        "culture", "arts", "music", "literature", "cuisine",
        "sport", "sports", "media",
    ),
    "festivals_holidays": (
        "festivals", "holidays", "public holidays", "celebrations",
        "traditions",
    ),
    "heritage_landmarks": (
        "world heritage", "unesco", "heritage", "architecture",
        "landmarks", "tourist attractions", "monuments",
    ),
    "notable_people": (
        "notable people", "notable persons", "people",
    ),
    "economy": (
        "economy", "agriculture", "industry", "trade", "tourism",
    ),
    "economic_drivers": (
        "oil industry", "petroleum industry", "energy", "agriculture",
        "fisheries", "manufacturing", "industry", "services", "tourism",
        "exports", "trade", "shipping", "finance",
    ),
    "infrastructure": (
        "infrastructure", "transport", "transportation", "roads", "rail",
    ),
    "transport_network": (
        "transport", "transportation", "roads", "rail", "railways",
        "air transport", "aviation", "airports", "ports", "shipping",
    ),
    "energy_connectivity": (
        "energy", "electricity", "power", "telecommunications",
        "communications", "internet", "mobile", "broadband",
    ),
    "education_science": (
        "education", "science and technology", "science", "technology",
        "research", "innovation",
    ),
    "government": (
        "government", "politics", "law",
    ),
    "administrative_divisions": (
        "administrative divisions", "regions", "districts", "provinces",
        "departments", "counties", "municipalities",
    ),
    "international_relations": (
        "foreign relations", "international relations",
    ),
}

EARLY_HISTORY_HEADINGS = (
    "prehistory", "early history", "ancient history", "pre-colonial",
    "precolonial", "origins", "antiquity", "early states",
)

_DYNAMIC_HEADINGS = (
    "population", "demographics", "health", "media", "economy",
    "education", "science", "technology", "employment", "poverty",
)

_CURRENT_YEAR = datetime.now(timezone.utc).year


@dataclass(frozen=True)
class ArticleSection:
    heading: str
    level: int
    body: str


def _clean_heading(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip().lower()
    return re.sub(r"[^a-z0-9 -]", "", value)


def canonical_overview_text(
    value: object,
    *,
    max_chars: int = 900,
) -> str:
    """Keep stable country description while removing competing live metrics."""
    text = str(value or "").strip()
    if not text:
        return "Not available"

    sentences = _sentences(text)
    dynamic_terms = (
        "inhabitants",
        "population",
        "gdp",
        "gross domestic product",
        "head of state",
        "president",
        "prime minister",
        "unemployment",
        "inflation",
    )
    stable = [
        sentence
        for sentence in sentences
        if not any(term in sentence.lower() for term in dynamic_terms)
    ]

    selected = stable[:4] or sentences[:2]
    result = " ".join(selected).strip()
    if len(result) > max_chars:
        result = result[:max_chars].rsplit(" ", 1)[0].rstrip() + "…"
    return result or "Not available"


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


def fetch_topic_article(
    country_name: str,
    topic: str,
    *,
    timeout: float = 10.0,
) -> tuple[str, str] | None:
    """Fetch a dedicated topic article with exact-title then search fallback."""
    query = f"{topic} of {country_name}"

    try:
        return fetch_country_article(query, timeout=timeout)
    except (requests.RequestException, LookupError, ValueError):
        pass

    try:
        response = requests.get(
            WIKIPEDIA_API,
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": 5,
                "format": "json",
                "formatversion": 2,
            },
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=timeout,
        )
        response.raise_for_status()
        results = response.json().get("query", {}).get("search", [])

        preferred = query.lower()
        candidates = [
            str(item.get("title") or "").strip()
            for item in results
            if str(item.get("title") or "").strip()
        ]
        country_token = country_name.lower()
        topic_token = topic.lower()
        title = next(
            (
                candidate
                for candidate in candidates
                if candidate.lower() == preferred
            ),
            next(
                (
                    candidate
                    for candidate in candidates
                    if country_token in candidate.lower()
                    and topic_token in candidate.lower()
                ),
                None,
            ),
        )
        if not title:
            return None

        return fetch_country_article(title, timeout=timeout)
    except (requests.RequestException, LookupError, ValueError):
        return None


def split_article_sections_detailed(text: str) -> list[ArticleSection]:
    """Split plaintext while preserving MediaWiki heading depth."""
    heading_re = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.MULTILINE)
    matches = list(heading_re.finditer(text))

    if not matches:
        return [ArticleSection("overview", 1, text.strip())] if text.strip() else []

    sections: list[ArticleSection] = []
    lead = text[: matches[0].start()].strip()
    if lead:
        sections.append(ArticleSection("overview", 1, lead))

    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections.append(
            ArticleSection(
                heading=match.group(2).strip(),
                level=len(match.group(1)),
                body=body,
            )
        )
    return sections


def split_article_sections(text: str) -> list[tuple[str, str]]:
    """Backward-compatible heading/body view used by tests and helpers."""
    return [
        (section.heading, section.body)
        for section in split_article_sections_detailed(text)
        if section.body
    ]


def _matches_alias(heading: str, aliases: Iterable[str]) -> bool:
    normalized = _clean_heading(heading)
    for alias in aliases:
        candidate = _clean_heading(alias)
        if normalized == candidate or normalized.startswith(candidate + " "):
            return True
        if candidate in normalized and len(candidate) >= 7:
            return True
    return False


def _sentences(text: str) -> list[str]:
    compact = " ".join(str(text).split())
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", compact)
        if sentence.strip()
    ]


def _latest_year(text: str) -> int | None:
    years = [
        int(value)
        for value in re.findall(r"(?<!\d)(1[89]\d{2}|20\d{2})(?!\d)", text)
    ]
    return max(years) if years else None


def _compact_body(
    heading: str,
    body: str,
    *,
    max_sentences: int = 2,
    max_chars: int = 760,
) -> str:
    """Create a short extractive learning summary without generating facts."""
    sentences = _sentences(body)
    if not sentences:
        return ""

    normalized = _clean_heading(heading)
    dynamic = any(token in normalized for token in _DYNAMIC_HEADINGS)

    if dynamic:
        recent = []
        qualitative = []
        for sentence in sentences:
            year = _latest_year(sentence)
            if year is not None and year >= _CURRENT_YEAR - 8:
                recent.append((year, sentence))
            elif year is None:
                qualitative.append(sentence)

        recent.sort(key=lambda item: item[0], reverse=True)
        selected = [sentence for _, sentence in recent[:1]]
        selected.extend(qualitative[: max_sentences - len(selected)])

        # Do not present old dynamic statistics as if they were current.
        if not selected:
            return ""
    else:
        selected = sentences[:max_sentences]

    text = " ".join(selected[:max_sentences]).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].rstrip() + "…"
    return text


def _domain_records(
    sections: list[ArticleSection],
    domain: str,
) -> list[ArticleSection]:
    """Return matching sections, including children of matching parent sections."""
    aliases = SECTION_ALIASES.get(domain, ())
    selected: list[ArticleSection] = []
    active_parent_level: int | None = None

    for section in sections:
        matches = _matches_alias(section.heading, aliases)

        if active_parent_level is not None and section.level <= active_parent_level:
            active_parent_level = None

        if matches:
            selected.append(section)
            # Top-level sections such as History, Culture and Economy frequently
            # store their useful content in child headings.
            active_parent_level = section.level
            continue

        if active_parent_level is not None and section.level > active_parent_level:
            selected.append(section)

    # Preserve order and remove duplicate headings/body pairs.
    deduped: list[ArticleSection] = []
    seen: set[tuple[str, str]] = set()
    for section in selected:
        key = (section.heading, section.body)
        if key not in seen:
            seen.add(key)
            deduped.append(section)
    return deduped


def collect_domain_text(
    sections: list[tuple[str, str]],
    domain: str,
    *,
    max_chars: int = 4200,
) -> str:
    """Compatibility helper for simple heading/body lists."""
    aliases = SECTION_ALIASES.get(domain, ())
    blocks: list[str] = []
    for heading, body in sections:
        if _matches_alias(heading, aliases):
            compact = _compact_body(heading, body)
            if compact:
                blocks.append(f"{heading}: {compact}")
    combined = "\n\n".join(blocks)
    if len(combined) <= max_chars:
        return combined
    return combined[:max_chars].rsplit(" ", 1)[0].rstrip() + "…"


def collect_domain_text_detailed(
    sections: list[ArticleSection],
    domain: str,
    *,
    max_chars: int = 2600,
    max_blocks: int = 6,
) -> str:
    """Create concise subheaded learning context for one domain."""
    blocks: list[str] = []

    for section in _domain_records(sections, domain):
        compact = _compact_body(section.heading, section.body)
        if not compact:
            continue
        blocks.append(f"{section.heading}: {compact}")
        if len(blocks) >= max_blocks:
            break

    combined = "\n\n".join(blocks)
    if len(combined) <= max_chars:
        return combined
    return combined[:max_chars].rsplit(" ", 1)[0].rstrip() + "…"


def collect_origins(
    sections: list[tuple[str, str]] | list[ArticleSection],
    source_url: str,
) -> tuple[TimelineEvent, ...]:
    """Capture explicit early/origin sections without inventing eras."""
    events: list[TimelineEvent] = []

    for raw in sections:
        if isinstance(raw, ArticleSection):
            heading, body = raw.heading, raw.body
        else:
            heading, body = raw

        normalized = _clean_heading(heading)
        if not any(alias in normalized for alias in EARLY_HISTORY_HEADINGS):
            continue

        summary = _compact_body(
            heading,
            body,
            max_sentences=3,
            max_chars=1000,
        )
        if not summary:
            continue

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
    "crisis", "conflict", "peace", "referendum", "transition",
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
    sentences = _sentences(history_text)

    for sentence in sentences:
        year_match = _DATE_TOKEN.search(sentence)
        if not year_match:
            continue
        lower = sentence.lower()
        if not any(keyword in lower for keyword in _HISTORY_KEYWORDS):
            continue
        if len(sentence) < 35:
            continue
        if len(sentence) > 560:
            sentence = sentence[:557].rsplit(" ", 1)[0] + "…"
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
            label="Historical event",
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


def _set_context(
    target: dict[str, Evidence],
    key: str,
    sections: list[ArticleSection],
    domain: str,
    source_url: str,
    *,
    max_chars: int = 1800,
    max_blocks: int = 5,
) -> None:
    text = collect_domain_text_detailed(
        sections,
        domain,
        max_chars=max_chars,
        max_blocks=max_blocks,
    )
    item = _domain_evidence(text, source_url)
    if item is not None:
        target.setdefault(key, item)


def _topic_sections(
    country_name: str,
    topic: str,
    fallback_sections: list[ArticleSection],
    fallback_url: str,
    *,
    timeout: float,
) -> tuple[list[ArticleSection], str]:
    article = fetch_topic_article(
        country_name,
        topic,
        timeout=timeout,
    )
    if article is None:
        return fallback_sections, fallback_url

    text, title = article
    return (
        split_article_sections_detailed(text),
        WIKIPEDIA_PAGE + quote(
            title.replace(" ", "_"),
            safe="()_-",
        ),
    )


def enrich_from_encyclopedia(
    record: CountryIntelligence,
    *,
    title: str | None = None,
    timeout: float = 15.0,
) -> CountryIntelligence:
    """Enrich a record with concise, sourced educational context."""
    article_text, canonical_title = fetch_country_article(
        title or record.name,
        timeout=timeout,
    )
    source_url = WIKIPEDIA_PAGE + quote(
        canonical_title.replace(" ", "_"),
        safe="()_-",
    )
    detailed = split_article_sections_detailed(article_text)

    # History uses the entire History subtree instead of only the empty
    # parent heading. This is essential for country articles with subsections.
    history_sections = _domain_records(detailed, "history")
    history_text = " ".join(
        f"{section.heading}. {section.body}"
        for section in history_sections
        if section.body
    )

    origins = record.origins or collect_origins(detailed, source_url)
    timeline = record.historical_timeline
    extracted_timeline = extract_timeline(history_text, source_url)

    legacy_only = (
        len(timeline) == 1
        and timeline[0].label == "Legacy historical context"
    )
    if extracted_timeline and (not timeline or legacy_only):
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
        domain_text = collect_domain_text_detailed(detailed, domain)
        item = _domain_evidence(domain_text, source_url)
        if item is not None:
            target.setdefault("context", item)

    # Social detail: use a dedicated Demographics article when available.
    demographics_sections, demographics_url = _topic_sections(
        record.name,
        "Demographics",
        detailed,
        source_url,
        timeout=timeout,
    )
    _set_context(
        record.people_society,
        "languages_religion",
        demographics_sections,
        "languages_religion",
        demographics_url,
    )
    _set_context(
        record.people_society,
        "health_system",
        demographics_sections,
        "health_system",
        demographics_url,
    )

    # Culture detail: cuisine/music stay in the main culture context, while
    # festivals and heritage get their own learning blocks.
    culture_sections, culture_url = _topic_sections(
        record.name,
        "Culture",
        detailed,
        source_url,
        timeout=timeout,
    )
    _set_context(
        record.culture,
        "festivals_holidays",
        culture_sections,
        "festivals_holidays",
        culture_url,
    )
    _set_context(
        record.culture,
        "heritage_landmarks",
        culture_sections,
        "heritage_landmarks",
        culture_url,
    )
    _set_context(
        record.culture,
        "notable_people",
        culture_sections,
        "notable_people",
        culture_url,
        max_chars=1400,
        max_blocks=4,
    )

    # Government/territorial organization is normally present in the country
    # article and should remain distinct from institutions.
    _set_context(
        record.government,
        "administrative_divisions",
        detailed,
        "administrative_divisions",
        source_url,
    )

    # Transport article improves ports, airports, road/rail information.
    transport_sections, transport_url = _topic_sections(
        record.name,
        "Transport",
        detailed,
        source_url,
        timeout=timeout,
    )
    _set_context(
        record.infrastructure,
        "transport_network",
        transport_sections,
        "transport_network",
        transport_url,
        max_chars=2200,
        max_blocks=6,
    )
    _set_context(
        record.infrastructure,
        "energy_connectivity",
        transport_sections,
        "energy_connectivity",
        transport_url,
    )

    # Dedicated geography/economy articles usually contain the physical
    # details absent from the general country article.
    geography_article = fetch_topic_article(
        record.name,
        "Geography",
        timeout=timeout,
    )
    geography_sections = detailed
    geography_source_url = source_url
    if geography_article is not None:
        geography_text, geography_title = geography_article
        geography_sections = split_article_sections_detailed(geography_text)
        geography_source_url = WIKIPEDIA_PAGE + quote(
            geography_title.replace(" ", "_"),
            safe="()_-",
        )

    physical_domains = {
        "climate_seasons": ("climate_seasons", record.environment),
        "rivers_lakes": ("rivers_lakes", record.geography),
        "mountains_relief": ("mountains_relief", record.geography),
        "natural_resources": ("natural_resources", record.environment),
    }
    for domain, (key, target) in physical_domains.items():
        text = collect_domain_text_detailed(
            geography_sections,
            domain,
            max_chars=1800,
            max_blocks=5,
        )
        item = _domain_evidence(text, geography_source_url)
        if item is not None:
            target.setdefault(key, item)

    # Keep a concise general physical-geography context as a fallback.
    environment_text = collect_domain_text_detailed(
        geography_sections,
        "geography",
        max_chars=1600,
        max_blocks=4,
    )
    environment_item = _domain_evidence(
        environment_text,
        geography_source_url,
    )
    if environment_item is not None:
        record.environment.setdefault("context", environment_item)

    economy_article = fetch_topic_article(
        record.name,
        "Economy",
        timeout=timeout,
    )
    if economy_article is not None:
        economy_text, economy_title = economy_article
        economy_sections = split_article_sections_detailed(economy_text)
        economy_source_url = WIKIPEDIA_PAGE + quote(
            economy_title.replace(" ", "_"),
            safe="()_-",
        )
        drivers_text = collect_domain_text_detailed(
            economy_sections,
            "economic_drivers",
            max_chars=2200,
            max_blocks=6,
        )
        drivers_item = _domain_evidence(
            drivers_text,
            economy_source_url,
        )
        if drivers_item is not None:
            record.economy.setdefault("economic_drivers", drivers_item)

    if history_text:
        compact_history = " ".join(
            event.summary
            for event in extracted_timeline[:8]
        )
        history_item = _domain_evidence(compact_history[:2600], source_url)
        if history_item is not None:
            record.sovereignty.setdefault("historical_context", history_item)

    record.origins = origins
    record.historical_timeline = timeline
    return record
