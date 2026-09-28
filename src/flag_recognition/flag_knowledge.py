"""Flag-specific educational enrichment for Flag Intelligence."""

from __future__ import annotations

from datetime import datetime, timezone
import re
from urllib.parse import quote

import requests

from .country_intelligence import Evidence, FlagProfile, PARTIAL, evidence
from .country_knowledge import (
    ArticleSection,
    WIKIPEDIA_API,
    WIKIPEDIA_PAGE,
    USER_AGENT,
    extract_timeline,
    split_article_sections_detailed,
)


FLAG_SECTION_ALIASES = {
    "design": (
        "design", "description", "construction", "specifications",
        "design and symbolism", "colours", "colors",
    ),
    "symbolism": (
        "symbolism", "meaning", "design and symbolism", "colours", "colors",
    ),
    "history": (
        "history", "historical flags", "previous flags", "origins",
    ),
}


def _search_flag_article(country_name: str, timeout: float = 12.0) -> str:
    """Resolve a likely dedicated national-flag article."""
    queries = [
        f"Flag of {country_name}",
        f"{country_name} flag",
    ]

    for query in queries:
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
        for item in results:
            title = str(item.get("title") or "").strip()
            if title.lower().startswith("flag of "):
                return title

    raise LookupError(f"No flag article found for {country_name!r}")


def _fetch_article(title: str, timeout: float = 12.0) -> tuple[str, str]:
    response = requests.get(
        WIKIPEDIA_API,
        params={
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "redirects": 1,
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
        raise LookupError(title)
    page = pages[0]
    extract = str(page.get("extract") or "").strip()
    canonical = str(page.get("title") or title).strip()
    if not extract:
        raise LookupError(title)
    return extract, canonical


def _matches(heading: str, aliases: tuple[str, ...]) -> bool:
    normalized = heading.strip().lower()
    return any(
        normalized == alias
        or normalized.startswith(alias + " ")
        or alias in normalized
        for alias in aliases
    )


def _collect(
    sections: list[ArticleSection],
    aliases: tuple[str, ...],
    max_chars: int = 2200,
) -> str:
    """Collect concise direct sections plus children of matching parent sections."""
    blocks: list[str] = []
    active_level: int | None = None

    for section in sections:
        if active_level is not None and section.level <= active_level:
            active_level = None

        matches = _matches(section.heading, aliases)
        if matches:
            active_level = section.level

        if not matches and not (
            active_level is not None and section.level > active_level
        ):
            continue

        sentences = [
            sentence.strip()
            for sentence in re.split(
                r"(?<=[.!?])\s+",
                " ".join(section.body.split()),
            )
            if sentence.strip()
        ]
        if not sentences:
            continue

        summary = " ".join(sentences[:2])
        if len(summary) > 700:
            summary = summary[:697].rsplit(" ", 1)[0] + "…"
        blocks.append(f"{section.heading}: {summary}")

        if len(blocks) >= 5:
            break

    text = "\n\n".join(blocks)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


def _fact(value: str, source_url: str) -> Evidence | None:
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


def _extract_adoption(article: str, source_url: str) -> Evidence | None:
    sentences = [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+",
            " ".join(article.split()),
        )
        if sentence.strip()
    ]

    for sentence in sentences[:12]:
        lower = sentence.lower()
        if "adopt" not in lower:
            continue

        match = re.search(
            r"(?:(January|February|March|April|May|June|July|August|"
            r"September|October|November|December)\s+\d{1,2},?\s+)?"
            r"(1[89]\d{2}|20\d{2})",
            sentence,
        )
        if match:
            return _fact(match.group(0), source_url)
    return None


def _extract_proportion(article: str, source_url: str) -> Evidence | None:
    patterns = [
        r"(?:ratio|proportion|dimensions?)\s+(?:of\s+)?(\d+\s*:\s*\d+)",
        r"(\d+\s*:\s*\d+)\s+(?:ratio|proportion)",
    ]
    for pattern in patterns:
        match = re.search(pattern, article, flags=re.IGNORECASE)
        if match:
            return _fact(match.group(1).replace(" ", ""), source_url)
    return None


def enrich_flag_profile(
    current: FlagProfile,
    country_name: str,
    *,
    similar_flags: tuple[str, ...] = (),
    timeout: float = 12.0,
) -> FlagProfile:
    """Add sourced design, symbolism, adoption and history where available."""
    title = _search_flag_article(country_name, timeout=timeout)
    article, canonical = _fetch_article(title, timeout=timeout)
    source_url = WIKIPEDIA_PAGE + quote(
        canonical.replace(" ", "_"),
        safe="()_-",
    )
    sections = split_article_sections_detailed(article)

    design = _collect(sections, FLAG_SECTION_ALIASES["design"])
    symbolism = _collect(sections, FLAG_SECTION_ALIASES["symbolism"])
    history = _collect(
        sections,
        FLAG_SECTION_ALIASES["history"],
        max_chars=7000,
    )

    design_fact = _fact(design, source_url)
    symbolism_fact = _fact(symbolism, source_url)
    history_events = extract_timeline(
        history,
        source_url,
        max_events=8,
    )

    return FlagProfile(
        adoption_date=current.adoption_date or _extract_adoption(
            article,
            source_url,
        ),
        proportion=current.proportion or _extract_proportion(
            article,
            source_url,
        ),
        colors=current.colors,
        symbolism=(
            current.symbolism
            if current.symbolism
            else ((symbolism_fact,) if symbolism_fact else ())
        ),
        design_origin=(
            current.design_origin
            if current.design_origin
            else ((design_fact,) if design_fact else ())
        ),
        historical_flags=(
            current.historical_flags
            if current.historical_flags
            else history_events
        ),
        similar_flags=similar_flags or current.similar_flags,
    )
