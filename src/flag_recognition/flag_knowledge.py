"""Flag-specific educational enrichment for Flag Intelligence V2."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from urllib.parse import quote

import requests

from .country_intelligence import Evidence, FlagProfile, PARTIAL, TimelineEvent, evidence
from .country_knowledge import (
    WIKIPEDIA_API,
    WIKIPEDIA_PAGE,
    USER_AGENT,
    extract_timeline,
    split_article_sections,
)


FLAG_SECTION_ALIASES = {
    "design": ("design", "description", "construction", "specifications"),
    "symbolism": ("symbolism", "meaning", "colours", "colors"),
    "history": ("history", "historical flags", "previous flags"),
}


def _search_flag_article(country_name: str, timeout: float = 12.0) -> str:
    """Resolve a likely flag article title using Wikipedia search."""
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


def _collect(
    sections: list[tuple[str, str]],
    aliases: tuple[str, ...],
    max_chars: int = 3000,
) -> str:
    blocks: list[str] = []
    for heading, body in sections:
        normalized = heading.strip().lower()
        if any(
            normalized == alias
            or normalized.startswith(alias + " ")
            or alias in normalized
            for alias in aliases
        ):
            blocks.append(f"{heading}: {' '.join(body.split())}")
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


def enrich_flag_profile(
    current: FlagProfile,
    country_name: str,
    *,
    similar_flags: tuple[str, ...] = (),
    timeout: float = 12.0,
) -> FlagProfile:
    """Add sourced flag design, symbolism and history where available."""
    title = _search_flag_article(country_name, timeout=timeout)
    article, canonical = _fetch_article(title, timeout=timeout)
    source_url = WIKIPEDIA_PAGE + quote(
        canonical.replace(" ", "_"),
        safe="()_-",
    )
    sections = split_article_sections(article)

    design = _collect(sections, FLAG_SECTION_ALIASES["design"])
    symbolism = _collect(sections, FLAG_SECTION_ALIASES["symbolism"])
    history = _collect(
        sections,
        FLAG_SECTION_ALIASES["history"],
        max_chars=12000,
    )

    design_fact = _fact(design, source_url)
    symbolism_fact = _fact(symbolism, source_url)
    history_events = extract_timeline(
        history,
        source_url,
        max_events=10,
    )

    return FlagProfile(
        adoption_date=current.adoption_date,
        proportion=current.proportion,
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
