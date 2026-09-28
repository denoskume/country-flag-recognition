"""Flag-specific educational enrichment for Flag Intelligence."""

from __future__ import annotations

from datetime import datetime, timezone
import re
import time
from urllib.parse import quote

import requests

def _get_with_retry(
    url: str,
    *,
    attempts: int = 4,
    backoff: float = 0.75,
    **kwargs,
):
    """HTTP GET with bounded retries for transient network/rate-limit failures."""
    last_error = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, **kwargs)
            status_code = getattr(response, "status_code", 200)
            if status_code not in (429, 500, 502, 503, 504):
                return response

            if attempt >= attempts - 1:
                return response

            headers = getattr(response, "headers", {}) or {}
            retry_after = headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else backoff * (2 ** attempt)
            except (TypeError, ValueError):
                delay = backoff * (2 ** attempt)
            time.sleep(min(max(delay, 0.0), 12.0))
        except (
            requests.Timeout,
            requests.ConnectionError,
        ) as exc:
            last_error = exc
            if attempt >= attempts - 1:
                raise
            time.sleep(min(backoff * (2 ** attempt), 12.0))

    if last_error is not None:
        raise last_error
    raise RuntimeError("HTTP retry loop exited unexpectedly")



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
        "adoption",
    ),
}


def _search_flag_article(country_name: str, timeout: float = 12.0) -> str:
    """Resolve a dedicated national-flag article deterministically first."""
    exact_title = f"Flag of {country_name}"
    try:
        _fetch_article(exact_title, timeout=timeout)
        return exact_title
    except (requests.RequestException, LookupError, ValueError):
        pass

    queries = [
        exact_title,
        f"{country_name} flag",
    ]

    for query in queries:
        response = _get_with_retry(
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


def _fetch_flag_wikitext(
    title: str,
    timeout: float = 12.0,
) -> tuple[str, str]:
    """Fetch raw flag-page wikitext through MediaWiki action=parse."""
    response = _get_with_retry(
        WIKIPEDIA_API,
        params={
            "action": "parse",
            "page": title,
            "prop": "wikitext|displaytitle",
            "redirects": 1,
            "format": "json",
        },
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=timeout,
    )
    response.raise_for_status()

    parsed = response.json().get("parse") or {}
    raw_wikitext = parsed.get("wikitext")
    if isinstance(raw_wikitext, dict):
        text = str(raw_wikitext.get("*") or "").strip()
    else:
        text = str(raw_wikitext or "").strip()

    canonical = str(
        parsed.get("title")
        or parsed.get("displaytitle")
        or title
    ).strip()

    if not text:
        raise LookupError(title)
    return text, canonical


def _infobox_value(wikitext: str, field: str) -> str:
    """Extract one infobox field, including common multiline values."""
    pattern = re.compile(
        rf"^\|\s*{re.escape(field)}\s*=\s*(.*)$",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    match = pattern.search(wikitext)
    if not match:
        return ""

    lines = [match.group(1).strip()]
    cursor = match.end()
    for raw_line in wikitext[cursor:].splitlines():
        if re.match(r"^\|\s*[A-Za-z0-9 _-]+\s*=", raw_line):
            break
        if re.match(r"^\}\}\s*$", raw_line):
            break
        lines.append(raw_line.strip())
        if len(lines) >= 12:
            break

    value = "\n".join(line for line in lines if line).strip()

    start_date = re.fullmatch(
        r"\{\{Start date(?: and age)?\|(\d{4})\|(\d{1,2})\|(\d{1,2})(?:\|[^{}]*)?\}\}",
        value,
        flags=re.IGNORECASE,
    )
    if start_date:
        year, month, day = map(int, start_date.groups())
        month_names = (
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        )
        if 1 <= month <= 12:
            return f"{day} {month_names[month - 1]} {year}"

    value = re.sub(
        r"\{\{\s*(?:plainlist|ubl|unbulleted list)\s*\|",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = value.replace("\n*", "; ").replace("\n", " ")
    value = re.sub(r"\}\}\s*$", "", value).strip()

    # Remove common wiki markup while preserving the factual text.
    value = re.sub(r"<ref[^>]*>.*?</ref>", "", value, flags=re.DOTALL)
    value = re.sub(r"<ref[^>]*/>", "", value)
    value = re.sub(r"\{\{(?:nowrap|small)\|([^{}]+)\}\}", r"\1", value)
    value = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", value)
    value = re.sub(r"\[\[([^\]]+)\]\]", r"\1", value)
    value = re.sub(r"''+", "", value)
    value = re.sub(r"<[^>]+>", "", value)
    return " ".join(value.split())


def _fetch_flag_page_extract(
    title: str,
    timeout: float = 12.0,
) -> tuple[str, str]:
    """Fetch readable flag-page plaintext without relying on section names."""
    response = _get_with_retry(
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
    text = str(page.get("extract") or "").strip()
    canonical = str(page.get("title") or title).strip()
    if not text:
        raise LookupError(title)
    return text, canonical


def _fetch_flag_rest_summary(
    country_name: str,
    timeout: float = 12.0,
) -> tuple[str, str]:
    """Fetch the exact flag page through Wikimedia's REST summary endpoint."""
    title = f"Flag_of_{country_name.replace(' ', '_')}"
    url = (
        "https://en.wikipedia.org/api/rest_v1/page/summary/"
        + quote(title, safe="()_'")
    )
    response = _get_with_retry(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    text = str(payload.get("extract") or "").strip()
    canonical = str(payload.get("title") or f"Flag of {country_name}").strip()
    if not text:
        raise LookupError(canonical)
    return text, canonical


def _flag_metadata_from_rest_summary(
    country_name: str,
    timeout: float = 12.0,
) -> tuple[Evidence | None, Evidence | None, Evidence | None]:
    article, canonical = _fetch_flag_rest_summary(
        country_name,
        timeout=timeout,
    )
    source_url = WIKIPEDIA_PAGE + quote(
        canonical.replace(" ", "_"),
        safe="()_-",
    )
    adoption = _extract_adoption(article, source_url)
    proportion = _extract_proportion(article, source_url)

    sentences = [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+",
            " ".join(article.split()),
        )
        if sentence.strip()
    ]
    design_sentence = next(
        (
            sentence
            for sentence in sentences[:8]
            if any(
                token in sentence.lower()
                for token in (
                    "tricolour",
                    "tricolor",
                    "vertical",
                    "horizontal",
                    "flag is",
                    "flag consists",
                )
            )
        ),
        "",
    )
    design = _fact(design_sentence, source_url)
    return adoption, proportion, design


def _flag_metadata_from_extract(
    country_name: str,
    timeout: float = 12.0,
) -> tuple[Evidence | None, Evidence | None, Evidence | None]:
    """Extract basic flag metadata from the exact flag page plaintext."""
    exact_title = f"Flag of {country_name}"
    article, canonical = _fetch_flag_page_extract(
        exact_title,
        timeout=timeout,
    )
    source_url = WIKIPEDIA_PAGE + quote(
        canonical.replace(" ", "_"),
        safe="()_-",
    )

    adoption = _extract_adoption(article, source_url)
    proportion = _extract_proportion(article, source_url)

    sentences = [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+",
            " ".join(article.split()),
        )
        if sentence.strip()
    ]
    design_sentence = next(
        (
            sentence
            for sentence in sentences[:12]
            if any(
                token in sentence.lower()
                for token in (
                    "tricolour",
                    "tricolor",
                    "vertical bands",
                    "horizontal bands",
                    "flag consists",
                    "flag is",
                )
            )
        ),
        "",
    )
    design = _fact(design_sentence, source_url)
    return adoption, proportion, design


def _flag_metadata_from_wikitext(
    country_name: str,
    timeout: float = 12.0,
) -> tuple[Evidence | None, Evidence | None, Evidence | None]:
    """Return adoption, proportion and design from the flag-page infobox."""
    exact_title = f"Flag of {country_name}"
    text, canonical = _fetch_flag_wikitext(
        exact_title,
        timeout=timeout,
    )
    source_url = WIKIPEDIA_PAGE + quote(
        canonical.replace(" ", "_"),
        safe="()_-",
    )

    adoption_raw = _infobox_value(text, "adoption")
    proportion_raw = _infobox_value(text, "proportion")
    design_raw = _infobox_value(text, "design")

    adoption = _fact(adoption_raw, source_url) if adoption_raw else None
    proportion = _fact(proportion_raw, source_url) if proportion_raw else None
    design = _fact(design_raw, source_url) if design_raw else None
    return adoption, proportion, design


def _fetch_article(title: str, timeout: float = 12.0) -> tuple[str, str]:
    response = _get_with_retry(
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

    month_names = (
        "January|February|March|April|May|June|July|August|"
        "September|October|November|December"
    )
    patterns = [
        rf"\b\d{{1,2}}\s+(?:{month_names})\s+(?:1[89]\d{{2}}|20\d{{2}})\b",
        rf"\b(?:{month_names})\s+\d{{1,2}},?\s+(?:1[89]\d{{2}}|20\d{{2}})\b",
        r"\b(?:1[89]\d{2}|20\d{2})\b",
    ]

    for sentence in sentences[:16]:
        if "adopt" not in sentence.lower():
            continue
        for pattern in patterns:
            match = re.search(pattern, sentence)
            if match:
                return _fact(match.group(0), source_url)
    return None


def _extract_proportion(article: str, source_url: str) -> Evidence | None:
    patterns = [
        r"(?:ratio|proportion|dimensions?)\s+(?:of\s+)?(\d+\s*:\s*\d+)",
        r"(\d+\s*:\s*\d+)(?:\s+[a-z-]+){0,4}\s+(?:ratio|proportion)",
        r"\b(\d+\s*:\s*\d+)\b",
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
    """Add sourced flag metadata with independent structured fallbacks."""

    adoption = current.adoption_date
    proportion = current.proportion
    symbolism = current.symbolism
    design_origin = current.design_origin
    historical_flags = current.historical_flags

    # First try the readable flag article.
    try:
        title = _search_flag_article(country_name, timeout=timeout)
        article, canonical = _fetch_article(title, timeout=timeout)
        source_url = WIKIPEDIA_PAGE + quote(
            canonical.replace(" ", "_"),
            safe="()_-",
        )
        sections = split_article_sections_detailed(article)

        design = _collect(sections, FLAG_SECTION_ALIASES["design"])
        symbolism_text = _collect(
            sections,
            FLAG_SECTION_ALIASES["symbolism"],
        )
        history = _collect(
            sections,
            FLAG_SECTION_ALIASES["history"],
            max_chars=7000,
        )

        design_fact = _fact(design, source_url)
        symbolism_fact = _fact(symbolism_text, source_url)

        if design_fact is None:
            lead = next(
                (
                    section.body
                    for section in sections
                    if section.heading == "overview" and section.body
                ),
                "",
            )
            lead_sentences = [
                sentence.strip()
                for sentence in re.split(
                    r"(?<=[.!?])\s+",
                    " ".join(lead.split()),
                )
                if sentence.strip()
            ]
            design_sentence = next(
                (
                    sentence
                    for sentence in lead_sentences
                    if any(
                        token in sentence.lower()
                        for token in (
                            "tricolour",
                            "tricolor",
                            "flag",
                            "bands",
                        )
                    )
                ),
                "",
            )
            design_fact = _fact(design_sentence, source_url)

        adoption = adoption or _extract_adoption(article, source_url)
        proportion = proportion or _extract_proportion(
            article,
            source_url,
        )
        if not symbolism and symbolism_fact is not None:
            symbolism = (symbolism_fact,)
        if not design_origin and design_fact is not None:
            design_origin = (design_fact,)
        if not historical_flags:
            historical_flags = extract_timeline(
                history,
                source_url,
                max_events=8,
            )
    except (requests.RequestException, LookupError, ValueError):
        pass

    # Independent structured fallback: flag-page infobox. This path does not
    # depend on narrative section parsing and prevents false "Flag Intelligence
    # missing" results when the article extract changes format.
    if adoption is None or proportion is None or not design_origin:
        try:
            (
                fallback_adoption,
                fallback_proportion,
                fallback_design,
            ) = _flag_metadata_from_wikitext(
                country_name,
                timeout=timeout,
            )
            adoption = adoption or fallback_adoption
            proportion = proportion or fallback_proportion
            if not design_origin and fallback_design is not None:
                design_origin = (fallback_design,)
        except (requests.RequestException, LookupError, ValueError):
            pass

    if adoption is None or proportion is None or not design_origin:
        try:
            (
                fallback_adoption,
                fallback_proportion,
                fallback_design,
            ) = _flag_metadata_from_extract(
                country_name,
                timeout=timeout,
            )
            adoption = adoption or fallback_adoption
            proportion = proportion or fallback_proportion
            if not design_origin and fallback_design is not None:
                design_origin = (fallback_design,)
        except (requests.RequestException, LookupError, ValueError):
            pass

    if adoption is None or proportion is None or not design_origin:
        try:
            (
                fallback_adoption,
                fallback_proportion,
                fallback_design,
            ) = _flag_metadata_from_rest_summary(
                country_name,
                timeout=timeout,
            )
            adoption = adoption or fallback_adoption
            proportion = proportion or fallback_proportion
            if not design_origin and fallback_design is not None:
                design_origin = (fallback_design,)
        except (requests.RequestException, LookupError, ValueError):
            pass

    return FlagProfile(
        adoption_date=adoption,
        proportion=proportion,
        colors=current.colors,
        symbolism=symbolism,
        design_origin=design_origin,
        historical_flags=historical_flags,
        similar_flags=similar_flags or current.similar_flags,
    )
