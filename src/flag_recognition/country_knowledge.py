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
import time
from typing import Iterable
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
            if response.status_code not in (429, 500, 502, 503, 504):
                return response

            if attempt >= attempts - 1:
                return response

            retry_after = response.headers.get("Retry-After")
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
        "list of sites", "location of sites",
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
        "energy", "electricity", "power", "renewable energy", "solar energy",
        "hydropower", "generation", "telecommunications", "communications",
        "internet", "mobile", "broadband",
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
    response = _get_with_retry(
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
    """Resolve common Wikipedia topic-title conventions deterministically."""
    candidate_titles = (
        f"{topic} of {country_name}",
        f"{topic} in {country_name}",
        f"List of {topic} in {country_name}",
        f"List of {topic} of {country_name}",
    )

    for title in candidate_titles:
        try:
            return fetch_country_article(title, timeout=timeout)
        except (requests.RequestException, LookupError, ValueError):
            continue

    for query in candidate_titles[:2]:
        try:
            response = _get_with_retry(
                WIKIPEDIA_API,
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": 8,
                    "format": "json",
                    "formatversion": 2,
                },
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/json",
                },
                timeout=timeout,
            )
            response.raise_for_status()
            results = response.json().get("query", {}).get("search", [])

            country_token = country_name.lower()
            topic_words = {
                word
                for word in re.findall(r"[a-z0-9]+", topic.lower())
                if len(word) >= 4
            }

            candidates = [
                str(item.get("title") or "").strip()
                for item in results
                if str(item.get("title") or "").strip()
            ]

            title = next(
                (
                    candidate
                    for candidate in candidates
                    if country_token in candidate.lower()
                    and all(
                        word in candidate.lower()
                        for word in topic_words
                    )
                ),
                None,
            )
            if title:
                return fetch_country_article(title, timeout=timeout)
        except (requests.RequestException, LookupError, ValueError):
            continue

    return None


def _fetch_topic_wikitext(
    title: str,
    *,
    timeout: float = 10.0,
) -> str:
    """Fetch structured wikitext for a known Wikipedia topic page."""
    response = _get_with_retry(
        WIKIPEDIA_API,
        params={
            "action": "parse",
            "page": title,
            "prop": "wikitext",
            "redirects": 1,
            "format": "json",
        },
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=timeout,
    )
    response.raise_for_status()
    parsed = response.json().get("parse") or {}
    raw = parsed.get("wikitext")
    if isinstance(raw, dict):
        return str(raw.get("*") or "").strip()
    return str(raw or "").strip()


def _clean_wikivalue(value: str) -> str:
    """Remove common wiki markup from compact infobox values."""
    text = str(value or "").strip()
    if not text:
        return ""

    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^>]*/>", "", text)
    text = re.sub(
        r"\{\{(?:nowrap|small|plainlist)\|([^{}]+)\}\}",
        r"\1",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\{\{[^{}]+\}\}", "", text)
    text = re.sub(r"''+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("<br />", "; ").replace("<br/>", "; ")
    text = text.replace("&nbsp;", " ")
    return " ".join(text.split()).strip(" ;,")


def split_wikitext_sections_detailed(
    wikitext: str,
) -> list[ArticleSection]:
    """Split MediaWiki source into clean section records with stable headings."""
    heading_re = re.compile(
        r"^(={2,6})\s*(.+?)\s*\1\s*$",
        re.MULTILINE,
    )
    matches = list(heading_re.finditer(wikitext))

    def clean_body(body: str) -> str:
        body = re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL)
        body = re.sub(r"<ref[^>]*>.*?</ref>", "", body, flags=re.DOTALL)
        body = re.sub(r"<ref[^>]*/>", "", body)
        body = re.sub(
            r"\[\[([^\]|]+)\|([^\]]+)\]\]",
            r"\2",
            body,
        )
        body = re.sub(r"\[\[([^\]]+)\]\]", r"\1", body)
        body = re.sub(r"\{\{[^{}]*\}\}", "", body)
        body = re.sub(r"''+", "", body)
        body = re.sub(r"<[^>]+>", "", body)
        body = re.sub(r"\{\|.*?\|\}", "", body, flags=re.DOTALL)
        return " ".join(body.split()).strip()

    if not matches:
        cleaned = clean_body(wikitext)
        return (
            [ArticleSection("overview", 1, cleaned)]
            if cleaned
            else []
        )

    sections: list[ArticleSection] = []
    lead = clean_body(wikitext[: matches[0].start()])
    if lead:
        sections.append(ArticleSection("overview", 1, lead))

    for index, match in enumerate(matches):
        start = match.end()
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(wikitext)
        )
        body = clean_body(wikitext[start:end])
        sections.append(
            ArticleSection(
                heading=match.group(2).strip(),
                level=len(match.group(1)),
                body=body,
            )
        )
    return sections


def _infobox_field(
    wikitext: str,
    names: tuple[str, ...],
) -> str:
    """Extract a compact infobox parameter, including common multiline values."""
    for name in names:
        start_pattern = re.compile(
            rf"^\|\s*{re.escape(name)}\s*=\s*(.*)$",
            flags=re.IGNORECASE | re.MULTILINE,
        )
        match = start_pattern.search(wikitext)
        if not match:
            continue

        lines = [match.group(1).strip()]
        cursor = match.end()
        for raw_line in wikitext[cursor:].splitlines():
            if re.match(r"^\|\s*[A-Za-z0-9 _-]+\s*=", raw_line):
                break
            if re.match(r"^\}\}\s*$", raw_line):
                break
            lines.append(raw_line.strip())
            if len(lines) >= 16:
                break

        value = "\n".join(line for line in lines if line).strip()
        if not value:
            continue

        # Normalize common list templates before generic cleanup.
        value = re.sub(
            r"\{\{\s*(?:plainlist|ubl|unbulleted list)\s*\|",
            "",
            value,
            flags=re.IGNORECASE,
        )
        value = value.replace("\n*", "; ")
        value = value.replace("\n", " ")
        value = re.sub(r"\}\}\s*$", "", value).strip()

        cleaned = _clean_wikivalue(value)
        if cleaned:
            return cleaned
    return ""


def _structured_economy_context(
    country_name: str,
    *,
    timeout: float = 10.0,
) -> tuple[str, str]:
    """Read stable industries/export goods from the Economy page infobox."""
    article = fetch_topic_article(
        country_name,
        "Economy",
        timeout=timeout,
    )
    if article is None:
        return "", ""

    _, title = article
    try:
        wikitext = _fetch_topic_wikitext(title, timeout=timeout)
    except (requests.RequestException, ValueError, LookupError):
        return "", ""

    industries = _infobox_field(
        wikitext,
        ("industries", "industry"),
    )
    export_goods = _infobox_field(
        wikitext,
        ("export-goods", "export_goods", "exports goods"),
    )
    import_goods = _infobox_field(
        wikitext,
        ("import-goods", "import_goods", "imports goods"),
    )

    blocks: list[str] = []
    if industries:
        blocks.append(f"Key industries: {industries}")
    if export_goods:
        blocks.append(f"Main export goods: {export_goods}")
    if import_goods:
        blocks.append(f"Main import goods: {import_goods}")

    source_url = WIKIPEDIA_PAGE + quote(
        title.replace(" ", "_"),
        safe="()_-",
    )
    return "\n\n".join(blocks), source_url


def _heritage_sites_from_wikitext(
    country_names: tuple[str, ...],
    *,
    timeout: float = 10.0,
) -> tuple[str, str]:
    """Extract site names from a country-specific World Heritage list page."""
    article: tuple[str, str] | None = None
    for country_name in country_names:
        if not country_name:
            continue
        article = fetch_topic_article(
            country_name,
            "World Heritage Sites",
            timeout=timeout,
        )
        if article is not None:
            break

    if article is None:
        return "", ""

    _, title = article
    try:
        wikitext = _fetch_topic_wikitext(title, timeout=timeout)
    except (requests.RequestException, LookupError, ValueError):
        return "", ""

    sections = split_wikitext_sections_detailed(wikitext)
    list_heading = next(
        (
            section.heading
            for section in sections
            if _clean_heading(section.heading) in {
                "list of sites",
                "world heritage sites",
                "sites",
            }
        ),
        "",
    )

    # Use the raw table body under the matching heading so that the first
    # linked item of each table row can be retained as the site name.
    heading_pattern = re.compile(
        rf"^==+\s*{re.escape(list_heading)}\s*==+\s*$",
        flags=re.IGNORECASE | re.MULTILINE,
    ) if list_heading else None

    body = wikitext
    if heading_pattern is not None:
        match = heading_pattern.search(wikitext)
        if match:
            start = match.end()
            next_heading = re.search(
                r"^==+\s*.+?\s*==+\s*$",
                wikitext[start:],
                flags=re.MULTILINE,
            )
            end = (
                start + next_heading.start()
                if next_heading
                else len(wikitext)
            )
            body = wikitext[start:end]

    sites: list[str] = []
    for row in body.split("|-"):
        link = re.search(
            r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|([^\]]+))?\]\]",
            row,
        )
        if not link:
            continue
        target = link.group(1).strip()
        display = (link.group(2) or target).strip()
        lowered = target.lower()
        if any(
            lowered.startswith(prefix)
            for prefix in (
                "file:", "image:", "unesco", "world heritage",
                "ivory coast", "côte d'ivoire",
            )
        ):
            continue
        display = re.sub(r"\{\{.*?\}\}", "", display).strip()
        if display and display not in sites:
            sites.append(display)
        if len(sites) >= 12:
            break

    if not sites:
        return "", ""

    source_url = WIKIPEDIA_PAGE + quote(
        title.replace(" ", "_"),
        safe="()_-'",
    )
    return "World Heritage Sites: " + "; ".join(sites), source_url


def _structured_geography_facts(
    country_name: str,
    *,
    timeout: float = 10.0,
) -> tuple[dict[str, str], str]:
    """Read stable physical-geography facts from a dedicated Geography infobox."""
    article = fetch_topic_article(
        country_name,
        "Geography",
        timeout=timeout,
    )
    if article is None:
        return {}, ""

    _, title = article
    try:
        wikitext = _fetch_topic_wikitext(title, timeout=timeout)
    except (requests.RequestException, ValueError, LookupError):
        return {}, ""

    fields: dict[str, str] = {}

    climate = _infobox_field(wikitext, ("climate",))
    terrain = _infobox_field(
        wikitext,
        ("terrain", "topography", "relief"),
    )
    resources = _infobox_field(
        wikitext,
        (
            "natural_resources",
            "natural resources",
            "resources",
        ),
    )
    longest_river = _infobox_field(
        wikitext,
        ("longest river", "longest_river"),
    )
    largest_lake = _infobox_field(
        wikitext,
        ("largest lake", "largest_lake"),
    )

    if climate:
        fields["climate_seasons"] = climate
    if terrain:
        fields["mountains_relief"] = terrain
    if resources:
        fields["natural_resources"] = resources

    waterways: list[str] = []
    if longest_river:
        waterways.append(f"Longest river: {longest_river}")
    if largest_lake:
        waterways.append(f"Largest lake: {largest_lake}")
    if waterways:
        fields["rivers_lakes"] = "; ".join(waterways)

    source_url = WIKIPEDIA_PAGE + quote(
        title.replace(" ", "_"),
        safe="()_-",
    )
    return fields, source_url


def _structured_geography_resources(
    country_name: str,
    *,
    timeout: float = 10.0,
) -> tuple[str, str]:
    """Read natural-resource fields from a dedicated Geography page."""
    article = fetch_topic_article(
        country_name,
        "Geography",
        timeout=timeout,
    )
    if article is None:
        return "", ""

    _, title = article
    try:
        wikitext = _fetch_topic_wikitext(title, timeout=timeout)
    except (requests.RequestException, ValueError, LookupError):
        return "", ""

    resources = _infobox_field(
        wikitext,
        (
            "natural_resources",
            "natural resources",
            "resources",
        ),
    )
    if not resources:
        return "", ""

    source_url = WIKIPEDIA_PAGE + quote(
        title.replace(" ", "_"),
        safe="()_-",
    )
    return f"Natural resources: {resources}", source_url


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


def collect_keyword_context(
    sections: list[ArticleSection],
    keywords: tuple[str, ...],
    *,
    max_sentences: int = 4,
    max_chars: int = 1800,
) -> str:
    """Fallback extractor when article headings do not match our taxonomy.

    It only reuses source sentences containing explicit domain keywords.
    No generated facts are introduced.
    """
    matches: list[str] = []
    seen: set[str] = set()

    normalized_keywords = tuple(keyword.lower() for keyword in keywords)

    for section in sections:
        for sentence in _sentences(section.body):
            lowered = sentence.lower()
            if not any(keyword in lowered for keyword in normalized_keywords):
                continue
            compact = sentence.strip()
            if compact and compact not in seen:
                seen.add(compact)
                matches.append(compact)
            if len(matches) >= max_sentences:
                break
        if len(matches) >= max_sentences:
            break

    text = " ".join(matches)
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].rstrip() + "…"
    return text


def _set_context_with_keywords(
    target: dict[str, Evidence],
    key: str,
    sections: list[ArticleSection],
    domain: str,
    source_url: str,
    *,
    keywords: tuple[str, ...],
    max_chars: int = 1800,
    max_blocks: int = 5,
) -> None:
    """Populate a domain from headings first, then sentence-level keywords."""
    if key in target:
        return

    text = collect_domain_text_detailed(
        sections,
        domain,
        max_chars=max_chars,
        max_blocks=max_blocks,
    )
    if not text:
        text = collect_keyword_context(
            sections,
            keywords,
            max_chars=max_chars,
        )

    item = _domain_evidence(text, source_url)
    if item is not None:
        target.setdefault(key, item)


STRICT_DOMAIN_HEADINGS: dict[str, tuple[str, ...]] = {
    "climate_seasons": ("climate", "climate and weather", "seasons"),
    "rivers_lakes": ("rivers", "hydrography", "drainage", "lakes"),
    "mountains_relief": (
        "terrain and topography", "topography", "terrain", "relief",
        "mountains", "mountain ranges",
    ),
    "natural_resources": (
        "natural resources", "mineral resources", "minerals", "mining",
        "cropland",
    ),
    "economic_drivers": (
        "agriculture", "industry", "industries", "trade", "exports",
        "tourism", "services", "economic sectors", "production",
    ),
    "energy_connectivity": (
        "energy", "electricity", "power", "solar energy", "hydroelectric",
        "renewable energy", "telecommunications", "internet",
    ),
    "transport_network": (
        "roads", "rail", "railways", "ports", "airports", "aviation",
        "shipping", "transport",
    ),
}


def collect_strict_domain_text(
    sections: list[ArticleSection],
    domain: str,
    *,
    max_chars: int = 2200,
    max_blocks: int = 6,
) -> str:
    """Extract only explicitly compatible headings for high-risk domains."""
    allowed = {
        _clean_heading(value)
        for value in STRICT_DOMAIN_HEADINGS.get(domain, ())
    }
    if not allowed:
        return ""

    blocks: list[str] = []
    for section in sections:
        heading = _clean_heading(section.heading)
        if not any(
            heading == candidate
            or heading.startswith(candidate + " ")
            for candidate in allowed
        ):
            continue

        summary = _compact_body(
            section.heading,
            section.body,
            max_sentences=3,
            max_chars=900,
        )
        if summary:
            blocks.append(f"{section.heading}: {summary}")
        if len(blocks) >= max_blocks:
            break

    combined = "\n\n".join(blocks)
    if len(combined) <= max_chars:
        return combined
    return combined[:max_chars].rsplit(" ", 1)[0].rstrip() + "…"


def _set_strict_context(
    target: dict[str, Evidence],
    key: str,
    sections: list[ArticleSection],
    domain: str,
    source_url: str,
    *,
    max_chars: int = 2200,
    max_blocks: int = 6,
) -> None:
    text = collect_strict_domain_text(
        sections,
        domain,
        max_chars=max_chars,
        max_blocks=max_blocks,
    )
    item = _domain_evidence(text, source_url)
    if item is not None:
        target[key] = item


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


def _dedicated_topic_sections(
    country_name: str,
    topic: str,
    *,
    timeout: float,
) -> tuple[list[ArticleSection], str]:
    """Return a dedicated topic page with headings preserved from wikitext."""
    article = fetch_topic_article(
        country_name,
        topic,
        timeout=timeout,
    )
    if article is None:
        return [], ""

    text, title = article
    source_url = WIKIPEDIA_PAGE + quote(
        title.replace(" ", "_"),
        safe="()_-",
    )

    try:
        wikitext = _fetch_topic_wikitext(
            title,
            timeout=timeout,
        )
        sections = split_wikitext_sections_detailed(wikitext)
        if sections:
            return sections, source_url
    except (requests.RequestException, ValueError, LookupError):
        pass

    return split_article_sections_detailed(text), source_url


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
    canonical_title_fact = evidence(
        canonical_title,
        "Wikipedia canonical article title",
        retrieved_at=datetime.now(timezone.utc).date().isoformat(),
        confidence=1.0,
    )
    if canonical_title_fact is not None:
        record.identity.setdefault(
            "encyclopedia_title",
            canonical_title_fact,
        )

    # Prefer a dedicated History article. The canonical Wikipedia title is
    # important for countries whose displayed/native name differs from the
    # English article naming convention (e.g. Côte d'Ivoire -> Ivory Coast).
    history_article = fetch_topic_article(
        canonical_title,
        "History",
        timeout=timeout,
    )
    history_sections = detailed
    history_source_url = source_url
    if history_article is not None:
        history_article_text, history_title = history_article
        history_sections = split_article_sections_detailed(
            history_article_text
        )
        history_source_url = WIKIPEDIA_PAGE + quote(
            history_title.replace(" ", "_"),
            safe="()_-",
        )
    else:
        history_sections = _domain_records(detailed, "history")

    reference_headings = {
        "references",
        "bibliography",
        "further reading",
        "notes",
        "sources",
        "citations",
        "works cited",
        "external links",
        "see also",
    }
    filtered_history_sections = [
        section
        for section in history_sections
        if (
            section.body
            and _clean_heading(section.heading)
            not in reference_headings
        )
    ]

    history_text = " ".join(
        f"{section.heading}. {section.body}"
        for section in filtered_history_sections
    )

    origins = record.origins or collect_origins(
        filtered_history_sections,
        history_source_url,
    )
    if not origins and filtered_history_sections:
        lead = next(
            (
                section
                for section in filtered_history_sections
                if section.heading == "overview" and section.body
            ),
            None,
        )
        if lead is not None:
            lower_lead = lead.body.lower()
            if any(
                token in lower_lead
                for token in (
                    "earliest", "prehistory", "prehistoric",
                    "paleolithic", "neolithic", "ancient",
                    "first inhabitants", "human arrival",
                )
            ):
                summary = _compact_body(
                    "Early history & origins",
                    lead.body,
                    max_sentences=3,
                    max_chars=1200,
                )
                if summary:
                    origins = (
                        TimelineEvent(
                            label="Early history & origins",
                            period="Early history",
                            summary=summary,
                            sources=("Wikipedia",),
                            source_urls=(history_source_url,),
                            confidence=0.75,
                        ),
                    )
    timeline = record.historical_timeline
    extracted_timeline = extract_timeline(
        history_text,
        history_source_url,
    )

    if not extracted_timeline:
        fallback_events: list[TimelineEvent] = []
        for section in filtered_history_sections:
            if section.heading == "overview" or not section.body:
                continue
            summary = _compact_body(
                section.heading,
                section.body,
                max_sentences=2,
                max_chars=700,
            )
            if not summary:
                continue
            year_match = _DATE_TOKEN.search(summary)
            period = (
                year_match.group("year")
                if year_match is not None
                else section.heading
            )
            fallback_events.append(
                TimelineEvent(
                    label=section.heading,
                    period=period,
                    summary=summary,
                    sources=("Wikipedia",),
                    source_urls=(history_source_url,),
                    confidence=0.70,
                )
            )
            if len(fallback_events) >= 12:
                break
        if fallback_events:
            extracted_timeline = tuple(fallback_events)

    legacy_only = (
        len(timeline) == 1
        and timeline[0].label == "Legacy historical context"
    )
    if extracted_timeline and (not timeline or legacy_only):
        timeline = extracted_timeline

    # Commit core history immediately. All later enrichments are optional and
    # must never make a valid History article disappear from the final record.
    record.origins = origins
    record.historical_timeline = timeline

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
        canonical_title,
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
        canonical_title,
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
    try:
        heritage_sites, heritage_sites_url = _heritage_sites_from_wikitext(
            (record.name, canonical_title),
            timeout=timeout,
        )
        if heritage_sites:
            heritage_item = _domain_evidence(
                heritage_sites,
                heritage_sites_url,
            )
            if heritage_item is not None:
                record.culture["heritage_landmarks"] = heritage_item
        elif "heritage_landmarks" not in record.culture:
            heritage_sections, heritage_url = _dedicated_topic_sections(
                canonical_title,
                "World Heritage Sites",
                timeout=timeout,
            )
            if not heritage_sections and record.name != canonical_title:
                heritage_sections, heritage_url = _dedicated_topic_sections(
                    record.name,
                    "World Heritage Sites",
                    timeout=timeout,
                )
            _set_context(
                record.culture,
                "heritage_landmarks",
                heritage_sections,
                "heritage_landmarks",
                heritage_url,
                max_chars=2200,
                max_blocks=8,
            )
    except (requests.RequestException, LookupError, ValueError):
        pass
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
    try:
        transport_sections, transport_url = _dedicated_topic_sections(
            canonical_title,
            "Transport",
            timeout=timeout,
        )
        _set_strict_context(
            record.infrastructure,
            "transport_network",
            transport_sections,
            "transport_network",
            transport_url,
            max_chars=2200,
            max_blocks=6,
        )
        if "transport_network" not in record.infrastructure:
            _set_strict_context(
                record.infrastructure,
                "transport_network",
                detailed,
                "transport_network",
                source_url,
                max_chars=2200,
                max_blocks=6,
            )
    except (requests.RequestException, LookupError, ValueError):
        pass

    try:
        energy_sections, energy_url = _dedicated_topic_sections(
            canonical_title,
            "Energy",
            timeout=timeout,
        )
        energy_text = collect_strict_domain_text(
            energy_sections,
            "energy_connectivity",
            max_chars=2200,
            max_blocks=6,
        )
        if not energy_text:
            energy_lead = next(
                (
                    section.body
                    for section in energy_sections
                    if section.heading == "overview" and section.body
                ),
                "",
            )
            energy_text = _compact_body(
                "Energy",
                energy_lead,
                max_sentences=3,
                max_chars=1200,
            )
        energy_item = _domain_evidence(energy_text, energy_url)
        if energy_item is not None:
            record.infrastructure["energy_connectivity"] = energy_item

        if "energy_connectivity" not in record.infrastructure:
            _set_strict_context(
                record.infrastructure,
                "energy_connectivity",
                detailed,
                "energy_connectivity",
                source_url,
                max_chars=2200,
                max_blocks=6,
            )
    except (requests.RequestException, LookupError, ValueError):
        pass

    # Dedicated geography article. Keep this optional so a source failure
    # cannot abort economy, transport, energy or the final report.
    try:
        geography_sections, geography_source_url = _dedicated_topic_sections(
            canonical_title,
            "Geography",
            timeout=timeout,
        )

        physical_domains = {
            "climate_seasons": ("climate_seasons", record.environment),
            "rivers_lakes": ("rivers_lakes", record.geography),
            "mountains_relief": ("mountains_relief", record.geography),
            "natural_resources": ("natural_resources", record.environment),
        }
        for domain, (key, target) in physical_domains.items():
            _set_strict_context(
                target,
                key,
                geography_sections,
                domain,
                geography_source_url,
                max_chars=2200,
                max_blocks=6,
            )

        structured_geo, structured_geo_url = _structured_geography_facts(
            canonical_title,
            timeout=timeout,
        )
        if structured_geo:
            structured_targets = {
                "climate_seasons": record.environment,
                "rivers_lakes": record.geography,
                "mountains_relief": record.geography,
                "natural_resources": record.environment,
            }
            for key, value in structured_geo.items():
                target = structured_targets[key]
                item = _domain_evidence(
                    value,
                    structured_geo_url or geography_source_url,
                )
                if item is not None:
                    target.setdefault(key, item)

        if geography_sections and geography_source_url:
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
                record.environment["context"] = environment_item
    except (requests.RequestException, LookupError, ValueError):
        pass

    try:
        economy_sections, economy_source_url = _dedicated_topic_sections(
            canonical_title,
            "Economy",
            timeout=timeout,
        )
        if economy_sections and economy_source_url:
            _set_strict_context(
                record.economy,
                "economic_drivers",
                economy_sections,
                "economic_drivers",
                economy_source_url,
                max_chars=2400,
                max_blocks=7,
            )

            structured_economy, structured_economy_url = (
                _structured_economy_context(
                    canonical_title,
                    timeout=timeout,
                )
            )
            if structured_economy:
                existing = record.economy.get("economic_drivers")
                combined = structured_economy
                if existing is not None and str(existing.value).strip():
                    combined = (
                        f"{existing.value}\n\n{structured_economy}"
                    )
                record.economy["economic_drivers"] = _domain_evidence(
                    combined[:3200],
                    structured_economy_url or economy_source_url,
                )

            # Some countries describe mineral/agricultural resources primarily in
            # the Economy article rather than in Geography.
            if "natural_resources" not in record.environment:
                _set_strict_context(
                    record.environment,
                    "natural_resources",
                    economy_sections,
                    "natural_resources",
                    economy_source_url,
                    max_chars=1800,
                    max_blocks=5,
                )

        if "natural_resources" not in record.environment:
            structured_resources, structured_resources_url = (
                _structured_geography_resources(
                    canonical_title,
                    timeout=timeout,
                )
            )
            if structured_resources:
                resource_item = _domain_evidence(
                    structured_resources,
                    structured_resources_url,
                )
                if resource_item is not None:
                    record.environment["natural_resources"] = resource_item
    except (requests.RequestException, LookupError, ValueError):
        pass

    if history_text:
        compact_history = " ".join(
            event.summary
            for event in extracted_timeline[:8]
        )
        history_item = _domain_evidence(
            compact_history[:2600],
            history_source_url,
        )
        if history_item is not None:
            record.sovereignty.setdefault("historical_context", history_item)

    record.origins = origins
    record.historical_timeline = timeline
    return record
