"""Local country/territory label helpers."""

from __future__ import annotations

import re
import unicodedata

import pycountry


NAME_OVERRIDES = {
    "xk": "Kosovo",
}

TEXT_ALIASES = {
    "kosovo": "xk",
    "ivory coast": "ci",
    "cote d ivoire": "ci",
    "south korea": "kr",
    "north korea": "kp",
    "russia": "ru",
    "iran": "ir",
    "bolivia": "bo",
    "venezuela": "ve",
    "tanzania": "tz",
    "moldova": "md",
    "laos": "la",
    "brunei": "bn",
    "syria": "sy",
    "palestine": "ps",
    "taiwan": "tw",
    "vietnam": "vn",
    "czechia": "cz",
    "cape verde": "cv",
}


def _normalize_country_text(value: str) -> str:
    """Normalize human country names without changing their meaning."""
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.casefold().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def country_name_from_code(code: str) -> str:
    """Return a human-readable name for a project class code."""
    normalized = code.lower().strip()

    if normalized in NAME_OVERRIDES:
        return NAME_OVERRIDES[normalized]

    country = pycountry.countries.get(
        alpha_2=normalized.upper()
    )

    if country is None:
        return normalized.upper()

    return str(country.name)


def country_code_from_text(value: str) -> str | None:
    """Resolve a typed country name or alpha code to a project class code.

    Matching is deterministic and accent/punctuation-insensitive. Common
    everyday names are supported alongside ISO names, official names and
    alpha-2/alpha-3 codes. Unknown free text is rejected rather than guessed.
    """
    normalized = _normalize_country_text(value)
    if not normalized:
        return None

    if normalized in TEXT_ALIASES:
        return TEXT_ALIASES[normalized]

    compact = normalized.replace(" ", "")
    if len(compact) == 2:
        if compact == "xk":
            return "xk"
        match = pycountry.countries.get(alpha_2=compact.upper())
        if match is not None:
            return str(match.alpha_2).lower()

    if len(compact) == 3:
        match = pycountry.countries.get(alpha_3=compact.upper())
        if match is not None:
            return str(match.alpha_2).lower()

    for country in pycountry.countries:
        candidates = {
            str(country.name),
            str(getattr(country, "official_name", "")),
            str(getattr(country, "common_name", "")),
        }
        if normalized in {
            _normalize_country_text(candidate)
            for candidate in candidates
            if candidate
        }:
            return str(country.alpha_2).lower()

    return None
