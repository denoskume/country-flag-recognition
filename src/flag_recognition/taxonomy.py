"""Local country/territory label helpers."""

from __future__ import annotations

import pycountry


NAME_OVERRIDES = {
    "xk": "Kosovo",
}


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
