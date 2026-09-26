"""Live country enrichment for the Streamlit dashboard."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import requests


WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
WORLD_BANK_BASE = "https://api.worldbank.org/v2"

# Kosovo has no official ISO 3166-1 alpha-2 assignment.
WIKIDATA_OVERRIDES = {
    "xk": "Q1246",
}
WORLD_BANK_OVERRIDES = {
    "xk": "XKX",
}


@dataclass(frozen=True)
class PopulationRecord:
    value: int | None
    year: str | None
    source: str


@dataclass(frozen=True)
class CountryProfile:
    code: str
    name: str
    continent: str
    capital: str
    currency: str
    government_form: str
    head_of_state: str
    head_of_state_office: str
    head_of_government: str
    head_of_government_office: str
    latitude: float | None
    longitude: float | None
    population: PopulationRecord
    political_source: str
    population_source: str


def _unique_join(values: Iterable[str]) -> str:
    cleaned = []
    for value in values:
        value = str(value).strip()
        if value and value not in cleaned:
            cleaned.append(value)
    return ", ".join(cleaned) if cleaned else "Not available"


def _parse_point(value: str | None) -> tuple[float | None, float | None]:
    if not value:
        return None, None

    match = re.fullmatch(
        r"Point\((-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\)",
        value.strip(),
    )
    if not match:
        return None, None

    longitude = float(match.group(1))
    latitude = float(match.group(2))
    return latitude, longitude


def _country_selector(code: str) -> str:
    code = code.lower().strip()

    if code in WIKIDATA_OVERRIDES:
        return f"VALUES ?country {{ wd:{WIKIDATA_OVERRIDES[code]} }}"

    return (
        f'?country wdt:P297 "{code.upper()}" .'
    )


def fetch_wikidata_profile(
    code: str,
    timeout: float = 12.0,
) -> dict[str, object]:
    """Fetch current political/geographic country metadata from Wikidata."""
    selector = _country_selector(code)

    query = f"""
    SELECT
      ?countryLabel
      ?continentLabel
      ?capitalLabel
      ?currencyLabel
      ?governmentLabel
      ?headOfStateLabel
      ?headOfStateOfficeLabel
      ?headOfGovernmentLabel
      ?headOfGovernmentOfficeLabel
      ?coord
    WHERE {{
      {selector}

      OPTIONAL {{ ?country wdt:P30 ?continent. }}
      OPTIONAL {{ ?country wdt:P36 ?capital. }}
      OPTIONAL {{ ?country wdt:P38 ?currency. }}
      OPTIONAL {{ ?country wdt:P122 ?government. }}
      OPTIONAL {{ ?country wdt:P1906 ?headOfStateOffice. }}
      OPTIONAL {{ ?country wdt:P1313 ?headOfGovernmentOffice. }}
      OPTIONAL {{ ?country wdt:P625 ?coord. }}

      OPTIONAL {{
        ?country p:P35 ?headOfStateStatement.
        ?headOfStateStatement ps:P35 ?headOfState.
        FILTER NOT EXISTS {{
          ?headOfStateStatement pq:P582 ?headOfStateEnd.
        }}
      }}

      OPTIONAL {{
        ?country p:P6 ?headOfGovernmentStatement.
        ?headOfGovernmentStatement ps:P6 ?headOfGovernment.
        FILTER NOT EXISTS {{
          ?headOfGovernmentStatement pq:P582 ?headOfGovernmentEnd.
        }}
      }}

      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "en,fr".
      }}
    }}
    """

    response = requests.get(
        WIKIDATA_ENDPOINT,
        params={
            "query": query,
            "format": "json",
        },
        headers={
            "Accept": "application/sparql-results+json",
            "User-Agent": (
                "country-flag-recognition/0.1 "
                "(educational portfolio project)"
            ),
        },
        timeout=timeout,
    )
    response.raise_for_status()

    bindings = (
        response.json()
        .get("results", {})
        .get("bindings", [])
    )

    if not bindings:
        raise LookupError(
            f"No Wikidata country profile found for '{code}'."
        )

    def values(field: str) -> list[str]:
        return [
            row[field]["value"]
            for row in bindings
            if field in row
            and row[field].get("value")
        ]

    latitude, longitude = _parse_point(
        next(iter(values("coord")), None)
    )

    return {
        "name": _unique_join(values("countryLabel")),
        "continent": _unique_join(values("continentLabel")),
        "capital": _unique_join(values("capitalLabel")),
        "currency": _unique_join(values("currencyLabel")),
        "government_form": _unique_join(values("governmentLabel")),
        "head_of_state": _unique_join(values("headOfStateLabel")),
        "head_of_state_office": _unique_join(
            values("headOfStateOfficeLabel")
        ),
        "head_of_government": _unique_join(
            values("headOfGovernmentLabel")
        ),
        "head_of_government_office": _unique_join(
            values("headOfGovernmentOfficeLabel")
        ),
        "latitude": latitude,
        "longitude": longitude,
    }


def fetch_latest_population(
    code: str,
    timeout: float = 12.0,
) -> PopulationRecord:
    """Fetch the most recent non-empty World Bank population observation."""
    lookup_code = WORLD_BANK_OVERRIDES.get(
        code.lower().strip(),
        code.upper().strip(),
    )

    response = requests.get(
        (
            f"{WORLD_BANK_BASE}/country/{lookup_code}"
            "/indicator/SP.POP.TOTL"
        ),
        params={
            "format": "json",
            "mrnev": 1,
            "per_page": 1,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()

    if (
        not isinstance(payload, list)
        or len(payload) < 2
        or not payload[1]
    ):
        return PopulationRecord(
            value=None,
            year=None,
            source="World Bank",
        )

    observation = payload[1][0]
    value = observation.get("value")
    year = observation.get("date")

    return PopulationRecord(
        value=int(value) if value is not None else None,
        year=str(year) if year is not None else None,
        source="World Bank",
    )


def fetch_country_profile(
    code: str,
    timeout: float = 12.0,
) -> CountryProfile:
    """Combine live Wikidata metadata with latest World Bank population."""
    wikidata = fetch_wikidata_profile(
        code,
        timeout=timeout,
    )

    try:
        population = fetch_latest_population(
            code,
            timeout=timeout,
        )
    except (requests.RequestException, ValueError):
        population = PopulationRecord(
            value=None,
            year=None,
            source="World Bank",
        )

    return CountryProfile(
        code=code.lower(),
        name=str(wikidata["name"]),
        continent=str(wikidata["continent"]),
        capital=str(wikidata["capital"]),
        currency=str(wikidata["currency"]),
        government_form=str(wikidata["government_form"]),
        head_of_state=str(wikidata["head_of_state"]),
        head_of_state_office=str(wikidata["head_of_state_office"]),
        head_of_government=str(wikidata["head_of_government"]),
        head_of_government_office=str(
            wikidata["head_of_government_office"]
        ),
        latitude=wikidata["latitude"],
        longitude=wikidata["longitude"],
        population=population,
        political_source="Wikidata",
        population_source=population.source,
    )
