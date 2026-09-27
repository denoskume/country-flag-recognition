"""Live country enrichment for the Streamlit dashboard."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable
from urllib.parse import quote, unquote, urlparse

import requests


WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
WORLD_BANK_BASE = "https://api.worldbank.org/v2"
WIKIPEDIA_SUMMARY_BASE = (
    "https://en.wikipedia.org/api/rest_v1/page/summary"
)

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
    official_languages: str
    head_of_state: str
    head_of_state_office: str
    head_of_government: str
    head_of_government_office: str
    area_km2: float | None
    calling_code: str
    internet_domain: str
    driving_side: str
    latitude: float | None
    longitude: float | None
    population: PopulationRecord
    overview: str
    national_day: str
    independence_day: str
    political_source: str
    population_source: str
    overview_source: str


def _unique_join(values: Iterable[str]) -> str:
    cleaned = []

    for value in values:
        value = str(value).strip()

        if value and value not in cleaned:
            cleaned.append(value)

    return (
        ", ".join(cleaned)
        if cleaned
        else "Not available"
    )


def _first_float(
    values: Iterable[str],
) -> float | None:
    for value in values:
        try:
            return float(value)
        except (TypeError, ValueError):
            continue

    return None


def _parse_point(
    value: str | None,
) -> tuple[
    float | None,
    float | None,
]:
    if not value:
        return None, None

    match = re.fullmatch(
        r"Point\((-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\)",
        value.strip(),
    )

    if not match:
        return None, None

    longitude = float(
        match.group(1)
    )
    latitude = float(
        match.group(2)
    )

    return latitude, longitude


def _wikipedia_title_from_url(
    url: str | None,
) -> str | None:
    if not url:
        return None

    path = urlparse(
        url
    ).path

    if "/wiki/" not in path:
        return None

    return unquote(
        path.split(
            "/wiki/",
            1,
        )[1]
    )


def _country_selector(
    code: str,
) -> str:
    code = (
        code.lower().strip()
    )

    if code in WIKIDATA_OVERRIDES:
        return (
            "VALUES ?country "
            f"{{ wd:{WIKIDATA_OVERRIDES[code]} }}"
        )

    return (
        f'?country wdt:P297 "{code.upper()}" .'
    )


def fetch_wikidata_profile(
    code: str,
    timeout: float = 12.0,
) -> dict[str, object]:
    """Fetch political, geographic and practical country metadata."""
    selector = (
        _country_selector(
            code
        )
    )

    query = f"""
    SELECT
      ?countryLabel
      ?continentLabel
      ?capitalLabel
      ?currencyLabel
      ?governmentLabel
      ?officialLanguageLabel
      ?headOfStateLabel
      ?headOfStateOfficeLabel
      ?headOfGovernmentLabel
      ?headOfGovernmentOfficeLabel
      ?area
      ?callingCode
      ?internetDomainLabel
      ?drivingSideLabel
      ?coord
      ?article
    WHERE {{
      {selector}

      OPTIONAL {{ ?country wdt:P30 ?continent. }}
      OPTIONAL {{ ?country wdt:P36 ?capital. }}
      OPTIONAL {{ ?country wdt:P38 ?currency. }}
      OPTIONAL {{ ?country wdt:P122 ?government. }}
      OPTIONAL {{ ?country wdt:P37 ?officialLanguage. }}
      OPTIONAL {{ ?country wdt:P2046 ?area. }}
      OPTIONAL {{ ?country wdt:P474 ?callingCode. }}
      OPTIONAL {{ ?country wdt:P78 ?internetDomain. }}
      OPTIONAL {{ ?country wdt:P1622 ?drivingSide. }}
      OPTIONAL {{ ?country wdt:P1906 ?headOfStateOffice. }}
      OPTIONAL {{ ?country wdt:P1313 ?headOfGovernmentOffice. }}
      OPTIONAL {{ ?country wdt:P625 ?coord. }}

      OPTIONAL {{
        ?article schema:about ?country;
                 schema:isPartOf <https://en.wikipedia.org/>.
      }}

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
            "Accept": (
                "application/sparql-results+json"
            ),
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
        .get(
            "results",
            {},
        )
        .get(
            "bindings",
            [],
        )
    )

    if not bindings:
        raise LookupError(
            "No Wikidata country profile "
            f"found for '{code}'."
        )

    def values(
        field: str,
    ) -> list[str]:
        return [
            row[field]["value"]
            for row in bindings
            if field in row
            and row[field].get(
                "value"
            )
        ]

    latitude, longitude = (
        _parse_point(
            next(
                iter(
                    values(
                        "coord"
                    )
                ),
                None,
            )
        )
    )

    article_url = next(
        iter(
            values(
                "article"
            )
        ),
        None,
    )

    return {
        "name": _unique_join(
            values(
                "countryLabel"
            )
        ),
        "continent": _unique_join(
            values(
                "continentLabel"
            )
        ),
        "capital": _unique_join(
            values(
                "capitalLabel"
            )
        ),
        "currency": _unique_join(
            values(
                "currencyLabel"
            )
        ),
        "government_form": _unique_join(
            values(
                "governmentLabel"
            )
        ),
        "official_languages": _unique_join(
            values(
                "officialLanguageLabel"
            )
        ),
        "head_of_state": _unique_join(
            values(
                "headOfStateLabel"
            )
        ),
        "head_of_state_office": _unique_join(
            values(
                "headOfStateOfficeLabel"
            )
        ),
        "head_of_government": _unique_join(
            values(
                "headOfGovernmentLabel"
            )
        ),
        "head_of_government_office": _unique_join(
            values(
                "headOfGovernmentOfficeLabel"
            )
        ),
        "area_km2": _first_float(
            values(
                "area"
            )
        ),
        "calling_code": _unique_join(
            values(
                "callingCode"
            )
        ),
        "internet_domain": _unique_join(
            values(
                "internetDomainLabel"
            )
        ),
        "driving_side": _unique_join(
            values(
                "drivingSideLabel"
            )
        ),
        "latitude": latitude,
        "longitude": longitude,
        "wikipedia_title": (
            _wikipedia_title_from_url(
                article_url
            )
        ),
    }


def _format_wikidata_date(value: str | None) -> str:
    if not value:
        return "Not available"

    match = re.match(
        r"^([+-]?\d{4,})-(\d{2})-(\d{2})T",
        value,
    )
    if not match:
        return value

    year, month, day = match.groups()
    months = [
        "January", "February", "March", "April",
        "May", "June", "July", "August",
        "September", "October", "November", "December",
    ]

    try:
        month_name = months[int(month) - 1]
        return f"{month_name} {int(day)}, {int(year)}"
    except (ValueError, IndexError):
        return value


def fetch_country_dates(
    code: str,
    timeout: float = 12.0,
) -> dict[str, str]:
    """Fetch national-day and independence-day dates from Wikidata."""
    selector = _country_selector(code)

    query = f"""
    SELECT
      ?nationalDayDate
      ?independenceDayDate
      ?independenceInception
    WHERE {{
      {selector}

      OPTIONAL {{
        ?nationalDay wdt:P17|wdt:P1001 ?country;
                     wdt:P31/wdt:P279* wd:Q57598;
                     wdt:P837 ?nationalDayDate.
      }}

      OPTIONAL {{
        ?independenceDay wdt:P17|wdt:P1001 ?country;
                         wdt:P31/wdt:P279* wd:Q14914657.
        OPTIONAL {{ ?independenceDay wdt:P837 ?independenceDayDate. }}
        OPTIONAL {{ ?independenceDay wdt:P571 ?independenceInception. }}
      }}
    }}
    LIMIT 50
    """

    response = requests.get(
        WIKIDATA_ENDPOINT,
        params={"query": query, "format": "json"},
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

    national_day = "Not available"
    independence_day = "Not available"

    for row in bindings:
        if (
            national_day == "Not available"
            and "nationalDayDate" in row
        ):
            national_day = _format_wikidata_date(
                row["nationalDayDate"].get("value")
            )

        if independence_day == "Not available":
            if "independenceInception" in row:
                independence_day = _format_wikidata_date(
                    row["independenceInception"].get("value")
                )
            elif "independenceDayDate" in row:
                independence_day = _format_wikidata_date(
                    row["independenceDayDate"].get("value")
                )

    return {
        "national_day": national_day,
        "independence_day": independence_day,
    }


def fetch_wikipedia_overview(
    title: str | None,
    timeout: float = 12.0,
) -> str:
    """Fetch a concise encyclopedic overview from Wikipedia."""
    if not title:
        return "Not available"

    endpoint = (
        WIKIPEDIA_SUMMARY_BASE
        + "/"
        + quote(
            title,
            safe="",
        )
    )

    response = requests.get(
        endpoint,
        headers={
            "Accept": "application/json",
            "User-Agent": (
                "country-flag-recognition/0.1 "
                "(educational portfolio project)"
            ),
        },
        timeout=timeout,
    )
    response.raise_for_status()

    payload = (
        response.json()
    )

    extract = str(
        payload.get(
            "extract",
            "",
        )
    ).strip()

    return (
        extract
        if extract
        else "Not available"
    )


def fetch_latest_population(
    code: str,
    timeout: float = 12.0,
) -> PopulationRecord:
    """Fetch the most recent non-empty World Bank population observation."""
    lookup_code = (
        WORLD_BANK_OVERRIDES.get(
            code.lower().strip(),
            code.upper().strip(),
        )
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

    payload = (
        response.json()
    )

    if (
        not isinstance(
            payload,
            list,
        )
        or len(payload) < 2
        or not payload[1]
    ):
        return PopulationRecord(
            value=None,
            year=None,
            source="World Bank",
        )

    observation = (
        payload[1][0]
    )
    value = (
        observation.get(
            "value"
        )
    )
    year = (
        observation.get(
            "date"
        )
    )

    return PopulationRecord(
        value=(
            int(value)
            if value is not None
            else None
        ),
        year=(
            str(year)
            if year is not None
            else None
        ),
        source="World Bank",
    )


def fetch_country_profile(
    code: str,
    timeout: float = 12.0,
) -> CountryProfile:
    """Combine live country facts, leadership, population and overview."""
    wikidata = (
        fetch_wikidata_profile(
            code,
            timeout=timeout,
        )
    )

    try:
        population = (
            fetch_latest_population(
                code,
                timeout=timeout,
            )
        )
    except (
        requests.RequestException,
        ValueError,
    ):
        population = (
            PopulationRecord(
                value=None,
                year=None,
                source="World Bank",
            )
        )

    try:
        country_dates = fetch_country_dates(
            code,
            timeout=timeout,
        )
    except (
        requests.RequestException,
        ValueError,
        LookupError,
    ):
        country_dates = {
            "national_day": "Not available",
            "independence_day": "Not available",
        }

    try:
        overview = (
            fetch_wikipedia_overview(
                str(
                    wikidata.get(
                        "wikipedia_title"
                    )
                )
                if wikidata.get(
                    "wikipedia_title"
                )
                else None,
                timeout=timeout,
            )
        )
    except (
        requests.RequestException,
        ValueError,
    ):
        overview = (
            "Not available"
        )

    return CountryProfile(
        code=code.lower(),
        name=str(
            wikidata[
                "name"
            ]
        ),
        continent=str(
            wikidata[
                "continent"
            ]
        ),
        capital=str(
            wikidata[
                "capital"
            ]
        ),
        currency=str(
            wikidata[
                "currency"
            ]
        ),
        government_form=str(
            wikidata[
                "government_form"
            ]
        ),
        official_languages=str(
            wikidata[
                "official_languages"
            ]
        ),
        head_of_state=str(
            wikidata[
                "head_of_state"
            ]
        ),
        head_of_state_office=str(
            wikidata[
                "head_of_state_office"
            ]
        ),
        head_of_government=str(
            wikidata[
                "head_of_government"
            ]
        ),
        head_of_government_office=str(
            wikidata[
                "head_of_government_office"
            ]
        ),
        area_km2=(
            wikidata[
                "area_km2"
            ]
        ),
        calling_code=str(
            wikidata[
                "calling_code"
            ]
        ),
        internet_domain=str(
            wikidata[
                "internet_domain"
            ]
        ),
        driving_side=str(
            wikidata[
                "driving_side"
            ]
        ),
        latitude=(
            wikidata[
                "latitude"
            ]
        ),
        longitude=(
            wikidata[
                "longitude"
            ]
        ),
        population=population,
        overview=overview,
        national_day=str(
            country_dates["national_day"]
        ),
        independence_day=str(
            country_dates["independence_day"]
        ),
        political_source=(
            "Wikidata"
        ),
        population_source=(
            population.source
        ),
        overview_source=(
            "Wikipedia"
        ),
    )
