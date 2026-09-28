"""Live country enrichment for the Streamlit dashboard."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable
from urllib.parse import quote, unquote, urlparse

import requests
import pycountry


WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
WORLD_BANK_BASE = "https://api.worldbank.org/v2"
REST_COUNTRIES_BASE = "https://restcountries.com/v3.1"
EMERGENCY_NUMBERS_DATA_URL = (
    "https://raw.githubusercontent.com/"
    "EmergencyNumberAPI/data/master/data.json"
)
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

# Verified fallbacks for fields that are inconsistently exposed by Wikidata.
# These values are used only when the live source is missing or ambiguous.
COUNTRY_PROFILE_OVERRIDES = {
    "fr": {
        "currency": "Euro (EUR)",
        "national_day": "July 14",
        "independence_day": "Not applicable",
        "national_motto": "Liberté, Égalité, Fraternité",
        "national_anthem": "La Marseillaise",
    },
    "tl": {
        "national_day": "May 20",
        "independence_day": "May 20, 2002",
    },
    "gb": {
        "national_day": "No single official national day",
        "independence_day": "Not applicable",
    },
    "ci": {
        "emergency_numbers": (
            "Police: 170 | Fire: 180 | Ambulance (SAMU): 185"
        ),
        "national_day": "August 7",
        "independence_day": "August 7, 1960",
        "former_colonial_powers": "France",
        "colonial_period": (
            "Official French colony from 1893; "
            "part of French West Africa from 1904"
        ),
        "independence_leader": "Félix Houphouët-Boigny",
        "historical_context": (
            "France formally established Côte d'Ivoire as a colony in 1893. "
            "The territory became part of French West Africa in 1904. "
            "Côte d'Ivoire became independent from France on August 7, 1960, "
            "with Félix Houphouët-Boigny as the central independence-era "
            "political leader and first president."
        ),
        "national_motto": "Union – Discipline – Travail",
    },
}


@dataclass(frozen=True)
class PopulationRecord:
    value: int | None
    year: str | None
    source: str


@dataclass(frozen=True)
class GDPRecord:
    value_usd: float | None
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
    emergency_numbers: str
    internet_domain: str
    driving_side: str
    latitude: float | None
    longitude: float | None
    population: PopulationRecord
    overview: str
    national_day: str
    independence_day: str
    colonial_history: str
    former_colonial_powers: str
    colonial_period: str
    independence_leader: str
    historical_context: str
    national_motto: str
    national_anthem: str
    region: str
    subregion: str
    demonym: str
    iso_alpha3: str
    timezones: str
    borders: str
    largest_cities: str
    international_organizations: str
    official_religion: str
    highest_point: str
    lowest_point: str
    gdp: GDPRecord
    political_source: str
    population_source: str
    overview_source: str


def _is_raw_wikidata_identifier(value: str) -> bool:
    """Return True for unresolved Wikidata entity identifiers."""
    stripped = value.strip()
    return bool(
        re.fullmatch(r"Q\d+", stripped)
        or re.fullmatch(
            r"https?://www\.wikidata\.org/entity/Q\d+",
            stripped,
        )
    )


def _unique_join(values: Iterable[str]) -> str:
    """Join distinct human-readable values and discard unresolved IDs."""
    cleaned = []

    for value in values:
        value = str(value).strip()

        if (
            value
            and not _is_raw_wikidata_identifier(value)
            and value not in cleaned
        ):
            cleaned.append(value)

    return (
        ", ".join(cleaned)
        if cleaned
        else "Not available"
    )


def _preferred_wikidata_literal(
    bindings: list[dict[str, object]],
    field: str,
    preferred_languages: tuple[str, ...] = ("en", "fr"),
) -> str:
    """Select one readable Wikidata literal instead of concatenating translations."""
    candidates: list[tuple[str, str]] = []

    for row in bindings:
        raw = row.get(field)
        if not isinstance(raw, dict):
            continue

        value = str(raw.get("value", "")).strip()
        if not value or _is_raw_wikidata_identifier(value):
            continue

        language = str(
            raw.get("xml:lang")
            or raw.get("lang")
            or ""
        ).lower()

        pair = (language, value)
        if pair not in candidates:
            candidates.append(pair)

    if not candidates:
        return "Not available"

    # Prefer one canonical translation instead of returning every language.
    for language in preferred_languages:
        for candidate_language, value in candidates:
            if candidate_language == language:
                return value

    # Prefer a Latin-script value for the PDF/report when no EN/FR label exists.
    latin_pattern = re.compile(
        r"^[\x00-\x7FÀ-ÖØ-öø-ÿĀ-ž’'“”–—·.,;:!?()\-\s]+$"
    )
    for _, value in candidates:
        if latin_pattern.fullmatch(value):
            return value

    # Last resort: exactly one value, never the multilingual concatenation.
    return candidates[0][1]


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
      ?nationalMotto
      ?nationalAnthemLabel
      ?officialReligionLabel
      ?highestPointLabel
      ?lowestPointLabel
      ?organizationLabel
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
      OPTIONAL {{ ?country wdt:P1451 ?nationalMotto. }}
      OPTIONAL {{ ?country wdt:P85 ?nationalAnthem. }}
      OPTIONAL {{ ?country wdt:P3075 ?officialReligion. }}
      OPTIONAL {{ ?country wdt:P610 ?highestPoint. }}
      OPTIONAL {{ ?country wdt:P1589 ?lowestPoint. }}
      OPTIONAL {{ ?country wdt:P463 ?organization. }}
      OPTIONAL {{ ?country wdt:P1906 ?headOfStateOffice. }}
      OPTIONAL {{ ?country wdt:P1313 ?headOfGovernmentOffice. }}
      OPTIONAL {{ ?country wdt:P625 ?coord. }}

      OPTIONAL {{
        ?article schema:about ?country;
                 schema:isPartOf <https://en.wikipedia.org/>.
      }}

      OPTIONAL {{ ?country wdt:P35 ?headOfState. }}
      OPTIONAL {{ ?country wdt:P6 ?headOfGovernment. }}

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
        "national_motto": _preferred_wikidata_literal(
            bindings,
            "nationalMotto",
        ),
        "national_anthem": _preferred_wikidata_literal(
            bindings,
            "nationalAnthemLabel",
        ),
        "official_religion": _unique_join(
            values(
                "officialReligionLabel"
            )
        ),
        "highest_point": _unique_join(
            values(
                "highestPointLabel"
            )
        ),
        "lowest_point": _unique_join(
            values(
                "lowestPointLabel"
            )
        ),
        "international_organizations": _unique_join(
            values(
                "organizationLabel"
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
        return "Not available"

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
        ?country wdt:P832 ?nationalDay.
        ?nationalDay wdt:P31/wdt:P279* wd:Q57598.
        OPTIONAL {{ ?nationalDay wdt:P837 ?nationalDayDate. }}
      }}

      OPTIONAL {{
        {{
          ?country wdt:P832 ?independenceDay.
        }}
        UNION
        {{
          ?independenceDay wdt:P17|wdt:P1001 ?country.
        }}
        ?independenceDay wdt:P31/wdt:P279* wd:Q14914657.
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


def fetch_rest_country_profile(
    code: str,
    timeout: float = 12.0,
) -> dict[str, str]:
    """Fetch stable geographic and identity metadata from REST Countries."""
    normalized_code = code.lower().strip()

    if normalized_code == "xk":
        return {
            "region": "Europe",
            "subregion": "Southeast Europe",
            "demonym": "Kosovan",
            "iso_alpha3": "XKX",
            "timezones": "UTC+01:00",
            "borders": "Albania, Montenegro, North Macedonia, Serbia",
        }

    response = requests.get(
        f"{REST_COUNTRIES_BASE}/alpha/{normalized_code}",
        params={
            "fields": (
                "region,subregion,demonyms,cca3,"
                "timezones,borders"
            )
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()

    if isinstance(payload, list):
        item = payload[0] if payload else {}
    elif isinstance(payload, dict):
        item = payload
    else:
        item = {}

    demonyms = item.get("demonyms") or {}
    eng_demonym = demonyms.get("eng") or {}
    demonym = (
        eng_demonym.get("m")
        or eng_demonym.get("f")
        or "Not available"
    )

    border_codes = item.get("borders") or []
    border_names = []
    for border_code in border_codes:
        country = pycountry.countries.get(alpha_3=str(border_code))
        border_names.append(
            country.name if country is not None else str(border_code)
        )

    return {
        "region": str(item.get("region") or "Not available"),
        "subregion": str(item.get("subregion") or "Not available"),
        "demonym": str(demonym),
        "iso_alpha3": str(item.get("cca3") or "Not available"),
        "timezones": _unique_join(item.get("timezones") or []),
        "borders": _unique_join(border_names),
    }


def fetch_latest_gdp(
    code: str,
    timeout: float = 12.0,
) -> GDPRecord:
    """Fetch latest nominal GDP (current US$) from the World Bank."""
    normalized_code = code.lower().strip()

    if normalized_code in WORLD_BANK_OVERRIDES:
        lookup_code = WORLD_BANK_OVERRIDES[normalized_code]
    else:
        country = pycountry.countries.get(alpha_2=normalized_code.upper())
        lookup_code = (
            country.alpha_3
            if country is not None
            else normalized_code.upper()
        )

    response = requests.get(
        (
            f"{WORLD_BANK_BASE}/country/{lookup_code}"
            "/indicator/NY.GDP.MKTP.CD"
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
        return GDPRecord(
            value_usd=None,
            year=None,
            source="World Bank",
        )

    observation = payload[1][0]
    value = observation.get("value")
    year = observation.get("date")

    return GDPRecord(
        value_usd=float(value) if value is not None else None,
        year=str(year) if year is not None else None,
        source="World Bank",
    )


def fetch_largest_cities(
    code: str,
    timeout: float = 12.0,
) -> str:
    """Fetch up to five largest populated cities from Wikidata."""
    selector = _country_selector(code)

    query = f"""
    SELECT ?cityLabel ?population WHERE {{
      {selector}
      ?city wdt:P17 ?country;
            wdt:P31/wdt:P279* wd:Q515;
            wdt:P1082 ?population.
      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "en,fr".
      }}
    }}
    ORDER BY DESC(?population)
    LIMIT 5
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

    cities = [
        row["cityLabel"]["value"]
        for row in bindings
        if row.get("cityLabel", {}).get("value")
    ]
    return _unique_join(cities)


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


def fetch_wikipedia_history_text(
    title: str | None,
    timeout: float = 12.0,
) -> str:
    """Fetch broad Wikipedia article text for historical extraction."""
    if not title:
        return "Not available"

    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "redirects": 1,
            "titles": title,
            "format": "json",
            "formatversion": 2,
        },
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

    pages = (
        response.json()
        .get("query", {})
        .get("pages", [])
    )

    if not pages:
        return "Not available"

    extract = str(
        pages[0].get("extract", "")
    ).strip()

    if not extract:
        return "Not available"

    # Keep enough context for colonial/independence extraction while
    # avoiding unnecessarily huge payloads.
    return extract[:50000]


def fetch_wikidata_population(
    code: str,
    timeout: float = 12.0,
) -> PopulationRecord:
    """Fetch the latest dated Wikidata population as a secondary fallback."""
    selector = _country_selector(code)

    query = f"""
    SELECT ?population ?date WHERE {{
      {selector}
      ?country p:P1082 ?populationStatement.
      ?populationStatement ps:P1082 ?population.
      OPTIONAL {{ ?populationStatement pq:P585 ?date. }}
      FILTER NOT EXISTS {{
        ?populationStatement wikibase:rank wikibase:DeprecatedRank.
      }}
    }}
    ORDER BY DESC(?date)
    LIMIT 1
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

    if not bindings:
        return PopulationRecord(
            value=None,
            year=None,
            source="Wikidata",
        )

    row = bindings[0]
    raw_value = row.get("population", {}).get("value")
    raw_date = row.get("date", {}).get("value")

    try:
        value = int(float(raw_value)) if raw_value is not None else None
    except (TypeError, ValueError):
        value = None

    year = None
    if raw_date:
        match = re.match(r"^([+-]?\d{4,})-", str(raw_date))
        if match:
            year = match.group(1)

    return PopulationRecord(
        value=value,
        year=year,
        source="Wikidata",
    )


def fetch_latest_population(
    code: str,
    timeout: float = 12.0,
) -> PopulationRecord:
    """Fetch the most recent non-empty World Bank population observation."""
    normalized_code = code.lower().strip()

    if normalized_code in WORLD_BANK_OVERRIDES:
        lookup_code = WORLD_BANK_OVERRIDES[normalized_code]
    else:
        country = pycountry.countries.get(
            alpha_2=normalized_code.upper()
        )
        lookup_code = (
            country.alpha_3
            if country is not None
            else normalized_code.upper()
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


COLONIAL_POWER_PATTERNS = {
    "British": "United Kingdom",
    "French": "France",
    "Portuguese": "Portugal",
    "Spanish": "Spain",
    "Belgian": "Belgium",
    "Dutch": "Netherlands",
    "German": "Germany",
    "Italian": "Italy",
    "Danish": "Denmark",
    "Swedish": "Sweden",
    "Norwegian": "Norway",
    "Russian": "Russia",
    "Soviet": "Soviet Union",
    "Ottoman": "Ottoman Empire",
    "Japanese": "Japan",
    "American": "United States",
    "Australian": "Australia",
    "New Zealand": "New Zealand",
    "South African": "South Africa",
}


def _extract_colonial_history(
    overview: str,
    independence_day: str,
) -> str:
    """Extract concise colonial-independence context from the country overview."""
    if not overview or overview == "Not available":
        return "Not applicable"

    text = " ".join(overview.split())
    lower = text.lower()

    colonial_powers: list[str] = []

    # Explicit independence wording.
    explicit_patterns = [
        r"(?:gained|achieved|declared|won|obtained) independence from "
        r"([A-Z][A-Za-z .'-]+?)(?: on| in|,|\.|;)",
        r"became independent from "
        r"([A-Z][A-Za-z .'-]+?)(?: on| in|,|\.|;)",
    ]

    for pattern in explicit_patterns:
        for match in re.finditer(pattern, text):
            value = match.group(1).strip()
            if value and value not in colonial_powers:
                colonial_powers.append(value)

    # Colonial-adjective wording covers introductions such as
    # "a British colony/protectorate" even when the independence sentence
    # omits the former power.
    for adjective, country in COLONIAL_POWER_PATTERNS.items():
        if re.search(
            rf"\b{adjective.lower()}\b.*\b"
            r"(?:colony|protectorate|territory|rule|administration)\b",
            lower,
        ):
            if country not in colonial_powers:
                colonial_powers.append(country)

    independence_year = None

    if independence_day != "Not available":
        year_match = re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", independence_day)
        if year_match:
            independence_year = year_match.group(1)

    if independence_year is None:
        year_patterns = [
            r"(?:independence|independent)[^\.]{0,80}\b"
            r"(1[5-9]\d{2}|20\d{2})\b",
            r"\b(1[5-9]\d{2}|20\d{2})\b[^\.]{0,80}"
            r"(?:independence|independent)",
        ]
        for pattern in year_patterns:
            match = re.search(pattern, lower)
            if match:
                independence_year = match.group(1)
                break

    if not colonial_powers:
        return "Not applicable"

    power_text = ", ".join(colonial_powers)

    if independence_year:
        return (
            f"{independence_year}; former colonial power: "
            f"{power_text}"
        )

    return f"Former colonial power: {power_text}"


def _extract_structured_history(
    overview: str,
    independence_day: str,
) -> dict[str, str]:
    """Extract concise colonial and sovereignty facts without inventing data."""
    text = " ".join(str(overview or "").split())
    lower = text.lower()

    colonial_powers: list[str] = []

    for adjective, country in COLONIAL_POWER_PATTERNS.items():
        if re.search(
            rf"\b{re.escape(adjective.lower())}\b[^.]{{0,120}}\b"
            r"(?:colony|colonial|protectorate|territory|rule|administration|sovereignty)\b",
            lower,
        ):
            if country not in colonial_powers:
                colonial_powers.append(country)

    explicit_power_patterns = [
        r"(?:independence|independent) from ([A-Z][A-Za-z .'-]+?)(?: on| in|,|\.|;)",
        r"(?:colony|protectorate) of ([A-Z][A-Za-z .'-]+?)(?: from| in|,|\.|;)",
        r"under ([A-Z][A-Za-z .'-]+?) sovereignty",
    ]

    for pattern in explicit_power_patterns:
        for match in re.finditer(pattern, text):
            value = match.group(1).strip()
            if value and value not in colonial_powers:
                colonial_powers.append(value)

    colonial_period = "Not applicable"

    period_patterns = [
        r"officially became (?:a|an) ([^.]{1,90}?(?:colony|protectorate)) in (\d{4})",
        r"became (?:a|an) ([^.]{1,90}?(?:colony|protectorate)) in (\d{4})",
        r"was (?:a|an) ([^.]{1,90}?(?:colony|protectorate)) from (\d{4}) to (\d{4})",
        r"from (\d{4}) to (\d{4}),? [^.]{0,90}(?:colony|protectorate|mandate|territory)",
        r"(?:colonized|colonised|annexed|occupied) by ([A-Z][A-Za-z .'-]+?) in (\d{4})",
        r"became part of ([A-Z][A-Za-z .'-]+?) in (\d{4})",
        r"(?:protectorate|colony|mandate|territory) of ([A-Z][A-Za-z .'-]+?) from (\d{4})",
    ]

    for pattern in period_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            values = [str(value).strip() for value in match.groups()]
            colonial_period = " ".join(values)
            break

    if colonial_period == "Not applicable" and colonial_powers:
        year_match = re.search(
            r"(?:colony|protectorate|colonial)[^.]{0,100}\b(1[6-9]\d{2}|20\d{2})\b",
            lower,
        )
        if year_match:
            colonial_period = (
                f"Colonial rule documented by {year_match.group(1)}"
            )
        else:
            colonial_period = "Colonial rule documented"

    leader = "Not applicable"

    leader_patterns = [
        r"([A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’.-]+(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’.-]+){1,4}) "
        r"(?:became|was elected|served as) [^.]{0,50}first president",
        r"first president(?:,| was)? "
        r"([A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’.-]+(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’.-]+){1,4})",
    ]

    for pattern in leader_patterns:
        match = re.search(pattern, text)
        if match:
            leader = match.group(1).strip()
            break

    context_sentences: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        sentence_lower = sentence.lower()
        if any(
            keyword in sentence_lower
            for keyword in (
                "colony",
                "colonial",
                "protectorate",
                "independence",
                "independent",
                "sovereignty",
                "self-government",
            )
        ):
            cleaned = sentence.strip()
            if cleaned and cleaned not in context_sentences:
                context_sentences.append(cleaned)
        if len(context_sentences) >= 3:
            break

    historical_context = (
        " ".join(context_sentences)
        if context_sentences
        else (
            "No classical colonial-independence transition is documented "
            "in the available country overview."
        )
    )

    return {
        "former_colonial_powers": (
            ", ".join(colonial_powers)
            if colonial_powers
            else "Not applicable"
        ),
        "colonial_period": colonial_period,
        "independence_leader": leader,
        "historical_context": historical_context,
        "independence_day": (
            independence_day
            if independence_day != "Not available"
            else "Not applicable"
        ),
    }


MONTH_NAMES = {
    "january": "January",
    "february": "February",
    "march": "March",
    "april": "April",
    "may": "May",
    "june": "June",
    "july": "July",
    "august": "August",
    "september": "September",
    "october": "October",
    "november": "November",
    "december": "December",
}


def _national_day_from_independence(
    independence_day: str,
) -> str:
    """Convert a dated independence value into a reusable national-day date."""
    if (
        not independence_day
        or independence_day in ("Not available", "Not applicable")
    ):
        return "Not available"

    match = re.search(
        r"\b("
        + "|".join(MONTH_NAMES.keys())
        + r")\s+(\d{1,2})\b",
        independence_day.lower(),
    )
    if not match:
        return "Not available"

    month = MONTH_NAMES[match.group(1)]
    day = int(match.group(2))
    return f"{month} {day}"


def _extract_national_day_from_overview(
    overview: str,
) -> str:
    """Infer the country's principal national-day date from encyclopedic text."""
    if not overview or overview == "Not available":
        return "Not available"

    text = " ".join(overview.split())
    month_pattern = (
        r"(January|February|March|April|May|June|July|August|"
        r"September|October|November|December)"
    )

    patterns = [
        rf"(?:independence|independent)[^.]{0,120}?"
        rf"{month_pattern}\s+(\d{{1,2}})",
        rf"{month_pattern}\s+(\d{{1,2}})[^.]{0,120}?"
        r"(?:independence|independent)",
        rf"(?:national day|republic day|constitution day)[^.]{0,100}?"
        rf"{month_pattern}\s+(\d{{1,2}})",
        rf"{month_pattern}\s+(\d{{1,2}})[^.]{0,100}?"
        r"(?:national day|republic day|constitution day)",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
        if not match:
            continue

        groups = match.groups()
        month = None
        day = None

        for value in groups:
            if value is None:
                continue
            if value.lower() in MONTH_NAMES:
                month = MONTH_NAMES[value.lower()]
            elif value.isdigit():
                day = int(value)

        if month and day:
            return f"{month} {day}"

    return "Not available"


def _normalize_emergency_number_label(
    value: str,
    calling_code: str | None = None,
) -> str:
    """Normalize emergency numbers to local dialing form."""
    normalized = str(value).strip()
    if not normalized:
        return normalized

    digits = re.sub(r"[^0-9+]", "", normalized)

    if calling_code:
        prefix = re.sub(r"\D", "", str(calling_code))
        if prefix:
            for candidate in (
                f"+{prefix}",
                f"00{prefix}",
                prefix,
            ):
                if digits.startswith(candidate):
                    digits = digits[len(candidate):]
                    break

    return digits.lstrip("0") or digits


def fetch_emergency_numbers(
    code: str,
    timeout: float = 12.0,
) -> str:
    """Fetch emergency telephone numbers from Wikidata."""
    selector = _country_selector(code)

    query = f"""
    SELECT DISTINCT
      ?emergencyNumber
      ?emergencyNumberLabel
      ?telephone
    WHERE {{
      {selector}
      ?country wdt:P2852 ?emergencyNumber.
      OPTIONAL {{
        ?emergencyNumber wdt:P1329 ?telephone.
      }}
      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "en".
      }}
    }}
    ORDER BY ?telephone ?emergencyNumberLabel
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

    numbers: list[str] = []

    for row in bindings:
        telephone = (
            row.get("telephone", {})
            .get("value", "")
            .strip()
        )
        label = (
            row.get("emergencyNumberLabel", {})
            .get("value", "")
            .strip()
        )

        value = telephone or label

        # Avoid exposing raw Wikidata entity identifiers/URLs.
        if (
            not value
            or value.startswith("http://")
            or value.startswith("https://")
            or re.fullmatch(r"Q\d+", value)
        ):
            continue

        if value not in numbers:
            numbers.append(value)

    return (
        ", ".join(numbers)
        if numbers
        else "Not available"
    )


def _clean_emergency_values(
    section: object,
) -> list[str]:
    """Flatten emergency-number section values while removing empty entries."""
    if not isinstance(section, dict):
        return []

    values: list[str] = []

    for raw in section.values():
        if not isinstance(raw, list):
            continue

        for item in raw:
            if item is None:
                continue

            value = str(item).strip()
            if not value:
                continue

            if value not in values:
                values.append(value)

    return values


def fetch_emergency_numbers_fallback(
    code: str,
    timeout: float = 12.0,
) -> str:
    """Fetch emergency numbers by ISO code from EmergencyNumberAPI data."""
    normalized = code.upper().strip()

    response = requests.get(
        EMERGENCY_NUMBERS_DATA_URL,
        headers={
            "User-Agent": (
                "country-flag-recognition/0.1 "
                "(educational portfolio project)"
            ),
        },
        timeout=timeout,
    )
    response.raise_for_status()

    payload = response.json()
    records = (
        payload.get("data", [])
        if isinstance(payload, dict)
        else []
    )

    record = next(
        (
            item
            for item in records
            if str(
                (item.get("Country") or {}).get("ISOCode", "")
            ).upper() == normalized
        ),
        None,
    )

    if not isinstance(record, dict):
        return "No national emergency number documented"

    dispatch = _clean_emergency_values(
        record.get("Dispatch")
    )
    police = _clean_emergency_values(
        record.get("Police")
    )
    ambulance = _clean_emergency_values(
        record.get("Ambulance")
    )
    fire = _clean_emergency_values(
        record.get("Fire")
    )

    # 112 membership is meaningful even when category-specific fields
    # are absent in the source record.
    if (
        not dispatch
        and bool(record.get("Member_112"))
    ):
        dispatch = ["112"]

    parts: list[str] = []

    if dispatch:
        parts.append(
            "General: " + ", ".join(dispatch)
        )
    if police:
        parts.append(
            "Police: " + ", ".join(police)
        )
    if ambulance:
        parts.append(
            "Ambulance: " + ", ".join(ambulance)
        )
    if fire:
        parts.append(
            "Fire: " + ", ".join(fire)
        )

    if parts:
        return " | ".join(parts)

    if bool(record.get("LocalOnly")):
        return (
            "Local emergency services only; "
            "no single national number documented"
        )

    if bool(record.get("NoData")):
        return "No national emergency number documented"

    return "No national emergency number documented"


def format_emergency_numbers(
    emergency_numbers: str,
    calling_code: str,
) -> str:
    """Remove country calling code repetition and keep service labels."""
    value = str(emergency_numbers).strip()
    if not value or value == "Not available":
        return value

    parts = [
        part.strip()
        for part in value.split("|")
        if part.strip()
    ]

    formatted: list[str] = []

    for part in parts:
        if ":" in part:
            label, raw_numbers = part.split(":", 1)
            cleaned_numbers: list[str] = []

            for raw in raw_numbers.split(","):
                cleaned = _normalize_emergency_number_label(
                    raw,
                    calling_code,
                )
                if cleaned and cleaned not in cleaned_numbers:
                    cleaned_numbers.append(cleaned)

            if cleaned_numbers:
                formatted.append(
                    f"{label.strip()}: {', '.join(cleaned_numbers)}"
                )
        else:
            cleaned = _normalize_emergency_number_label(
                part,
                calling_code,
            )
            if cleaned:
                formatted.append(cleaned)

    return " | ".join(formatted) if formatted else value


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
        population = fetch_latest_population(
            code,
            timeout=timeout,
        )
    except (
        requests.RequestException,
        ValueError,
    ):
        population = PopulationRecord(
            value=None,
            year=None,
            source="World Bank",
        )

    if population.value is None:
        try:
            population = fetch_wikidata_population(
                code,
                timeout=timeout,
            )
        except (
            requests.RequestException,
            ValueError,
            LookupError,
        ):
            pass

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
        supplemental = fetch_rest_country_profile(
            code,
            timeout=timeout,
        )
    except (
        requests.RequestException,
        ValueError,
        LookupError,
    ):
        supplemental = {
            "region": "Not available",
            "subregion": "Not available",
            "demonym": "Not available",
            "iso_alpha3": "Not available",
            "timezones": "Not available",
            "borders": "Not available",
        }

    try:
        gdp = fetch_latest_gdp(
            code,
            timeout=timeout,
        )
    except (
        requests.RequestException,
        ValueError,
    ):
        gdp = GDPRecord(
            value_usd=None,
            year=None,
            source="World Bank",
        )

    try:
        largest_cities = fetch_largest_cities(
            code,
            timeout=timeout,
        )
    except (
        requests.RequestException,
        ValueError,
        LookupError,
    ):
        largest_cities = "Not available"

    try:
        emergency_numbers = fetch_emergency_numbers(
            code,
            timeout=timeout,
        )
    except (
        requests.RequestException,
        ValueError,
        LookupError,
    ):
        emergency_numbers = "Not available"

    if emergency_numbers == "Not available":
        try:
            emergency_numbers = fetch_emergency_numbers_fallback(
                code,
                timeout=timeout,
            )
        except (
            requests.RequestException,
            ValueError,
            LookupError,
        ):
            emergency_numbers = (
                "No national emergency number documented"
            )

    emergency_override = str(
        COUNTRY_PROFILE_OVERRIDES.get(
            code.lower().strip(),
            {},
        ).get(
            "emergency_numbers",
            "",
        )
    ).strip()
    if emergency_override:
        emergency_numbers = emergency_override

    wikipedia_title = (
        str(wikidata.get("wikipedia_title"))
        if wikidata.get("wikipedia_title")
        else None
    )

    try:
        overview = fetch_wikipedia_overview(
            wikipedia_title,
            timeout=timeout,
        )
    except (
        requests.RequestException,
        ValueError,
    ):
        overview = "Not available"

    try:
        history_text = fetch_wikipedia_history_text(
            wikipedia_title,
            timeout=timeout,
        )
    except (
        requests.RequestException,
        ValueError,
    ):
        history_text = overview

    normalized_code = code.lower().strip()
    overrides = COUNTRY_PROFILE_OVERRIDES.get(
        normalized_code,
        {},
    )

    currency = str(wikidata["currency"])
    if (
        normalized_code in COUNTRY_PROFILE_OVERRIDES
        or re.search(r"\bQ\d+\b", currency)
    ):
        currency = str(
            overrides.get(
                "currency",
                currency,
            )
        )

    colonial_history = _extract_colonial_history(
        history_text,
        str(country_dates["independence_day"]),
    )

    independence_day = str(
        overrides.get(
            "independence_day",
            country_dates["independence_day"],
        )
    )

    national_day = str(
        overrides.get(
            "national_day",
            country_dates["national_day"],
        )
    )

    if national_day == "Not available":
        national_day = _national_day_from_independence(
            independence_day
        )

    if national_day == "Not available":
        national_day = _extract_national_day_from_overview(
            history_text
        )

    if national_day == "Not available":
        # Some states genuinely do not designate one unique national day.
        national_day = "No single official national day documented"

    structured_history = _extract_structured_history(
        history_text,
        independence_day,
    )

    former_colonial_powers = str(
        overrides.get(
            "former_colonial_powers",
            structured_history["former_colonial_powers"],
        )
    )
    colonial_period = str(
        overrides.get(
            "colonial_period",
            structured_history["colonial_period"],
        )
    )
    independence_leader = str(
        overrides.get(
            "independence_leader",
            structured_history["independence_leader"],
        )
    )
    historical_context = str(
        overrides.get(
            "historical_context",
            structured_history["historical_context"],
        )
    )

    # Historical coherence guard: a verified sovereignty date/power must
    # never coexist with generic "not applicable" placeholders.
    if (
        independence_day not in ("Not available", "Not applicable")
        and national_day in (
            "Not available",
            "No single official national day documented",
        )
    ):
        inferred_day = _national_day_from_independence(
            independence_day
        )
        if inferred_day != "Not available":
            national_day = inferred_day

    if (
        former_colonial_powers not in ("Not available", "Not applicable")
        and colonial_period == "Not applicable"
    ):
        colonial_period = "Colonial rule documented"

    if (
        independence_day not in ("Not available", "Not applicable")
        and historical_context.startswith(
            "No classical colonial-independence transition"
        )
    ):
        historical_context = (
            f"Sovereignty/independence documented on {independence_day}."
        )

    national_motto = str(
        overrides.get(
            "national_motto",
            wikidata["national_motto"],
        )
    )

    national_anthem = str(
        overrides.get(
            "national_anthem",
            wikidata["national_anthem"],
        )
    )

    country_record = pycountry.countries.get(
        alpha_2=normalized_code.upper()
    )

    name = str(wikidata["name"])
    if name == "Not available":
        name = (
            str(country_record.name)
            if country_record is not None
            else normalized_code.upper()
        )

    internet_domain = str(wikidata["internet_domain"])
    if internet_domain == "Not available":
        internet_domain = f".{normalized_code}"

    emergency_numbers = format_emergency_numbers(
        emergency_numbers,
        str(wikidata["calling_code"]),
    )

    return CountryProfile(
        code=normalized_code,
        name=name,
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
        currency=currency,
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
        emergency_numbers=emergency_numbers,
        internet_domain=internet_domain,
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
        national_day=national_day,
        independence_day=independence_day,
        colonial_history=colonial_history,
        former_colonial_powers=former_colonial_powers,
        colonial_period=colonial_period,
        independence_leader=independence_leader,
        historical_context=historical_context,
        national_motto=national_motto,
        national_anthem=national_anthem,
        region=str(supplemental["region"]),
        subregion=str(supplemental["subregion"]),
        demonym=str(supplemental["demonym"]),
        iso_alpha3=str(supplemental["iso_alpha3"]),
        timezones=str(supplemental["timezones"]),
        borders=str(supplemental["borders"]),
        largest_cities=largest_cities,
        international_organizations=str(
            wikidata["international_organizations"]
        ),
        official_religion=str(
            wikidata["official_religion"]
        ),
        highest_point=str(
            wikidata["highest_point"]
        ),
        lowest_point=str(
            wikidata["lowest_point"]
        ),
        gdp=gdp,
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
