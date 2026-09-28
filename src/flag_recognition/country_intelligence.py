"""Core data model for Flag Intelligence country knowledge.

This module intentionally separates country knowledge from flag recognition.
Recognition answers "which country is this?". Country Intelligence answers
"what verified, non-sensitive information can we teach about that country?".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable


VERIFIED = "verified"
PARTIAL = "partial"
UNVERIFIED = "unverified"
MISSING = "missing"

_ALLOWED_STATUSES = {VERIFIED, PARTIAL, UNVERIFIED, MISSING}


@dataclass(frozen=True)
class Evidence:
    """One factual value with provenance and freshness metadata."""

    value: Any
    source: str
    reference_year: str | None = None
    retrieved_at: str | None = None
    confidence: float | None = None
    status: str = VERIFIED
    source_url: str | None = None

    def __post_init__(self) -> None:
        if self.status not in _ALLOWED_STATUSES:
            raise ValueError(f"Unsupported evidence status: {self.status}")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True)
class TimelineEvent:
    """A dated or period-based historical event."""

    label: str
    period: str
    summary: str
    sources: tuple[str, ...] = ()
    source_urls: tuple[str, ...] = ()
    confidence: float | None = None


@dataclass(frozen=True)
class EmergencyContact:
    """Public emergency number and its exact purpose."""

    service: str
    number: str
    coverage: str = "national"
    notes: str = ""
    source: str = ""
    source_url: str | None = None
    verified_at: str | None = None


@dataclass(frozen=True)
class FlagProfile:
    """Meaning, history and technical metadata of the national flag."""

    adoption_date: Evidence | None = None
    proportion: Evidence | None = None
    colors: tuple[Evidence, ...] = ()
    symbolism: tuple[Evidence, ...] = ()
    design_origin: tuple[Evidence, ...] = ()
    historical_flags: tuple[TimelineEvent, ...] = ()
    similar_flags: tuple[str, ...] = ()


@dataclass
class CountryIntelligence:
    """Normalized, source-aware country knowledge record."""

    code: str
    name: str
    official_name: Evidence | None = None

    identity: dict[str, Evidence] = field(default_factory=dict)
    flag: FlagProfile = field(default_factory=FlagProfile)
    geography: dict[str, Evidence] = field(default_factory=dict)
    origins: tuple[TimelineEvent, ...] = ()
    historical_timeline: tuple[TimelineEvent, ...] = ()
    sovereignty: dict[str, Evidence] = field(default_factory=dict)
    national_identity: dict[str, Evidence] = field(default_factory=dict)
    government: dict[str, Evidence] = field(default_factory=dict)
    people_society: dict[str, Evidence] = field(default_factory=dict)
    culture: dict[str, Evidence] = field(default_factory=dict)
    economy: dict[str, Evidence] = field(default_factory=dict)
    infrastructure: dict[str, Evidence] = field(default_factory=dict)
    education_science: dict[str, Evidence] = field(default_factory=dict)
    environment: dict[str, Evidence] = field(default_factory=dict)
    practical: dict[str, Evidence] = field(default_factory=dict)
    emergency: tuple[EmergencyContact, ...] = ()
    international_relations: dict[str, Evidence] = field(default_factory=dict)
    notable_people: tuple[dict[str, Evidence], ...] = ()
    did_you_know: tuple[Evidence, ...] = ()
    recognition: dict[str, Evidence] = field(default_factory=dict)

    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evidence(
    value: Any,
    source: str,
    *,
    reference_year: str | None = None,
    retrieved_at: str | None = None,
    confidence: float | None = None,
    status: str = VERIFIED,
    source_url: str | None = None,
) -> Evidence | None:
    """Create evidence only for meaningful values."""

    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {
        "",
        "not available",
        "n/a",
        "unknown",
    }:
        return None

    return Evidence(
        value=value,
        source=source,
        reference_year=reference_year,
        retrieved_at=retrieved_at,
        confidence=confidence,
        status=status,
        source_url=source_url,
    )


def _add(
    target: dict[str, Evidence],
    key: str,
    value: Any,
    source: str,
    *,
    reference_year: str | None = None,
) -> None:
    item = evidence(
        value,
        source,
        reference_year=reference_year,
    )
    if item is not None:
        target[key] = item


def _split_emergency_numbers(
    raw: str | None,
    *,
    source: str = "EmergencyNumberAPI",
) -> tuple[EmergencyContact, ...]:
    if not raw:
        return ()

    contacts: list[EmergencyContact] = []
    for part in str(raw).split("|"):
        if ":" not in part:
            continue
        service, number = [chunk.strip() for chunk in part.split(":", 1)]
        if service and number:
            contacts.append(
                EmergencyContact(
                    service=service,
                    number=number,
                    source=source,
                )
            )
    return tuple(contacts)


def build_from_legacy_profile(
    profile: dict[str, Any],
    recognition: dict[str, Any] | None = None,
) -> CountryIntelligence:
    """Convert the existing CountryProfile payload into the V2 schema.

    This adapter lets the application migrate incrementally without breaking
    the current classifier, Streamlit UI, or PDF generation.
    """

    code = str(profile.get("code") or profile.get("country_code") or "").upper()
    name = str(profile.get("name") or "Unknown country")
    item = CountryIntelligence(code=code, name=name)

    population_year = profile.get("population_year")
    gdp_year = profile.get("gdp_year")

    identity_source = "REST Countries / Wikidata"
    _add(item.identity, "continent", profile.get("continent"), identity_source)
    _add(item.identity, "region", profile.get("region"), identity_source)
    _add(item.identity, "subregion", profile.get("subregion"), identity_source)
    _add(item.identity, "capital", profile.get("capital"), "Wikidata")
    _add(item.identity, "demonym", profile.get("demonym"), "REST Countries")
    _add(item.identity, "iso_alpha3", profile.get("iso_alpha3"), "REST Countries")
    _add(item.identity, "currency", profile.get("currency"), "Wikidata")
    _add(
        item.identity,
        "languages",
        profile.get("official_languages"),
        "Wikidata",
    )
    _add(
        item.identity,
        "population",
        profile.get("population"),
        profile.get("population_source") or "World Bank",
        reference_year=str(population_year) if population_year else None,
    )
    _add(item.identity, "area_km2", profile.get("area_km2"), "Wikidata")

    _add(item.geography, "largest_cities", profile.get("largest_cities"), "Wikidata")
    _add(item.geography, "borders", profile.get("borders"), "REST Countries")
    _add(item.geography, "timezones", profile.get("timezones"), "REST Countries")
    _add(item.geography, "highest_point", profile.get("highest_point"), "Wikidata")
    _add(item.geography, "lowest_point", profile.get("lowest_point"), "Wikidata")
    _add(
        item.geography,
        "latitude",
        profile.get("latitude"),
        "REST Countries / country reference point",
    )
    _add(
        item.geography,
        "longitude",
        profile.get("longitude"),
        "REST Countries / country reference point",
    )

    _add(
        item.sovereignty,
        "former_colonial_powers",
        profile.get("former_colonial_powers"),
        profile.get("political_source") or "Wikidata",
    )
    _add(
        item.sovereignty,
        "colonial_period",
        profile.get("colonial_period"),
        profile.get("political_source") or "Wikidata",
    )
    _add(
        item.sovereignty,
        "independence_or_sovereignty_date",
        profile.get("independence_day"),
        profile.get("political_source") or "Wikidata",
    )
    _add(
        item.sovereignty,
        "independence_figure",
        profile.get("independence_leader"),
        profile.get("political_source") or "Wikidata",
    )

    historical_context = evidence(
        profile.get("historical_context"),
        profile.get("political_source") or "Wikidata / curated fallback",
        status=PARTIAL,
    )
    if historical_context is not None:
        item.historical_timeline = (
            TimelineEvent(
                label="Legacy historical context",
                period="Historical period",
                summary=str(historical_context.value),
                sources=(historical_context.source,),
            ),
        )

    _add(
        item.national_identity,
        "national_day",
        profile.get("national_day"),
        "Wikidata / curated fallback",
    )
    _add(
        item.national_identity,
        "national_motto",
        profile.get("national_motto"),
        "Wikidata / curated fallback",
    )
    _add(
        item.national_identity,
        "national_anthem",
        profile.get("national_anthem"),
        "Wikidata",
    )

    _add(item.government, "government_form", profile.get("government_form"), "Wikidata")
    _add(item.government, "head_of_state", profile.get("head_of_state"), "Wikidata")
    _add(
        item.government,
        "head_of_state_office",
        profile.get("head_of_state_office"),
        "Wikidata",
    )
    _add(
        item.government,
        "head_of_government",
        profile.get("head_of_government"),
        "Wikidata",
    )
    _add(
        item.government,
        "head_of_government_office",
        profile.get("head_of_government_office"),
        "Wikidata",
    )

    _add(
        item.economy,
        "gdp_current_usd",
        profile.get("gdp_value_usd", profile.get("gdp_usd")),
        profile.get("gdp_source") or "World Bank",
        reference_year=str(gdp_year) if gdp_year else None,
    )

    _add(item.practical, "calling_code", profile.get("calling_code"), "Wikidata")
    _add(item.practical, "internet_domain", profile.get("internet_domain"), "Wikidata")
    _add(item.practical, "driving_side", profile.get("driving_side"), "Wikidata")

    item.emergency = _split_emergency_numbers(
        profile.get("emergency_numbers"),
    )

    _add(
        item.international_relations,
        "international_organizations",
        profile.get("international_organizations"),
        "Wikidata",
    )

    if recognition:
        for key in (
            "recognition_status",
            "confidence",
            "top1_margin",
            "decision_mode",
            "top_candidates",
        ):
            if key in recognition:
                source = "Flag Intelligence recognition model"
                _add(item.recognition, key, recognition[key], source)

    return item


SENSITIVE_KEYS = {
    "private_address",
    "personal_phone",
    "personal_email",
    "medical_record",
    "financial_account",
    "precise_private_location",
}


def validate_country_intelligence(
    record: CountryIntelligence,
) -> list[str]:
    """Return actionable validation issues before UI/PDF publication."""

    issues: list[str] = []

    if not record.code:
        issues.append("Missing country code.")
    if not record.name or record.name == "Unknown country":
        issues.append("Missing country name.")

    for section_name in (
        "identity",
        "geography",
        "sovereignty",
        "government",
        "economy",
        "practical",
    ):
        section = getattr(record, section_name)
        for key, fact in section.items():
            if key in SENSITIVE_KEYS:
                issues.append(
                    f"Sensitive field must not be published: {section_name}.{key}"
                )
            if not fact.source:
                issues.append(
                    f"Missing source: {section_name}.{key}"
                )

    population = record.identity.get("population")
    if population is not None and population.reference_year is None:
        issues.append("Population must include a reference year.")

    gdp = record.economy.get("gdp_current_usd")
    if gdp is not None and gdp.reference_year is None:
        issues.append("GDP must include a reference year.")

    if not record.historical_timeline:
        issues.append(
            "Historical timeline is incomplete: no structured events are available."
        )

    return issues


def _usable_context(section: dict[str, Evidence], minimum_chars: int = 180) -> bool:
    context = section.get("context")
    if context is None:
        return False
    return len(str(context.value).strip()) >= minimum_chars


def _has_keys(section: dict[str, Evidence], keys: tuple[str, ...], minimum: int) -> bool:
    return sum(key in section for key in keys) >= minimum


def section_completion(record: CountryIntelligence) -> dict[str, bool]:
    """Expose strict learning-domain completion for QA and publication.

    A domain is complete only when it contains enough useful, sourced content;
    a single placeholder or generic paragraph is not sufficient.
    """

    timeline_years = [
        int(event.period)
        for event in record.historical_timeline
        if str(event.period).isdigit()
    ]
    timeline_is_broad = (
        len(record.historical_timeline) >= 4
        and (
            len(timeline_years) < 2
            or max(timeline_years) - min(timeline_years) >= 30
        )
    )

    flag_signals = sum([
        bool(record.flag.adoption_date),
        bool(record.flag.symbolism),
        bool(record.flag.design_origin),
        bool(record.flag.historical_flags),
    ])

    return {
        "identity": _has_keys(
            record.identity,
            ("capital", "currency", "languages", "population", "area_km2"),
            4,
        ),
        "flag": flag_signals >= 2,
        "geography": (
            _has_keys(
                record.geography,
                ("borders", "largest_cities", "highest_point", "lowest_point"),
                2,
            )
            and (
                "latitude" in record.geography
                and "longitude" in record.geography
            )
        ),
        "origins": bool(record.origins),
        "historical_timeline": timeline_is_broad,
        "sovereignty": _has_keys(
            record.sovereignty,
            (
                "independence_or_sovereignty_date",
                "former_colonial_powers",
                "colonial_period",
                "historical_context",
            ),
            2,
        ),
        "national_identity": _has_keys(
            record.national_identity,
            ("national_day", "national_motto", "national_anthem"),
            2,
        ),
        "government": _has_keys(
            record.government,
            (
                "government_form",
                "head_of_state",
                "head_of_government",
                "context",
            ),
            3,
        ),
        "people_society": _usable_context(record.people_society),
        "culture": _usable_context(record.culture),
        "economy": (
            "gdp_current_usd" in record.economy
            and _usable_context(record.economy, minimum_chars=120)
        ),
        "infrastructure": _usable_context(record.infrastructure),
        "education_science": _usable_context(record.education_science),
        "environment": _usable_context(record.environment),
        "practical": _has_keys(
            record.practical,
            ("calling_code", "internet_domain", "driving_side"),
            3,
        ),
        "emergency": (
            len(record.emergency) >= 2
            and all(contact.number for contact in record.emergency[:2])
        ),
        "international_relations": (
            "international_organizations" in record.international_relations
            and _usable_context(
                record.international_relations,
                minimum_chars=120,
            )
        ),
    }


def iter_evidence(record: CountryIntelligence) -> Iterable[tuple[str, Evidence]]:
    """Iterate evidence fields for provenance tables and QA."""

    for section_name in (
        "identity",
        "geography",
        "sovereignty",
        "national_identity",
        "government",
        "people_society",
        "culture",
        "economy",
        "infrastructure",
        "education_science",
        "environment",
        "practical",
        "international_relations",
        "recognition",
    ):
        section = getattr(record, section_name)
        for key, value in section.items():
            yield f"{section_name}.{key}", value
