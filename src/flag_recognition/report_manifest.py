"""Deterministic report manifest for Flag Intelligence.

The report generator should never decide ad hoc what exists. This module
defines the official knowledge chapters and evaluates whether each chapter has
publishable source-backed content before PDF layout begins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReportSectionSpec:
    key: str
    title: str
    required: bool = False


OFFICIAL_REPORT_SECTIONS: tuple[ReportSectionSpec, ...] = (
    ReportSectionSpec("overview", "Country Overview", True),
    ReportSectionSpec("geography", "Geography — Key Facts", True),
    ReportSectionSpec("climate", "Climate & Seasons"),
    ReportSectionSpec("rivers", "Rivers, Lakes & Waterways"),
    ReportSectionSpec("relief", "Mountains & Relief"),
    ReportSectionSpec("resources", "Natural Resources & Raw Materials"),
    ReportSectionSpec("flag", "Flag Intelligence", True),
    ReportSectionSpec("origins", "Origins & Early History"),
    ReportSectionSpec("history", "Historical Journey", True),
    ReportSectionSpec("sovereignty", "State Formation & Sovereignty", True),
    ReportSectionSpec("identity", "National Identity", True),
    ReportSectionSpec("government", "Government & Institutions", True),
    ReportSectionSpec("administration", "Administrative Divisions"),
    ReportSectionSpec("society", "People & Society", True),
    ReportSectionSpec("languages_religion", "Languages & Religion"),
    ReportSectionSpec("health", "Health System & Public Health"),
    ReportSectionSpec("culture", "Culture, Cuisine, Music & Sport", True),
    ReportSectionSpec("festivals", "Festivals, Holidays & Traditions"),
    ReportSectionSpec("heritage", "Heritage, UNESCO & Major Landmarks"),
    ReportSectionSpec("economy", "Economic Structure & Trade", True),
    ReportSectionSpec("economic_drivers", "Economic Drivers, Industries & Exports"),
    ReportSectionSpec("infrastructure", "Infrastructure Overview"),
    ReportSectionSpec("transport", "Transport Network, Ports & Airports"),
    ReportSectionSpec("energy", "Energy & Connectivity"),
    ReportSectionSpec("education", "Education, Science & Innovation"),
    ReportSectionSpec("environment", "Environment & Biodiversity"),
    ReportSectionSpec("practical", "Practical & Emergency Information", True),
    ReportSectionSpec("international", "International Relations"),
    ReportSectionSpec("notable_people", "Notable Public Figures"),
)


def _value(record: dict[str, Any], section: str, key: str = "context") -> Any:
    item = record.get(section)
    if not isinstance(item, dict):
        return None
    fact = item.get(key)
    if isinstance(fact, dict):
        value = fact.get("value")
        if value not in (None, "", "Not available"):
            return value
    return None


def _has_flag(record: dict[str, Any]) -> bool:
    flag = record.get("flag")
    if not isinstance(flag, dict):
        return False
    return any(
        flag.get(key)
        for key in (
            "adoption_date",
            "proportion",
            "symbolism",
            "design_origin",
            "historical_flags",
        )
    )


def build_report_manifest(
    intelligence: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, bool]:
    """Evaluate all official report chapters before PDF rendering."""
    timeline = intelligence.get("historical_timeline")
    origins = intelligence.get("origins")

    return {
        "overview": bool(profile.get("overview")),
        "geography": any(
            profile.get(key)
            for key in (
                "largest_cities",
                "borders",
                "highest_point",
                "lowest_point",
            )
        ),
        "climate": bool(_value(intelligence, "environment", "climate_seasons")),
        "rivers": bool(_value(intelligence, "geography", "rivers_lakes")),
        "relief": bool(_value(intelligence, "geography", "mountains_relief")),
        "resources": bool(_value(intelligence, "environment", "natural_resources")),
        "flag": _has_flag(intelligence),
        "origins": isinstance(origins, list) and bool(origins),
        "history": isinstance(timeline, list) and bool(timeline),
        "sovereignty": bool(intelligence.get("sovereignty")),
        "identity": bool(intelligence.get("national_identity")),
        "government": bool(intelligence.get("government")),
        "administration": bool(
            _value(intelligence, "government", "administrative_divisions")
        ),
        "society": bool(_value(intelligence, "people_society")),
        "languages_religion": bool(
            _value(intelligence, "people_society", "languages_religion")
        ),
        "health": bool(_value(intelligence, "people_society", "health_system")),
        "culture": bool(_value(intelligence, "culture")),
        "festivals": bool(_value(intelligence, "culture", "festivals_holidays")),
        "heritage": bool(_value(intelligence, "culture", "heritage_landmarks")),
        "economy": bool(_value(intelligence, "economy")),
        "economic_drivers": bool(
            _value(intelligence, "economy", "economic_drivers")
        ),
        "infrastructure": bool(_value(intelligence, "infrastructure")),
        "transport": bool(
            _value(intelligence, "infrastructure", "transport_network")
        ),
        "energy": bool(
            _value(intelligence, "infrastructure", "energy_connectivity")
        ),
        "education": bool(_value(intelligence, "education_science")),
        "environment": bool(_value(intelligence, "environment")),
        "practical": any(
            profile.get(key)
            for key in (
                "calling_code",
                "emergency_numbers",
                "driving_side",
                "internet_domain",
            )
        ),
        "international": bool(
            _value(intelligence, "international_relations")
        ),
        "notable_people": bool(
            _value(intelligence, "culture", "notable_people")
        ),
    }


def missing_required_sections(manifest: dict[str, bool]) -> list[str]:
    """Return required official chapters that still have no content."""
    missing: list[str] = []
    for spec in OFFICIAL_REPORT_SECTIONS:
        if spec.required and not manifest.get(spec.key, False):
            missing.append(spec.title)
    return missing
