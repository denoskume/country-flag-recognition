from flag_recognition.country_intelligence import (
    CountryIntelligence,
    Evidence,
    build_from_legacy_profile,
    section_completion,
    validate_country_intelligence,
)


def test_evidence_rejects_invalid_confidence():
    try:
        Evidence(value="x", source="test", confidence=1.2)
    except ValueError as exc:
        assert "confidence" in str(exc)
    else:
        raise AssertionError("invalid confidence should raise ValueError")


def test_legacy_profile_maps_core_sections():
    profile = {
        "code": "CI",
        "name": "Côte d'Ivoire",
        "continent": "Africa",
        "capital": "Yamoussoukro",
        "currency": "West African CFA franc",
        "official_languages": "French",
        "population": 32711547,
        "population_year": "2025",
        "population_source": "World Bank",
        "area_km2": 322463,
        "region": "Africa",
        "subregion": "Western Africa",
        "demonym": "Ivorian",
        "iso_alpha3": "CIV",
        "largest_cities": "Abidjan, Bouaké, Korhogo",
        "borders": "Burkina Faso, Ghana, Guinea, Liberia, Mali",
        "timezones": "UTC",
        "historical_context": (
            "Côte d'Ivoire became independent from France on August 7, 1960."
        ),
        "political_source": "Wikidata / curated fallback",
        "independence_day": "August 7, 1960",
        "former_colonial_powers": "France",
        "government_form": "republic",
        "gdp_value_usd": 99_773_555_666,
        "gdp_year": "2025",
        "gdp_source": "World Bank",
        "calling_code": "+225",
        "emergency_numbers": "Police: 170 | Fire: 180 | Ambulance (SAMU): 185",
        "internet_domain": ".ci",
        "driving_side": "right",
    }

    record = build_from_legacy_profile(profile)

    assert isinstance(record, CountryIntelligence)
    assert record.code == "CI"
    assert record.identity["capital"].value == "Yamoussoukro"
    assert record.identity["population"].reference_year == "2025"
    assert record.economy["gdp_current_usd"].reference_year == "2025"
    assert len(record.emergency) == 3
    assert record.emergency[0].service == "Police"


def test_validation_requires_reference_years_and_history():
    record = CountryIntelligence(code="FR", name="France")
    record.identity["population"] = Evidence(
        value=68_000_000,
        source="Example",
    )
    record.economy["gdp_current_usd"] = Evidence(
        value=3_000_000_000_000,
        source="Example",
    )

    issues = validate_country_intelligence(record)

    assert "Population must include a reference year." in issues
    assert "GDP must include a reference year." in issues
    assert any("Historical timeline is incomplete" in issue for issue in issues)


def test_completion_exposes_missing_knowledge_domains():
    record = CountryIntelligence(code="CI", name="Côte d'Ivoire")
    record.identity["capital"] = Evidence(value="Yamoussoukro", source="Wikidata")
    record.identity["currency"] = Evidence(value="XOF", source="Wikidata")
    record.identity["languages"] = Evidence(value="French", source="Wikidata")
    record.identity["population"] = Evidence(
        value=32_000_000,
        source="World Bank",
        reference_year="2025",
    )

    completion = section_completion(record)

    assert completion["identity"] is True
    assert completion["culture"] is False
    assert completion["origins"] is False
