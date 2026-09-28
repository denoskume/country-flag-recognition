from flag_recognition.report_manifest import (
    OFFICIAL_REPORT_SECTIONS,
    build_report_manifest,
    missing_required_sections,
)


def _fact(value):
    return {"value": value, "source": "test"}


def test_manifest_detects_complete_core_report():
    intelligence = {
        "flag": {
            "adoption_date": _fact("1960"),
            "symbolism": [_fact("symbolism")],
        },
        "historical_timeline": [
            {"period": "1960", "summary": "Independence"}
        ],
        "sovereignty": {"independence": _fact("1960")},
        "national_identity": {"national_day": _fact("August 7")},
        "government": {"government_form": _fact("republic")},
        "people_society": {"context": _fact("society context")},
        "culture": {"context": _fact("culture context")},
        "economy": {"context": _fact("economy context")},
        "education_science": {"context": _fact("education context")},
        "environment": {
            "climate_seasons": _fact("tropical climate"),
            "context": _fact("environment context"),
        },
        "geography": {
            "rivers_lakes": _fact("major rivers"),
            "mountains_relief": _fact("plateau"),
        },
        "practical": {"calling_code": _fact("+225")},
        "international_relations": {"context": _fact("regional relations")},
    }
    profile = {
        "overview": "Country overview",
        "largest_cities": "Abidjan",
        "highest_point": "Mount Nimba",
        "calling_code": "+225",
        "emergency_numbers": "Police: 170",
    }

    manifest = build_report_manifest(intelligence, profile)

    assert manifest["flag"] is True
    assert manifest["history"] is True
    assert manifest["climate"] is True
    assert manifest["rivers"] is True
    assert manifest["relief"] is True


def test_missing_required_sections_are_explicit():
    manifest = {spec.key: False for spec in OFFICIAL_REPORT_SECTIONS}

    missing = missing_required_sections(manifest)

    assert "Country Overview" in missing
    assert "Flag Intelligence" not in missing
    assert "Historical Journey" in missing
    assert "Government & Institutions" in missing


def test_optional_missing_sections_do_not_count_as_required():
    manifest = {spec.key: True for spec in OFFICIAL_REPORT_SECTIONS}
    manifest["heritage"] = False
    manifest["notable_people"] = False
    manifest["energy"] = False

    missing = missing_required_sections(manifest)

    assert "Heritage, UNESCO & Major Landmarks" not in missing
    assert "Notable Public Figures" not in missing
    assert "Energy & Connectivity" not in missing


def test_manifest_accepts_tuple_backed_history_and_flag_collections():
    intelligence = {
        "flag": {
            "symbolism": ({"value": "National symbolism", "source": "test"},),
            "design_origin": (),
            "historical_flags": (),
            "adoption_date": None,
            "proportion": None,
        },
        "historical_timeline": (
            {"period": "1960", "summary": "Independence"},
        ),
        "origins": (
            {"period": "Early history", "summary": "Early settlement"},
        ),
        "sovereignty": {"independence": _fact("1960")},
        "national_identity": {"national_day": _fact("August 7")},
        "government": {"government_form": _fact("republic")},
        "people_society": {"context": _fact("society")},
        "culture": {"context": _fact("culture")},
        "economy": {"context": _fact("economy")},
    }
    profile = {
        "overview": "Overview",
        "largest_cities": "Abidjan",
        "calling_code": "+225",
    }

    manifest = build_report_manifest(intelligence, profile)

    assert manifest["flag"] is True
    assert manifest["history"] is True
    assert manifest["origins"] is True


def test_history_tuple_is_publishable_and_should_render():
    intelligence = {
        "flag": {},
        "historical_timeline": (
            {
                "period": "1960",
                "label": "Independence",
                "summary": "The country became independent.",
            },
        ),
        "origins": (),
        "sovereignty": {"independence": _fact("1960")},
        "national_identity": {"national_day": _fact("August 7")},
        "government": {"government_form": _fact("republic")},
        "people_society": {"context": _fact("society")},
        "culture": {"context": _fact("culture")},
        "economy": {"context": _fact("economy")},
    }
    profile = {
        "overview": "Overview",
        "largest_cities": "Abidjan",
        "calling_code": "+225",
    }

    manifest = build_report_manifest(intelligence, profile)

    assert manifest["history"] is True
    assert "Historical Journey" not in missing_required_sections(manifest)
