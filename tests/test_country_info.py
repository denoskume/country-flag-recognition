from flag_recognition.country_info import (
    _format_wikidata_date,
    _is_raw_wikidata_identifier,
    _parse_point,
    _unique_join,
    _wikipedia_title_from_url,
)


def test_parse_wikidata_point():
    latitude, longitude = _parse_point(
        "Point(2.2137 46.2276)"
    )

    assert latitude == 46.2276
    assert longitude == 2.2137


def test_unique_join_removes_duplicates_and_empty_values():
    assert _unique_join(
        [
            "France",
            "",
            "France",
            "French Republic",
        ]
    ) == "France, French Republic"


def test_parse_point_handles_missing_value():
    assert _parse_point(None) == (
        None,
        None,
    )


def test_extract_wikipedia_title_from_article_url():
    title = _wikipedia_title_from_url(
        "https://en.wikipedia.org/wiki/C%C3%B4te_d%27Ivoire"
    )

    assert title == "Côte_d'Ivoire"



def test_unique_join_discards_unresolved_wikidata_ids():
    assert _unique_join(
        ["Euro", "Q4916", "Euro"]
    ) == "Euro"


def test_unique_join_returns_not_available_for_only_raw_ids():
    assert _unique_join(
        ["Q4916", "https://www.wikidata.org/entity/Q123"]
    ) == "Not available"


def test_detect_raw_wikidata_identifiers():
    assert _is_raw_wikidata_identifier("Q4916")
    assert _is_raw_wikidata_identifier(
        "https://www.wikidata.org/entity/Q4916"
    )
    assert not _is_raw_wikidata_identifier("Euro")


def test_format_wikidata_date():
    assert _format_wikidata_date(
        "+1960-08-07T00:00:00Z"
    ) == "August 7, 1960"
