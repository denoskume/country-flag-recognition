from flag_recognition.country_info import (
    _parse_point,
    _unique_join,
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
