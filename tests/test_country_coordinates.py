from flag_recognition import country_info


class _Response:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "region": "Europe",
            "subregion": "Northern Europe",
            "demonyms": {"eng": {"m": "Norwegian", "f": "Norwegian"}},
            "cca3": "NOR",
            "timezones": ["UTC+01:00"],
            "borders": ["FIN", "SWE", "RUS"],
            "latlng": [62.0, 10.0],
        }


def test_rest_country_profile_uses_country_reference_coordinates(monkeypatch):
    monkeypatch.setattr(
        country_info.requests,
        "get",
        lambda *args, **kwargs: _Response(),
    )

    profile = country_info.fetch_rest_country_profile("NO")

    assert profile["latitude"] == 62.0
    assert profile["longitude"] == 10.0
    assert profile["iso_alpha3"] == "NOR"
