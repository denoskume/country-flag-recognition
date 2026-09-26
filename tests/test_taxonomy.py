from flag_recognition.taxonomy import country_name_from_code


def test_iso_country_name():
    assert country_name_from_code("fr") == "France"


def test_kosovo_override():
    assert country_name_from_code("xk") == "Kosovo"


def test_unknown_code_falls_back_to_code():
    assert country_name_from_code("zz") == "ZZ"
