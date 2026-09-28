from flag_recognition.taxonomy import country_code_from_text, country_name_from_code


def test_iso_country_name():
    assert country_name_from_code("fr") == "France"


def test_kosovo_override():
    assert country_name_from_code("xk") == "Kosovo"


def test_unknown_code_falls_back_to_code():
    assert country_name_from_code("zz") == "ZZ"



def test_country_text_accepts_common_names_and_accents():
    assert country_code_from_text("France") == "fr"
    assert country_code_from_text("Côte d’Ivoire") == "ci"
    assert country_code_from_text("Ivory Coast") == "ci"
    assert country_code_from_text("South Korea") == "kr"


def test_country_text_accepts_iso_codes():
    assert country_code_from_text("FR") == "fr"
    assert country_code_from_text("FRA") == "fr"
    assert country_code_from_text("XK") == "xk"


def test_country_text_rejects_unknown_free_text():
    assert country_code_from_text("Atlantis") is None
