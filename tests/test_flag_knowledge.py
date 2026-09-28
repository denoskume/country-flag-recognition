from flag_recognition.country_knowledge import split_article_sections_detailed
from flag_recognition.flag_knowledge import (
    _collect,
    _extract_adoption,
    _extract_proportion,
    FLAG_SECTION_ALIASES,
)


FLAG_ARTICLE = """
The national flag was adopted in 1959 and has a proportion of 2:3.

== Design ==
The flag uses three vertical bands.

== Symbolism ==
The colours represent several national ideas.

== History ==
In 1960 the flag remained the national symbol after independence.
"""


def test_flag_sections_are_kept_separate():
    sections = split_article_sections_detailed(FLAG_ARTICLE)

    design = _collect(sections, FLAG_SECTION_ALIASES["design"])
    symbolism = _collect(sections, FLAG_SECTION_ALIASES["symbolism"])
    history = _collect(sections, FLAG_SECTION_ALIASES["history"])

    assert "three vertical bands" in design
    assert "national ideas" in symbolism
    assert "1960" in history
    assert "1960" not in design


def test_flag_metadata_extracts_adoption_and_proportion():
    url = "https://en.wikipedia.org/wiki/Flag_of_Example"

    adoption = _extract_adoption(FLAG_ARTICLE, url)
    proportion = _extract_proportion(FLAG_ARTICLE, url)

    assert adoption is not None
    assert adoption.value == "1959"
    assert proportion is not None
    assert proportion.value == "2:3"
