from flag_recognition.country_knowledge import split_article_sections_detailed
from flag_recognition.flag_knowledge import (
    _collect,
    _extract_adoption,
    _extract_proportion,
    _infobox_value,
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


def test_flag_lead_supports_day_month_year_and_width_length_ratio():
    article = """
The national flag is a vertical tricolour of orange, white and green.
It was adopted on 3 December 1959 with a 2:3 width-to-length ratio.
"""
    url = "https://en.wikipedia.org/wiki/Flag_of_Ivory_Coast"

    adoption = _extract_adoption(article, url)
    proportion = _extract_proportion(article, url)

    assert adoption is not None
    assert adoption.value == "3 December 1959"
    assert proportion is not None
    assert proportion.value == "2:3"


def test_flag_infobox_fields_are_extracted_without_narrative_parser():
    wikitext = """
{{Infobox flag
| Name = Republic of Côte d'Ivoire
| proportion = 2:3
| adoption = 3 December 1959
| design = A vertical tricolour of orange, white, and green
}}
"""

    assert _infobox_value(wikitext, "adoption") == "3 December 1959"
    assert _infobox_value(wikitext, "proportion") == "2:3"
    assert (
        _infobox_value(wikitext, "design")
        == "A vertical tricolour of orange, white, and green"
    )
