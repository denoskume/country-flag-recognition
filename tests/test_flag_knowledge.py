from flag_recognition.country_knowledge import split_article_sections
from flag_recognition.flag_knowledge import _collect, FLAG_SECTION_ALIASES


FLAG_ARTICLE = """
Lead text.

== Design ==
The flag uses three vertical bands.

== Symbolism ==
The colours represent several national ideas.

== History ==
In 1960 the national flag was adopted after independence.
"""


def test_flag_sections_are_kept_separate():
    sections = split_article_sections(FLAG_ARTICLE)

    design = _collect(sections, FLAG_SECTION_ALIASES["design"])
    symbolism = _collect(sections, FLAG_SECTION_ALIASES["symbolism"])
    history = _collect(sections, FLAG_SECTION_ALIASES["history"])

    assert "three vertical bands" in design
    assert "national ideas" in symbolism
    assert "1960" in history
    assert "1960" not in design
