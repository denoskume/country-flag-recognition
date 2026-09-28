from flag_recognition.country_intelligence import CountryIntelligence
from flag_recognition.country_knowledge import (
    collect_domain_text,
    collect_origins,
    extract_timeline,
    split_article_sections,
)


ARTICLE = """
Country lead paragraph.

== History ==
Early communities lived in the region for centuries.

=== Pre-colonial history ===
Several political communities developed before European colonial rule.

=== Colonial period ===
In 1893 the territory was established as a French colony.
In 1960 the country became independent and established a republic.

== Geography ==
The country has coastal and inland regions.

== Demographics ==
The population includes multiple linguistic communities.

== Culture ==
Music, cuisine and literature are important cultural expressions.

== Economy ==
Agriculture and services are major parts of the economy.

== Education ==
The education system includes primary, secondary and tertiary levels.
"""


def test_split_article_sections_preserves_subsections():
    sections = split_article_sections(ARTICLE)
    headings = [heading for heading, _ in sections]

    assert "History" in headings
    assert "Pre-colonial history" in headings
    assert "Culture" in headings


def test_domain_collection_is_section_based():
    sections = split_article_sections(ARTICLE)

    culture = collect_domain_text(sections, "culture")
    economy = collect_domain_text(sections, "economy")

    assert "Music, cuisine and literature" in culture
    assert "Agriculture and services" in economy
    assert "Agriculture and services" not in culture


def test_origins_require_explicit_early_history_heading():
    sections = split_article_sections(ARTICLE)
    origins = collect_origins(
        sections,
        "https://en.wikipedia.org/wiki/Example",
    )

    assert origins
    assert origins[0].label == "Pre-colonial history"
    assert origins[0].sources == ("Wikipedia",)


def test_timeline_extracts_explicit_dated_history():
    sections = split_article_sections(ARTICLE)
    history = collect_domain_text(sections, "history", max_chars=10000)
    timeline = extract_timeline(
        history,
        "https://en.wikipedia.org/wiki/Example",
    )

    periods = [event.period for event in timeline]

    assert "1893" in periods
    assert "1960" in periods
    assert periods == sorted(periods)


def test_empty_history_does_not_invent_events():
    assert extract_timeline(
        "",
        "https://en.wikipedia.org/wiki/Example",
    ) == ()
