from flag_recognition.country_knowledge import (
    collect_domain_text,
    collect_domain_text_detailed,
    collect_origins,
    extract_timeline,
    split_article_sections,
    split_article_sections_detailed,
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
According to the 2021 census, the population was 29 million.
The population includes multiple linguistic communities.

=== Languages ===
French is the official language and several local languages are widely spoken.

== Culture ==
Music, cuisine and literature are important cultural expressions.

=== Music ===
Several popular music genres developed in the country.

=== Cuisine ===
Cassava and plantain are common ingredients in many dishes.

== Economy ==
In 2009 farmers earned substantial export revenue.
Agriculture and services are major parts of the economy.

== Education ==
According to a 2019 literacy estimate, literacy exceeded 80 percent.
The education system includes primary, secondary and tertiary levels.
"""


def test_split_article_sections_preserves_subsections():
    sections = split_article_sections(ARTICLE)
    headings = [heading for heading, _ in sections]

    assert "History" in headings
    assert "Pre-colonial history" in headings
    assert "Culture" in headings


def test_detailed_domain_collection_includes_child_sections():
    sections = split_article_sections_detailed(ARTICLE)

    culture = collect_domain_text_detailed(sections, "culture")
    history = collect_domain_text_detailed(
        sections,
        "history",
        max_chars=10000,
    )

    assert "Music:" in culture
    assert "Cuisine:" in culture
    assert "1893" in history
    assert "1960" in history


def test_domain_collection_keeps_domains_separate():
    sections = split_article_sections(ARTICLE)

    culture = collect_domain_text(sections, "culture")
    economy = collect_domain_text(sections, "economy")

    assert "Music, cuisine and literature" in culture
    assert "Agriculture and services" in economy
    assert "Agriculture and services" not in culture


def test_origins_require_explicit_early_history_heading():
    sections = split_article_sections_detailed(ARTICLE)
    origins = collect_origins(
        sections,
        "https://en.wikipedia.org/wiki/Example",
    )

    assert origins
    assert origins[0].label == "Pre-colonial history"
    assert origins[0].sources == ("Wikipedia",)


def test_timeline_extracts_history_subtree_events():
    sections = split_article_sections_detailed(ARTICLE)
    history = collect_domain_text_detailed(
        sections,
        "history",
        max_chars=10000,
        max_blocks=10,
    )
    timeline = extract_timeline(
        history,
        "https://en.wikipedia.org/wiki/Example",
    )

    periods = [event.period for event in timeline]

    assert "1893" in periods
    assert "1960" in periods
    assert periods == sorted(periods)


def test_old_dynamic_stat_is_not_promoted_as_current_context():
    sections = split_article_sections_detailed(ARTICLE)
    economy = collect_domain_text_detailed(sections, "economy")

    assert "2009 farmers" not in economy
    assert "Agriculture and services" in economy


def test_empty_history_does_not_invent_events():
    assert extract_timeline(
        "",
        "https://en.wikipedia.org/wiki/Example",
    ) == ()
