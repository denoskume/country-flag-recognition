from flag_recognition.country_knowledge import (
    canonical_overview_text,
    collect_domain_text,
    collect_keyword_context,
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

=== Climate ===
The coastal region has mild winters and cool summers.

=== Rivers ===
The River Alpha drains the central plateau.

=== Mountains ===
The Beta Range contains the country's highest peaks.

=== Natural resources ===
The country has petroleum, forests and fisheries.

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

=== Oil industry ===
Oil and gas exports are a major source of export revenue.

=== Fisheries ===
Fishing and aquaculture support coastal employment and exports.

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


def test_deep_geography_domains_are_separated():
    sections = split_article_sections_detailed(ARTICLE)

    climate = collect_domain_text_detailed(sections, "climate_seasons")
    rivers = collect_domain_text_detailed(sections, "rivers_lakes")
    relief = collect_domain_text_detailed(sections, "mountains_relief")
    resources = collect_domain_text_detailed(sections, "natural_resources")

    assert "mild winters" in climate
    assert "River Alpha" in rivers
    assert "Beta Range" in relief
    assert "petroleum" in resources


def test_economic_drivers_capture_key_industries():
    sections = split_article_sections_detailed(ARTICLE)
    drivers = collect_domain_text_detailed(sections, "economic_drivers")

    assert "Oil industry" in drivers
    assert "Fisheries" in drivers
    assert "export revenue" in drivers


DEEP_ARTICLE = """
== Geography ==
General geography.

=== Climate ===
The country has a tropical climate with a dry season and a rainy season.

=== Rivers ===
The Bandama and Comoé are major rivers.

=== Mountains ===
Mount Nimba and other uplands shape the western relief.

=== Natural resources ===
The country has petroleum, natural gas, gold, manganese and forest resources.

== Demographics ==
Population overview.

=== Languages ===
French is the official language and many indigenous languages are spoken.

=== Religion ===
Islam and Christianity are major religions.

=== Health ===
The health system includes public hospitals and regional health services.

== Culture ==
General cultural overview.

=== Festivals ===
National and local festivals are celebrated throughout the year.

=== World Heritage ===
Several cultural and natural sites are recognized by UNESCO.

=== Notable people ===
The country has internationally known writers, athletes and artists.

== Economy ==
General economy.

=== Agriculture ===
Cocoa and cashew production are important export activities.

=== Industry ===
Food processing, energy and construction are important sectors.

=== Exports ===
Cocoa, petroleum products and gold are major exports.

== Transport ==
General transport overview.

=== Roads ===
The road network links major cities.

=== Ports ===
The main commercial port handles regional trade.

=== Airports ===
International airports connect the country to other regions.

=== Telecommunications ===
Mobile and internet services are widely used.

== Administrative divisions ==
The country is divided into districts and regions.
"""


def test_extended_country_domains_are_extractable():
    sections = split_article_sections_detailed(DEEP_ARTICLE)

    assert "dry season" in collect_domain_text_detailed(
        sections, "climate_seasons"
    )
    assert "Bandama" in collect_domain_text_detailed(
        sections, "rivers_lakes"
    )
    assert "Mount Nimba" in collect_domain_text_detailed(
        sections, "mountains_relief"
    )
    assert "petroleum" in collect_domain_text_detailed(
        sections, "natural_resources"
    )
    assert "French" in collect_domain_text_detailed(
        sections, "languages_religion"
    )
    assert "public hospitals" in collect_domain_text_detailed(
        sections, "health_system"
    )
    assert "festivals" in collect_domain_text_detailed(
        sections, "festivals_holidays"
    ).lower()
    assert "UNESCO" in collect_domain_text_detailed(
        sections, "heritage_landmarks"
    )
    assert "Cocoa" in collect_domain_text_detailed(
        sections, "economic_drivers"
    )
    assert "Ports" in collect_domain_text_detailed(
        sections, "transport_network"
    )
    assert "internet" in collect_domain_text_detailed(
        sections, "energy_connectivity"
    ).lower()
    assert "districts" in collect_domain_text_detailed(
        sections, "administrative_divisions"
    )


def test_canonical_overview_removes_competing_live_population():
    overview = (
        "Exampleland is a country on the coast. "
        "Its capital is Example City. "
        "With 33.2 million inhabitants in 2026, it is densely populated. "
        "French is an official language."
    )

    cleaned = canonical_overview_text(overview)

    assert "33.2 million" not in cleaned
    assert "2026" not in cleaned
    assert "Exampleland is a country on the coast." in cleaned
    assert "French is an official language." in cleaned


def test_keyword_fallback_extracts_domain_without_matching_heading():
    text = """
== Production ==
The country exports cocoa, petroleum products and gold.
Electricity generation relies on gas-fired plants and hydropower.
"""
    sections = split_article_sections_detailed(text)

    economy = collect_keyword_context(
        sections,
        ("cocoa", "petroleum", "exports"),
    )
    energy = collect_keyword_context(
        sections,
        ("electricity", "hydropower", "gas-fired"),
    )

    assert "cocoa" in economy.lower()
    assert "petroleum" in economy.lower()
    assert "electricity" in energy.lower()
    assert "hydropower" in energy.lower()
