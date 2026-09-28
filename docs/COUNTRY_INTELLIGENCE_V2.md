# Flag Intelligence — Country Intelligence V2

## Product goal

Flag Intelligence is not only a flag classifier. The target experience is:

> Upload a flag → identify the country → understand the country from its origins to today.

The system should provide broad, useful, non-sensitive public knowledge about a country while preserving provenance, reference years, freshness and uncertainty.

## Core principles

1. **Recognition and knowledge are separate layers.**
   - Recognition decides which flag/country is visible.
   - Country Intelligence explains the country.

2. **No unsupported fact reaches the final report.**
   - Every factual field carries a source.
   - Time-varying facts carry a reference year or verification date.

3. **Historical coverage must be chronological.**
   - Origins / early societies
   - Pre-modern political formations
   - External contact
   - Colonial / occupation / territorial transitions where applicable
   - Sovereignty / independence / state formation
   - Major post-independence eras
   - Contemporary period

4. **The schema must work for all countries.**
   - Never assume every country was colonized.
   - Never assume every territory had kingdoms/empires.
   - Allow multiple sovereignty transitions and multiple colonial powers.

5. **Sensitive personal information is excluded.**
   - No private addresses, personal phone numbers, medical records, financial accounts or precise private locations.
   - Public institutional and emergency information is allowed.

## Knowledge domains

The V2 schema contains:

1. Recognition Intelligence
2. Country Identity
3. Flag Intelligence
4. Geography
5. Origins
6. Historical Timeline
7. Sovereignty / Colonization / State Formation
8. National Identity
9. Government & Institutions
10. People & Society
11. Culture
12. Economy
13. Infrastructure
14. Education & Science
15. Environment
16. Practical Life
17. Emergency Information
18. International Relations
19. Notable Public Figures
20. Did You Know?
21. Sources & Data Freshness

## Evidence model

Each factual value should be represented as:

- value
- source
- source URL when available
- reference year when relevant
- retrieval / verification date
- confidence
- status: verified / partial / unverified / missing

This solves the current problem where one paragraph can mix data from different years without making that visible.

## Historical timeline rules

A publishable timeline should:

- include at least one pre-modern/origin event when sources support it;
- explicitly mark uncertainty instead of filling gaps;
- include sovereignty/state-formation events;
- include the major modern political periods;
- continue to the contemporary era;
- attach sources to each event or event group.

The former single `Historical Background` paragraph should be treated only as a migration fallback and should never be considered complete V2 history.

## Dynamic vs stable facts

### Stable or slow-changing
- independence / sovereignty dates
- flag adoption
- historical leaders
- major historical events
- national symbols

### Dynamic
- head of state
- head of government
- population
- GDP
- inflation
- travel information
- emergency information where regulations change

Dynamic fields should expose freshness metadata.

## Target PDF structure

The PDF becomes adaptive instead of fixed to three pages:

1. Cover + Recognition Result
2. Country at a Glance
3. Flag Intelligence
4. Geography
5. Origins
6. Historical Journey
7. Sovereignty / State Formation
8. National Identity
9. Government & Institutions
10. People & Society
11. Culture
12. Economy
13. Infrastructure
14. Education & Science
15. Environment
16. Practical Information
17. Emergency Information
18. International Relations
19. Did You Know?
20. Sources & Data Freshness

Sections with no verified data should be omitted or clearly marked incomplete; they should not be fabricated.

## Learning modes

The web application should ultimately expose:

- **Quick View** — essential country facts
- **Explore** — structured country profile
- **Timeline** — chronological history
- **Deep Dive** — full Country Intelligence
- **Compare** — two-country comparison
- **Quiz** — generated only from verified facts
- **Ask Flag Intelligence** — answers grounded in the country knowledge record

## Migration strategy

The current `CountryProfile` remains operational.

`build_from_legacy_profile()` converts it into the V2 schema so the application can migrate without breaking recognition, Streamlit or the existing PDF.

The JSON report now exposes:

- `country_intelligence_v2`
- `country_intelligence_completion`
- `country_intelligence_validation`

This makes missing domains visible instead of silently pretending the report is complete.

## Acceptance criteria for a country report

A report is considered **Country Intelligence Complete** only when:

- core identity is sourced;
- population and GDP include reference years;
- emergency numbers identify the service;
- government facts are fresh enough for publication;
- origins/history are represented as structured events;
- sovereignty/state-formation context is present when applicable;
- culture, people/society and economy contain sourced information;
- no sensitive personal fields are present;
- source provenance is available at field or event level;
- no contradictory values are presented without explanation;
- incomplete domains are explicitly marked.

## Côte d'Ivoire migration note

The current report correctly includes items such as capital, population, language, currency, emergency numbers, colonial period, independence date, government and GDP, but its historical coverage is still a short colonial/independence summary.

Under V2, Côte d'Ivoire should gain a chronological history beginning with supported early societies and political formations, continuing through French colonial rule, independence, the Houphouët-Boigny era, later political transitions and the contemporary period.
