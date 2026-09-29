<p>
  <img align="left" src="assets/flag_intelligence_logo.svg" alt="Flag Intelligence" height="72">
</p>
<br clear="both">

<h1 align="center">FLAG INTELLIGENCE</h1>

<p align="center">
  <b>Recognize a flag — or type a country name — and generate a structured country intelligence report.</b>
</p>

<p align="center">
  Version 0.1.0
</p>

<p align="center">
  <a href="https://flag-intelligence.streamlit.app/"><b>Open Flag Intelligence</b></a>
</p>

Flag Intelligence is a personal computer-vision and country-intelligence project that connects worldwide flag recognition with structured country knowledge and professional report generation.

A user can either:

- upload a flag image and let the vision model identify the country; or
- type a country name or ISO code directly.

Both input modes converge on the same country-report pipeline.

---

## Product goal

> **Identify a country from its flag or name, then provide a clear, structured and educational report about that country.**

The application is designed to move beyond simple flag classification. For an accepted country, it aims to provide useful information covering geography, history, government, society, culture, economy, infrastructure, science, environment and practical information.

The final output can be exported as both JSON and PDF.

---

## Input modes

### 1. Flag image

```text
Upload flag image
      ↓
Worldwide vision model
      ↓
Robust prediction
      ↓
Open-set decision
      ↓
Accepted country
      ↓
Country Intelligence report
```

The image workflow includes:

- Top-1 prediction;
- confidence score;
- decision margin;
- ranked alternatives;
- similar / visually equivalent flag handling;
- accepted or ambiguous decision state.

### 2. Country name or ISO code

```text
Type country name or ISO code
      ↓
Deterministic country resolver
      ↓
Country Intelligence report
```

Examples:

```text
France
Côte d’Ivoire
Ivory Coast
Lebanon
India
Nigeria
FR
FRA
CI
LB
```

Text input does not simulate image recognition and does not create an artificial confidence score.

---

## Country report pipeline

Once a country is resolved, the application builds a structured profile and generates the final report through a shared pipeline.

```text
Country resolved
      ↓
Structured country profile
      ↓
Historical / practical context
      ↓
OpenAI-authored country report
      ↓
Deterministic editorial QA
      ↓
Automatic text sanitization
      ↓
Professional PDF + JSON export
```

The report writer receives the country identity and available structured context, then authors the final educational report in professional English.

The application does not require a separate user-facing web-research step before writing the report.

### Reliability behavior

Report generation is designed to avoid unnecessary failures:

- extended OpenAI writer timeout;
- one controlled SDK retry;
- cached successful authored reports;
- automatic cleanup of harmless formatting artefacts;
- deterministic QA after generation;
- local fallback generation if the external writer is unavailable;
- progress feedback shown immediately after country recognition.

A transient writer failure must not prevent the application from returning usable country output.

---

## Report content

The authored report currently supports the following sections:

1. Introduction
2. Physical Geography
3. Climate, Water & Natural Resources
4. Flag Design, Adoption & Symbolism
5. Origins & Early History
6. Historical Journey
7. State Formation & National Identity
8. Government & Administrative Structure
9. Leadership Through Time
10. People & Society
11. Languages & Religion
12. Health System & Public Health
13. Culture, Cuisine, Music & Sport
14. Festivals, Holidays & Traditions
15. Heritage, UNESCO & Major Landmarks
16. Literature, Philosophy & Thought
17. Economy, Trade & Key Industries
18. Infrastructure, Transport & Energy
19. Education & Research
20. Universities & Higher Education
21. Science, Discovery & Invention
22. Environment & Biodiversity
23. Cost of Living & Everyday Prices
24. Practical & Emergency Information
25. International Relations
26. Notable Public Figures
27. Conclusion

The exact depth of a section depends on the country and the reliability of the available context.

---

## Country snapshot

The report also includes a concise structured snapshot where available:

- capital;
- population and reference year;
- area;
- official language(s);
- currency;
- national day;
- calling code;
- driving side;
- internet domain;
- emergency numbers;
- geographic location.

Emergency numbers are displayed by service where possible, for example:

```text
Police: 170 · Fire: 180 · Ambulance: 185
```

The international calling code is not repeated inside emergency service numbers.

---

## Historical coverage

History is treated as a core part of the product rather than an optional enrichment block.

The report aims to explain:

- early societies and historical origins;
- major kingdoms, states or political formations;
- external rule or colonial history where applicable;
- major political transitions;
- independence or sovereignty milestones;
- important independence-era figures;
- civil conflict, constitutional change or state reorganization where relevant;
- modern state formation.

The system does not force every country into the same colonial-independence template.

Countries with different historical trajectories are handled accordingly.

---

## Flag Intelligence

For the selected country, the report can describe:

- flag layout;
- colours;
- symbols;
- adoption context;
- historical development;
- commonly accepted symbolism;
- visually similar flags where relevant.

The application also handles known visually similar or identical flags to reduce misleading prediction fragmentation.

Examples include:

- Côte d’Ivoire / Ireland;
- Chad / Romania;
- Indonesia / Monaco;
- Netherlands / Luxembourg;
- Australia / New Zealand.

---

## Government and institutions

Where reliable context is available, the report can cover:

- form of government;
- head of state;
- head of government;
- legislative structure;
- judiciary;
- administrative divisions;
- institutional development;
- international organizations.

Current office holders are time-sensitive facts and should be interpreted together with the report generation date.

---

## People, society and culture

The report can include:

- demographic context;
- languages;
- religion;
- health-system context;
- cuisine;
- music;
- sport;
- literature;
- traditions;
- public holidays;
- heritage sites;
- UNESCO sites;
- notable public figures.

Notable figures are written as short explanatory biographies rather than raw name lists.

---

## Universities & higher education

University coverage is treated as a dedicated report domain.

Where reliable information exists, the report should identify both:

- historically significant universities or predecessor institutions;
- currently prominent or internationally recognized universities.

For the most important institutions, the report aims to explain:

- official institution name;
- city or location;
- founding year or historical origin where well established;
- important predecessor, merger, closure or renaming history where relevant;
- role in the development of national higher education;
- major academic or research strengths;
- notable scientific, cultural or public contributions;
- broader national or international significance.

Exact contemporary ranking positions are included only when the ranking body, edition/year and position are supported by the available context. Otherwise, the report uses neutral descriptions such as **major national university**, **widely recognized institution**, or **internationally prominent university**.

Countries without medieval or early universities are not forced into an artificial ancient-university narrative; the report instead identifies the earliest major modern higher-education institutions.

---

## Economy, infrastructure and innovation

Coverage may include:

- major economic sectors;
- agriculture;
- industry;
- services;
- exports and imports;
- major production sectors;
- transport;
- ports;
- rail;
- roads;
- airports;
- electricity and energy;
- education systems;
- universities and higher education;
- historically significant universities;
- currently prominent and internationally recognized universities;
- founding context, location and institutional evolution;
- academic strengths and major contributions;
- research;
- science;
- technology;
- environment and biodiversity.

---

## Reports and exports

For an accepted or directly selected country, the application can generate:

- downloadable JSON;
- professional A4 PDF;
- country snapshot;
- geographic map;
- long-form authored country report.

The PDF and JSON use the same underlying country result.

Text input can generate the same report as image input without requiring an uploaded flag image.

---

## Report quality gate

The final report is checked before publication.

Mechanical QA detects issues such as:

- duplicated words;
- duplicated year constructions;
- repeated sentences;
- source or navigation residue;
- malformed editorial fragments.

Harmless formatting artefacts are repaired automatically before QA.

For example:

```text
Police: 110 | Ambulance: 113
```

is normalized to:

```text
Police: 110 · Ambulance: 113
```

The goal is to block genuinely malformed content, not to reject a report because of a repairable separator.

---

## Worldwide coverage

The project taxonomy contains **250 classes**:

- 249 ISO 3166-1 country / territory codes;
- Kosovo as the additional project class `XK`.

Both the visual-recognition layer and the country-resolution layer are designed around this worldwide taxonomy.

Coverage depth can vary by country, so worldwide quality validation remains part of the project.

---

## Recognition model

The current deployment model is a lightweight **MobileNetV3-Small** classifier.

Why MobileNetV3-Small:

- CPU-friendly;
- modest model size;
- fast inference;
- practical for Streamlit deployment;
- suitable baseline for worldwide flag recognition.

Deployment checkpoint:

```text
artifacts/models/worldwide_mobilenet_v3_small.pt
```

---

## Recognition evaluation

The project evaluates more than Top-1 accuracy.

### Classification

- Top-1 accuracy;
- Top-5 accuracy;
- macro precision;
- macro recall;
- macro F1;
- per-country recall;
- confusion matrix.

### Calibration and open-set behavior

- Negative Log-Likelihood;
- Expected Calibration Error;
- AUROC;
- AUPR;
- FPR@95TPR;
- confidence / entropy analysis;
- accepted vs rejected predictions.

### Efficiency

- parameter count;
- model size;
- inference latency;
- throughput.

---

## Real-world robustness

Current human-reviewed real-world challenge results:

| System | Top-1 | Top-5 |
|---|---:|---:|
| V1 worldwide classifier | 19.61% | 27.45% |
| Scene-aware V2 | 17.65% | 30.39% |
| Detector + V1 | 19.61% | 30.39% |

Flag-detector localization rate:

```text
73.53%
```

The scene-aware model and detector-first pipeline were not promoted because they did not improve Top-1 accuracy over the V1 reference.

The Wikimedia challenge set remains evaluation-only.

---

## Architecture

```text
                         ┌────────────────────┐
                         │   Flag image input │
                         └─────────┬──────────┘
                                   │
                            Vision inference
                                   │
                         Open-set decision
                                   │
                                   ▼
┌────────────────────┐     ┌────────────────────┐
│ Country-name input │────▶│ Canonical country  │
│ or ISO code        │     │ identity / code    │
└────────────────────┘     └─────────┬──────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │ Structured country    │
                         │ profile + context     │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │ OpenAI report writer  │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │ Sanitization + QA     │
                         └───────────┬───────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
               JSON export                      PDF report
```

---

## Main modules

```text
src/flag_recognition/
├── country_info.py
├── country_intelligence.py
├── country_knowledge.py
├── flag_knowledge.py
├── learning.py
├── report_manifest.py
├── report_writer.py
├── taxonomy.py
├── inference.py
├── model.py
├── detection.py
├── dataset.py
├── metrics.py
├── splits.py
└── transforms.py
```

Main responsibilities:

- `country_info.py` — structured country facts and source integration;
- `country_intelligence.py` — normalized country-intelligence model;
- `country_knowledge.py` — country learning context;
- `flag_knowledge.py` — flag-specific metadata and history;
- `report_writer.py` — final authored report generation and QA;
- `report_manifest.py` — report-structure validation;
- `taxonomy.py` — country-code and country-name resolution;
- `inference.py` — visual flag prediction pipeline;
- `learning.py` — Ask Flag Intelligence retrieval support.

---

## Repository structure

```text
country-flag-recognition/
├── app.py
├── serve.py
├── assets/
├── configs/
├── data/
├── artifacts/
│   ├── models/
│   ├── metrics/
│   └── figures/
├── scripts/
├── src/
│   └── flag_recognition/
├── tests/
├── .github/
│   └── workflows/
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Run the Streamlit application

```bash
pip install -r requirements.txt
streamlit run app.py
```

The Streamlit application is the main product interface for:

- image-based flag recognition;
- direct country-name input;
- Country Intelligence;
- report authoring;
- JSON export;
- PDF export;
- maps;
- Ask Flag Intelligence.

---

## OpenAI configuration

The report writer reads the OpenAI API key from the application environment.

For Streamlit Community Cloud, the application also supports `st.secrets`.

Expected configuration:

```text
OPENAI_API_KEY=...
FLAG_INTELLIGENCE_WRITER_MODEL=...
```

The writer model defaults to the configured project model when no override is provided.

API credentials must never be committed to the repository.

---

## Live application

Flag Intelligence V0.1.0 is publicly available here:

**https://flag-intelligence.streamlit.app/**

The deployed application currently provides the complete user workflow:

- flag-image recognition;
- direct country-name / ISO input;
- Country Intelligence report generation;
- JSON export;
- professional PDF export;
- geographic context;
- Ask Flag Intelligence.

---

## Local development

Run the Streamlit application locally with:

```bash
pip install -r requirements.txt
streamlit run app.py
```

A separate FastAPI-compatible development entry point is also available:

```bash
PYTHONPATH=src uvicorn serve:app --reload
```

Local FastAPI development is available at:

```text
http://127.0.0.1:8000
```

The public Streamlit deployment is the reference interface for Flag Intelligence V0.1.0.

---

## Tests

Run the unit test suite with:

```bash
pytest
```

Tests cover areas including:

- taxonomy;
- country-name resolution;
- country-profile extraction;
- country knowledge;
- historical timelines;
- flag metadata;
- report manifests;
- dataset splits;
- metrics.

---

## Worldwide Country Intelligence audit

The repository includes automated worldwide validation across the 250-class taxonomy.

The audit checks areas such as:

- expected country-code coverage;
- report structure;
- country-profile consistency;
- historical-field coherence;
- malformed content;
- raw MediaWiki leakage;
- report-manifest validity;
- pipeline exceptions.

A local shard can be run with:

```bash
PYTHONPATH=src python scripts/audit_country_intelligence_world.py \
  --shard-index 0 \
  --shard-count 5 \
  --timeout 12 \
  --output-dir artifacts/world_audit
```

---

## Current status

**Version 0.1.0** currently includes four main working layers:

1. **Worldwide visual flag recognition**
2. **Direct country-name / ISO resolution**
3. **Authored Country Intelligence reporting**
4. **Automated quality validation and worldwide auditing**

The reporting pipeline has already been validated on multiple countries including France, Côte d’Ivoire, India and Nigeria, while additional country-by-country testing continues.

The visual model still has room for improvement on uncontrolled real-world imagery. Report depth can also vary when country context is incomplete, which is why deterministic QA and fallback behavior remain part of the architecture.

---

## Design principles

- no fake confidence score for text-selected countries;
- no deliberate fabrication of unsupported numerical facts;
- history adapted to the country rather than forced into one template;
- practical emergency information presented clearly by service;
- image and text inputs converge on the same report pipeline;
- professional PDF and JSON outputs from one shared country result;
- repair harmless formatting defects instead of failing unnecessarily;
- keep a fallback path when external report generation is unavailable;
- validate the complete worldwide taxonomy rather than only a few example countries.

---

## Project direction

Flag Intelligence is evolving from a flag-recognition experiment into a compact country-intelligence product.

The next priorities are:

- broader country-by-country report validation;
- stronger consistency across all 250 classes;
- improved real-world visual recognition;
- continued report-quality refinement;
- production-grade reliability for V0.1.x.
