# Semantic Conversation Understanding Design

## Problem

Flag Intelligence currently understands report requests through a mixture of LLM output and deterministic keyword/range rules. The deterministic layer is too authoritative: it can reject valid contextual meaning when the user's wording does not match a predefined topic or calendar-year pattern.

The latest regression demonstrates the failure clearly. After the user asks for the history of the French Republics, a follow-up such as `1 to 5` should naturally resolve to the First through Fifth Republics. Instead, the application treats that fragment in isolation, asks for more scope, and becomes repetitive.

Adding more keywords or regex patterns (for cuisine, architecture, republic numbers, etc.) will not solve the general problem. The dialogue layer must understand arbitrary user wording in context.

## Goal

Make Flag Intelligence dynamically understand a user's report need from the whole conversation, preserve and refine that need across turns, ask a clarification only when ambiguity genuinely blocks a useful report, and hand a structured semantic brief to the report pipeline.

The chat remains a requirements-gathering interface. It must not become a general country-information chatbot.

## Non-goals

- Do not make the chat answer the requested country content in long prose before report generation.
- Do not require every possible topic to be registered in a keyword dictionary.
- Do not remove deterministic validation entirely; it remains a safety layer.
- Do not redesign the report PDF layout or flag-recognition pipeline in this change.

## Architecture

Use a hybrid semantic architecture:

1. **LLM semantic interpreter is authoritative for meaning.** It receives the current country, recent dialogue, previous semantic brief, report availability, and the latest user message. It returns a structured report brief plus a dialogue action.
2. **Deterministic code validates safety and state consistency.** It checks required schema fields, preserves country identity, prevents accidental loss of explicit user constraints, handles malformed model output, and prevents repeated clarification loops. It must not invent semantic requirements such as "history always needs calendar years."
3. **Report generation consumes the semantic brief.** Known report categories may still map to specialized section sets, but unknown or novel topics must fall through to the generic focused-report path rather than trigger a clarification simply because no keyword exists.

## Semantic Brief

The persistent brief should represent meaning, not only keywords.

```text
country: canonical country derived from flag/context
subject: free-text report subject
scope: free-text resolved scope
period: optional free-text temporal scope
entity_range: optional free-text non-calendar range, e.g. "First through Fifth Republic"
topics: free-text topic list
angles: optional focus areas
exclusions: optional exclusions
depth: optional requested depth
current_events: boolean
other_constraints: free-text constraints
confidence: 0..1
ready: boolean
ambiguities: unresolved interpretations that materially affect the report
normalized_request: concise natural-language report brief used by the writer
changed_fields: fields explicitly changed by the latest user turn
```

Calendar periods and semantic/entity ranges are deliberately separate. `1950 to 2020` is a calendar period; `Republics 1 to 5`, `the first three presidents`, and `chapters 2–4` are contextual entity ranges.

## Conversation Decision Policy

The semantic interpreter decides one of four actions:

- `clarify`: one short question because multiple plausible interpretations would materially change the report.
- `generate`: the need is sufficiently clear; start the report automatically.
- `status`: answer only about an existing report or generation state.
- `converse`: short non-substantive interaction such as greeting; never use it to explain the requested country topic.

### Readiness rule

A report is ready when the interpreter can state a coherent `normalized_request` that a writer could execute without guessing a materially different intent.

Optional preferences must not block generation. Angle, depth, exclusions, and exact subtopics are refinements unless the user's wording explicitly makes them necessary.

There is no global rule that a history request requires calendar years. A bounded historical subject such as `French Republics I–V`, `the Meiji era`, or `post-independence Ghana` is already a valid scope.

### Clarification rule

Ask only when ambiguity is real and consequential.

Examples:

- `1 to 5` after `history of the French Republics` -> resolve as First through Fifth Republics; do not ask again.
- `1 to 5` with no relevant preceding context -> clarify what the range refers to.
- `traditional only` after a cuisine request -> add a traditional-cuisine constraint and generate.
- `whatever is fine`, `anything`, `you choose` -> accept a reasonable default and generate; do not repeat the same question.
- `focus on women` after a history request -> preserve the prior subject and add an angle.
- `actually compare both` -> resolve `both` from the recent dialogue; clarify only if two referents cannot be identified.

The controller must not ask the same semantic question twice after the user has answered or accepted a default.

## State and Corrections

The latest explicit user correction overrides stale state. Unchanged fields are preserved.

Examples:

- `1950–2020` followed by `focus on policy impact` preserves the period and adds the angle.
- `remove the policy angle` clears the angle without restoring it later.
- `actually, make it about traditional cuisine` replaces the old topic rather than merging history and cuisine accidentally.

`changed_fields` is the explicit mutation contract used to distinguish a genuine correction from a model omission.

## Dynamic Topic Handling

Keyword taxonomies may be used only for optimization or section selection. They are not a gate for understanding.

If the user asks for an unregistered topic such as:

- semiconductor sovereignty,
- colonial architecture,
- women's football development,
- constitutional symbolism,
- literary movements during a named dynasty,

then a semantically complete brief should generate a focused report using the generic focused-report section set if no specialized mapping exists.

No new keyword entry should be required merely to make the conversation understand the topic.

## Deterministic Safety Layer

Deterministic checks may:

- validate JSON/schema and action values;
- ensure the identified country remains consistent with the current flag unless the user explicitly switches it;
- preserve explicit constraints that the model accidentally omits;
- reject empty `normalized_request` when `generate` is requested;
- prevent duplicate/repeated clarification questions;
- fall back safely when the semantic model call fails or returns malformed output;
- enforce explicit calendar-year boundaries in the final report when a calendar period exists.

Deterministic checks must not:

- require a topic to exist in a keyword table;
- require calendar years merely because the topic is history;
- reinterpret contextual numeric fragments without dialogue context;
- overwrite an LLM-resolved semantic/entity range with a generic `scope` question.

## Report Writer Handoff

The report pipeline receives the final `normalized_request` and structured brief.

For recognized broad categories, existing specialized section selection may be reused when it improves report quality. For arbitrary topics, the writer uses the generic focused-report sections and writes specifically to the semantic brief.

The report writer must preserve all explicit constraints from the brief. Calendar-year enforcement applies only to actual calendar periods, not ordinal/entity ranges.

## Failure Handling

If semantic interpretation fails:

1. preserve the previous valid brief;
2. use deterministic parsing only for high-confidence facts such as an explicit four-digit year range;
3. ask one neutral clarification only if the latest turn cannot safely be interpreted;
4. never fall into a loop of repeated generic questions.

A model/network failure must not erase accumulated user requirements.

## Acceptance Criteria

The implementation is complete when automated regressions prove all of the following:

1. `history about French republics` + `1 to 5` resolves to the First through Fifth Republics and generates without demanding calendar years.
2. `economic history` + `1950 to 2020` + `policy impact` preserves all three constraints and generates.
3. `Japan cuisine` / `traditional cuisine` works without keyword-dependent repeated clarification.
4. An unseen topic not present in the taxonomy can form a valid report brief and generate through the generic focused-report path.
5. Contextual fragments (`the first three`, `after independence`, `both`, `traditional only`) are resolved from recent dialogue when their referent is clear.
6. Truly ambiguous fragments with no usable context cause exactly one targeted clarification.
7. `anything is fine` or equivalent acceptance stops clarification and generates with a reasonable default.
8. Explicit corrections replace or clear stale constraints rather than being merged back in.
9. Long substantive country explanations never appear in pre-report chat.
10. Malformed/failed semantic-model output preserves the previous brief and degrades safely.
11. Existing report generation, flag upload preview, PDF scope enforcement, and report-status behavior continue to pass their regressions.

## Migration

The existing PR `fix/dynamic-contextual-dialogue` already contains partial movement toward semantic state authority (`changed_fields`, arbitrary semantic scope, and removal of the hard-coded history-period requirement). Keep those useful changes, but treat them as an incomplete migration.

The final implementation should move conversation interpretation into a focused component rather than continuing to grow `report_writer.py`. Existing keyword helpers may remain for report-section optimization and deterministic fallback, but they must no longer decide whether a user's request is understandable.
