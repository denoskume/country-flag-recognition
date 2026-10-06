# Semantic Conversation Understanding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Flag Intelligence understand arbitrary report needs from conversation context instead of keyword/range rules, while ensuring a flag upload alone only identifies the country and never starts report generation.

**Architecture:** Extract semantic conversation interpretation and persistent brief reconciliation into a focused `report_conversation.py` component. The LLM is authoritative for semantic meaning; deterministic code validates schema/state, preserves explicit constraints, prevents clarification loops, and enforces safe fallbacks. `report_writer.py` keeps report writing and exposes a compatibility adapter; `app.py` treats flag recognition as country context rather than a synthetic report request.

**Tech Stack:** Python 3.11, existing Flag Intelligence LLM backend, Streamlit, pytest.

**Spec:** `docs/superpowers/specs/2026-10-06-semantic-conversation-understanding-design.md` and `docs/superpowers/specs/2026-10-06-semantic-conversation-flag-entry-addendum.md`

## Global Constraints

- Pre-report chat gathers requirements; it does not explain substantive country content.
- Unknown topics must work without adding keyword registrations.
- Deterministic rules may validate but may not invent semantic requirements.
- A flag upload with no accompanying text establishes country identity only and never generates a report.
- A flag plus user text uses the flag as country context and the text as the first semantic requirement turn.
- Preserve existing flag preview, recognition, PDF generation, report-status, and explicit calendar-period enforcement behavior.

## Review Focus

- Short contextual numeric/ordinal fragments such as `1 to 5` resolve against recent dialogue rather than calendar-year regex.
- Explicit user corrections replace stale brief fields instead of being merged back.
- Unknown/unregistered topics can become ready semantic briefs and use generic focused report sections.
- Model failure or malformed output preserves prior valid brief and asks at most one neutral clarification.
- New flag uploads cannot inherit or trigger stale scope from the previous country.

---

### Task 1: Focused semantic conversation component

**Files:**
- Create: `src/flag_recognition/report_conversation.py`
- Create: `tests/test_semantic_report_conversation.py`

**Interfaces:**
- Produces: `continue_semantic_conversation(country_name, existing_request, latest_message, turn_number=1, existing_state=None, conversation_history=None, report_available=False) -> dict[str, object]`
- Produces: `reconcile_semantic_brief(previous_state, model_state, *, latest_message, existing_request) -> dict[str, object]`
- Produces: `semantic_brief_has_scope(brief_state) -> bool`
- Produces: `opening_question(country_name) -> str`

- [ ] **Step 1: Write failing semantic-context tests**
  Cover French Republics `1 to 5`, unseen topic, `traditional only`, `after independence`, explicit correction/clearing, broad acceptance, true ambiguity, and malformed-model fallback.

- [ ] **Step 2: Run the focused tests and confirm they fail**
  Run: `pytest tests/test_semantic_report_conversation.py -q`
  Expected: FAIL because the component does not exist.

- [ ] **Step 3: Implement the semantic component**
  Use structured model output with free-text `subject`, `scope`, `period`, `entity_range`, topic/angle lists, confidence, ambiguities, changed-fields and readiness. Use semantic actions `clarify`, `generate`, `status`, `converse`. Deterministic fallback may extract explicit four-digit year ranges but must not gate topics or force calendar periods.

- [ ] **Step 4: Run the focused tests**
  Run: `pytest tests/test_semantic_report_conversation.py -q`
  Expected: PASS.

- [ ] **Step 5: Commit**
  Commit message: `feat: add context-adaptive report conversation engine`

### Task 2: Integrate semantic engine with report writer without breaking report generation

**Files:**
- Modify: `src/flag_recognition/report_writer.py`
- Modify: `tests/test_report_generation_intent.py`
- Modify: `tests/test_report_brief_readiness.py`
- Modify: `tests/test_no_canned_chat.py`

**Interfaces:**
- Consumes: `continue_semantic_conversation(...)` from Task 1.
- Produces: existing public `continue_report_conversation(...)` compatibility API used by `app.py` and existing tests.

- [ ] **Step 1: Add failing integration tests**
  Assert semantic readiness is authoritative, unknown topics no longer need keyword registration, French Republics I–V generates without calendar years, and legacy callers still receive `generate` / clarification / reply-compatible actions.

- [ ] **Step 2: Run report conversation tests and confirm failure before integration**
  Run: `pytest tests/test_report_generation_intent.py tests/test_report_brief_readiness.py tests/test_no_canned_chat.py -q`
  Expected: at least the new integration assertions FAIL.

- [ ] **Step 3: Replace controller internals with a thin adapter**
  Delegate semantic interpretation to Task 1. Keep taxonomy helpers only for report-section optimization and calendar-period report enforcement. Map semantic `clarify` to the existing app clarification action and `status`/`converse` to non-generation replies while preserving `generate`.

- [ ] **Step 4: Run integration tests**
  Run: `pytest tests/test_report_generation_intent.py tests/test_report_brief_readiness.py tests/test_no_canned_chat.py -q`
  Expected: PASS.

- [ ] **Step 5: Commit**
  Commit message: `refactor: delegate report dialogue to semantic engine`

### Task 3: Make flag upload a country-only entry event

**Files:**
- Modify: `app.py`
- Create: `tests/test_flag_upload_entry_behavior.py`
- Modify: `tests/test_new_flag_session_reset.py`

**Interfaces:**
- Consumes: `opening_question(country_name)` from Task 1 and existing session reset behavior.
- Produces: `awaiting_interest` state with empty report request/state after flag-only recognition.

- [ ] **Step 1: Write failing flag-entry regressions**
  Assert the app no longer sends the synthetic `The country was identified from an uploaded flag...` message into report readiness, flag-only upload cannot surface `Generating your report now.`, and a flag plus text preserves the text as the first report requirement.

- [ ] **Step 2: Run flag-entry tests and confirm failure**
  Run: `pytest tests/test_flag_upload_entry_behavior.py tests/test_new_flag_session_reset.py tests/test_chat_uploaded_flag_rendering.py -q`
  Expected: FAIL on the new entry behavior.

- [ ] **Step 3: Implement country-only entry**
  On flag-only recognition, set country/session state, keep report brief empty, render the uploaded image, and use `opening_question(country)`. Do not call `continue_report_conversation` until the user submits genuine report text. When image and text arrive together, use the recognized country plus that text as the first semantic turn.

- [ ] **Step 4: Run flag-entry tests**
  Run: `pytest tests/test_flag_upload_entry_behavior.py tests/test_new_flag_session_reset.py tests/test_chat_uploaded_flag_rendering.py -q`
  Expected: PASS.

- [ ] **Step 5: Commit**
  Commit message: `fix: require user need after flag recognition`

### Task 4: Whole behavior regression verification

**Files:**
- Modify only if a regression exposes a defect in Tasks 1–3.

**Interfaces:**
- Consumes all prior tasks.
- Produces a merge-ready branch with the approved semantic behavior.

- [ ] **Step 1: Run the complete conversation/report regression set**
  Run: `pytest tests/test_semantic_report_conversation.py tests/test_report_generation_intent.py tests/test_report_brief_readiness.py tests/test_live_conversation_state.py tests/test_dialogue_latency.py tests/test_no_canned_chat.py tests/test_new_flag_session_reset.py tests/test_chat_uploaded_flag_rendering.py tests/test_flag_upload_entry_behavior.py tests/test_scoped_report_period_guard.py -q`
  Expected: PASS.

- [ ] **Step 2: Compile affected production files**
  Run: `python -m py_compile src/flag_recognition/report_conversation.py src/flag_recognition/report_writer.py app.py`
  Expected: exit code 0.

- [ ] **Step 3: Review branch diff against both specs**
  Verify no keyword-only readiness gate remains, no flag-only generation path remains, and no substantive country prose is added to pre-report chat.

- [ ] **Step 4: Commit any verification-only fixes if needed**
  Commit message: `test: verify adaptive report conversation flow`
