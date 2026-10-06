from pathlib import Path


APP = Path("app.py")
WRITER = Path("src/flag_recognition/report_writer.py")

app_source = APP.read_text(encoding="utf-8")
writer_source = WRITER.read_text(encoding="utf-8")

state_anchor = '''if "fi_report_available" not in st.session_state:
    st.session_state.fi_report_available = False

for message in st.session_state.fi_messages:
'''
state_replacement = '''if "fi_report_available" not in st.session_state:
    st.session_state.fi_report_available = False


def _reset_flag_intelligence_session_for_new_upload() -> None:
    """Start a clean country session while preserving the current upload turn."""
    st.session_state.fi_stage = "idle"
    st.session_state.fi_country_code = None
    st.session_state.fi_country_name = None
    st.session_state.fi_report_request = ""
    st.session_state.fi_report_state = {}
    st.session_state.fi_messages = []
    st.session_state.fi_image_bytes = None
    st.session_state.fi_clarification_turn = 0
    st.session_state.fi_pending_input = None
    st.session_state.fi_report_pdf = None
    st.session_state.fi_report_filename = ""
    st.session_state.fi_report_available = False


for message in st.session_state.fi_messages:
'''
if state_anchor not in app_source:
    raise RuntimeError("app session-state anchor not found")
app_source = app_source.replace(state_anchor, state_replacement, 1)

upload_anchor = '''    if prompt_files:
        uploaded_image = prompt_files[0]
'''
upload_replacement = '''    if prompt_files:
        _reset_flag_intelligence_session_for_new_upload()
        uploaded_image = prompt_files[0]
'''
if upload_anchor not in app_source:
    raise RuntimeError("flag upload anchor not found")
app_source = app_source.replace(upload_anchor, upload_replacement, 1)

period_helper_anchor = '''def _has_explicit_time_range(user_request: str) -> bool:
    return _extract_requested_year_range(user_request) is not None
'''
period_helper = '''def _enforce_requested_year_range(
    report: dict[str, Any],
    year_range: tuple[int, int] | None,
) -> dict[str, Any]:
    """Remove explicitly dated material that falls outside a requested time window."""
    if year_range is None:
        return dict(report)

    start_year, end_year = year_range
    year_pattern = re.compile(r"\\b((?:1[5-9]|20)\\d{2})(?:s)?\\b")
    scoped: dict[str, Any] = {}

    for key, raw_value in report.items():
        if str(key).startswith("__") or not isinstance(raw_value, str):
            scoped[key] = raw_value
            continue

        paragraphs: list[str] = []
        for raw_paragraph in re.split(r"\\n{2,}", raw_value):
            paragraph = raw_paragraph.strip()
            if not paragraph:
                continue

            clauses: list[str] = []
            for raw_clause in re.split(r";\\s*", paragraph):
                sentences = [
                    item.strip()
                    for item in re.split(r"(?<=[.!?])\\s+", raw_clause.strip())
                    if item.strip()
                ]
                kept_sentences: list[str] = []
                for sentence in sentences:
                    years = [int(value) for value in year_pattern.findall(sentence)]
                    if years and any(year < start_year or year > end_year for year in years):
                        continue
                    kept_sentences.append(sentence)
                if kept_sentences:
                    clauses.append(" ".join(kept_sentences))

            if clauses:
                paragraphs.append("; ".join(clauses))

        scoped[key] = "\\n\\n".join(paragraphs).strip()

    return scoped


def _has_explicit_time_range(user_request: str) -> bool:
    return _extract_requested_year_range(user_request) is not None
'''
if period_helper_anchor not in writer_source:
    raise RuntimeError("period helper anchor not found")
writer_source = writer_source.replace(period_helper_anchor, period_helper, 1)

old_range_rule = '''        range_rule = (
            f"The requested time window is strictly {start_year}-{end_year}. "
            "Do not add a separate early-history section or substantive events before "
            f"{start_year}. Use earlier history only as one or two contextual sentences "
            "in the introduction if indispensable. "
        )
'''
new_range_rule = '''        range_rule = (
            f"The requested time window is strictly {start_year}-{end_year}. "
            f"Do not include any explicitly dated fact, event, law, statistic, timeline entry, "
            f"or person-specific milestone before {start_year} or after {end_year}. "
            "Do not mention out-of-range years even as background. If earlier context is essential, "
            "state it briefly without dated details. "
        )
'''
if old_range_rule not in writer_source:
    raise RuntimeError("focused range rule anchor not found")
writer_source = writer_source.replace(old_range_rule, new_range_rule, 1)

old_review = '''            "Review the section below as a subject-matter expert. Independently verify "
            "names, dates, chronology, attribution, institutions, works, concepts, places, "
            "and causal claims. Remove anything uncertain, generic, unsupported, outside "
            "the requested scope, or temporally misleading. Preserve useful concrete facts "
'''
new_review = '''            "Review the section below as a subject-matter expert. Verify every concrete factual claim "
            "with browser search before retaining it, especially names, dates, laws, institutions, "
            "works, concepts, places, quantitative statements, chronology, attribution, and causal claims. "
            "Prefer primary or official sources, then major international institutions and reputable "
            "academic or reference sources. Remove or qualify any claim you cannot verify. Remove anything "
            "generic, unsupported, outside the requested scope, or temporally misleading. Preserve useful concrete facts "
'''
if old_review not in writer_source:
    raise RuntimeError("scoped review prompt anchor not found")
writer_source = writer_source.replace(old_review, new_review, 1)

pre_review_anchor = '''    if not result:
        raise RuntimeError("LLM scoped report generation failed: empty report")

    missing_sections = [
'''
pre_review_replacement = '''    if not result:
        raise RuntimeError("LLM scoped report generation failed: empty report")

    result = _enforce_requested_year_range(result, year_range)

    missing_sections = [
'''
if pre_review_anchor not in writer_source:
    raise RuntimeError("pre-review scope guard anchor not found")
writer_source = writer_source.replace(pre_review_anchor, pre_review_replacement, 1)

post_review_anchor = '''    result = _review_scoped_report(
        api_key=api_key,
        model=model,
        country_name=country_name,
        user_request=user_request,
        section_keys=section_keys,
        draft=result,
        brief_state=brief_state,
    )
    result["__qa_passed"] = True
'''
post_review_replacement = '''    result = _review_scoped_report(
        api_key=api_key,
        model=model,
        country_name=country_name,
        user_request=user_request,
        section_keys=section_keys,
        draft=result,
        brief_state=brief_state,
    )
    result = _enforce_requested_year_range(result, year_range)
    result["__qa_passed"] = True
'''
if post_review_anchor not in writer_source:
    raise RuntimeError("post-review scope guard anchor not found")
writer_source = writer_source.replace(post_review_anchor, post_review_replacement, 1)

APP.write_text(app_source, encoding="utf-8")
WRITER.write_text(writer_source, encoding="utf-8")
