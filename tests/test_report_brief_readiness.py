from flag_recognition import report_writer


def test_history_topic_with_explicit_period_is_ready_without_optional_refinements():
    assert report_writer._brief_is_sufficiently_specific(
        "",
        "Economical history from 1950 to 2020",
    )


def test_history_topic_with_explicit_period_has_no_required_follow_up_dimension():
    assert report_writer._brief_missing_dimension(
        "",
        "Economical history from 1950 to 2020",
    ) == ""


def test_cuisine_is_a_valid_report_topic_without_extra_scope_question():
    assert report_writer._requested_topic_groups(
        "I would like to explore Japan cuisine",
    ) == ("cuisine",)
    assert report_writer._brief_is_sufficiently_specific(
        "",
        "I would like to explore Japan cuisine",
    )


def test_traditional_cuisine_refinement_does_not_require_another_question():
    assert report_writer._requested_topic_groups("traditional cuisine") == ("cuisine",)
    assert report_writer._brief_missing_dimension(
        "Japan cuisine",
        "traditional cuisine",
    ) == ""


def test_cuisine_request_stays_focused_on_cuisine_sections():
    sections = report_writer._requested_report_sections("Japan cuisine")

    assert "culture_cuisine_music_sport" in sections
    assert "heritage_landmarks" not in sections
    assert "literature_philosophy_thought" not in sections
