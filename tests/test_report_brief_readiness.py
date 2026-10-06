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
