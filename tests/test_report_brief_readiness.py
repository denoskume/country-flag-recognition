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


def test_bounded_historical_subject_does_not_force_a_calendar_period():
    brief_state = {
        "subject": "French Republics",
        "topics": ["history"],
        "period": "",
        "angles": ["First through Fifth Republic"],
        "depth": "",
        "exclusions": [],
        "current_events": False,
        "other_constraints": [],
        "ready": True,
        "missing": [],
    }

    assert report_writer._brief_missing_dimension_from_state(
        brief_state,
        "history about French republics",
        "from 1 to 5",
    ) == ""


def test_model_declared_missing_dimension_is_respected_without_hardcoded_topic_rules():
    brief_state = {
        "subject": "French Republics",
        "topics": ["history"],
        "period": "",
        "angles": [],
        "depth": "",
        "exclusions": [],
        "current_events": False,
        "other_constraints": [],
        "ready": False,
        "missing": ["scope"],
    }

    assert report_writer._brief_missing_dimension_from_state(
        brief_state,
        "history about French republics",
        "",
    ) == "scope"


def test_arbitrary_model_topic_can_form_a_report_scope_without_keyword_registration():
    brief_state = {
        "subject": "French semiconductor sovereignty",
        "topics": ["semiconductor supply chain"],
        "period": "",
        "angles": ["industrial policy"],
        "depth": "balanced overview",
        "exclusions": [],
        "current_events": False,
        "other_constraints": [],
        "ready": True,
        "missing": [],
    }

    assert report_writer._semantic_brief_has_scope(brief_state)


def test_new_model_scope_replaces_stale_previous_topic_when_user_changes_direction():
    previous_state = {
        "subject": "French Republics",
        "topics": ["history"],
        "period": "",
        "angles": ["First through Fifth Republic"],
        "depth": "",
        "exclusions": [],
        "current_events": False,
        "other_constraints": [],
        "ready": True,
        "missing": [],
    }
    model_state = {
        "subject": "French cuisine",
        "topics": ["traditional cuisine"],
        "period": "",
        "angles": [],
        "depth": "",
        "exclusions": [],
        "current_events": False,
        "other_constraints": [],
        "ready": True,
        "missing": [],
    }

    reconciled = report_writer._reconcile_report_brief_state(
        "history about French republics from 1 to 5",
        "actually, focus only on traditional cuisine",
        previous_state,
        model_state,
    )

    assert reconciled["topics"] == ["traditional cuisine"]
    assert reconciled["subject"] == "French cuisine"
