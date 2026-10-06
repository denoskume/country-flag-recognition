from flag_recognition.report_writer import _enforce_requested_year_range


def test_period_guard_removes_dated_facts_before_and_after_requested_window():
    report = {
        "historical_journey": (
            "The 1949 Dodge Plan stabilized the economy. "
            "In 1950 reconstruction accelerated. "
            "The 1964 Tokyo Olympics marked a major milestone. "
            "A 2021 reform falls outside the requested window."
        ),
        "key_historical_timeline": (
            "1949 – Dodge Plan; 1950 – reconstruction; 1989 – asset bubble peak; "
            "2020 – pandemic shock; 2021 – later reform"
        ),
    }

    scoped = _enforce_requested_year_range(report, (1950, 2020))

    combined = " ".join(scoped.values())
    assert "1949" not in combined
    assert "2021" not in combined
    assert "1950" in combined
    assert "1964" in combined
    assert "1989" in combined
    assert "2020" in combined


def test_period_guard_leaves_undated_context_and_in_range_decades():
    report = {
        "introduction": (
            "Japan entered the period with a damaged industrial base. "
            "During the 1950s growth accelerated rapidly."
        )
    }

    scoped = _enforce_requested_year_range(report, (1950, 2020))

    assert "damaged industrial base" in scoped["introduction"]
    assert "1950s" in scoped["introduction"]
