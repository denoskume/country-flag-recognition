from flag_recognition.learning import answer_country_question


INTELLIGENCE = {
    "identity": {
        "capital": {
            "value": "Yamoussoukro",
            "source": "Wikidata",
            "reference_year": None,
            "source_url": "",
        }
    },
    "government": {
        "head_of_state": {
            "value": "Example President",
            "source": "Official source",
            "reference_year": "2026",
            "source_url": "",
        }
    },
    "historical_timeline": [
        {
            "label": "Historical event — 1960",
            "period": "1960",
            "summary": "In 1960 the country became independent.",
            "sources": ["Wikipedia"],
            "source_urls": ["https://example.com/history"],
        }
    ],
    "flag": {
        "symbolism": [
            {
                "value": "The colours have documented national symbolism.",
                "source": "Wikipedia",
                "source_url": "https://example.com/flag",
            }
        ]
    },
    "emergency": [
        {
            "service": "Police",
            "number": "170",
            "source": "Official emergency source",
            "source_url": "",
        }
    ],
}


def test_question_about_capital_returns_grounded_fact():
    results = answer_country_question("What is the capital?", INTELLIGENCE)

    assert results
    assert results[0]["value"] == "Yamoussoukro"


def test_question_about_history_returns_timeline():
    results = answer_country_question(
        "What happened in the history in 1960?",
        INTELLIGENCE,
    )

    assert results
    assert "independent" in results[0]["value"]


def test_question_about_emergency_number_returns_service():
    results = answer_country_question(
        "What is the emergency police number?",
        INTELLIGENCE,
    )

    assert results
    assert results[0]["value"] == "170"


def test_unknown_question_returns_no_invented_answer():
    results = answer_country_question(
        "What is the national flower?",
        INTELLIGENCE,
    )

    assert results == []
