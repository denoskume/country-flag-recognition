from pathlib import Path


def test_normal_chat_contains_no_canned_assistant_fallbacks():
    root = Path(__file__).resolve().parents[1]
    app = (root / "app.py").read_text(encoding="utf-8")
    for phrase in (
        "I couldn't read that image. Please upload another flag image.",
        "The flag recognition model is currently unavailable.",
        "The language model could not answer this turn.",
        "The language model returned an empty conversational response.",
        "The language model could not start the country conversation.",
        "The language model did not produce a usable report.",
        "The language model returned a country reference that the application could not resolve.",
    ):
        assert phrase not in app, phrase

def test_model_prompts_require_fresh_contextual_dialogue():
    root = Path(__file__).resolve().parents[1]
    writer = (root / "src/flag_recognition/report_writer.py").read_text(encoding="utf-8")
    assert "Every reply must be freshly generated" in writer
    assert "never use stock dialogue" in writer
    assert '["reply", "ask", "generate"]' in writer
