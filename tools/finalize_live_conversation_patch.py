from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"
WRITER = ROOT / "src/flag_recognition/report_writer.py"
KNOWLEDGE = ROOT / "src/flag_recognition/country_knowledge.py"
DIALOGUE_TESTS = ROOT / "tests/test_dialogue_latency.py"
LIVE_TEST = ROOT / "tests/test_live_conversation_state.py"
NO_CANNED_TEST = ROOT / "tests/test_no_canned_chat.py"


def replace_if_present(text: str, old: str, new: str) -> str:
    if old in text:
        return text.replace(old, new, 1)
    return text


# ---------------------------------------------------------------------------
# Streamlit: normal assistant dialogue must come from the live model only.
# Operational failures remain UI status notices and are never stored as chat.
# ---------------------------------------------------------------------------
app = APP.read_text(encoding="utf-8")

app = replace_if_present(
    app,
    '''        except Exception:\n            assistant_text = "I couldn't read that image. Please upload another flag image."\n            st.session_state.fi_messages.append(\n                {"role": "assistant", "content": assistant_text}\n            )\n            _render_chat_message("assistant", assistant_text)\n''',
    '''        except Exception:\n            st.error("Image processing failed. Please upload a valid flag image.")\n''',
)

app = replace_if_present(
    app,
    '''            else:\n                assistant_text = "The flag recognition model is currently unavailable."\n                st.session_state.fi_messages.append(\n                    {"role": "assistant", "content": assistant_text}\n                )\n                _render_chat_message("assistant", assistant_text)\n''',
    '''            else:\n                st.error("Flag recognition model unavailable.")\n''',
)

app = replace_if_present(
    app,
    '''            except Exception as exc:\n                assistant_text = (\n                    "The language model could not answer this turn. "\n                    f"{type(exc).__name__}: {str(exc)[:220]}"\n                )\n                st.session_state.fi_messages.append(\n                    {"role": "assistant", "content": assistant_text}\n                )\n                _render_chat_message("assistant", assistant_text)\n''',
    '''            except Exception as exc:\n                st.error(\n                    "LLM request failed: "\n                    f"{type(exc).__name__}: {str(exc)[:220]}"\n                )\n''',
)

app = replace_if_present(
    app,
    '''                else:\n                    assistant_text = (\n                        "The language model returned an empty conversational response."\n                    )\n                    st.session_state.fi_messages.append(\n                        {"role": "assistant", "content": assistant_text}\n                    )\n                    _render_chat_message("assistant", assistant_text)\n''',
    '''                else:\n                    st.error("LLM returned an empty response.")\n''',
)

app = replace_if_present(
    app,
    '''            except Exception as exc:\n                assistant_text = (\n                    "The language model could not start the country conversation. "\n                    f"{type(exc).__name__}: {str(exc)[:220]}"\n                )\n\n        st.session_state.fi_messages.append(\n            {"role": "assistant", "content": assistant_text}\n        )\n        _render_chat_message("assistant", assistant_text)\n        return\n''',
    '''            except Exception as exc:\n                st.error(\n                    "LLM request failed while starting the country conversation: "\n                    f"{type(exc).__name__}: {str(exc)[:220]}"\n                )\n                assistant_text = ""\n\n        if assistant_text:\n            st.session_state.fi_messages.append(\n                {"role": "assistant", "content": assistant_text}\n            )\n            _render_chat_message("assistant", assistant_text)\n        return\n''',
)

app = replace_if_present(
    app,
    '''    if accepted and not authored_ready:\n        writer_error = str(report.get("authored_report_error") or "").strip()\n        assistant_text = (\n            "The language model did not produce a usable report. "\n            + (writer_error if writer_error else "No report content was returned.")\n        )\n        st.session_state.fi_messages.append(\n            {"role": "assistant", "content": assistant_text}\n        )\n        _render_chat_message("assistant", assistant_text)\n''',
    '''    if accepted and not authored_ready:\n        writer_error = str(report.get("authored_report_error") or "").strip()\n        st.error(\n            "Report generation failed. "\n            + (writer_error if writer_error else "No report content was returned.")\n        )\n''',
)

app = replace_if_present(
    app,
    '''    if resolved_code is None:\n        assistant_text = (\n            "The language model returned a country reference that the application could not resolve."\n        )\n        st.session_state.fi_messages.append(\n            {"role": "assistant", "content": assistant_text}\n        )\n        _render_chat_message("assistant", assistant_text)\n''',
    '''    if resolved_code is None:\n        st.error("Country resolution failed for the model response.")\n''',
)

app = replace_if_present(
    app,
    '''        except Exception as exc:\n            assistant_text = (\n                "The language model could not answer this turn. "\n                f"{type(exc).__name__}: {str(exc)[:220]}"\n            )\n            st.session_state.fi_messages.append(\n                {"role": "assistant", "content": assistant_text}\n            )\n            _render_chat_message("assistant", assistant_text)\n''',
    '''        except Exception as exc:\n            st.error(\n                "LLM request failed: "\n                f"{type(exc).__name__}: {str(exc)[:220]}"\n            )\n''',
)

app = replace_if_present(
    app,
    '''    except Exception as exc:\n        assistant_text = (\n            "The language model could not answer this turn. "\n            f"{type(exc).__name__}: {str(exc)[:220]}"\n        )\n        st.session_state.fi_messages.append(\n            {"role": "assistant", "content": assistant_text}\n        )\n        _render_chat_message("assistant", assistant_text)\n''',
    '''    except Exception as exc:\n        st.error(\n            "LLM request failed: "\n            f"{type(exc).__name__}: {str(exc)[:220]}"\n        )\n''',
)

APP.write_text(app, encoding="utf-8")


# ---------------------------------------------------------------------------
# Model controller: preserve broad-report behavior and explicitly require fresh
# context-derived wording rather than stock conversational text.
# ---------------------------------------------------------------------------
writer = WRITER.read_text(encoding="utf-8")
writer = replace_if_present(
    writer,
    '''        "complete report",\n        "full report",\n''',
    '''        "complete report",\n        "complete country report",\n        "full report",\n        "full country report",\n''',
)
writer = replace_if_present(
    writer,
    '''        "country_only and ask naturally what they want to know. "\n''',
    '''        "country_only and choose a natural context-aware response without using a stock question. "\n''',
)
writer = replace_if_present(
    writer,
    '''        "language. For greetings or genuinely country-independent questions, reply naturally "\n        "and briefly. Do not use canned wording and do not invent a country.\\n\\n"\n''',
    '''        "language. For greetings or genuinely country-independent questions, reply naturally "\n        "and briefly. Every reply must be freshly generated from the user's exact message "\n        "and current context. Never use a stock greeting, stock clarification, canned "\n        "confirmation, or template wording. Do not invent a country.\\n\\n"\n''',
)
writer = replace_if_present(
    writer,
    '''        "canned wording and do not behave like a questionnaire.\\n\\n"\n''',
    '''        "canned wording and do not behave like a questionnaire. Every reply must be freshly "\n        "generated from the exact latest message and accumulated context; never use stock dialogue.\\n\\n"\n''',
)
WRITER.write_text(writer, encoding="utf-8")


# Keep the existing Nobel syntax repair idempotent.
knowledge = KNOWLEDGE.read_text(encoding="utf-8")
knowledge = replace_if_present(
    knowledge,
    '''            if motivation:\n                detail += f" for {motivation.strip('"')}"\n''',
    '''            if motivation:\n                clean_motivation = motivation.strip('"')\n                detail += f" for {clean_motivation}"\n''',
)
KNOWLEDGE.write_text(knowledge, encoding="utf-8")


# ---------------------------------------------------------------------------
# Tests: fake model calls should never require a real Groq secret.
# ---------------------------------------------------------------------------
tests = DIALOGUE_TESTS.read_text(encoding="utf-8")
fixture = '''\n\n@pytest.fixture(autouse=True)\ndef _mock_report_writer_auth(monkeypatch):\n    monkeypatch.setattr(report_writer, "llm_auth_token", lambda: "test-key")\n'''
if "def _mock_report_writer_auth" not in tests:
    anchor = "from flag_recognition import report_writer\n"
    if anchor not in tests:
        raise RuntimeError("dialogue test import anchor missing")
    tests = tests.replace(anchor, anchor + fixture, 1)
DIALOGUE_TESTS.write_text(tests, encoding="utf-8")


LIVE_TEST.write_text(
    '''from pathlib import Path\n\n\ndef test_app_persists_report_and_live_context():\n    root = Path(__file__).resolve().parents[1]\n    app = (root / "app.py").read_text(encoding="utf-8")\n    assert '\"report_ready\"' in app\n    assert 'persistent_report_download' in app\n    assert 'conversation_history=st.session_state.fi_messages[:-1]' in app\n    assert 'report_available=bool(st.session_state.fi_report_available)' in app\n    assert 'turn.get(\"country\")' in app\n    assert 'if st.session_state.fi_report_available' in app\n    assert 'st.session_state.fi_country_code = None' not in app.split('if conversation_process:')[-1]\n\ndef test_country_knowledge_module_has_valid_nobel_motivation_code():\n    root = Path(__file__).resolve().parents[1]\n    text = (root / "src/flag_recognition/country_knowledge.py").read_text(encoding="utf-8")\n    assert "clean_motivation = motivation.strip" in text\n''',
    encoding="utf-8",
)

NO_CANNED_TEST.write_text(
    '''from pathlib import Path\n\n\ndef test_normal_chat_contains_no_canned_assistant_fallbacks():\n    root = Path(__file__).resolve().parents[1]\n    app = (root / "app.py").read_text(encoding="utf-8")\n    for phrase in (\n        "I couldn't read that image. Please upload another flag image.",\n        "The flag recognition model is currently unavailable.",\n        "The language model could not answer this turn.",\n        "The language model returned an empty conversational response.",\n        "The language model could not start the country conversation.",\n        "The language model did not produce a usable report.",\n        "The language model returned a country reference that the application could not resolve.",\n    ):\n        assert phrase not in app, phrase\n\ndef test_model_prompts_require_fresh_contextual_dialogue():\n    root = Path(__file__).resolve().parents[1]\n    writer = (root / "src/flag_recognition/report_writer.py").read_text(encoding="utf-8")\n    assert "Every reply must be freshly generated" in writer\n    assert "never use stock dialogue" in writer\n    assert '["reply", "ask", "generate"]' in writer\n''',
    encoding="utf-8",
)
