from pathlib import Path

path = Path("app.py")
text = path.read_text(encoding="utf-8")

# Import the deterministic country-only opening question at top level, before
# the optional report-writer try/except block.
import_line = "from flag_recognition.report_conversation import opening_question\n"
if import_line not in text:
    marker = "try:\n    from flag_recognition.report_writer import"
    index = text.find(marker)
    if index < 0:
        raise SystemExit("Could not find top-level report_writer try block")
    text = text[:index] + import_line + text[index:]

# Allow flag recognition to establish country context without emitting an opening
# question when the same user turn already contains report-requirement text.
show_start = text.find("def show_result(")
if show_start < 0:
    raise SystemExit("Could not find show_result")
show_signature_end = text.find("\n):", show_start)
if show_signature_end < 0:
    raise SystemExit("Could not find show_result signature end")
signature = text[show_start:show_signature_end]
if "defer_opening: bool = False" not in signature:
    text = (
        text[:show_signature_end]
        + "\n    defer_opening: bool = False,"
        + text[show_signature_end:]
    )

# Remove the synthetic upload instruction from the report-requirements controller.
synthetic = '"The country was identified from an uploaded flag. "'
synthetic_pos = text.find(synthetic)
if synthetic_pos >= 0:
    opening_start = text.rfind("        assistant_text = str(opening_reply or \"\").strip()", show_start, synthetic_pos)
    assistant_if = text.find("        if assistant_text:", synthetic_pos)
    if opening_start < 0 or assistant_if < 0:
        raise SystemExit("Could not isolate synthetic opening block")
    replacement = (
        "        assistant_text = \"\"\n"
        "        if not defer_opening:\n"
        "            assistant_text = str(opening_reply or opening_question(country)).strip()\n\n"
    )
    text = text[:opening_start] + replacement + text[assistant_if:]

if synthetic in text:
    raise SystemExit("Synthetic flag-upload instruction still present")

# A flag submitted with text uses that text as the first real semantic turn. A
# flag-only submission asks the open country question and waits for the user.
old_upload = "if process and image is not None:\n    show_result(image=image)"
new_upload = (
    "if process and image is not None:\n"
    "    show_result(image=image, defer_opening=bool(prompt_text))\n"
    "    if prompt_text:\n"
    "        conversation_text = prompt_text\n"
    "        conversation_process = True"
)
if old_upload in text:
    text = text.replace(old_upload, new_upload, 1)
elif "defer_opening=bool(prompt_text)" not in text:
    raise SystemExit("Could not find flag upload show_result call")

required = (
    "defer_opening: bool = False",
    "assistant_text = str(opening_reply or opening_question(country)).strip()",
    "defer_opening=bool(prompt_text)",
    "conversation_text = prompt_text",
    "conversation_process = True",
    'fi_stage = "awaiting_interest"',
    'fi_report_request = ""',
    "fi_report_state = {}",
)
missing = [value for value in required if value not in text]
if missing:
    raise SystemExit(f"Missing required app markers: {missing}")

path.write_text(text, encoding="utf-8")
