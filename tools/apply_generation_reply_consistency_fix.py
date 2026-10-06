from pathlib import Path


WRITER = Path(__file__).resolve().parents[1] / "src" / "flag_recognition" / "report_writer.py"
source = WRITER.read_text(encoding="utf-8")

old = '''    else:
        action = "reply"

    if action == "ask":
'''

new = '''    else:
        action = "reply"

    # The deterministic controller may promote a model clarification turn to
    # generation once the report brief is already complete. Do not surface the
    # obsolete clarification text beside a generated PDF.
    if action == "generate" and requested_action != "generate":
        reply = "Generating your report now."

    if action == "ask":
'''

if old not in source:
    raise RuntimeError("generation action anchor not found")

WRITER.write_text(source.replace(old, new, 1), encoding="utf-8")
