from pathlib import Path


APP = Path(__file__).resolve().parents[1] / "app.py"

source = APP.read_text(encoding="utf-8")
old_style = (
    "display:block;max-width:280px;max-height:190px;width:auto;height:auto;"
    "object-fit:contain;border-radius:12px;"
)
new_style = (
    "display:block;width:2cm;height:1cm;"
    "object-fit:contain;border-radius:6px;"
)

if new_style not in source:
    if old_style not in source:
        raise RuntimeError("Uploaded flag preview style anchor not found")
    source = source.replace(old_style, new_style, 1)

APP.write_text(source, encoding="utf-8")
