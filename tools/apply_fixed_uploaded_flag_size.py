from pathlib import Path


APP = Path(__file__).resolve().parents[1] / "app.py"

source = APP.read_text(encoding="utf-8")
old_size = "max-width:280px;max-height:190px;width:auto;height:auto;"
new_size = "width:2cm;height:1cm;"

if new_size not in source:
    if old_size not in source:
        raise RuntimeError("Uploaded flag preview size anchor not found")
    source = source.replace(old_size, new_size, 1)

APP.write_text(source, encoding="utf-8")
