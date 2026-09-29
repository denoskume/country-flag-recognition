"""Stable Streamlit Cloud entrypoint.

This wrapper keeps deployment configuration independent from the internal
application filename. The actual UI and application logic live in app.py.
"""

from pathlib import Path

APP_PATH = Path(__file__).with_name("app.py")
exec(compile(APP_PATH.read_text(encoding="utf-8"), str(APP_PATH), "exec"))
