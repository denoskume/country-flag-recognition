"""Compatibility entrypoint for hosted Streamlit deployments."""

from pathlib import Path

APP_PATH = Path(__file__).with_name("app.py")
exec(compile(APP_PATH.read_text(encoding="utf-8"), str(APP_PATH), "exec"))
