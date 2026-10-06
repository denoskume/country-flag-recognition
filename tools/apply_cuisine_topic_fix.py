from pathlib import Path

path = Path("src/flag_recognition/report_writer.py")
text = path.read_text(encoding="utf-8")

old_groups = '''    "economy": (\n        "economy_trade_industries",\n        "infrastructure_transport_energy",\n    ),\n    "culture": (\n'''
new_groups = '''    "economy": (\n        "economy_trade_industries",\n        "infrastructure_transport_energy",\n    ),\n    "cuisine": (\n        "culture_cuisine_music_sport",\n    ),\n    "culture": (\n'''

old_markers = '''        "economy": ("economy", "economic", "economical", "trade", "industry", "industrial"),\n        "culture": ("culture", "cultural", "arts", "literature", "music", "cinema"),\n'''
new_markers = '''        "economy": ("economy", "economic", "economical", "trade", "industry", "industrial"),\n        "cuisine": ("cuisine", "culinary", "gastronomy", "food"),\n        "culture": ("culture", "cultural", "arts", "literature", "music", "cinema"),\n'''

old_labels = '''            "economy": "economy",\n            "culture": "culture",\n'''
new_labels = '''            "economy": "economy",\n            "cuisine": "cuisine",\n            "culture": "culture",\n'''

for old, new, name in (
    (old_groups, new_groups, "topic section group"),
    (old_markers, new_markers, "topic markers"),
    (old_labels, new_labels, "canonical label"),
):
    if text.count(old) != 1:
        raise SystemExit(f"Expected exactly one {name} target, found {text.count(old)}")
    text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")
