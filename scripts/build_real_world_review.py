"""Build a local HTML contact sheet for real-world challenge-set review."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/external_benchmark/real_world_manifest.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/external_benchmark/real_world_review.html"),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    cards = []
    for index, row in enumerate(rows, start=1):
        image_path = Path(row["path"])
        try:
            relative = image_path.relative_to(args.output.parent)
        except ValueError:
            relative = image_path

        cards.append(f"""
        <article class="card">
          <img src="{html.escape(relative.as_posix())}" loading="lazy">
          <div class="body">
            <strong>#{index} · {html.escape(row['class_code'].upper())}
              — {html.escape(row['country_name'])}</strong>
            <p>{html.escape(row.get('description', '')[:240])}</p>
            <a href="{html.escape(row['source_page'])}" target="_blank">Source</a>
            <div class="path">{html.escape(row['path'])}</div>
          </div>
        </article>
        """)

    document = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Real-world challenge review</title>
<style>
body{{font-family:system-ui;margin:24px;background:#f5f7fb;color:#172033}}
h1{{margin-bottom:4px}} .lead{{color:#667085;margin-top:0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px}}
.card{{background:white;border:1px solid #e3e7ef;border-radius:14px;overflow:hidden}}
.card img{{width:100%;height:190px;object-fit:contain;background:#eef1f5}}
.body{{padding:14px}} p{{font-size:13px;color:#596579;min-height:36px}}
.path{{font:11px monospace;color:#8a94a6;margin-top:8px;word-break:break-all}}
</style>
</head>
<body>
<h1>Real-world challenge review</h1>
<p class="lead">{len(rows)} candidate images. Verify that each visible flag matches the proposed label.</p>
<div class="grid">{''.join(cards)}</div>
</body>
</html>"""

    args.output.write_text(document, encoding="utf-8")
    print(f"Review page: {args.output}")
    print(f"Images     : {len(rows)}")


if __name__ == "__main__":
    main()
