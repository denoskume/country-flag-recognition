"""Local review server for the real-world challenge-set manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import uvicorn


DEFAULT_MANIFEST = Path("data/external_benchmark/real_world_manifest.csv")


class ReviewUpdate(BaseModel):
    index: int
    status: str
    note: str = ""


def load_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    return rows, fieldnames


def save_rows(
    path: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_app(manifest_path: Path) -> FastAPI:
    app = FastAPI(title="Challenge Set Review", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse("""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Challenge Set Review</title>
<style>
body{font-family:system-ui;margin:0;background:#f4f6fa;color:#172033}
.wrap{max-width:1100px;margin:0 auto;padding:24px}
.top{display:flex;justify-content:space-between;gap:16px;align-items:center;margin-bottom:18px}
.card{background:#fff;border:1px solid #e2e7ef;border-radius:16px;padding:18px}
img{width:100%;max-height:520px;object-fit:contain;background:#eef2f6;border-radius:12px}
.meta{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:14px}
.meta div{background:#f7f9fc;padding:10px;border-radius:10px}
.actions{display:flex;gap:10px;margin-top:16px;flex-wrap:wrap}
button{border:0;border-radius:10px;padding:11px 18px;font-weight:650;cursor:pointer}
.approve{background:#16803a;color:#fff}.reject{background:#b42318;color:#fff}.skip{background:#e6eaf0;color:#172033}
.note{width:100%;box-sizing:border-box;margin-top:14px;padding:10px;border:1px solid #d8dee8;border-radius:10px}
.progress{font-weight:700}.muted{color:#667085}.source{display:inline-block;margin-top:12px}
@media(max-width:720px){.meta{grid-template-columns:1fr}.top{align-items:flex-start;flex-direction:column}}
</style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <div>
      <h1>Real-world challenge review</h1>
      <div class="muted">Approve only when the visible flag clearly matches the proposed class.</div>
    </div>
    <div class="progress" id="progress">Loading…</div>
  </div>
  <div class="card">
    <img id="image" alt="Challenge image">
    <div class="meta">
      <div><small>Class</small><br><strong id="label"></strong></div>
      <div><small>Status</small><br><strong id="status"></strong></div>
      <div><small>Index</small><br><strong id="index"></strong></div>
    </div>
    <a class="source" id="source" target="_blank">Open Wikimedia source</a>
    <textarea class="note" id="note" rows="3" placeholder="Optional review note"></textarea>
    <div class="actions">
      <button class="approve" onclick="submitReview('approved')">Approve</button>
      <button class="reject" onclick="submitReview('rejected')">Reject</button>
      <button class="skip" onclick="nextPending()">Skip</button>
    </div>
  </div>
</div>
<script>
let rows=[];
let current=-1;

async function refresh(){
  const res=await fetch('/api/items');
  rows=await res.json();
  showNext();
}

function pendingIndices(){
  return rows.map((r,i)=>[r,i]).filter(([r])=>r.review_status==='pending').map(([,i])=>i);
}

function showNext(){
  const pending=pendingIndices();
  document.getElementById('progress').textContent=
    'Pending '+pending.length+' / '+rows.length;

  if(!pending.length){
    document.querySelector('.card').innerHTML=
      '<h2>Review complete</h2><p>All images have been reviewed.</p>';
    return;
  }

  current=pending[0];
  render();
}

function nextPending(){
  if(current<0)return;
  const pending=pendingIndices().filter(i=>i!==current);
  if(!pending.length){showNext();return;}
  current=pending[0];
  render();
}

function render(){
  const r=rows[current];
  document.getElementById('image').src='/image/'+current;
  document.getElementById('label').textContent=
    r.class_code.toUpperCase()+' — '+r.country_name;
  document.getElementById('status').textContent=r.review_status;
  document.getElementById('index').textContent=(current+1)+' / '+rows.length;
  document.getElementById('source').href=r.source_page;
  document.getElementById('note').value=r.review_note||'';
}

async function submitReview(status){
  const note=document.getElementById('note').value;
  const res=await fetch('/api/review',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({index:current,status,note})
  });
  if(!res.ok){
    alert('Could not save review.');
    return;
  }
  rows[current].review_status=status;
  rows[current].review_note=note;
  showNext();
}

refresh();
</script>
</body>
</html>
""")

    @app.get("/api/items")
    def items():
        rows, _ = load_rows(manifest_path)
        return rows

    @app.get("/image/{index}")
    def image(index: int):
        rows, _ = load_rows(manifest_path)
        if index < 0 or index >= len(rows):
            raise HTTPException(status_code=404, detail="Image index not found.")

        path = Path(rows[index]["path"])
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Image file not found.")
        return FileResponse(path)

    @app.post("/api/review")
    def review(update: ReviewUpdate):
        if update.status not in {"approved", "rejected", "pending"}:
            raise HTTPException(status_code=400, detail="Invalid review status.")

        rows, fieldnames = load_rows(manifest_path)
        if update.index < 0 or update.index >= len(rows):
            raise HTTPException(status_code=404, detail="Review index not found.")

        rows[update.index]["review_status"] = update.status
        rows[update.index]["review_note"] = update.note.strip()
        save_rows(manifest_path, rows, fieldnames)

        approved = sum(
            row["review_status"] == "approved"
            for row in rows
        )
        rejected = sum(
            row["review_status"] == "rejected"
            for row in rows
        )
        pending = sum(
            row["review_status"] == "pending"
            for row in rows
        )

        return {
            "ok": True,
            "approved": approved,
            "rejected": rejected,
            "pending": pending,
        }

    return app


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.manifest.is_file():
        raise SystemExit(
            f"Manifest not found: {args.manifest}\n"
            "Run collect_real_world_challenge.py first."
        )

    app = create_app(args.manifest)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
