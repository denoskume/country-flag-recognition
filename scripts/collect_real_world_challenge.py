"""Collect a real-world flag challenge set from Wikimedia Commons.

The collector targets difficult classes plus a reproducible broad sample. Search
results are never trusted as labels automatically: every downloaded image enters
the manifest with review_status=pending and must be approved before evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import re
import urllib.parse
import urllib.request
import urllib.error
import time

import numpy as np
from PIL import Image, ImageOps


API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "country-flag-recognition/1.0 (external benchmark research)"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
HARD_CODES = [
    "ae", "bv", "gf", "gp", "hk", "id", "lv", "mq", "mp", "nc",
    "no", "pm", "ro", "sj", "sv", "sy",
    "ci", "ie", "fr", "yt", "td", "mc", "ps", "tr",
]
EXCLUDE_TERMS = {
    "map", "icon", "svg", "coat of arms", "emblem", "roundel",
    "diagram", "construction sheet", "color sheet", "colour sheet",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--taxonomy", type=Path, default=Path("data/taxonomy.csv"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/external_benchmark/real_world_known"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/external_benchmark/real_world_manifest.csv"),
    )
    parser.add_argument("--images-per-class", type=int, default=3)
    parser.add_argument("--broad-extra-classes", type=int, default=26)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--near-duplicate-hamming",
        type=int,
        default=5,
        help="Reject candidate dHash within this Hamming distance of data/raw.",
    )
    return parser.parse_args()


def api_get(
    params: dict[str, str | int],
    retries: int = 6,
    base_delay: float = 2.0,
) -> dict:
    encoded = urllib.parse.urlencode({
        "format": "json",
        "formatversion": "2",
        **params,
    })
    request = urllib.request.Request(
        f"{API}?{encoded}",
        headers={"User-Agent": USER_AGENT},
    )

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.load(response)
            time.sleep(0.8)
            return payload
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == retries - 1:
                raise

            retry_after = exc.headers.get("Retry-After")
            delay = (
                float(retry_after)
                if retry_after and retry_after.isdigit()
                else base_delay * (2 ** attempt)
            )
            delay = min(delay, 60.0)
            print(
                f"Wikimedia rate limit (429). "
                f"Retrying in {delay:.0f}s..."
            )
            time.sleep(delay)

    raise RuntimeError("Wikimedia request failed after retries.")


def load_taxonomy(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["class_code"].lower(): row["name"]
            for row in csv.DictReader(handle)
        }


def iter_images(root: Path):
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def dhash_image(image: Image.Image, hash_size: int = 16) -> int:
    gray = ImageOps.grayscale(image).resize(
        (hash_size + 1, hash_size),
        Image.Resampling.LANCZOS,
    )
    array = np.asarray(gray, dtype=np.int16)
    bits = array[:, 1:] > array[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def dhash_path(path: Path) -> int | None:
    try:
        with Image.open(path) as image:
            return dhash_image(image)
    except (OSError, ValueError):
        return None


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def raw_hashes(raw_dir: Path) -> list[int]:
    output = []
    for path in iter_images(raw_dir):
        value = dhash_path(path)
        if value is not None:
            output.append(value)
    return output


def search_files(country_name: str, limit: int = 30) -> list[dict]:
    queries = [
        f'"flag of {country_name}"',
        f'{country_name} flag waving',
        f'{country_name} flag building',
    ]

    seen = set()
    results = []

    for query in queries:
        data = api_get({
            "action": "query",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": 6,
            "gsrlimit": min(limit, 50),
            "prop": "imageinfo",
            "iiprop": "url|mime|extmetadata",
            "iiurlwidth": 1000,
            "iiextmetadatafilter": "LicenseShortName|Artist|ImageDescription",
        })

        for page in data.get("query", {}).get("pages", []):
            title = page.get("title", "")
            key = title.lower()
            if key in seen:
                continue
            seen.add(key)

            info = (page.get("imageinfo") or [{}])[0]
            mime = info.get("mime", "")
            if not mime.startswith("image/"):
                continue

            title_lower = title.lower()
            if any(term in title_lower for term in EXCLUDE_TERMS):
                continue

            thumb = info.get("thumburl") or info.get("url")
            if not thumb:
                continue

            results.append({
                "title": title,
                "page_url": "https://commons.wikimedia.org/wiki/"
                    + urllib.parse.quote(title.replace(" ", "_")),
                "image_url": thumb,
                "metadata": info.get("extmetadata", {}),
            })

    return results


def metadata_value(metadata: dict, key: str) -> str:
    value = metadata.get(key, {})
    if isinstance(value, dict):
        return re.sub(r"<[^>]+>", "", str(value.get("value", "")))
    return ""


def download_image(
    url: str,
    retries: int = 5,
    base_delay: float = 2.0,
) -> tuple[bytes, Image.Image]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT},
    )

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = response.read(12 * 1024 * 1024)
            time.sleep(0.5)
            image = Image.open(io.BytesIO(payload)).convert("RGB")
            return payload, image
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == retries - 1:
                raise

            retry_after = exc.headers.get("Retry-After")
            delay = (
                float(retry_after)
                if retry_after and retry_after.isdigit()
                else base_delay * (2 ** attempt)
            )
            delay = min(delay, 60.0)
            print(
                f"Image rate limit (429). "
                f"Retrying in {delay:.0f}s..."
            )
            time.sleep(delay)

    raise RuntimeError("Image download failed after retries.")


def safe_name(title: str) -> str:
    stem = title.removeprefix("File:")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    return stem[:120] or "image"


def main():
    args = parse_args()
    taxonomy = load_taxonomy(args.taxonomy)
    rng = random.Random(args.seed)

    hard = [code for code in HARD_CODES if code in taxonomy]
    remaining = sorted(set(taxonomy) - set(hard))
    broad = rng.sample(
        remaining,
        min(args.broad_extra_classes, len(remaining)),
    )
    target_codes = hard + broad

    training_hashes = raw_hashes(args.raw_dir)
    accepted_hashes: list[int] = []
    rows = []

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    if args.manifest.is_file():
        with args.manifest.open(newline="", encoding="utf-8") as handle:
            existing_rows = list(csv.DictReader(handle))
        rows.extend(existing_rows)

        for row in existing_rows:
            try:
                accepted_hashes.append(int(row["dhash16"], 16))
            except (KeyError, ValueError, TypeError):
                pass

        print(
            f"Resuming existing manifest: {len(existing_rows)} image(s)"
        )

    fieldnames = [
        "path", "class_code", "country_name", "source", "source_page",
        "source_image", "license", "artist", "description",
        "sha256_download", "dhash16", "review_status", "review_note",
    ]

    def save_manifest() -> None:
        with args.manifest.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            writer.writerows(rows)

    for class_index, code in enumerate(target_codes, start=1):
        name = taxonomy[code]
        class_dir = args.output_dir / code
        class_dir.mkdir(parents=True, exist_ok=True)

        existing_for_class = [
            row
            for row in rows
            if row.get("class_code") == code
            and Path(row.get("path", "")).is_file()
        ]
        saved = len(existing_for_class)

        if saved >= args.images_per_class:
            print(
                f"[{class_index:02d}/{len(target_codes):02d}] "
                f"{code} {name}: {saved}/{args.images_per_class} (already complete)"
            )
            continue

        try:
            candidates = search_files(name)
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                print(
                    f"[{class_index:02d}/{len(target_codes):02d}] "
                    f"{code} {name}: API rate-limited, skipping for this run"
                )
                save_manifest()
                continue
            raise

        for candidate in candidates:
            if saved >= args.images_per_class:
                break

            try:
                payload, image = download_image(candidate["image_url"])
            except Exception as exc:
                print(f"{code}: skip download error: {exc}")
                continue

            if min(image.size) < 120:
                continue

            candidate_hash = dhash_image(image)
            if any(
                hamming(candidate_hash, known_hash)
                <= args.near_duplicate_hamming
                for known_hash in training_hashes
            ):
                continue
            if any(
                hamming(candidate_hash, known_hash) <= 2
                for known_hash in accepted_hashes
            ):
                continue

            accepted_hashes.append(candidate_hash)

            filename = f"{saved + 1:02d}_{safe_name(candidate['title'])}.jpg"
            path = class_dir / filename
            image.save(path, format="JPEG", quality=92)

            rows.append({
                "path": path.as_posix(),
                "class_code": code,
                "country_name": name,
                "source": "Wikimedia Commons",
                "source_page": candidate["page_url"],
                "source_image": candidate["image_url"],
                "license": metadata_value(candidate["metadata"], "LicenseShortName"),
                "artist": metadata_value(candidate["metadata"], "Artist"),
                "description": metadata_value(candidate["metadata"], "ImageDescription"),
                "sha256_download": hashlib.sha256(payload).hexdigest(),
                "dhash16": format(candidate_hash, "064x"),
                "review_status": "pending",
                "review_note": "",
            })
            saved += 1
            save_manifest()

        print(
            f"[{class_index:02d}/{len(target_codes):02d}] "
            f"{code} {name}: {saved}/{args.images_per_class}"
        )

        time.sleep(1.5)

    save_manifest()

    print()
    print(f"Target classes       : {len(target_codes)}")
    print(f"Images downloaded    : {len(rows)}")
    print(f"Manifest             : {args.manifest}")
    print("Review status        : all pending")
    print()
    print(
        "Next: review the contact sheet and mark only visually correct "
        "country labels as approved before evaluation."
    )


if __name__ == "__main__":
    main()
