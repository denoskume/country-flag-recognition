"""Worldwide QA audit for Flag Intelligence country reports.

Audits every project country class (ISO 3166-1 + Kosovo/XK) against the
Country Intelligence pipeline. The audit is source-aware and checks both
coverage and common content-quality regressions before a country is considered
ready for an official report.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import re
import time
from typing import Any

import pycountry
import requests

from flag_recognition.country_info import fetch_country_profile
from flag_recognition.country_intelligence import (
    build_from_legacy_profile,
    validate_country_intelligence,
)
from flag_recognition.country_knowledge import enrich_from_encyclopedia
from flag_recognition.flag_knowledge import enrich_flag_profile
from flag_recognition.report_manifest import (
    OFFICIAL_REPORT_SECTIONS,
    build_report_manifest,
    missing_required_sections,
)


RAW_TEMPLATE_RE = re.compile(r"\{\{[^{}]+\}\}")
RAW_WIKIDATA_RE = re.compile(r"\bQ\d+\b")
BIBLIOGRAPHY_RE = re.compile(
    r"\b(?:university press|academic journal|publications of|isbn|doi:|"
    r"bibliography|references)\b",
    re.IGNORECASE,
)


def all_project_codes() -> list[str]:
    codes = sorted(country.alpha_2.lower() for country in pycountry.countries)
    if "xk" not in codes:
        codes.append("xk")
    return sorted(codes)


def profile_payload(profile: Any) -> dict[str, Any]:
    payload = asdict(profile)
    population = payload.pop("population", {}) or {}
    gdp = payload.pop("gdp", {}) or {}

    payload["population"] = population.get("value")
    payload["population_year"] = population.get("year")
    payload["population_source"] = (
        population.get("source")
        or payload.get("population_source")
        or "World Bank"
    )
    payload["gdp_value_usd"] = gdp.get("value_usd")
    payload["gdp_year"] = gdp.get("year")
    payload["gdp_source"] = gdp.get("source") or "World Bank"
    return payload


def evidence_value(section: Any, key: str = "context") -> str:
    if not isinstance(section, dict):
        return ""
    item = section.get(key)
    if not isinstance(item, dict):
        return ""
    return str(item.get("value") or "").strip()


def scan_strings(value: Any) -> list[str]:
    out: list[str] = []
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for child in value.values():
            out.extend(scan_strings(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            out.extend(scan_strings(child))
    return out


def content_quality_issues(
    intelligence: dict[str, Any],
    profile: dict[str, Any],
) -> list[str]:
    issues: list[str] = []

    strings = scan_strings(intelligence) + scan_strings(profile)
    if any(RAW_TEMPLATE_RE.search(text) for text in strings):
        issues.append("raw_mediawiki_template")
    if any(RAW_WIKIDATA_RE.search(text) for text in strings):
        issues.append("raw_wikidata_identifier")

    timeline = intelligence.get("historical_timeline") or ()
    for event in timeline:
        if not isinstance(event, dict):
            continue
        summary = str(event.get("summary") or "")
        if BIBLIOGRAPHY_RE.search(summary):
            issues.append("bibliography_in_timeline")
            break

    independence = str(profile.get("independence_day") or "").strip()
    powers = str(profile.get("former_colonial_powers") or "").strip()
    if independence in {"Not available", "Not applicable"} and powers not in {
        "",
        "Not available",
        "Not applicable",
    }:
        issues.append("colonial_power_without_independence_transition")

    flag = intelligence.get("flag")
    if isinstance(flag, dict):
        adoption = flag.get("adoption_date")
        if isinstance(adoption, dict):
            value = str(adoption.get("value") or "")
            if "{{" in value or "}}" in value:
                issues.append("raw_flag_adoption_template")

    domain_expectations = (
        ("environment", "climate_seasons", "climate"),
        ("geography", "rivers_lakes", "rivers"),
        ("geography", "mountains_relief", "relief"),
        ("environment", "natural_resources", "resources"),
        ("economy", "economic_drivers", "economic_drivers"),
        ("infrastructure", "transport_network", "transport"),
        ("infrastructure", "energy_connectivity", "energy"),
    )
    for section, key, label in domain_expectations:
        value = evidence_value(intelligence.get(section), key)
        if value and len(value) < 20:
            issues.append(f"{label}_suspiciously_short")

    return sorted(set(issues))


def _retry_source_call(
    operation,
    *,
    attempts: int,
    backoff_seconds: float,
):
    """Retry transient public-source failures with exponential backoff."""
    last_error: Exception | None = None

    for attempt in range(attempts):
        try:
            return operation()
        except (
            requests.Timeout,
            requests.ConnectionError,
            requests.HTTPError,
        ) as exc:
            last_error = exc

            status = (
                exc.response.status_code
                if isinstance(exc, requests.HTTPError)
                and exc.response is not None
                else None
            )
            retryable = (
                status is None
                or status in {408, 425, 429, 500, 502, 503, 504}
            )
            if not retryable or attempt + 1 >= attempts:
                raise

            delay = backoff_seconds * (2 ** attempt)
            time.sleep(delay)

    if last_error is not None:
        raise last_error
    raise RuntimeError("Retry helper exited unexpectedly")


def audit_country(
    code: str,
    timeout: float,
    *,
    retries: int,
    backoff_seconds: float,
) -> dict[str, Any]:
    started = time.monotonic()
    result: dict[str, Any] = {
        "code": code.upper(),
        "status": "error",
        "name": "",
        "elapsed_seconds": 0.0,
        "missing_required": [],
        "missing_optional": [],
        "validation_issues": [],
        "quality_issues": [],
        "source_issues": [],
        "coverage_percent": 0.0,
        "exception": "",
    }

    try:
        profile = _retry_source_call(
            lambda: fetch_country_profile(code, timeout=timeout),
            attempts=retries,
            backoff_seconds=backoff_seconds,
        )
        payload = profile_payload(profile)
        result["name"] = profile.name

        def build_enriched_record():
            fresh = build_from_legacy_profile(payload)
            return enrich_from_encyclopedia(
                fresh,
                title=profile.name,
                timeout=timeout,
            )

        try:
            record = _retry_source_call(
                build_enriched_record,
                attempts=retries,
                backoff_seconds=backoff_seconds,
            )
        except (
            requests.RequestException,
            LookupError,
            ValueError,
        ) as exc:
            record = build_from_legacy_profile(payload)
            result["source_issues"].append(
                f"encyclopedia_enrichment:{type(exc).__name__}:{exc}"
            )

        canonical_fact = record.identity.get("encyclopedia_title")
        flag_country_name = (
            str(canonical_fact.value)
            if canonical_fact is not None
            else profile.name
        )

        try:
            enriched_flag = _retry_source_call(
                lambda: enrich_flag_profile(
                    record.flag,
                    flag_country_name,
                    timeout=timeout,
                ),
                attempts=retries,
                backoff_seconds=backoff_seconds,
            )
            record.flag = enriched_flag
        except (
            requests.RequestException,
            LookupError,
            ValueError,
        ) as exc:
            result["source_issues"].append(
                f"flag_enrichment:{type(exc).__name__}:{exc}"
            )

        intelligence = record.to_dict()
        manifest = build_report_manifest(intelligence, payload)
        required_missing = missing_required_sections(manifest)
        optional_missing = [
            spec.title
            for spec in OFFICIAL_REPORT_SECTIONS
            if not spec.required and not manifest.get(spec.key, False)
        ]

        validation = validate_country_intelligence(record)
        quality = content_quality_issues(intelligence, payload)

        present = sum(
            1
            for spec in OFFICIAL_REPORT_SECTIONS
            if manifest.get(spec.key, False)
        )
        total = len(OFFICIAL_REPORT_SECTIONS)
        coverage = round(100.0 * present / total, 1) if total else 0.0

        if result["source_issues"]:
            status = "source_error"
        elif not required_missing and not quality:
            status = "pass"
        else:
            status = "review"

        result.update(
            {
                "status": status,
                "missing_required": required_missing,
                "missing_optional": optional_missing,
                "validation_issues": sorted(set(validation)),
                "quality_issues": quality,
                "coverage_percent": coverage,
            }
        )
    except (
        requests.RequestException,
        LookupError,
        ValueError,
    ) as exc:
        result["status"] = "source_error"
        result["source_issues"].append(
            f"profile:{type(exc).__name__}:{exc}"
        )
        result["exception"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001 - audit must record every country
        result["exception"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["elapsed_seconds"] = round(
            time.monotonic() - started,
            2,
        )

    return result


def write_outputs(rows: list[dict[str, Any]], output_dir: Path, label: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"country_intelligence_audit_{label}.json"
    csv_path = output_dir / f"country_intelligence_audit_{label}.csv"
    md_path = output_dir / f"country_intelligence_audit_{label}.md"

    json_path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    fieldnames = [
        "code",
        "name",
        "status",
        "coverage_percent",
        "missing_required",
        "missing_optional",
        "validation_issues",
        "quality_issues",
        "source_issues",
        "exception",
        "elapsed_seconds",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flattened = dict(row)
            for key in (
                "missing_required",
                "missing_optional",
                "validation_issues",
                "quality_issues",
                "source_issues",
            ):
                flattened[key] = " | ".join(row.get(key) or [])
            writer.writerow({key: flattened.get(key, "") for key in fieldnames})

    total = len(rows)
    passed = sum(row.get("status") == "pass" for row in rows)
    review = sum(row.get("status") == "review" for row in rows)
    source_errors = sum(
        row.get("status") == "source_error"
        for row in rows
    )
    errors = sum(row.get("status") == "error" for row in rows)
    avg_coverage = (
        round(
            sum(float(row.get("coverage_percent") or 0) for row in rows)
            / total,
            1,
        )
        if total
        else 0.0
    )

    lines = [
        f"# Flag Intelligence Worldwide Audit — {label}",
        "",
        f"- Countries/classes audited: **{total}**",
        f"- Pass: **{passed}**",
        f"- Review required: **{review}**",
        f"- Source errors after retry: **{source_errors}**",
        f"- Code/data errors: **{errors}**",
        f"- Mean report coverage: **{avg_coverage}%**",
        "",
        "## Countries requiring review",
        "",
        "| Code | Country | Coverage | Required missing | Quality issues | Source issues | Error |",
        "|---|---|---:|---|---|---|---|",
    ]

    for row in rows:
        if row.get("status") == "pass":
            continue
        lines.append(
            "| {code} | {name} | {coverage}% | {missing} | {quality} | {source} | {error} |".format(
                code=row.get("code", ""),
                name=str(row.get("name", "")).replace("|", "/"),
                coverage=row.get("coverage_percent", 0),
                missing=", ".join(row.get("missing_required") or []),
                quality=", ".join(row.get("quality_issues") or []),
                source=", ".join(row.get("source_issues") or []).replace("|", "/"),
                error=str(row.get("exception") or "").replace("|", "/"),
            )
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def merge_shards(input_dir: Path, output_dir: Path) -> int:
    rows: list[dict[str, Any]] = []
    for path in sorted(input_dir.rglob("country_intelligence_audit_shard-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows.extend(payload)

    by_code = {str(row["code"]): row for row in rows}
    expected = {code.upper() for code in all_project_codes()}
    missing = sorted(expected - set(by_code))

    for code in missing:
        by_code[code] = {
            "code": code,
            "name": "",
            "status": "error",
            "coverage_percent": 0.0,
            "missing_required": [],
            "missing_optional": [],
            "validation_issues": [],
            "quality_issues": ["missing_from_audit_shards"],
            "source_issues": [],
            "exception": "Country was not returned by any audit shard.",
            "elapsed_seconds": 0.0,
        }

    merged = [by_code[code] for code in sorted(by_code)]
    write_outputs(merged, output_dir, "worldwide")

    summary = {
        "classes_expected": len(expected),
        "classes_returned": len(merged),
        "pass": sum(row["status"] == "pass" for row in merged),
        "review": sum(row["status"] == "review" for row in merged),
        "source_error": sum(
            row["status"] == "source_error"
            for row in merged
        ),
        "error": sum(row["status"] == "error" for row in merged),
        "missing_codes": missing,
    }
    (output_dir / "country_intelligence_audit_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))

    return 0 if not missing else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/audit"))
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--backoff-seconds", type=float, default=1.0)
    parser.add_argument("--country-delay", type=float, default=0.25)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--merge-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.merge_dir is not None:
        return merge_shards(args.merge_dir, args.output_dir)

    codes = all_project_codes()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be within shard-count")

    selected = [
        code
        for index, code in enumerate(codes)
        if index % args.shard_count == args.shard_index
    ]

    rows: list[dict[str, Any]] = []
    for index, code in enumerate(selected, start=1):
        print(
            f"[{index}/{len(selected)}] auditing {code.upper()}",
            flush=True,
        )
        rows.append(
            audit_country(
                code,
                timeout=args.timeout,
                retries=args.retries,
                backoff_seconds=args.backoff_seconds,
            )
        )
        if args.country_delay > 0:
            time.sleep(args.country_delay)

    label = f"shard-{args.shard_index:02d}"
    write_outputs(rows, args.output_dir, label)

    source_errors = sum(
        row["status"] == "source_error"
        for row in rows
    )
    errors = sum(row["status"] == "error" for row in rows)
    print(
        json.dumps(
            {
                "shard": args.shard_index,
                "countries": len(rows),
                "source_errors": source_errors,
                "errors": errors,
            },
            indent=2,
        )
    )
    # Country-source outages are reported in artifacts rather than failing the
    # shard. The merge job verifies that all 250 classes were returned.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
