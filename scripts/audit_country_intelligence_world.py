"""Audit Country Intelligence across the complete worldwide taxonomy.

The audit runs the same source-aware country pipeline used by the application,
but without Streamlit. It is designed for sharded GitHub Actions execution so
all 249 ISO 3166-1 entities plus Kosovo (XK) can be validated regularly.

Outputs:
- one JSON result per shard;
- one CSV row per audited country;
- a merged worldwide JSON/CSV summary;
- machine-readable coverage rates and critical defects.

A country may legitimately lack optional deep-learning sections. The audit only
marks structural/data-quality defects as critical; optional coverage gaps are
reported separately so they can be improved without falsifying content.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pycountry
import requests

from flag_recognition.country_info import fetch_country_profile
from flag_recognition.country_intelligence import (
    build_from_legacy_profile,
    section_completion,
    validate_country_intelligence,
)
from flag_recognition.country_knowledge import (
    canonical_overview_text,
    enrich_from_encyclopedia,
)
from flag_recognition.flag_knowledge import enrich_flag_profile
from flag_recognition.report_manifest import (
    OFFICIAL_REPORT_SECTIONS,
    build_report_manifest,
    missing_required_sections,
)


EXPECTED_WORLD_CLASS_COUNT = 250
OPTIONAL_DEEP_SECTIONS = (
    "climate",
    "rivers",
    "relief",
    "resources",
    "flag",
    "origins",
    "administration",
    "languages_religion",
    "health",
    "festivals",
    "heritage",
    "economic_drivers",
    "infrastructure",
    "transport",
    "energy",
    "education",
    "environment",
    "international",
    "notable_people",
)
RAW_WIKI_PATTERNS = (
    r"\{\{[^{}]+\}\}",
    r"\[\[[^\]]+\]\]",
    r"<ref\b",
)
REFERENCE_HEADINGS = (
    "references",
    "bibliography",
    "further reading",
    "works cited",
    "external links",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/world_audit"))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument(
        "--inter-country-delay",
        type=float,
        default=0.4,
        help="Pause between countries to reduce public-source rate limiting.",
    )
    parser.add_argument("--merge-root", type=Path)
    parser.add_argument(
        "--fail-on-critical",
        action="store_true",
        help="Exit non-zero when any country has a critical structural defect.",
    )
    return parser.parse_args()


def worldwide_codes() -> list[str]:
    codes = sorted({country.alpha_2.lower() for country in pycountry.countries})
    if "xk" not in codes:
        codes.append("xk")
    codes = sorted(codes)
    if len(codes) != EXPECTED_WORLD_CLASS_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_WORLD_CLASS_COUNT} worldwide classes, got {len(codes)}"
        )
    return codes


def _profile_payload(code: str, profile: Any) -> dict[str, Any]:
    return {
        "code": code.upper(),
        "name": profile.name,
        "capital": profile.capital,
        "population": profile.population.value,
        "population_year": profile.population.year,
        "population_source": profile.population.source,
        "currency": profile.currency,
        "official_languages": profile.official_languages,
        "continent": profile.continent,
        "area_km2": profile.area_km2,
        "overview": canonical_overview_text(profile.overview),
        "national_day": profile.national_day,
        "independence_day": profile.independence_day,
        "colonial_history": getattr(profile, "colonial_history", "Not applicable"),
        "former_colonial_powers": getattr(
            profile, "former_colonial_powers", "Not applicable"
        ),
        "colonial_period": getattr(profile, "colonial_period", "Not applicable"),
        "independence_leader": getattr(
            profile, "independence_leader", "Not applicable"
        ),
        "historical_context": getattr(
            profile,
            "historical_context",
            "No classical colonial-independence transition is documented "
            "in the available country overview.",
        ),
        "national_motto": profile.national_motto,
        "national_anthem": profile.national_anthem,
        "region": getattr(profile, "region", "Not available"),
        "subregion": getattr(profile, "subregion", "Not available"),
        "demonym": getattr(profile, "demonym", "Not available"),
        "iso_alpha3": getattr(profile, "iso_alpha3", "Not available"),
        "timezones": getattr(profile, "timezones", "Not available"),
        "borders": getattr(profile, "borders", "Not available"),
        "largest_cities": getattr(profile, "largest_cities", "Not available"),
        "international_organizations": getattr(
            profile, "international_organizations", "Not available"
        ),
        "official_religion": getattr(
            profile, "official_religion", "Not available"
        ),
        "highest_point": getattr(profile, "highest_point", "Not available"),
        "lowest_point": getattr(profile, "lowest_point", "Not available"),
        "gdp_usd": getattr(getattr(profile, "gdp", None), "value_usd", None),
        "gdp_year": getattr(getattr(profile, "gdp", None), "year", None),
        "gdp_source": getattr(
            getattr(profile, "gdp", None), "source", "World Bank"
        ),
        "government_form": profile.government_form,
        "head_of_state": profile.head_of_state,
        "head_of_state_office": profile.head_of_state_office,
        "head_of_government": profile.head_of_government,
        "head_of_government_office": profile.head_of_government_office,
        "calling_code": profile.calling_code,
        "emergency_numbers": getattr(profile, "emergency_numbers", "Not available"),
        "internet_domain": profile.internet_domain,
        "driving_side": profile.driving_side,
        "latitude": profile.latitude,
        "longitude": profile.longitude,
    }


def _string_values(value: Any):
    if isinstance(value, dict):
        for nested in value.values():
            yield from _string_values(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _string_values(nested)
    elif isinstance(value, str):
        yield value


def _raw_markup_issues(payload: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    for text in _string_values(payload):
        for pattern in RAW_WIKI_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE):
                issues.append(f"Raw wiki markup detected: {text[:140]}")
                break
        if len(issues) >= 5:
            break
    return issues


def _timeline_reference_issues(intelligence: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    timeline = intelligence.get("historical_timeline") or ()
    for event in timeline:
        if not isinstance(event, dict):
            continue
        summary = str(event.get("summary") or "")
        label = str(event.get("label") or "")
        lower = f"{label} {summary}".lower()
        if any(token in lower for token in REFERENCE_HEADINGS):
            issues.append(f"Reference-like timeline event: {summary[:150]}")
        if re.search(
            r"\b(?:university press|academic journal|publications of|isbn|doi:)\b",
            lower,
        ):
            issues.append(f"Bibliographic timeline event: {summary[:150]}")
    return issues[:5]


def _sovereignty_issues(profile: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    independence = str(profile.get("independence_day") or "")
    powers = str(profile.get("former_colonial_powers") or "")
    status = str(profile.get("colonial_period") or "")

    if independence in ("Not available", "Not applicable"):
        if powers not in ("", "Not available", "Not applicable"):
            issues.append(
                "Colonial powers populated without a documented independence transition."
            )
        if status not in (
            "",
            "Not available",
            "Not applicable",
            "No classical colonial-independence transition",
        ):
            issues.append(
                "Colonial status populated without a documented independence transition."
            )
    return issues


def audit_country(code: str, timeout: float) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    result: dict[str, Any] = {
        "code": code.upper(),
        "status": "ok",
        "critical_issues": [],
        "warnings": [],
        "missing_required_sections": [],
        "missing_optional_sections": [],
        "manifest": {},
        "completion": {},
        "validation": [],
    }

    try:
        profile = fetch_country_profile(code, timeout=timeout)
        payload = _profile_payload(code, profile)
        intelligence = build_from_legacy_profile(payload)

        try:
            intelligence = enrich_from_encyclopedia(
                intelligence,
                title=profile.name,
                timeout=timeout,
            )
        except Exception as exc:
            result["warnings"].append(
                f"Encyclopedia enrichment failed: {type(exc).__name__}: {exc}"
            )

        try:
            canonical_fact = intelligence.identity.get("encyclopedia_title")
            flag_country_name = (
                str(canonical_fact.value)
                if canonical_fact is not None
                else profile.name
            )
            intelligence.flag = enrich_flag_profile(
                intelligence.flag,
                flag_country_name,
                timeout=timeout,
            )
        except Exception as exc:
            result["warnings"].append(
                f"Flag enrichment failed: {type(exc).__name__}: {exc}"
            )

        intelligence_dict = intelligence.to_dict()
        manifest = build_report_manifest(intelligence_dict, payload)
        required_missing = missing_required_sections(manifest)
        optional_missing = [
            key
            for key in OPTIONAL_DEEP_SECTIONS
            if not manifest.get(key, False)
        ]

        result["name"] = profile.name
        result["manifest"] = manifest
        result["completion"] = section_completion(intelligence)
        result["validation"] = validate_country_intelligence(intelligence)
        result["missing_required_sections"] = required_missing
        result["missing_optional_sections"] = optional_missing

        critical = []
        critical.extend(_raw_markup_issues(intelligence_dict))
        critical.extend(_timeline_reference_issues(intelligence_dict))
        critical.extend(_sovereignty_issues(payload))

        if required_missing:
            critical.append(
                "Missing required report sections: " + ", ".join(required_missing)
            )

        if not payload.get("capital") or payload.get("capital") == "Not available":
            result["warnings"].append("Capital unavailable.")
        if payload.get("population") is None:
            result["warnings"].append("Population unavailable.")
        if payload.get("latitude") is None or payload.get("longitude") is None:
            result["warnings"].append("Reference coordinates unavailable.")

        result["critical_issues"] = critical
        if critical:
            result["status"] = "critical"
        elif result["warnings"] or optional_missing:
            result["status"] = "partial"

    except Exception as exc:
        result["status"] = "error"
        result["critical_issues"] = [
            f"Pipeline exception: {type(exc).__name__}: {exc}"
        ]

    elapsed = datetime.now(timezone.utc) - started
    result["elapsed_seconds"] = round(elapsed.total_seconds(), 3)
    return result


def write_shard(
    rows: list[dict[str, Any]],
    output_dir: Path,
    shard_index: int,
    shard_count: int,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"country_intelligence_audit_shard_{shard_index}.json"
    csv_path = output_dir / f"country_intelligence_audit_shard_{shard_index}.csv"

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "countries": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    fieldnames = [
        "code",
        "name",
        "status",
        "critical_count",
        "warning_count",
        "missing_required_count",
        "missing_optional_count",
        "elapsed_seconds",
        "critical_issues",
        "warnings",
        "missing_required_sections",
        "missing_optional_sections",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "code": row.get("code", ""),
                "name": row.get("name", ""),
                "status": row.get("status", ""),
                "critical_count": len(row.get("critical_issues", [])),
                "warning_count": len(row.get("warnings", [])),
                "missing_required_count": len(row.get("missing_required_sections", [])),
                "missing_optional_count": len(row.get("missing_optional_sections", [])),
                "elapsed_seconds": row.get("elapsed_seconds", ""),
                "critical_issues": " | ".join(row.get("critical_issues", [])),
                "warnings": " | ".join(row.get("warnings", [])),
                "missing_required_sections": " | ".join(
                    row.get("missing_required_sections", [])
                ),
                "missing_optional_sections": " | ".join(
                    row.get("missing_optional_sections", [])
                ),
            })

    return json_path, csv_path


def merge_results(merge_root: Path, output_dir: Path) -> dict[str, Any]:
    files = sorted(merge_root.rglob("country_intelligence_audit_shard_*.json"))
    countries: list[dict[str, Any]] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        countries.extend(payload.get("countries", []))

    by_code = {row["code"]: row for row in countries}
    codes = worldwide_codes()
    missing_codes = [code.upper() for code in codes if code.upper() not in by_code]

    section_counts = {
        spec.key: sum(
            1
            for row in by_code.values()
            if row.get("manifest", {}).get(spec.key, False)
        )
        for spec in OFFICIAL_REPORT_SECTIONS
    }
    completion_counts: dict[str, int] = {}
    for row in by_code.values():
        for key, value in row.get("completion", {}).items():
            if value:
                completion_counts[key] = completion_counts.get(key, 0) + 1

    status_counts: dict[str, int] = {}
    for row in by_code.values():
        status = row.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    audited = len(by_code)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "expected_classes": EXPECTED_WORLD_CLASS_COUNT,
        "audited_classes": audited,
        "missing_codes": missing_codes,
        "status_counts": status_counts,
        "critical_countries": sorted(
            code
            for code, row in by_code.items()
            if row.get("status") in ("critical", "error")
        ),
        "section_coverage": {
            key: {
                "count": count,
                "rate": round(count / audited, 4) if audited else 0.0,
            }
            for key, count in section_counts.items()
        },
        "completion_coverage": {
            key: {
                "count": count,
                "rate": round(count / audited, 4) if audited else 0.0,
            }
            for key, count in sorted(completion_counts.items())
        },
        "countries": [by_code[code.upper()] for code in codes if code.upper() in by_code],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "country_intelligence_world_audit.json"
    csv_path = output_dir / "country_intelligence_world_audit.csv"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "code",
            "name",
            "status",
            "critical_issues",
            "warnings",
            "missing_required_sections",
            "missing_optional_sections",
        ])
        for row in summary["countries"]:
            writer.writerow([
                row.get("code", ""),
                row.get("name", ""),
                row.get("status", ""),
                " | ".join(row.get("critical_issues", [])),
                " | ".join(row.get("warnings", [])),
                " | ".join(row.get("missing_required_sections", [])),
                " | ".join(row.get("missing_optional_sections", [])),
            ])

    print(json.dumps({
        key: value
        for key, value in summary.items()
        if key != "countries"
    }, indent=2))
    return summary


def main() -> None:
    args = parse_args()

    if args.merge_root is not None:
        summary = merge_results(args.merge_root, args.output_dir)
        critical = summary.get("critical_countries", [])
        incomplete = summary.get("missing_codes", [])
        if args.fail_on_critical and (critical or incomplete):
            raise SystemExit(1)
        return

    codes = worldwide_codes()
    if args.shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must be within shard-count")

    selected = [
        code
        for index, code in enumerate(codes)
        if index % args.shard_count == args.shard_index
    ]
    rows = []
    for position, code in enumerate(selected, 1):
        print(
            f"[{position}/{len(selected)}] auditing {code.upper()}",
            flush=True,
        )
        result = audit_country(code, args.timeout)
        rows.append(result)
        if args.inter_country_delay > 0:
            import time
            time.sleep(args.inter_country_delay)
        print(
            f"  -> {result['status']} "
            f"critical={len(result.get('critical_issues', []))} "
            f"optional_missing={len(result.get('missing_optional_sections', []))}",
            flush=True,
        )

    json_path, csv_path = write_shard(
        rows,
        args.output_dir,
        args.shard_index,
        args.shard_count,
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")

    if args.fail_on_critical and any(
        row.get("status") in ("critical", "error")
        for row in rows
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
