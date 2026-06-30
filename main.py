"""
main.py
-------
CLI orchestrator for the Multi-Source Candidate Data Transformer.

Wires together the full pipeline end-to-end:
    ATS JSON + Recruiter Notes
        → ingestion.py   (parse_ats_json / parse_recruiter_notes)
        → merger.py      (run_merge — record linkage, confidence resolution, provenance)
        → projector.py   (project — runtime-config-driven reshaping)
        → stdout (JSON)

Designed per the blueprint's error-isolation principle: a corrupted or
missing source file is logged to stderr and the pipeline proceeds with
whatever sources successfully ingested, rather than aborting outright.
If projection itself fails (e.g., on_missing="error" on a genuinely
mandatory field), that is treated as a hard failure since it represents
an explicit validation contract being violated.

Usage:
    python main.py --ats ats_input.json --notes recruiter_notes.txt --config runtime_config.json
    python main.py --ats ats_input.json --config runtime_config.json   # notes optional
    python main.py --notes recruiter_notes.txt --config runtime_config.json  # ats optional
    python main.py --ats ats_input.json --notes recruiter_notes.txt --config runtime_config.json --pretty
    python main.py --ats ats_input.json --notes recruiter_notes.txt --config runtime_config.json --out result.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from ingestion import parse_ats_json, parse_recruiter_notes
from merger import run_merge
from projector import project


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the pipeline orchestrator.

    At least one of --ats / --notes must be supplied; --config is always
    required since the projection stage cannot run without a schema.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Multi-Source Candidate Data Transformer — runs the full "
            "ingest → normalize → merge → project pipeline end-to-end."
        )
    )
    parser.add_argument(
        "--ats",
        type=str,
        default=None,
        help="Path to the ATS structured JSON input file.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default=None,
        help="Path to the recruiter notes unstructured .txt input file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the runtime projection configuration JSON file.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional file path to write the projected JSON output. "
             "If omitted, output is printed to stdout only.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the output JSON with indentation (default: compact).",
    )
    parser.add_argument(
        "--all-candidates",
        action="store_true",
        help="Project and output ALL resolved candidate clusters as a JSON array, "
             "instead of only the first one (default: first candidate only).",
    )

    args = parser.parse_args()

    if not args.ats and not args.notes:
        parser.error("At least one of --ats or --notes must be supplied.")

    return args


def load_runtime_config(config_path: str) -> dict:
    """
    Load and parse the runtime projection configuration JSON file from disk.

    Args:
        config_path: Path to the config JSON file.

    Returns:
        Parsed config dict.

    Raises:
        SystemExit: If the file is missing or contains invalid JSON.
                    This is treated as a hard failure since the projection
                    stage cannot proceed without a valid schema.
    """
    path = Path(config_path).resolve()

    if not path.exists():
        print(f"[main] FATAL: Runtime config file not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        raw = path.read_text(encoding="utf-8")
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        print(
            f"[main] FATAL: Runtime config file is not valid JSON: {path}\n  Detail: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)


def run_ingestion_stage(ats_path: Optional[str], notes_path: Optional[str]) -> list[dict]:
    """
    Run the ingestion stage against whichever source paths were supplied.

    Per the blueprint's error-isolation principle: if a given source file
    fails to parse (missing, corrupted, or — for recruiter notes — an API
    failure), the error is logged to stderr and that source is skipped
    entirely. The pipeline proceeds with whatever sources succeeded.

    Args:
        ats_path:   Optional path to the ATS JSON file.
        notes_path: Optional path to the recruiter notes .txt file.

    Returns:
        List of raw profile dicts from all sources that ingested successfully.
        May be empty if every source failed.
    """
    raw_profiles: list[dict] = []

    if ats_path:
        print(f"[main] Ingesting ATS source: {ats_path}", file=sys.stderr)
        try:
            ats_profile = parse_ats_json(ats_path)
            raw_profiles.append(ats_profile)
            print(f"[main]   OK — ATS profile ingested.", file=sys.stderr)
        except FileNotFoundError as exc:
            print(f"[main]   SKIPPED — ATS file not found: {exc}", file=sys.stderr)
        except ValueError as exc:
            print(f"[main]   SKIPPED — ATS file invalid/corrupted: {exc}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 — deliberate broad catch for isolation
            print(f"[main]   SKIPPED — Unexpected ATS ingestion error: {exc}", file=sys.stderr)

    if notes_path:
        print(f"[main] Ingesting recruiter notes source: {notes_path}", file=sys.stderr)
        try:
            notes_profile = parse_recruiter_notes(notes_path)
            raw_profiles.append(notes_profile)
            print(f"[main]   OK — Recruiter notes profile ingested.", file=sys.stderr)
        except FileNotFoundError as exc:
            print(f"[main]   SKIPPED — Notes file not found: {exc}", file=sys.stderr)
        except RuntimeError as exc:
            print(f"[main]   SKIPPED — Claude API extraction failed: {exc}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 — deliberate broad catch for isolation
            print(f"[main]   SKIPPED — Unexpected notes ingestion error: {exc}", file=sys.stderr)

    if not raw_profiles:
        print(
            "[main] FATAL: All supplied source files failed to ingest. "
            "No data available to merge or project.",
            file=sys.stderr,
        )

    return raw_profiles


def run_merge_stage(raw_profiles: list[dict]) -> list:
    """
    Run the merge stage against the successfully ingested raw profiles.

    Args:
        raw_profiles: List of raw profile dicts from run_ingestion_stage().

    Returns:
        List of CanonicalProfile objects, one per resolved identity cluster.

    Raises:
        SystemExit: If the merge stage raises an unexpected exception —
                    treated as a hard failure since merge logic errors
                    indicate a pipeline bug, not a data-quality issue.
    """
    print(f"[main] Running merge stage on {len(raw_profiles)} raw profile(s)...", file=sys.stderr)
    try:
        canonical_profiles = run_merge(raw_profiles)
    except Exception as exc:  # noqa: BLE001
        print(f"[main] FATAL: Merge stage raised an unexpected error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(
        f"[main] Merge stage complete — {len(canonical_profiles)} canonical profile(s) resolved.",
        file=sys.stderr,
    )
    return canonical_profiles


def run_projection_stage(canonical_profiles: list, config: dict, all_candidates: bool) -> object:
    """
    Run the projection stage against one or all resolved canonical profiles.

    Args:
        canonical_profiles: List of CanonicalProfile objects from run_merge_stage().
        config:             Parsed runtime projection config dict.
        all_candidates:     If True, project every candidate and return a list.
                            If False, project only the first candidate and return a dict.

    Returns:
        A single projected dict (default), or a list of projected dicts
        (if all_candidates=True).

    Raises:
        SystemExit: If projection fails — typically an on_missing="error"
                    field that is genuinely absent, which is an explicit
                    validation failure the user must act on.
    """
    targets = canonical_profiles if all_candidates else canonical_profiles[:1]

    projected_results = []
    for idx, profile in enumerate(targets):
        print(
            f"[main] Projecting candidate {idx + 1}/{len(targets)} "
            f"(candidate_id={getattr(profile, 'candidate_id', 'unknown')})...",
            file=sys.stderr,
        )
        try:
            projected = project(profile, config)
        except ValueError as exc:
            print(f"[main] FATAL: Projection failed: {exc}", file=sys.stderr)
            sys.exit(1)
        except Exception as exc:  # noqa: BLE001
            print(f"[main] FATAL: Unexpected projection error: {exc}", file=sys.stderr)
            sys.exit(1)

        projected_results.append(projected)

    print("[main] Projection stage complete.", file=sys.stderr)

    return projected_results if all_candidates else projected_results[0]


def emit_output(result: object, out_path: Optional[str], pretty: bool) -> None:
    """
    Serialize the final projection result to JSON and write it to stdout,
    and optionally also to a file.

    Args:
        result:   The projected dict or list of dicts.
        out_path: Optional file path to additionally write the output to.
        pretty:   If True, indent the JSON for human readability.

    Raises:
        SystemExit: If the result cannot be serialized to JSON, or if writing
                    to the optional output file fails.
    """
    try:
        if pretty:
            serialized = json.dumps(result, indent=2, ensure_ascii=False, default=str)
        else:
            serialized = json.dumps(result, ensure_ascii=False, default=str)
    except (TypeError, ValueError) as exc:
        print(f"[main] FATAL: Could not serialize projection result to JSON: {exc}", file=sys.stderr)
        sys.exit(1)

    # Always print to stdout — this is the primary contract of the CLI.
    print(serialized)

    # Optionally also persist to disk.
    if out_path:
        target = Path(out_path).resolve()
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(serialized, encoding="utf-8")
            print(f"[main] Output also written to file: {target}", file=sys.stderr)
        except IOError as exc:
            print(f"[main] WARNING: Failed to write output file '{target}': {exc}", file=sys.stderr)
            # Non-fatal — stdout output already succeeded.


def main() -> None:
    """
    Orchestrate the full pipeline: parse args → ingest → merge → project → emit.
    """
    args = parse_args()

    print("=" * 70, file=sys.stderr)
    print("Multi-Source Candidate Data Transformer — Pipeline Run", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    # --- Stage 0: Load runtime config ---
    config = load_runtime_config(args.config)
    print(f"[main] Loaded runtime config: {args.config}", file=sys.stderr)

    # --- Stage 1: Ingestion ---
    raw_profiles = run_ingestion_stage(args.ats, args.notes)
    if not raw_profiles:
        sys.exit(1)

    # --- Stage 2: Merge ---
    canonical_profiles = run_merge_stage(raw_profiles)
    if not canonical_profiles:
        print(
            "[main] FATAL: Merge stage produced zero canonical profiles "
            "despite successful ingestion. Aborting.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Stage 3: Projection ---
    result = run_projection_stage(canonical_profiles, config, args.all_candidates)

    # --- Stage 4: Output ---
    emit_output(result, args.out, args.pretty)

    print("=" * 70, file=sys.stderr)
    print("Pipeline run complete.", file=sys.stderr)
    print("=" * 70, file=sys.stderr)


if __name__ == "__main__":
    main()