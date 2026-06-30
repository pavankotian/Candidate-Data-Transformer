"""
generate_runtime_config.py
---------------------------
Standalone script that writes a mock runtime projection configuration file
to disk, conforming to the schema consumed by projector.py.

This config exercises the full feature surface of the projection engine:
scalar fields, nested object fields, indexed array access, wildcard array
plucking, format transforms, and all three on_missing behaviours.

Usage:
    python generate_runtime_config.py
    python generate_runtime_config.py --output runtime_config.json
"""

import argparse
import json
import sys
from pathlib import Path


def build_runtime_config() -> dict:
    """
    Construct the mock runtime projection configuration dict.

    Deliberately exercises every on_missing mode and several format
    transforms so it can serve as both a working default config and a
    feature-coverage smoke test for the projection engine.

    Returns:
        Dict conforming to projector.py's expected configuration schema.
    """
    return {
        "flags": {
            "include_provenance": True,
            "include_overall_confidence": True,
        },
        "fields": [
            {
                "from": "candidate_id",
                "path": "id",
                "on_missing": "error",
                "format": None,
            },
            {
                "from": "full_name",
                "path": "name",
                "on_missing": "error",
                "format": "titlecase",
            },
            {
                "from": "emails[0]",
                "path": "primary_email",
                "on_missing": "null",
                "format": "lowercase",
            },
            {
                "from": "emails[1]",
                "path": "secondary_email",
                "on_missing": "omit",
                "format": "lowercase",
            },
            {
                "from": "phones[0]",
                "path": "phone",
                "on_missing": "null",
                "format": "e164",
            },
            {
                "from": "location.city",
                "path": "location.city",
                "on_missing": "null",
                "format": None,
            },
            {
                "from": "location.country",
                "path": "location.country_code",
                "on_missing": "null",
                "format": "iso-alpha2",
            },
            {
                "from": "skills[].name",
                "path": "skills_csv",
                "on_missing": "null",
                "format": "join_comma",
            },
            {
                "from": "skills[].name",
                "path": "skill_count",
                "on_missing": "null",
                "format": "count",
            },
            {
                "from": "experience[0].company",
                "path": "current_employer",
                "on_missing": "null",
                "format": None,
            },
            {
                "from": "experience[0].title",
                "path": "current_title",
                "on_missing": "null",
                "format": None,
            },
            {
                "from": "experience[0].start",
                "path": "current_role_start",
                "on_missing": "null",
                "format": "yyyy-mm",
            },
            {
                "from": "education[0].institution",
                "path": "education.university",
                "on_missing": "omit",
                "format": None,
            },
            {
                "from": "links.linkedin",
                "path": "linkedin_url",
                "on_missing": "omit",
                "format": None,
            },
            {
                "from": "links.portfolio",
                "path": "portfolio_url",
                "on_missing": "omit",
                "format": None,
            },
        ],
    }


def write_runtime_config(output_path: Path) -> Path:
    """
    Serialize the runtime config dict to disk as pretty-printed JSON.

    Args:
        output_path: Resolved Path object pointing to the target file.

    Returns:
        The resolved Path of the written file.

    Raises:
        IOError: If the file cannot be written.
    """
    config = build_runtime_config()

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, ensure_ascii=False)

    return output_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate a mock runtime projection configuration JSON file "
            "for the Multi-Source Candidate Data Transformer pipeline."
        )
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runtime_config.json",
        help="Path to write the generated config file (default: runtime_config.json).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: resolve output path, write the config file, report result."""
    args = parse_args()
    output_path = Path(args.output).resolve()

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"[ERROR] Could not create output directory '{output_path.parent}': {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        written_path = write_runtime_config(output_path)
    except IOError as exc:
        print(f"[FAIL] Could not write runtime config: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"[OK] Runtime projection config written to: {written_path}")
    print("Exercise coverage: scalar fields, nested fields, indexed array access,")
    print("wildcard plucking, format transforms, and all three on_missing modes.")


if __name__ == "__main__":
    main()