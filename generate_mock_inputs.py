"""
generate_mock_inputs.py
-----------------------
Standalone script to generate two mock input files representing the same candidate
(Jane Doe) across two heterogeneous data sources.

Outputs:
    ats_input.json        — Structured JSON with intentionally messy field names,
                            non-standard date formats, and unformatted phone numbers.
    recruiter_notes.txt   — Raw unstructured text simulating free-form recruiter notes.

Usage:
    python generate_mock_inputs.py
    python generate_mock_inputs.py --output-dir ./test_fixtures
"""

import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Mock Data Definitions
# ---------------------------------------------------------------------------

def build_ats_payload() -> dict:
    """
    Construct the mock ATS JSON payload for Jane Doe.

    Intentional messiness introduced to stress-test the normalization engine:
      - Field names use inconsistent casing and underscores (e.g., "Full_Name",
        "phone_RAW", "cntry").
      - Date formats are non-standard ("2024/05/12", "2022/10/01").
      - Phone number is unformatted with parentheses and spaces.
      - Country stored as a verbose string ("United States") rather than ISO code.
      - Skills list contains duplicates with inconsistent casing.
      - A second email is included to verify deduplication against recruiter notes.
    """
    return {
        # --- Identity Fields (messy casing and key names) ---
        "Full_Name": "Jane Doe",
        "email_PRIMARY": "jane.doe@email.com",
        "email_secondary": "j.doe@workmail.com",
        "phone_RAW": "(555) 019-2834",

        # --- Location (verbose country, no ISO code) ---
        "location": {
            "city_name": "San Francisco",
            "state_region": "California",
            "cntry": "United States"
        },

        # --- Professional Links (inconsistent nesting) ---
        "social_profiles": {
            "linkedin_url": "https://linkedin.com/in/janedoe",
            "gh": "https://github.com/janedoe"
        },

        # --- Work Experience (non-standard date formats) ---
        "work_history": [
            {
                "employer": "Acme Corp",
                "job_title": "Lead Software Engineer",
                "date_from": "2022/10/01",
                "date_to": "Present",
                "responsibilities": (
                    "Led a cross-functional team of 8 engineers to deliver "
                    "a real-time data pipeline handling 2M events/day. "
                    "Drove adoption of React for the front-end dashboard suite."
                )
            },
            {
                "employer": "Beta Systems Inc.",
                "job_title": "Software Engineer II",
                "date_from": "2019/03/15",
                "date_to": "2022/09/30",
                "responsibilities": (
                    "Built and maintained RESTful microservices in Python (FastAPI). "
                    "Owned CI/CD pipelines on GitHub Actions."
                )
            }
        ],

        # --- Education ---
        "education_history": [
            {
                "school": "University of California, Berkeley",
                "qualification": "Bachelor of Science",
                "subject": "Computer Science",
                "start_yr": "2015/08/01",
                "end_yr": "2019/05/15"
            }
        ],

        # --- Skills (duplicates + inconsistent casing to stress dedup logic) ---
        "skills_list": [
            "Python",
            "python",           # duplicate, different casing
            "React",
            "react.js",         # near-duplicate variant
            "FastAPI",
            "PostgreSQL",
            "Docker",
            "GitHub Actions"
        ],

        # --- ATS Metadata ---
        "ats_record_id": "ATS-00471823",
        "ats_created_at": "2024/05/12",
        "ats_source_system": "Greenhouse"
    }


def build_recruiter_notes() -> str:
    """
    Construct the mock recruiter notes text for Jane Doe.

    Intentional messiness introduced to stress-test the LLM extraction layer:
      - No structured schema; information is buried in prose.
      - Phone number uses a different formatting style than the ATS record.
      - Date expressions use natural language ("Oct 2022", "Present").
      - Skills mentioned inline without enumeration.
      - A second role (Beta Systems) is mentioned briefly, testing multi-role extraction.
      - Interviewer opinion and noise text included to test extraction robustness.
      - Email matches ATS primary email — critical for identity merge key matching.
    """
    return """\
Recruiter Call Notes — Jane Doe
Interviewed by: Marcus Webb
Date of Call: May 14, 2024
---

Spoke with Jane today. Really strong candidate, highly recommend moving forward.

She's currently working at Acme Corp as a Lead Engineer — been there since Oct 2022
and is still active there (Present). She described her work as leading a data
infrastructure project and building out their front-end in React. She mentioned Python
is her primary backend language and she's been using it for about 6 years. Also brought
up FastAPI and PostgreSQL unprompted, which was impressive.

Before Acme, she was at Beta Systems Inc. as a Software Engineer II. Rough dates she
gave were around March 2019 to September 2022. Sounded like solid foundational work,
mainly microservices.

Education: UC Berkeley, BS in Computer Science, graduated around May 2019.

Skills she highlighted: Python, React, FastAPI, PostgreSQL, Docker.
Also mentioned dabbling in Kubernetes but said she wouldn't call it a core skill yet.

Contact info:
  Email: jane.doe@email.com
  Phone: +1 (555) 019-2834

LinkedIn she said is linkedin.com/in/janedoe — haven't verified.

Overall impression: 9/10. Strong technical depth, communicates clearly.
Would be a great fit for the Principal Engineer opening on the data platform team.

-- End of Notes --
"""


# ---------------------------------------------------------------------------
# File Writers
# ---------------------------------------------------------------------------

def write_ats_json(output_dir: Path) -> Path:
    """
    Serialize the ATS payload to a JSON file at the specified output directory.

    Args:
        output_dir: Resolved Path object pointing to the target directory.

    Returns:
        The resolved Path of the written file.

    Raises:
        IOError: If the file cannot be written.
    """
    target_path = output_dir / "ats_input.json"
    payload = build_ats_payload()

    with target_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    return target_path


def write_recruiter_notes(output_dir: Path) -> Path:
    """
    Write the recruiter notes string to a plain text file.

    Args:
        output_dir: Resolved Path object pointing to the target directory.

    Returns:
        The resolved Path of the written file.

    Raises:
        IOError: If the file cannot be written.
    """
    target_path = output_dir / "recruiter_notes.txt"
    notes = build_recruiter_notes()

    with target_path.open("w", encoding="utf-8") as fh:
        fh.write(notes)

    return target_path


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate mock ATS JSON and recruiter notes text fixtures "
            "for the Multi-Source Candidate Data Transformer pipeline."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory in which to write fixture files (default: current directory).",
    )
    return parser.parse_args()


def main() -> None:
    """
    Entry point: resolve output directory, write both fixture files,
    and report results to stdout.
    """
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()

    # Guard: create output directory if it does not already exist.
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"[ERROR] Could not create output directory '{output_dir}': {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Writing mock fixtures to: {output_dir}\n")

    # Write ATS JSON fixture.
    try:
        ats_path = write_ats_json(output_dir)
        print(f"  [OK] {ats_path}")
    except IOError as exc:
        print(f"  [FAIL] ats_input.json — {exc}", file=sys.stderr)
        sys.exit(1)

    # Write recruiter notes fixture.
    try:
        notes_path = write_recruiter_notes(output_dir)
        print(f"  [OK] {notes_path}")
    except IOError as exc:
        print(f"  [FAIL] recruiter_notes.txt — {exc}", file=sys.stderr)
        sys.exit(1)

    print("\nMock fixture generation complete.")
    print("Both files represent the same candidate (Jane Doe) across two sources.")
    print("Identity merge key: jane.doe@email.com / +1 (555) 019-2834")


if __name__ == "__main__":
    main()