"""
ingestion.py
------------
Ingestion and extraction layer for the Multi-Source Candidate Data Transformer.

Provides two parsers that each produce a normalized raw profile dict conforming
to the pipeline's internal intermediate schema. Downstream components (merger.py)
consume this schema regardless of which parser produced it.

Internal raw profile schema (all keys optional at ingestion time):
{
    "full_name":   str | None,
    "emails":      list[str],
    "phones":      list[str],
    "location": {
        "city":    str | None,
        "region":  str | None,
        "country": str | None       # raw string; normalizer resolves to ISO-3166
    },
    "links": {
        "linkedin":  str | None,
        "github":    str | None,
        "portfolio": str | None,
        "other":     list[str]
    },
    "skills":      list[str],       # raw skill name strings
    "experience": [
        {
            "company": str,
            "title":   str,
            "start":   str,         # raw date string; normalizer resolves to YYYY-MM
            "end":     str | None,  # raw date string or "Present"
            "summary": str | None
        }
    ],
    "education": [
        {
            "institution":    str,
            "degree":         str | None,
            "field_of_study": str | None,
            "start":          str | None,
            "end":            str | None
        }
    ],
    "_source_id":   str,            # originating file path for provenance
    "_source_type": str             # "ats" | "recruiter_notes"
}

Modules:
    parse_ats_json(file_path)        → raw profile dict
    parse_recruiter_notes(file_path) → raw profile dict
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Internal Schema Helpers
# ---------------------------------------------------------------------------

def _empty_raw_profile(source_id: str, source_type: str) -> dict:
    """
    Return a blank raw profile dict with all required keys initialised to
    safe empty defaults. Every parser must populate from this base to ensure
    downstream consumers never encounter missing keys.

    Args:
        source_id:   Originating file path or identifier string.
        source_type: "ats" or "recruiter_notes".

    Returns:
        Dict conforming to the internal raw profile schema.
    """
    return {
        "full_name": None,
        "emails": [],
        "phones": [],
        "location": {
            "city": None,
            "region": None,
            "country": None,
        },
        "links": {
            "linkedin": None,
            "github": None,
            "portfolio": None,
            "other": [],
        },
        "skills": [],
        "experience": [],
        "education": [],
        "_source_id": source_id,
        "_source_type": source_type,
    }


def _safe_str(value: Any, fallback: Optional[str] = None) -> Optional[str]:
    """
    Coerce a value to a stripped string, returning fallback if None or empty.

    Args:
        value:    Any value from a parsed JSON dict.
        fallback: Value to return if coercion yields an empty string.

    Returns:
        Stripped string or fallback.
    """
    if value is None:
        return fallback
    coerced = str(value).strip()
    return coerced if coerced else fallback


def _safe_list_of_str(value: Any) -> list[str]:
    """
    Coerce a value to a list of non-empty stripped strings.
    Accepts lists, single strings, or None. Drops empty entries.

    Args:
        value: Any value from a parsed JSON dict.

    Returns:
        List of non-empty stripped strings.
    """
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


# ---------------------------------------------------------------------------
# ATS JSON Parser
# ---------------------------------------------------------------------------

def parse_ats_json(file_path: str) -> dict:
    """
    Parse the structured ATS JSON file into an internal raw profile dict.

    The ATS file uses non-standard, inconsistent key names (e.g., "Full_Name",
    "phone_RAW", "email_PRIMARY", "cntry"). This function handles all known
    key variants produced by generate_mock_inputs.py and maps them into the
    canonical internal schema.

    Args:
        file_path: Absolute or relative path to the ATS JSON file.

    Returns:
        Raw profile dict conforming to the internal schema.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError:        If the file is not valid JSON.
        RuntimeError:      If a critical parse error occurs.
    """
    path = Path(file_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"[ingestion] ATS JSON file not found: {path}")

    print(f"[ingestion] Parsing ATS JSON: {path}", file=sys.stderr)

    try:
        raw_json = path.read_text(encoding="utf-8")
        ats_data: dict = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"[ingestion] ATS file is not valid JSON: {path}\n  Detail: {exc}"
        ) from exc

    profile = _empty_raw_profile(source_id=str(path), source_type="ats")

    # --- Identity Fields ---
    # Support multiple key variants for name.
    profile["full_name"] = _safe_str(
        ats_data.get("Full_Name")
        or ats_data.get("full_name")
        or ats_data.get("name")
    )

    # Collect all email variants into a deduplicated list.
    email_candidates = [
        _safe_str(ats_data.get("email_PRIMARY")),
        _safe_str(ats_data.get("email_secondary")),
        _safe_str(ats_data.get("email")),
    ]
    profile["emails"] = [e for e in email_candidates if e]

    # Collect all phone variants.
    phone_candidates = [
        _safe_str(ats_data.get("phone_RAW")),
        _safe_str(ats_data.get("phone")),
        _safe_str(ats_data.get("phone_number")),
    ]
    profile["phones"] = [p for p in phone_candidates if p]

    # --- Location ---
    location_raw: dict = ats_data.get("location", {}) or {}
    profile["location"] = {
        "city": _safe_str(
            location_raw.get("city_name")
            or location_raw.get("city")
        ),
        "region": _safe_str(
            location_raw.get("state_region")
            or location_raw.get("region")
            or location_raw.get("state")
        ),
        "country": _safe_str(
            location_raw.get("cntry")
            or location_raw.get("country")
            or location_raw.get("country_name")
        ),
    }

    # --- Professional Links ---
    socials_raw: dict = ats_data.get("social_profiles", {}) or {}
    profile["links"] = {
        "linkedin": _safe_str(
            socials_raw.get("linkedin_url")
            or socials_raw.get("linkedin")
        ),
        "github": _safe_str(
            socials_raw.get("gh")
            or socials_raw.get("github")
            or socials_raw.get("github_url")
        ),
        "portfolio": _safe_str(
            socials_raw.get("portfolio")
            or socials_raw.get("portfolio_url")
        ),
        "other": [],
    }

    # --- Work Experience ---
    work_history: list = ats_data.get("work_history", []) or []
    for entry in work_history:
        if not isinstance(entry, dict):
            continue
        company = _safe_str(entry.get("employer") or entry.get("company"))
        title = _safe_str(entry.get("job_title") or entry.get("title"))
        if not company or not title:
            # Skip malformed entries that lack the minimum required fields.
            print(
                f"[ingestion] Skipping malformed work_history entry (missing company or title): {entry}",
                file=sys.stderr,
            )
            continue
        profile["experience"].append({
            "company": company,
            "title": title,
            "start": _safe_str(entry.get("date_from") or entry.get("start"), fallback=""),
            "end": _safe_str(entry.get("date_to") or entry.get("end")),
            "summary": _safe_str(entry.get("responsibilities") or entry.get("summary")),
        })

    # --- Education ---
    education_history: list = ats_data.get("education_history", []) or []
    for entry in education_history:
        if not isinstance(entry, dict):
            continue
        institution = _safe_str(entry.get("school") or entry.get("institution"))
        if not institution:
            print(
                f"[ingestion] Skipping malformed education entry (missing institution): {entry}",
                file=sys.stderr,
            )
            continue
        profile["education"].append({
            "institution": institution,
            "degree": _safe_str(entry.get("qualification") or entry.get("degree")),
            "field_of_study": _safe_str(entry.get("subject") or entry.get("field_of_study")),
            "start": _safe_str(entry.get("start_yr") or entry.get("start")),
            "end": _safe_str(entry.get("end_yr") or entry.get("end")),
        })

    # --- Skills ---
    profile["skills"] = _safe_list_of_str(ats_data.get("skills_list") or ats_data.get("skills"))

    print(
        f"[ingestion] ATS parse complete. "
        f"Emails: {len(profile['emails'])}, "
        f"Experience entries: {len(profile['experience'])}, "
        f"Skills: {len(profile['skills'])}",
        file=sys.stderr,
    )

    return profile


# ---------------------------------------------------------------------------
# Recruiter Notes Parser (LLM Extraction via Claude API)
# ---------------------------------------------------------------------------

# The JSON schema passed to the Claude API as a tool definition.
# Defines exactly what fields the model must extract from free-form text.
_EXTRACTION_TOOL_SCHEMA: dict = {
    "name": "extract_candidate_profile",
    "description": (
        "Extract all structured candidate information from the unstructured recruiter notes. "
        "Return only fields that are explicitly mentioned in the text. "
        "Do not invent or hallucinate values. "
        "For fields not mentioned, omit the key or return null."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "full_name": {
                "type": "string",
                "description": "Candidate's full name."
            },
            "emails": {
                "type": "array",
                "items": {"type": "string"},
                "description": "All email addresses mentioned."
            },
            "phones": {
                "type": "array",
                "items": {"type": "string"},
                "description": "All phone numbers mentioned, in any format."
            },
            "location": {
                "type": "object",
                "properties": {
                    "city":    {"type": "string"},
                    "region":  {"type": "string"},
                    "country": {"type": "string"}
                },
                "description": "Candidate's location if mentioned."
            },
            "links": {
                "type": "object",
                "properties": {
                    "linkedin":  {"type": "string"},
                    "github":    {"type": "string"},
                    "portfolio": {"type": "string"}
                },
                "description": "Professional profile URLs if mentioned."
            },
            "skills": {
                "type": "array",
                "items": {"type": "string"},
                "description": "All technical skills and technologies mentioned."
            },
            "experience": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "company": {"type": "string"},
                        "title":   {"type": "string"},
                        "start":   {"type": "string"},
                        "end":     {"type": "string"},
                        "summary": {"type": "string"}
                    },
                    "required": ["company", "title"]
                },
                "description": "Work experience entries extracted from the notes."
            },
            "education": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "institution":    {"type": "string"},
                        "degree":         {"type": "string"},
                        "field_of_study": {"type": "string"},
                        "start":          {"type": "string"},
                        "end":            {"type": "string"}
                    },
                    "required": ["institution"]
                },
                "description": "Education history extracted from the notes."
            }
        },
        "required": []
    }
}


def _call_claude_extraction_api(notes_text: str) -> dict:
    """
    Call the Anthropic Claude API with the recruiter notes text and the
    structured extraction tool definition. Returns the tool_use input dict.

    Uses the claude-sonnet-4-6 model with tool_choice forced to the extraction
    tool so the response is guaranteed to be structured JSON, not prose.

    Args:
        notes_text: Raw recruiter notes string.

    Returns:
        Extracted fields dict matching _EXTRACTION_TOOL_SCHEMA input_schema.

    Raises:
        RuntimeError: If the API call fails or returns an unexpected structure.
        ImportError:  If the anthropic package is not installed.
    """
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "The 'anthropic' package is required for live LLM extraction. "
            "Install it with: pip install anthropic"
        ) from exc

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Cannot perform live LLM extraction."
        )

    client = anthropic.Anthropic(api_key=api_key)

    print("[ingestion] Calling Claude API for recruiter notes extraction...", file=sys.stderr)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            tools=[_EXTRACTION_TOOL_SCHEMA],
            # Force the model to call our extraction tool rather than responding in prose.
            tool_choice={"type": "tool", "name": "extract_candidate_profile"},
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Please extract all structured candidate information from the "
                        "following recruiter notes. Use only information explicitly present "
                        "in the text — do not invent or infer values not stated.\n\n"
                        f"---\n{notes_text}\n---"
                    )
                }
            ]
        )
    except Exception as exc:
        raise RuntimeError(
            f"[ingestion] Claude API call failed: {exc}"
        ) from exc

    # Locate the tool_use block in the response content.
    tool_use_block = next(
        (block for block in response.content if block.type == "tool_use"),
        None
    )

    if tool_use_block is None:
        raise RuntimeError(
            "[ingestion] Claude API response contained no tool_use block. "
            f"Full response content: {response.content}"
        )

    extracted: dict = tool_use_block.input

    print(
        f"[ingestion] Claude extraction complete. "
        f"Fields returned: {list(extracted.keys())}",
        file=sys.stderr,
    )

    return extracted


def _build_mock_extraction_response() -> dict:
    """
    Return a hard-coded extraction result that mirrors what the Claude API
    would produce from recruiter_notes.txt.

    This mock is used when ANTHROPIC_API_KEY is not set, enabling full
    end-to-end pipeline testing without any API dependency.

    The values deliberately use the natural-language date formats and phone
    formatting present in the recruiter notes to stress-test the normalizer.

    Returns:
        Dict conforming to _EXTRACTION_TOOL_SCHEMA input_schema.
    """
    return {
        "full_name": "Jane Doe",
        "emails": ["jane.doe@email.com"],
        "phones": ["+1 (555) 019-2834"],
        "location": None,
        "links": {
            "linkedin": "https://linkedin.com/in/janedoe",
            "github": None,
            "portfolio": None,
        },
        "skills": ["Python", "React", "FastAPI", "PostgreSQL", "Docker", "Kubernetes"],
        "experience": [
            {
                "company": "Acme Corp",
                "title": "Lead Engineer",
                "start": "Oct 2022",
                "end": "Present",
                "summary": (
                    "Leading a data infrastructure project and building out "
                    "the front-end in React."
                ),
            },
            {
                "company": "Beta Systems Inc.",
                "title": "Software Engineer II",
                "start": "March 2019",
                "end": "September 2022",
                "summary": "Solid foundational work, mainly microservices.",
            },
        ],
        "education": [
            {
                "institution": "UC Berkeley",
                "degree": "Bachelor of Science",
                "field_of_study": "Computer Science",
                "start": None,
                "end": "May 2019",
            }
        ],
    }


def _map_extracted_dict_to_profile(extracted: dict, source_id: str) -> dict:
    """
    Map the raw dict returned by the Claude API (or mock) into the
    internal raw profile schema, applying the same safe coercion helpers
    used by the ATS parser.

    Args:
        extracted:  Dict returned by _call_claude_extraction_api or mock.
        source_id:  Originating file path for provenance tracking.

    Returns:
        Raw profile dict conforming to the internal schema.
    """
    profile = _empty_raw_profile(source_id=source_id, source_type="recruiter_notes")

    profile["full_name"] = _safe_str(extracted.get("full_name"))
    profile["emails"] = _safe_list_of_str(extracted.get("emails"))
    profile["phones"] = _safe_list_of_str(extracted.get("phones"))

    location_raw = extracted.get("location") or {}
    if isinstance(location_raw, dict):
        profile["location"] = {
            "city":    _safe_str(location_raw.get("city")),
            "region":  _safe_str(location_raw.get("region")),
            "country": _safe_str(location_raw.get("country")),
        }

    links_raw = extracted.get("links") or {}
    if isinstance(links_raw, dict):
        profile["links"] = {
            "linkedin":  _safe_str(links_raw.get("linkedin")),
            "github":    _safe_str(links_raw.get("github")),
            "portfolio": _safe_str(links_raw.get("portfolio")),
            "other":     _safe_list_of_str(links_raw.get("other")),
        }

    profile["skills"] = _safe_list_of_str(extracted.get("skills"))

    for entry in (extracted.get("experience") or []):
        if not isinstance(entry, dict):
            continue
        company = _safe_str(entry.get("company"))
        title = _safe_str(entry.get("title"))
        if not company or not title:
            continue
        profile["experience"].append({
            "company": company,
            "title":   title,
            "start":   _safe_str(entry.get("start"), fallback=""),
            "end":     _safe_str(entry.get("end")),
            "summary": _safe_str(entry.get("summary")),
        })

    for entry in (extracted.get("education") or []):
        if not isinstance(entry, dict):
            continue
        institution = _safe_str(entry.get("institution"))
        if not institution:
            continue
        profile["education"].append({
            "institution":    institution,
            "degree":         _safe_str(entry.get("degree")),
            "field_of_study": _safe_str(entry.get("field_of_study")),
            "start":          _safe_str(entry.get("start")),
            "end":            _safe_str(entry.get("end")),
        })

    return profile


def parse_recruiter_notes(file_path: str) -> dict:
    """
    Extract structured candidate data from a plain-text recruiter notes file.

    Execution path is determined by API key availability:
        - ANTHROPIC_API_KEY set → Live Claude API call with forced tool_use.
        - ANTHROPIC_API_KEY absent → Hard-coded mock response (for local testing).

    Both paths produce an identical output schema, so downstream components
    are completely unaware of which extraction method was used.

    Args:
        file_path: Absolute or relative path to the recruiter notes .txt file.

    Returns:
        Raw profile dict conforming to the internal schema.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        RuntimeError:      If the live API call fails.
    """
    path = Path(file_path).resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"[ingestion] Recruiter notes file not found: {path}"
        )

    notes_text = path.read_text(encoding="utf-8")
    print(f"[ingestion] Parsing recruiter notes: {path}", file=sys.stderr)

    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if api_key:
        print("[ingestion] ANTHROPIC_API_KEY detected → using live Claude API.", file=sys.stderr)
        try:
            extracted = _call_claude_extraction_api(notes_text)
        except Exception as exc:
            # Surface the error clearly but do not silently swallow it.
            # The caller (or CLI) decides whether to abort or continue with other sources.
            print(f"[ingestion] Live API extraction failed: {exc}", file=sys.stderr)
            raise
    else:
        print(
            "[ingestion] ANTHROPIC_API_KEY not set → using mock extraction response "
            "(safe for local testing without API access).",
            file=sys.stderr,
        )
        extracted = _build_mock_extraction_response()

    profile = _map_extracted_dict_to_profile(extracted, source_id=str(path))

    print(
        f"[ingestion] Recruiter notes parse complete. "
        f"Emails: {len(profile['emails'])}, "
        f"Experience entries: {len(profile['experience'])}, "
        f"Skills: {len(profile['skills'])}",
        file=sys.stderr,
    )

    return profile