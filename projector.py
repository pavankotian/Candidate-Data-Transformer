"""
projector.py
------------
Dynamic configuration projection engine for the Multi-Source Candidate Data
Transformer.

Accepts a fully populated CanonicalProfile (or its dict representation) and a
runtime projection configuration dict, then produces a reshaped output dict
whose structure is entirely governed by the configuration — not hardcoded here.

Projection configuration schema (JSON):
{
    "flags": {
        "include_provenance":         bool,   // default false
        "include_overall_confidence": bool    // default false
    },
    "fields": [
        {
            "from":       str,          // dot/bracket path into CanonicalProfile
            "path":       str,          // output key name in the projected dict
            "on_missing": str,          // "null" | "omit" | "error"
            "format":     str | null    // optional post-resolution transform
        },
        ...
    ]
}

Supported "from" path syntax:
    full_name                   → scalar field
    emails[0]                   → first element of an array
    emails[-1]                  → last element of an array
    skills[].name               → pluck "name" from every element of "skills" array
    experience[0].company       → sub-field of a specific array element
    location.city               → nested object field
    links.linkedin              → nested object field

Supported "format" values:
    "uppercase"     → str.upper()
    "lowercase"     → str.lower()
    "titlecase"     → str.title()
    "strip"         → str.strip()
    "e164"          → normalize_phone() → E.164
    "yyyy-mm"       → normalize_date()  → YYYY-MM
    "iso-alpha2"    → normalize_country() → ISO-3166 alpha-2
    "join_comma"    → ", ".join(list)   (list → str)
    "join_newline"  → "\n".join(list)   (list → str)
    "first"         → list[0] if non-empty, else None
    "count"         → len(list) as int

Public API:
    project(
        profile: CanonicalProfile | dict,
        config:  dict
    ) -> dict
"""

from __future__ import annotations

import re
import sys
from typing import Any, Optional

from models import CanonicalProfile
from normalizer import normalize_country, normalize_date, normalize_phone


# ---------------------------------------------------------------------------
# Sentinel object used to distinguish "key was missing" from "value is None"
# ---------------------------------------------------------------------------

_MISSING = object()


# ---------------------------------------------------------------------------
# Path Resolution
# ---------------------------------------------------------------------------

# Matches:  some_field[0]   some_field[-1]   some_field[]
_INDEXED_SEGMENT_RE = re.compile(r"^(.+?)\[(-?\d*)\]$")


def _resolve_path(data: dict, path: str) -> Any:
    """
    Resolve a dot/bracket path expression against a nested dict and return
    the value found, or the _MISSING sentinel if any segment cannot be
    traversed.

    Supported path forms
    --------------------
    Scalar field:
        "full_name"                 → data["full_name"]

    Nested object field:
        "location.city"             → data["location"]["city"]
        "links.linkedin"            → data["links"]["linkedin"]

    Specific array element:
        "emails[0]"                 → data["emails"][0]
        "emails[-1]"                → data["emails"][-1]
        "experience[0].company"     → data["experience"][0]["company"]

    Wildcard array pluck (returns a list):
        "skills[].name"             → [item["name"] for item in data["skills"]]
        "experience[].title"        → [item["title"] for item in data["experience"]]
        "phones[]"                  → all elements of data["phones"]

    Resolution is intentionally strict:
        - Any missing key at any depth returns _MISSING (not None), so callers
          can distinguish "field absent" from "field present but null".
        - An out-of-range index on an array returns _MISSING.
        - Type mismatches (e.g., indexing into a scalar) return _MISSING.

    Args:
        data: Nested dict (the model_dump() of a CanonicalProfile).
        path: Path expression string.

    Returns:
        Resolved value, or _MISSING sentinel if the path cannot be followed.
    """
    if not path or not path.strip():
        return _MISSING

    segments = _split_path(path)
    return _traverse(data, segments)


def _split_path(path: str) -> list[str]:
    """
    Split a path string into ordered traversal tokens.

    "experience[0].company" → ["experience[0]", "company"]
    "skills[].name"         → ["skills[]", "name"]
    "location.city"         → ["location", "city"]
    "full_name"             → ["full_name"]

    Args:
        path: Raw path expression string.

    Returns:
        Ordered list of segment tokens.
    """
    # Split on '.' but not when the dot is inside brackets (there are none in
    # our syntax, but guard defensively).
    return [seg for seg in path.split(".") if seg]


def _traverse(node: Any, segments: list[str]) -> Any:
    """
    Recursively traverse `node` by consuming segments from the front of the list.

    Args:
        node:     Current position in the data tree.
        segments: Remaining path segments to consume.

    Returns:
        Resolved value, or _MISSING if traversal cannot continue.
    """
    if not segments:
        return node

    segment = segments[0]
    remaining = segments[1:]

    # --- Bracket segment: "field[N]" or "field[]" ---
    bracket_match = _INDEXED_SEGMENT_RE.match(segment)
    if bracket_match:
        field_name = bracket_match.group(1)
        index_str  = bracket_match.group(2)

        # Resolve the field name against the current node.
        if not isinstance(node, dict):
            return _MISSING
        array_value = node.get(field_name, _MISSING)
        if array_value is _MISSING or array_value is None:
            return _MISSING
        if not isinstance(array_value, list):
            return _MISSING

        if index_str == "":
            # Wildcard "[]" — pluck remaining path from every element.
            if not remaining:
                # "phones[]" with no further segments → return entire list.
                return array_value
            # "skills[].name" → collect sub-value from each element.
            collected = []
            for element in array_value:
                sub_value = _traverse(element, remaining)
                if sub_value is not _MISSING and sub_value is not None:
                    collected.append(sub_value)
            # Return _MISSING only if the array itself was empty; empty result
            # list is a valid (empty) resolved value.
            return collected

        else:
            # Specific numeric index: "emails[0]", "emails[-1]".
            try:
                idx = int(index_str)
                element = array_value[idx]
            except (IndexError, ValueError):
                return _MISSING

            return _traverse(element, remaining)

    # --- Plain field segment (no brackets) ---
    if isinstance(node, dict):
        value = node.get(segment, _MISSING)
        if value is _MISSING:
            return _MISSING
        return _traverse(value, remaining)

    # Node is not a dict and we still have a plain field segment to consume.
    return _MISSING


# ---------------------------------------------------------------------------
# Format Transforms
# ---------------------------------------------------------------------------

# Registry mapping format name → callable(value) → transformed value.
# Each callable receives the already-resolved value and returns a new value.
# Callables may raise ValueError on invalid input.

def _fmt_uppercase(v: Any) -> Any:
    return str(v).upper() if v is not None else None

def _fmt_lowercase(v: Any) -> Any:
    return str(v).lower() if v is not None else None

def _fmt_titlecase(v: Any) -> Any:
    return str(v).title() if v is not None else None

def _fmt_strip(v: Any) -> Any:
    return str(v).strip() if v is not None else None

def _fmt_e164(v: Any) -> Any:
    if v is None:
        return None
    return normalize_phone(str(v))

def _fmt_yyyy_mm(v: Any) -> Any:
    if v is None:
        return None
    return normalize_date(str(v))

def _fmt_iso_alpha2(v: Any) -> Any:
    if v is None:
        return None
    return normalize_country(str(v))

def _fmt_join_comma(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, list):
        return ", ".join(str(item) for item in v if item is not None)
    return str(v)

def _fmt_join_newline(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, list):
        return "\n".join(str(item) for item in v if item is not None)
    return str(v)

def _fmt_first(v: Any) -> Any:
    if isinstance(v, list):
        return v[0] if v else None
    return v

def _fmt_count(v: Any) -> Any:
    if isinstance(v, list):
        return len(v)
    if v is None:
        return 0
    return 1

_FORMAT_REGISTRY: dict[str, Any] = {
    "uppercase":    _fmt_uppercase,
    "lowercase":    _fmt_lowercase,
    "titlecase":    _fmt_titlecase,
    "strip":        _fmt_strip,
    "e164":         _fmt_e164,
    "yyyy-mm":      _fmt_yyyy_mm,
    "iso-alpha2":   _fmt_iso_alpha2,
    "join_comma":   _fmt_join_comma,
    "join_newline": _fmt_join_newline,
    "first":        _fmt_first,
    "count":        _fmt_count,
}


def _apply_format(value: Any, format_name: str, from_path: str) -> Any:
    """
    Apply a named format transform to a resolved value.

    Args:
        value:       The resolved value to transform.
        format_name: Key into _FORMAT_REGISTRY.
        from_path:   Original "from" path, used in error messages only.

    Returns:
        Transformed value.

    Raises:
        ValueError: If format_name is not registered, or if the transform
                    itself raises ValueError (e.g., invalid phone for "e164").
    """
    if format_name not in _FORMAT_REGISTRY:
        raise ValueError(
            f"[projector] Unknown format '{format_name}' specified for "
            f"field '{from_path}'. "
            f"Valid formats: {sorted(_FORMAT_REGISTRY.keys())}"
        )
    try:
        return _FORMAT_REGISTRY[format_name](value)
    except ValueError as exc:
        raise ValueError(
            f"[projector] Format '{format_name}' failed for field "
            f"'{from_path}' with value {value!r}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Output Path Writing
# ---------------------------------------------------------------------------

def _set_nested(output: dict, dot_path: str, value: Any) -> None:
    """
    Write `value` into `output` at the location described by `dot_path`,
    creating intermediate dicts as needed.

    Examples:
        "primary_email"         → output["primary_email"] = value
        "contact.email"         → output["contact"]["email"] = value
        "meta.location.city"    → output["meta"]["location"]["city"] = value

    Args:
        output:   The mutable output dict being built.
        dot_path: Dot-separated output key path.
        value:    Value to write.
    """
    parts = [p for p in dot_path.split(".") if p]
    if not parts:
        return

    node = output
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]

    node[parts[-1]] = value


# ---------------------------------------------------------------------------
# Configuration Parsing & Validation
# ---------------------------------------------------------------------------

_VALID_ON_MISSING = frozenset({"null", "omit", "error"})


def _parse_flags(config: dict) -> dict:
    """
    Extract and validate the "flags" block from the projection config.

    Both flags default to False if absent. Unknown keys inside "flags" emit
    a warning to stderr but do not raise — forward-compatible with future flags.

    Args:
        config: Top-level projection configuration dict.

    Returns:
        Normalised flags dict with keys "include_provenance" and
        "include_overall_confidence".
    """
    raw_flags = config.get("flags", {}) or {}

    known_flags = {"include_provenance", "include_overall_confidence"}
    for key in raw_flags:
        if key not in known_flags:
            print(
                f"[projector] Warning: unknown flag '{key}' in config — ignored.",
                file=sys.stderr,
            )

    return {
        "include_provenance":         bool(raw_flags.get("include_provenance",         False)),
        "include_overall_confidence": bool(raw_flags.get("include_overall_confidence", False)),
    }


def _parse_field_spec(spec: Any, index: int) -> dict:
    """
    Validate a single field specification entry from the "fields" array.

    Required keys: "from", "path", "on_missing".
    Optional keys: "format".

    Args:
        spec:  Raw entry from the config "fields" array.
        index: Position in the array (used in error messages).

    Returns:
        Validated field spec dict.

    Raises:
        ValueError: If any required key is missing or has an invalid value.
    """
    if not isinstance(spec, dict):
        raise ValueError(
            f"[projector] Field spec at index {index} must be a dict, "
            f"got {type(spec).__name__}: {spec!r}"
        )

    # --- Required: "from" ---
    from_path = spec.get("from")
    if not from_path or not isinstance(from_path, str) or not from_path.strip():
        raise ValueError(
            f"[projector] Field spec at index {index} is missing a valid "
            f"'from' key. Got: {from_path!r}"
        )

    # --- Required: "path" ---
    output_path = spec.get("path")
    if not output_path or not isinstance(output_path, str) or not output_path.strip():
        raise ValueError(
            f"[projector] Field spec at index {index} (from='{from_path}') "
            f"is missing a valid 'path' key. Got: {output_path!r}"
        )

    # --- Required: "on_missing" ---
    on_missing = spec.get("on_missing")
    if not on_missing or on_missing not in _VALID_ON_MISSING:
        raise ValueError(
            f"[projector] Field spec at index {index} (from='{from_path}') "
            f"has invalid 'on_missing' value: {on_missing!r}. "
            f"Must be one of: {sorted(_VALID_ON_MISSING)}"
        )

    # --- Optional: "format" ---
    format_name = spec.get("format")
    if format_name is not None:
        if not isinstance(format_name, str) or not format_name.strip():
            raise ValueError(
                f"[projector] Field spec at index {index} (from='{from_path}') "
                f"has invalid 'format' value: {format_name!r}. "
                f"Must be a non-empty string or omitted/null."
            )

    return {
        "from":       from_path.strip(),
        "path":       output_path.strip(),
        "on_missing": on_missing,
        "format":     format_name.strip() if format_name else None,
    }


# ---------------------------------------------------------------------------
# Core Projection Logic
# ---------------------------------------------------------------------------

def _profile_to_dict(profile: CanonicalProfile | dict) -> dict:
    """
    Ensure the profile is a plain dict suitable for path traversal.

    Converts Pydantic CanonicalProfile instances via model_dump(); passes
    plain dicts through unchanged.

    Args:
        profile: CanonicalProfile instance or pre-serialised dict.

    Returns:
        Plain nested dict.

    Raises:
        TypeError: If the input is neither a CanonicalProfile nor a dict.
    """
    if isinstance(profile, CanonicalProfile):
        return profile.model_dump(mode="python")
    if isinstance(profile, dict):
        return profile
    raise TypeError(
        f"[projector] project() expects a CanonicalProfile or dict, "
        f"got {type(profile).__name__}."
    )


def _resolve_field(
    profile_dict: dict,
    spec: dict,
) -> tuple[Any, bool]:
    """
    Resolve the value for a single field spec against the profile dict and
    apply any declared format transform.

    Args:
        profile_dict: Flat/nested dict from _profile_to_dict().
        spec:         Validated field spec dict from _parse_field_spec().

    Returns:
        Tuple of (resolved_value, is_missing):
            resolved_value: The value after format transform (may be None).
            is_missing:     True if the source path resolved to _MISSING.

    Raises:
        ValueError: Propagated from _apply_format() on transform failure.
    """
    from_path   = spec["from"]
    format_name = spec["format"]

    raw_value = _resolve_path(profile_dict, from_path)

    # Distinguish truly absent (_MISSING sentinel) from explicitly null (None).
    if raw_value is _MISSING:
        return None, True

    # Apply format transform if requested.
    if format_name:
        transformed = _apply_format(raw_value, format_name, from_path)
    else:
        transformed = raw_value

    return transformed, False


def project(
    profile: CanonicalProfile | dict,
    config: dict,
) -> dict:
    """
    Project a CanonicalProfile into a custom output shape governed entirely
    by the runtime configuration dict.

    Processing steps
    ----------------
    1. Validate and parse the config (flags block + fields array).
    2. Convert the profile to a plain dict.
    3. For each field spec:
           a. Resolve the "from" path against the profile dict.
           b. Apply any declared "format" transform.
           c. Honour "on_missing" if the path resolved to _MISSING:
                  "null"  → write None at the output path.
                  "omit"  → skip; do not write the output key at all.
                  "error" → raise ValueError immediately.
           d. Write the transformed value to the output dict at "path".
    4. Conditionally append provenance and overall_confidence per flags.
    5. Return the assembled output dict.

    Args:
        profile: Populated CanonicalProfile instance or its dict equivalent.
        config:  Projection configuration dict (see module docstring for schema).

    Returns:
        Reshaped output dict conforming to the projection configuration.

    Raises:
        TypeError:  If profile is not a CanonicalProfile or dict.
        ValueError: If config is structurally invalid, if an "error" on_missing
                    fires, or if a format transform fails.
    """
    if not isinstance(config, dict):
        raise ValueError(
            f"[projector] config must be a dict, got {type(config).__name__}."
        )

    # --- Step 1: Parse configuration ---
    flags = _parse_flags(config)

    raw_fields = config.get("fields", [])
    if not isinstance(raw_fields, list):
        raise ValueError(
            f"[projector] config 'fields' must be a list, "
            f"got {type(raw_fields).__name__}."
        )

    field_specs = [_parse_field_spec(spec, idx) for idx, spec in enumerate(raw_fields)]

    # --- Step 2: Normalise profile to dict ---
    profile_dict = _profile_to_dict(profile)

    # --- Step 3: Resolve and project each field ---
    output: dict = {}

    for spec in field_specs:
        from_path   = spec["from"]
        output_path = spec["path"]
        on_missing  = spec["on_missing"]

        try:
            value, is_missing = _resolve_field(profile_dict, spec)
        except ValueError as exc:
            # Format transform failed — surface with output path context.
            raise ValueError(
                f"[projector] Format error on output field '{output_path}' "
                f"(from='{from_path}'): {exc}"
            ) from exc

        if is_missing:
            if on_missing == "omit":
                print(
                    f"[projector] Field '{from_path}' → '{output_path}': "
                    "value missing, on_missing='omit' — skipping.",
                    file=sys.stderr,
                )
                continue

            elif on_missing == "null":
                print(
                    f"[projector] Field '{from_path}' → '{output_path}': "
                    "value missing, on_missing='null' — writing null.",
                    file=sys.stderr,
                )
                _set_nested(output, output_path, None)
                continue

            elif on_missing == "error":
                raise ValueError(
                    f"[projector] Mandatory field '{from_path}' is missing from "
                    f"the CanonicalProfile but is required by the projection "
                    f"config (output key: '{output_path}', on_missing='error'). "
                    "Ensure the ingestion and merge pipeline populated this field."
                )

        else:
            _set_nested(output, output_path, value)
            print(
                f"[projector] Field '{from_path}' → '{output_path}': "
                f"resolved {_repr_value(value)}",
                file=sys.stderr,
            )

    # --- Step 4: Conditional flag-governed fields ---
    if flags["include_overall_confidence"]:
        confidence_raw = _resolve_path(profile_dict, "overall_confidence")
        confidence_val = None if confidence_raw is _MISSING else confidence_raw
        _set_nested(output, "overall_confidence", confidence_val)
        print(
            f"[projector] Flag 'include_overall_confidence' → "
            f"overall_confidence={confidence_val}",
            file=sys.stderr,
        )

    if flags["include_provenance"]:
        provenance_raw = _resolve_path(profile_dict, "provenance")
        provenance_val = [] if provenance_raw is _MISSING else provenance_raw
        _set_nested(output, "provenance", provenance_val)
        print(
            f"[projector] Flag 'include_provenance' → "
            f"{len(provenance_val) if isinstance(provenance_val, list) else '?'} entries",
            file=sys.stderr,
        )

    return output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _repr_value(value: Any, max_len: int = 80) -> str:
    """
    Produce a compact, truncated repr of a value for debug logging.

    Args:
        value:   Any resolved value.
        max_len: Maximum character length before truncation.

    Returns:
        Short human-readable string.
    """
    raw = repr(value)
    if len(raw) > max_len:
        return raw[:max_len - 3] + "..."
    return raw


# ---------------------------------------------------------------------------
# Smoke Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from models import (
        CanonicalProfile, ExperienceItem, SkillItem,
        Location, Links, ProvenanceItem, EducationItem,
    )

    print("=" * 60)
    print("projector.py — standalone smoke test")
    print("=" * 60)

    # Build a minimal CanonicalProfile that mirrors what merger.py produces
    # for Jane Doe after Sprint 3.
    mock_profile = CanonicalProfile(
        candidate_id="cand-abc123def456",
        full_name="Jane Doe",
        emails=["jane.doe@email.com", "j.doe@workmail.com"],
        phones=["+15550192834"],
        location=Location(city="San Francisco", region="California", country="US"),
        links=Links(
            linkedin="https://linkedin.com/in/janedoe",
            github="https://github.com/janedoe",
            portfolio=None,
            other=[],
        ),
        skills=[
            SkillItem(name="Python",     confidence=0.95, sources=["ats", "recruiter_notes"]),
            SkillItem(name="React",      confidence=0.95, sources=["ats", "recruiter_notes"]),
            SkillItem(name="FastAPI",    confidence=0.95, sources=["ats", "recruiter_notes"]),
            SkillItem(name="PostgreSQL", confidence=0.95, sources=["ats", "recruiter_notes"]),
            SkillItem(name="Docker",     confidence=0.95, sources=["ats", "recruiter_notes"]),
            SkillItem(name="Kubernetes", confidence=0.75, sources=["recruiter_notes"]),
        ],
        experience=[
            ExperienceItem(
                company="Acme Corp",
                title="Lead Software Engineer",
                start="2022-10",
                end=None,
                summary="Led cross-functional team, built real-time data pipeline.",
            ),
            ExperienceItem(
                company="Beta Systems Inc.",
                title="Software Engineer II",
                start="2019-03",
                end="2022-09",
                summary="RESTful microservices in Python (FastAPI).",
            ),
        ],
        education=[
            EducationItem(
                institution="University of California, Berkeley",
                degree="Bachelor of Science",
                field_of_study="Computer Science",
                start="2015-08",
                end="2019-05",
            )
        ],
        overall_confidence=0.95,
        provenance=[
            ProvenanceItem(
                field="full_name", source="ats",
                method="initial_write", value="Jane Doe",
            ),
            ProvenanceItem(
                field="emails", source="recruiter_notes",
                method="array_union", value="jane.doe@email.com",
            ),
        ],
        raw_sources=["ats_input.json", "recruiter_notes.txt"],
    )

    # -----------------------------------------------------------------------
    # Projection configuration exercising all major features:
    #   - Scalar fields, nested fields, indexed array access
    #   - Wildcard pluck with format transforms
    #   - All three on_missing behaviours
    #   - Both flags enabled
    # -----------------------------------------------------------------------
    projection_config = {
        "flags": {
            "include_provenance":         True,
            "include_overall_confidence": True,
        },
        "fields": [
            # Basic scalar
            {
                "from":       "candidate_id",
                "path":       "id",
                "on_missing": "error",
                "format":     None,
            },
            # Scalar with titlecase format
            {
                "from":       "full_name",
                "path":       "name",
                "on_missing": "error",
                "format":     "titlecase",
            },
            # First element of array
            {
                "from":       "emails[0]",
                "path":       "primary_email",
                "on_missing": "null",
                "format":     "lowercase",
            },
            # Second email (index 1)
            {
                "from":       "emails[1]",
                "path":       "secondary_email",
                "on_missing": "omit",
                "format":     None,
            },
            # E.164 phone (already E.164, format is idempotent here)
            {
                "from":       "phones[0]",
                "path":       "phone",
                "on_missing": "null",
                "format":     "e164",
            },
            # Nested object field
            {
                "from":       "location.city",
                "path":       "location.city",
                "on_missing": "null",
                "format":     None,
            },
            {
                "from":       "location.country",
                "path":       "location.country_code",
                "on_missing": "null",
                "format":     "uppercase",
            },
            # Wildcard pluck + join
            {
                "from":       "skills[].name",
                "path":       "skills_csv",
                "on_missing": "null",
                "format":     "join_comma",
            },
            # Skill count
            {
                "from":       "skills[].name",
                "path":       "skill_count",
                "on_missing": "null",
                "format":     "count",
            },
            # Specific experience sub-field
            {
                "from":       "experience[0].company",
                "path":       "current_employer",
                "on_missing": "null",
                "format":     None,
            },
            {
                "from":       "experience[0].title",
                "path":       "current_title",
                "on_missing": "null",
                "format":     None,
            },
            # LinkedIn link
            {
                "from":       "links.linkedin",
                "path":       "linkedin_url",
                "on_missing": "omit",
                "format":     None,
            },
            # Portfolio missing → omit (will not appear in output)
            {
                "from":       "links.portfolio",
                "path":       "portfolio_url",
                "on_missing": "omit",
                "format":     None,
            },
            # Education institution
            {
                "from":       "education[0].institution",
                "path":       "education.university",
                "on_missing": "null",
                "format":     None,
            },
            # A field guaranteed to be absent → null
            {
                "from":       "headline",
                "path":       "headline",
                "on_missing": "null",
                "format":     None,
            },
        ],
    }

    print("\n[Running projection...]\n")
    result = project(mock_profile, projection_config)

    print("\n" + "=" * 60)
    print("PROJECTED OUTPUT")
    print("=" * 60)

    # Serialize — replace provenance list with entry count for brevity.
    display = {
        k: (f"[{len(v)} provenance entries]" if k == "provenance" and isinstance(v, list) else v)
        for k, v in result.items()
    }
    print(json.dumps(display, indent=2, default=str))

    # -----------------------------------------------------------------------
    # on_missing="error" test
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Testing on_missing='error' for a mandatory missing field...")
    print("=" * 60)

    error_config = {
        "flags": {},
        "fields": [
            {
                "from":       "headline",       # Does not exist on profile.
                "path":       "headline",
                "on_missing": "error",
                "format":     None,
            }
        ],
    }

    try:
        project(mock_profile, error_config)
        print("[FAIL] Expected ValueError was not raised.")
    except ValueError as exc:
        print(f"[OK] ValueError correctly raised:\n  {exc}")