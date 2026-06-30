"""
merger.py
---------
Identity merge and conflict resolution engine for the Multi-Source Candidate
Data Transformer.

Responsibilities:
    1. Normalise contact identifiers (emails, phones) from every raw profile
       so that matching keys are comparable regardless of source formatting.
    2. Link records across sources using compound key intersection on
       normalised emails and phones (strict deterministic matching — no fuzzy logic).
    3. Resolve scalar field conflicts using source confidence weights:
           ATS JSON        → 0.95
           Recruiter Notes → 0.75
       The higher-confidence source always wins on collision.
    4. Merge array fields (skills, experience, education) with explicit
       deduplication logic.
    5. Write a ProvenanceItem for every field decision made, enabling a full
       downstream audit trail.
    6. Produce a finalised CanonicalProfile per resolved identity group.

Public API:
    run_merge(raw_profiles: list[dict]) -> list[CanonicalProfile]
        Accepts the list of raw profile dicts produced by ingestion.py and
        returns one CanonicalProfile per resolved unique candidate identity.
"""

from __future__ import annotations

import hashlib
import sys
import uuid
from typing import Any, Optional

from models import (
    CanonicalProfile,
    EducationItem,
    ExperienceItem,
    Links,
    Location,
    ProvenanceItem,
    SkillItem,
)
from normalizer import normalize_country, normalize_date, normalize_phone


# ---------------------------------------------------------------------------
# Source Confidence Registry
# ---------------------------------------------------------------------------

# Maps _source_type values (set by ingestion.py) to their base confidence weight.
# These are used both for conflict resolution and for overall_confidence calculation.
SOURCE_CONFIDENCE: dict[str, float] = {
    "ats": 0.95,
    "recruiter_notes": 0.75,
}

# Fallback confidence for any unrecognised source type.
_DEFAULT_CONFIDENCE: float = 0.70


def _source_confidence(source_type: str) -> float:
    """Return the confidence weight for a given source type string."""
    return SOURCE_CONFIDENCE.get(source_type, _DEFAULT_CONFIDENCE)


# ---------------------------------------------------------------------------
# Normalisation Helpers (contact-key normalisation for matching)
# ---------------------------------------------------------------------------

def _normalise_email_key(raw: str) -> str:
    """
    Produce a canonical email key for matching purposes.
    Lowercases and strips whitespace; does not validate RFC-5322 compliance.

    Args:
        raw: Raw email string from any source.

    Returns:
        Lowercase stripped string, or empty string if input is blank.
    """
    return raw.strip().lower() if raw else ""


def _normalise_phone_key(raw: str) -> Optional[str]:
    """
    Attempt E.164 normalisation of a phone string for matching purposes.
    Returns None (and logs a warning) if normalisation fails, so a bad phone
    never blocks the rest of the merge pipeline.

    Args:
        raw: Raw phone string from any source.

    Returns:
        E.164 string (e.g., "+15550192834") or None on failure.
    """
    if not raw or not raw.strip():
        return None
    try:
        return normalize_phone(raw)
    except ValueError as exc:
        print(
            f"[merger] Phone normalisation skipped for matching: {raw!r} — {exc}",
            file=sys.stderr,
        )
        return None


# ---------------------------------------------------------------------------
# Candidate ID Generation
# ---------------------------------------------------------------------------

def _generate_candidate_id(normalised_emails: list[str], normalised_phones: list[str]) -> str:
    """
    Produce a deterministic, stable candidate ID by hashing the sorted union
    of all normalised contact keys for this identity cluster.

    Using a hash (rather than a random UUID) means the same set of contact
    keys always yields the same ID, making re-runs idempotent.

    Args:
        normalised_emails: Deduplicated normalised email strings.
        normalised_phones: Deduplicated normalised E.164 phone strings.

    Returns:
        "cand-<first-12-hex-chars-of-sha256>" string.
    """
    all_keys = sorted(set(normalised_emails) | set(normalised_phones))
    if not all_keys:
        # No contact keys at all — fall back to a random UUID so we still
        # produce a unique identifier rather than crashing.
        return f"cand-{uuid.uuid4().hex[:12]}"
    digest = hashlib.sha256("|".join(all_keys).encode("utf-8")).hexdigest()
    return f"cand-{digest[:12]}"


# ---------------------------------------------------------------------------
# Record Linkage — Compound Key Matching
# ---------------------------------------------------------------------------

def _extract_match_keys(raw_profile: dict) -> tuple[set[str], set[str]]:
    """
    Extract and normalise the full set of email and phone keys from a raw
    profile dict, discarding any values that fail normalisation.

    Args:
        raw_profile: Raw profile dict produced by ingestion.py.

    Returns:
        Tuple of (email_key_set, phone_key_set).
    """
    email_keys: set[str] = set()
    for raw_email in raw_profile.get("emails", []):
        key = _normalise_email_key(raw_email)
        if key:
            email_keys.add(key)

    phone_keys: set[str] = set()
    for raw_phone in raw_profile.get("phones", []):
        key = _normalise_phone_key(raw_phone)
        if key:
            phone_keys.add(key)

    return email_keys, phone_keys


def _find_matching_cluster(
    email_keys: set[str],
    phone_keys: set[str],
    clusters: list[dict],
) -> Optional[int]:
    """
    Search existing identity clusters for one that shares at least one email
    or phone key with the supplied sets.

    Matching is intentionally strict (exact key intersection only) — no fuzzy
    name or address matching is performed, per the blueprint's scope boundaries.

    Args:
        email_keys: Normalised email keys from the incoming profile.
        phone_keys: Normalised phone keys from the incoming profile.
        clusters:   List of cluster dicts, each carrying accumulated key sets.

    Returns:
        Index of the first matching cluster, or None if no match found.
    """
    for idx, cluster in enumerate(clusters):
        cluster_emails: set[str] = cluster["email_keys"]
        cluster_phones: set[str] = cluster["phone_keys"]

        emails_intersect = bool(email_keys & cluster_emails)
        phones_intersect = bool(phone_keys & cluster_phones)

        if emails_intersect or phones_intersect:
            return idx

    return None


def _link_profiles(raw_profiles: list[dict]) -> list[list[dict]]:
    """
    Partition raw profiles into identity clusters.  Two profiles land in the
    same cluster if they share at least one normalised email or phone key.

    This implements a single-pass union-find using a list of cluster dicts.
    Transitive linkage (A matches B, B matches C → all three in one cluster)
    is handled naturally because each new profile searches the already-merged
    key sets of existing clusters.

    Args:
        raw_profiles: All raw profile dicts from all sources.

    Returns:
        List of clusters, where each cluster is a list of raw profile dicts
        that represent the same candidate identity.
    """
    # Each cluster dict holds:
    #   "profiles"   : list[dict]   — raw profiles grouped here
    #   "email_keys" : set[str]     — accumulated normalised emails
    #   "phone_keys" : set[str]     — accumulated normalised phones
    clusters: list[dict] = []

    for profile in raw_profiles:
        email_keys, phone_keys = _extract_match_keys(profile)

        match_idx = _find_matching_cluster(email_keys, phone_keys, clusters)

        if match_idx is not None:
            # Merge this profile into the existing cluster and expand its key sets.
            clusters[match_idx]["profiles"].append(profile)
            clusters[match_idx]["email_keys"].update(email_keys)
            clusters[match_idx]["phone_keys"].update(phone_keys)
            print(
                f"[merger] Profile from '{profile.get('_source_type')}' "
                f"linked to existing cluster (idx={match_idx}).",
                file=sys.stderr,
            )
        else:
            # No match — start a new cluster for this profile.
            clusters.append({
                "profiles": [profile],
                "email_keys": set(email_keys),
                "phone_keys": set(phone_keys),
            })
            print(
                f"[merger] Profile from '{profile.get('_source_type')}' "
                f"started new cluster (idx={len(clusters) - 1}).",
                file=sys.stderr,
            )

    return [c["profiles"] for c in clusters]


# ---------------------------------------------------------------------------
# Field-Level Conflict Resolution
# ---------------------------------------------------------------------------

def _resolve_scalar(
    current_value: Optional[str],
    current_confidence: float,
    incoming_value: Optional[str],
    incoming_confidence: float,
    field_name: str,
    source_id: str,
    source_type: str,
    provenance: list[ProvenanceItem],
) -> tuple[Optional[str], float]:
    """
    Resolve a single scalar string field between current and incoming values.

    Resolution rules (applied in order):
        1. If incoming is absent/blank   → keep current; no provenance entry.
        2. If current is absent/blank    → populate from incoming; log "initial_write".
        3. If values are identical       → no change; no provenance entry needed.
        4. If incoming confidence is higher → override; log "confidence_override".
        5. Otherwise (current wins)      → keep current; log "confidence_held".

    Args:
        current_value:      The field value already on the in-progress profile.
        current_confidence: Confidence weight of the source that set current_value.
        incoming_value:     The field value from the new source being merged in.
        incoming_confidence:Confidence weight of the new source.
        field_name:         Dot-notation field label for provenance (e.g., "full_name").
        source_id:          File path or identifier of the incoming source.
        source_type:        "ats" or "recruiter_notes" of the incoming source.
        provenance:         Mutable list to which audit entries are appended.

    Returns:
        Tuple of (resolved_value, resolved_confidence).
    """
    # Treat blank strings the same as None.
    incoming_clean = incoming_value.strip() if incoming_value else None
    current_clean = current_value.strip() if current_value else None

    # Rule 1: Nothing incoming — nothing to do.
    if not incoming_clean:
        return current_clean, current_confidence

    # Rule 2: Field is currently empty — initial population.
    if not current_clean:
        provenance.append(ProvenanceItem(
            field=field_name,
            source=source_type,
            method="initial_write",
            value=incoming_clean,
        ))
        return incoming_clean, incoming_confidence

    # Rule 3: Values are identical — silent no-op.
    if current_clean.lower() == incoming_clean.lower():
        return current_clean, current_confidence

    # Rule 4: Incoming confidence is strictly higher — override.
    if incoming_confidence > current_confidence:
        provenance.append(ProvenanceItem(
            field=field_name,
            source=source_type,
            method="confidence_override",
            value=incoming_clean,
        ))
        print(
            f"[merger] Field '{field_name}' overridden by '{source_type}' "
            f"({incoming_confidence:.2f} > {current_confidence:.2f}): "
            f"'{current_clean}' → '{incoming_clean}'",
            file=sys.stderr,
        )
        return incoming_clean, incoming_confidence

    # Rule 5: Current confidence is equal or higher — hold current.
    provenance.append(ProvenanceItem(
        field=field_name,
        source=source_type,
        method="confidence_held",
        value=current_clean,
    ))
    print(
        f"[merger] Field '{field_name}' held from existing source "
        f"({current_confidence:.2f} >= {incoming_confidence:.2f}): "
        f"keeping '{current_clean}', discarding '{incoming_clean}'",
        file=sys.stderr,
    )
    return current_clean, current_confidence


# ---------------------------------------------------------------------------
# Array Field Merging
# ---------------------------------------------------------------------------

def _merge_contact_list(
    current: list[str],
    incoming: list[str],
    normalise_fn,
    field_name: str,
    source_type: str,
    provenance: list[ProvenanceItem],
) -> list[str]:
    """
    Merge two lists of contact identifiers (emails or phones), deduplicating
    by their normalised forms. Preserves insertion order (first-seen wins).

    Args:
        current:      Already-accumulated list on the in-progress profile.
        incoming:     New values from the source being merged in.
        normalise_fn: Callable that produces a canonical key for deduplication.
        field_name:   Provenance field label (e.g., "emails", "phones").
        source_type:  Source identifier for provenance logging.
        provenance:   Mutable provenance list.

    Returns:
        Deduplicated merged list.
    """
    seen_keys: set[str] = set()
    result: list[str] = []

    for value in current:
        key = normalise_fn(value)
        if key and key not in seen_keys:
            seen_keys.add(key)
            result.append(value)

    for value in incoming:
        key = normalise_fn(value)
        if not key:
            continue
        if key not in seen_keys:
            seen_keys.add(key)
            result.append(value)
            provenance.append(ProvenanceItem(
                field=field_name,
                source=source_type,
                method="array_union",
                value=value,
            ))

    return result


def _merge_skills(
    current: list[SkillItem],
    incoming_skill_names: list[str],
    source_type: str,
    source_confidence: float,
    provenance: list[ProvenanceItem],
) -> list[SkillItem]:
    """
    Merge an incoming list of raw skill name strings into the current SkillItem list.

    Deduplication key is the lowercased, stripped skill name.  When a skill
    already exists, the incoming source is added to its `sources` list and its
    confidence is elevated to the max of the two source weights.  New skills
    are appended with the incoming source's confidence weight.

    Args:
        current:            Existing SkillItem list on the in-progress profile.
        incoming_skill_names: Raw skill name strings from the new source.
        source_type:        Source identifier string.
        source_confidence:  Confidence weight of the new source.
        provenance:         Mutable provenance list.

    Returns:
        Updated SkillItem list.
    """
    # Build a lookup from normalised name → index in current list.
    index: dict[str, int] = {
        item.name.strip().lower(): idx for idx, item in enumerate(current)
    }
    result = list(current)

    for raw_name in incoming_skill_names:
        if not raw_name or not raw_name.strip():
            continue
        normalised_name = raw_name.strip()
        lookup_key = normalised_name.lower()

        if lookup_key in index:
            # Skill already recorded — update sources and confidence.
            existing = result[index[lookup_key]]
            if source_type not in existing.sources:
                existing.sources.append(source_type)
                existing.confidence = max(existing.confidence, source_confidence)
                provenance.append(ProvenanceItem(
                    field=f"skills[{existing.name}]",
                    source=source_type,
                    method="skill_source_added",
                    value=existing.name,
                ))
        else:
            # New skill — append and register in the lookup index.
            new_skill = SkillItem(
                name=normalised_name,
                confidence=source_confidence,
                sources=[source_type],
            )
            index[lookup_key] = len(result)
            result.append(new_skill)
            provenance.append(ProvenanceItem(
                field=f"skills[{normalised_name}]",
                source=source_type,
                method="initial_write",
                value=normalised_name,
            ))

    return result


def _normalise_experience_key(company: str, title: str) -> str:
    """
    Produce a deduplication key for an experience entry.
    Lowercases and strips both company and title and joins them.
    """
    return f"{company.strip().lower()}||{title.strip().lower()}"


def _build_experience_items(
    experience_raw: list[dict],
    source_type: str,
    source_confidence: float,
    provenance: list[ProvenanceItem],
) -> list[ExperienceItem]:
    """
    Convert raw experience dicts from one source into validated ExperienceItem
    objects, applying date normalisation and logging provenance entries.

    Entries that fail date normalisation are included with their raw strings
    rather than dropped — a warning is emitted to stderr so nothing is silently lost.

    Args:
        experience_raw:   List of raw experience dicts from the source.
        source_type:      Source identifier string.
        source_confidence:Confidence weight (for logging only at this stage).
        provenance:       Mutable provenance list.

    Returns:
        List of ExperienceItem objects.
    """
    items: list[ExperienceItem] = []

    for raw in experience_raw:
        company = (raw.get("company") or "").strip()
        title   = (raw.get("title")   or "").strip()

        if not company or not title:
            print(
                f"[merger] Skipping experience entry missing company/title: {raw}",
                file=sys.stderr,
            )
            continue

        # Normalise start date.
        raw_start = (raw.get("start") or "").strip()
        if raw_start:
            try:
                start = normalize_date(raw_start)
            except ValueError as exc:
                print(
                    f"[merger] Start date normalisation failed for '{company}': "
                    f"{raw_start!r} — {exc}. Using raw value.",
                    file=sys.stderr,
                )
                start = raw_start
        else:
            start = ""

        # Normalise end date.
        raw_end = (raw.get("end") or "").strip()
        end: Optional[str] = None
        if raw_end:
            try:
                end = normalize_date(raw_end)
            except ValueError as exc:
                print(
                    f"[merger] End date normalisation failed for '{company}': "
                    f"{raw_end!r} — {exc}. Using raw value.",
                    file=sys.stderr,
                )
                end = raw_end

        item = ExperienceItem(
            company=company,
            title=title,
            start=start,
            end=end,
            summary=(raw.get("summary") or "").strip() or None,
        )
        items.append(item)

        provenance.append(ProvenanceItem(
            field=f"experience[{company}|{title}]",
            source=source_type,
            method="initial_write",
            value=f"{company} / {title} ({start} → {end or 'Present'})",
        ))

    return items


def _merge_experience(
    current: list[ExperienceItem],
    incoming: list[ExperienceItem],
    source_type: str,
    provenance: list[ProvenanceItem],
) -> list[ExperienceItem]:
    """
    Merge an incoming list of ExperienceItems into the current list,
    deduplicating by (company, title) compound key.

    When a duplicate entry is found, the existing record is kept (ATS data
    is higher confidence and parsed first) and a provenance entry is written
    to show the second source also confirmed the role.

    Args:
        current:     Existing ExperienceItem list.
        incoming:    New ExperienceItem list from the source being merged.
        source_type: Source identifier string (for provenance).
        provenance:  Mutable provenance list.

    Returns:
        Merged, deduplicated ExperienceItem list.
    """
    seen_keys: set[str] = {
        _normalise_experience_key(e.company, e.title) for e in current
    }
    result = list(current)

    for item in incoming:
        key = _normalise_experience_key(item.company, item.title)
        if key not in seen_keys:
            seen_keys.add(key)
            result.append(item)
        else:
            # Already present — write a confirmation provenance entry only.
            provenance.append(ProvenanceItem(
                field=f"experience[{item.company}|{item.title}]",
                source=source_type,
                method="duplicate_confirmed",
                value=f"{item.company} / {item.title}",
            ))

    return result


def _build_education_items(
    education_raw: list[dict],
    source_type: str,
    provenance: list[ProvenanceItem],
) -> list[EducationItem]:
    """
    Convert raw education dicts from one source into validated EducationItem
    objects with normalised dates.

    Args:
        education_raw: List of raw education dicts from the source.
        source_type:   Source identifier string.
        provenance:    Mutable provenance list.

    Returns:
        List of EducationItem objects.
    """
    items: list[EducationItem] = []

    for raw in education_raw:
        institution = (raw.get("institution") or "").strip()
        if not institution:
            print(
                f"[merger] Skipping education entry missing institution: {raw}",
                file=sys.stderr,
            )
            continue

        def _safe_normalise_date(raw_val: Optional[str]) -> Optional[str]:
            if not raw_val or not raw_val.strip():
                return None
            try:
                return normalize_date(raw_val.strip())
            except ValueError as exc:
                print(
                    f"[merger] Education date normalisation failed: "
                    f"{raw_val!r} — {exc}. Using raw value.",
                    file=sys.stderr,
                )
                return raw_val.strip()

        item = EducationItem(
            institution=institution,
            degree=(raw.get("degree") or "").strip() or None,
            field_of_study=(raw.get("field_of_study") or "").strip() or None,
            start=_safe_normalise_date(raw.get("start")),
            end=_safe_normalise_date(raw.get("end")),
        )
        items.append(item)

        provenance.append(ProvenanceItem(
            field=f"education[{institution}]",
            source=source_type,
            method="initial_write",
            value=institution,
        ))

    return items


def _merge_education(
    current: list[EducationItem],
    incoming: list[EducationItem],
    source_type: str,
    provenance: list[ProvenanceItem],
) -> list[EducationItem]:
    """
    Merge incoming EducationItems into the current list, deduplicating by
    normalised institution name.

    Args:
        current:     Existing EducationItem list.
        incoming:    New EducationItem list from the source being merged.
        source_type: Source identifier string (for provenance).
        provenance:  Mutable provenance list.

    Returns:
        Merged, deduplicated EducationItem list.
    """
    seen: set[str] = {e.institution.strip().lower() for e in current}
    result = list(current)

    for item in incoming:
        key = item.institution.strip().lower()
        if key not in seen:
            seen.add(key)
            result.append(item)
        else:
            provenance.append(ProvenanceItem(
                field=f"education[{item.institution}]",
                source=source_type,
                method="duplicate_confirmed",
                value=item.institution,
            ))

    return result


def _build_location(
    raw_location: dict,
    source_type: str,
    provenance: list[ProvenanceItem],
) -> Optional[Location]:
    """
    Construct a Location object from a raw location dict, applying country
    normalisation. Returns None if all location fields are absent.

    Args:
        raw_location: Dict with optional "city", "region", "country" keys.
        source_type:  Source identifier for provenance logging.
        provenance:   Mutable provenance list.

    Returns:
        Location object or None.
    """
    city    = (raw_location.get("city")    or "").strip() or None
    region  = (raw_location.get("region")  or "").strip() or None
    country_raw = (raw_location.get("country") or "").strip() or None

    country: Optional[str] = None
    if country_raw:
        try:
            country = normalize_country(country_raw)
        except ValueError as exc:
            print(
                f"[merger] Country normalisation failed: {country_raw!r} — {exc}. "
                "Storing raw value.",
                file=sys.stderr,
            )
            country = country_raw

    if not any([city, region, country]):
        return None

    location = Location(city=city, region=region, country=country)

    if city:
        provenance.append(ProvenanceItem(
            field="location.city", source=source_type,
            method="initial_write", value=city,
        ))
    if region:
        provenance.append(ProvenanceItem(
            field="location.region", source=source_type,
            method="initial_write", value=region,
        ))
    if country:
        provenance.append(ProvenanceItem(
            field="location.country", source=source_type,
            method="initial_write", value=country,
        ))

    return location


def _build_links(
    raw_links: dict,
    source_type: str,
    provenance: list[ProvenanceItem],
) -> Optional[Links]:
    """
    Construct a Links object from a raw links dict.
    Returns None if all link fields are absent.

    Args:
        raw_links:   Dict with optional "linkedin", "github", "portfolio", "other" keys.
        source_type: Source identifier for provenance logging.
        provenance:  Mutable provenance list.

    Returns:
        Links object or None.
    """
    linkedin  = (raw_links.get("linkedin")  or "").strip() or None
    github    = (raw_links.get("github")    or "").strip() or None
    portfolio = (raw_links.get("portfolio") or "").strip() or None
    other     = [s.strip() for s in (raw_links.get("other") or []) if s.strip()]

    if not any([linkedin, github, portfolio, other]):
        return None

    for field_name, value in [
        ("links.linkedin", linkedin),
        ("links.github", github),
        ("links.portfolio", portfolio),
    ]:
        if value:
            provenance.append(ProvenanceItem(
                field=field_name, source=source_type,
                method="initial_write", value=value,
            ))

    return Links(linkedin=linkedin, github=github, portfolio=portfolio, other=other)


# ---------------------------------------------------------------------------
# Single-Cluster Merge
# ---------------------------------------------------------------------------

def _merge_cluster(profiles: list[dict]) -> CanonicalProfile:
    """
    Merge a list of raw profile dicts that have been determined to represent
    the same candidate identity into a single CanonicalProfile.

    Processing order:
        1. Sort profiles by descending source confidence so the highest-quality
           source sets initial field values.
        2. Iterate sources: resolve scalars, merge arrays, accumulate provenance.
        3. Compute overall_confidence as the highest source confidence used.
        4. Assign a deterministic candidate_id from the normalised contact keys.

    Args:
        profiles: Non-empty list of raw profile dicts for one identity cluster.

    Returns:
        Fully populated CanonicalProfile.
    """
    # Sort highest-confidence source first so initial_write always comes from
    # the most reliable source, minimising confidence_override events.
    sorted_profiles = sorted(
        profiles,
        key=lambda p: _source_confidence(p.get("_source_type", "")),
        reverse=True,
    )

    # Mutable accumulator state.
    provenance: list[ProvenanceItem] = []
    raw_sources: list[str] = []

    # Scalar field accumulators: (value, confidence_of_setter).
    full_name_state:  tuple[Optional[str], float] = (None, 0.0)

    # Array accumulators.
    emails:     list[str]          = []
    phones:     list[str]          = []
    skills:     list[SkillItem]    = []
    experience: list[ExperienceItem] = []
    education:  list[EducationItem]  = []

    # Structured object accumulators.
    location: Optional[Location] = None
    links:    Optional[Links]    = None

    # Track the highest confidence source encountered for overall_confidence.
    max_confidence: float = 0.0

    for profile in sorted_profiles:
        source_type   = profile.get("_source_type", "unknown")
        source_id     = profile.get("_source_id",   "unknown")
        confidence    = _source_confidence(source_type)

        max_confidence = max(max_confidence, confidence)

        if source_id not in raw_sources:
            raw_sources.append(source_id)

        print(
            f"[merger] Processing source: '{source_type}' "
            f"(confidence={confidence:.2f}, id={source_id})",
            file=sys.stderr,
        )

        # --- Scalar: full_name ---
        full_name_state = _resolve_scalar(
            current_value=full_name_state[0],
            current_confidence=full_name_state[1],
            incoming_value=profile.get("full_name"),
            incoming_confidence=confidence,
            field_name="full_name",
            source_id=source_id,
            source_type=source_type,
            provenance=provenance,
        )

        # --- Array: emails ---
        emails = _merge_contact_list(
            current=emails,
            incoming=profile.get("emails", []),
            normalise_fn=_normalise_email_key,
            field_name="emails",
            source_type=source_type,
            provenance=provenance,
        )

        # --- Array: phones ---
        phones = _merge_contact_list(
            current=phones,
            incoming=profile.get("phones", []),
            normalise_fn=lambda p: _normalise_phone_key(p) or "",
            field_name="phones",
            source_type=source_type,
            provenance=provenance,
        )

        # --- Array: skills ---
        skills = _merge_skills(
            current=skills,
            incoming_skill_names=profile.get("skills", []),
            source_type=source_type,
            source_confidence=confidence,
            provenance=provenance,
        )

        # --- Array: experience ---
        incoming_exp = _build_experience_items(
            experience_raw=profile.get("experience", []),
            source_type=source_type,
            source_confidence=confidence,
            provenance=provenance,
        )
        experience = _merge_experience(
            current=experience,
            incoming=incoming_exp,
            source_type=source_type,
            provenance=provenance,
        )

        # --- Array: education ---
        incoming_edu = _build_education_items(
            education_raw=profile.get("education", []),
            source_type=source_type,
            provenance=provenance,
        )
        education = _merge_education(
            current=education,
            incoming=incoming_edu,
            source_type=source_type,
            provenance=provenance,
        )

        # --- Structured: location ---
        # Only attempt to build/merge location if the source provides one.
        raw_location = profile.get("location") or {}
        if isinstance(raw_location, dict) and any(raw_location.values()):
            if location is None:
                location = _build_location(raw_location, source_type, provenance)
            else:
                # Merge individual sub-fields using scalar resolution.
                new_loc = _build_location(raw_location, source_type, [])

                if new_loc:
                    city_resolved, _ = _resolve_scalar(
                        location.city, confidence,
                        new_loc.city, confidence,
                        "location.city", source_id, source_type, provenance,
                    )
                    region_resolved, _ = _resolve_scalar(
                        location.region, confidence,
                        new_loc.region, confidence,
                        "location.region", source_id, source_type, provenance,
                    )
                    country_resolved, _ = _resolve_scalar(
                        location.country, confidence,
                        new_loc.country, confidence,
                        "location.country", source_id, source_type, provenance,
                    )
                    location = Location(
                        city=city_resolved,
                        region=region_resolved,
                        country=country_resolved,
                    )

        # --- Structured: links ---
        raw_links = profile.get("links") or {}
        if isinstance(raw_links, dict) and any(raw_links.values()):
            if links is None:
                links = _build_links(raw_links, source_type, provenance)
            else:
                # Resolve each link field individually.
                new_lnks = _build_links(raw_links, source_type, [])
                if new_lnks:
                    li_val, _ = _resolve_scalar(
                        links.linkedin, confidence,
                        new_lnks.linkedin, confidence,
                        "links.linkedin", source_id, source_type, provenance,
                    )
                    gh_val, _ = _resolve_scalar(
                        links.github, confidence,
                        new_lnks.github, confidence,
                        "links.github", source_id, source_type, provenance,
                    )
                    pf_val, _ = _resolve_scalar(
                        links.portfolio, confidence,
                        new_lnks.portfolio, confidence,
                        "links.portfolio", source_id, source_type, provenance,
                    )
                    # Merge "other" URLs as a deduplicated union.
                    other_merged = list(
                        dict.fromkeys(links.other + (new_lnks.other or []))
                    )
                    links = Links(
                        linkedin=li_val,
                        github=gh_val,
                        portfolio=pf_val,
                        other=other_merged,
                    )

    # --- Finalise normalised contact arrays ---
    # Apply E.164 normalisation to the final phone list for storage.
    final_phones: list[str] = []
    seen_phone_keys: set[str] = set()
    for raw_phone in phones:
        key = _normalise_phone_key(raw_phone)
        if key and key not in seen_phone_keys:
            seen_phone_keys.add(key)
            final_phones.append(key)   # Store E.164 form.

    final_emails: list[str] = []
    seen_email_keys: set[str] = set()
    for raw_email in emails:
        key = _normalise_email_key(raw_email)
        if key and key not in seen_email_keys:
            seen_email_keys.add(key)
            final_emails.append(key)   # Store lowercased form.

    candidate_id = _generate_candidate_id(final_emails, list(seen_phone_keys))

    profile_out = CanonicalProfile(
        candidate_id=candidate_id,
        full_name=full_name_state[0],
        emails=final_emails,
        phones=final_phones,
        location=location,
        links=links,
        skills=skills,
        experience=experience,
        education=education,
        overall_confidence=round(max_confidence, 4),
        provenance=provenance,
        raw_sources=raw_sources,
    )

    print(
        f"[merger] Cluster resolved → candidate_id='{candidate_id}', "
        f"overall_confidence={profile_out.overall_confidence}, "
        f"provenance_entries={len(provenance)}",
        file=sys.stderr,
    )

    return profile_out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_merge(raw_profiles: list[dict]) -> list[CanonicalProfile]:
    """
    Full merge pipeline entry point.

    Accepts any number of raw profile dicts (from any mix of sources) and
    returns one CanonicalProfile per resolved unique candidate identity.

    Args:
        raw_profiles: List of raw profile dicts produced by ingestion.py parsers.
                      May be empty; returns an empty list in that case.

    Returns:
        List of CanonicalProfile objects, one per unique resolved identity.
    """
    if not raw_profiles:
        print("[merger] No profiles supplied — returning empty result.", file=sys.stderr)
        return []

    print(
        f"[merger] Starting merge for {len(raw_profiles)} raw profile(s).",
        file=sys.stderr,
    )

    # Step 1: Link profiles into identity clusters.
    clusters = _link_profiles(raw_profiles)
    print(
        f"[merger] Record linkage complete: "
        f"{len(raw_profiles)} profiles → {len(clusters)} identity cluster(s).",
        file=sys.stderr,
    )

    # Step 2: Merge each cluster into a CanonicalProfile.
    results: list[CanonicalProfile] = []
    for idx, cluster_profiles in enumerate(clusters):
        print(
            f"[merger] Merging cluster {idx + 1}/{len(clusters)} "
            f"({len(cluster_profiles)} source(s))...",
            file=sys.stderr,
        )
        canonical = _merge_cluster(cluster_profiles)
        results.append(canonical)

    print(
        f"[merger] Merge complete. {len(results)} canonical profile(s) produced.",
        file=sys.stderr,
    )

    return results


# ---------------------------------------------------------------------------
# Smoke Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("merger.py — standalone smoke test")
    print("=" * 60)

    # Two raw profiles representing the same candidate (Jane Doe) as produced
    # by ingestion.py — one from the ATS, one from recruiter notes.
    mock_ats_profile: dict = {
        "_source_type": "ats",
        "_source_id": "ats_input.json",
        "full_name": "Jane Doe",
        "emails": ["jane.doe@email.com", "j.doe@workmail.com"],
        "phones": ["(555) 019-2834"],
        "location": {
            "city": "San Francisco",
            "region": "California",
            "country": "United States",
        },
        "links": {
            "linkedin": "https://linkedin.com/in/janedoe",
            "github": "https://github.com/janedoe",
            "portfolio": None,
            "other": [],
        },
        "skills": ["Python", "python", "React", "react.js", "FastAPI", "PostgreSQL", "Docker"],
        "experience": [
            {
                "company": "Acme Corp",
                "title": "Lead Software Engineer",
                "start": "2022/10/01",
                "end": "Present",
                "summary": "Led cross-functional team, built real-time data pipeline.",
            },
            {
                "company": "Beta Systems Inc.",
                "title": "Software Engineer II",
                "start": "2019/03/15",
                "end": "2022/09/30",
                "summary": "RESTful microservices in Python (FastAPI).",
            },
        ],
        "education": [
            {
                "institution": "University of California, Berkeley",
                "degree": "Bachelor of Science",
                "field_of_study": "Computer Science",
                "start": "2015/08/01",
                "end": "2019/05/15",
            }
        ],
    }

    mock_notes_profile: dict = {
        "_source_type": "recruiter_notes",
        "_source_id": "recruiter_notes.txt",
        "full_name": "Jane Doe",
        "emails": ["jane.doe@email.com"],          # Overlapping key → same cluster.
        "phones": ["+1 (555) 019-2834"],
        "location": None,
        "links": {
            "linkedin": "https://linkedin.com/in/janedoe",
            "github": None,
            "portfolio": None,
            "other": [],
        },
        "skills": ["Python", "React", "FastAPI", "PostgreSQL", "Docker", "Kubernetes"],
        "experience": [
            {
                "company": "Acme Corp",
                "title": "Lead Engineer",           # Slightly different title → new entry.
                "start": "Oct 2022",
                "end": "Present",
                "summary": "Leading data infrastructure and front-end React work.",
            },
            {
                "company": "Beta Systems Inc.",
                "title": "Software Engineer II",    # Exact match → deduped.
                "start": "March 2019",
                "end": "September 2022",
                "summary": "Microservices work.",
            },
        ],
        "education": [
            {
                "institution": "UC Berkeley",       # Different name → new entry.
                "degree": "Bachelor of Science",
                "field_of_study": "Computer Science",
                "start": None,
                "end": "May 2019",
            }
        ],
    }

    raw_profiles = [mock_ats_profile, mock_notes_profile]
    canonical_profiles = run_merge(raw_profiles)

    print("\n" + "=" * 60)
    print(f"RESULTS: {len(canonical_profiles)} canonical profile(s) produced")
    print("=" * 60)

    for cp in canonical_profiles:
        output = cp.model_dump(mode="json", exclude_none=False)
        print(json.dumps(output, indent=2))

    print("\n--- Provenance Summary ---")
    for cp in canonical_profiles:
        print(f"\nCandidate: {cp.candidate_id}")
        for prov in cp.provenance:
            print(
                f"  [{prov.method:<22}]  {prov.field:<45}  "
                f"source={prov.source:<18}  value={prov.value!r}"
            )