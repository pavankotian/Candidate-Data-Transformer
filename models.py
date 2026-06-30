"""
models.py
---------
Pydantic v2 canonical data models for the Multi-Source Candidate Data Transformer.

All models enforce strict typing. Fields that may be absent across sources are
declared Optional with explicit None defaults. Array fields default to empty lists
to avoid mutable default pitfalls and simplify downstream merging logic.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class Location(BaseModel):
    """
    Represents a candidate's geographic location.
    Country must be a valid ISO-3166 alpha-2 code (e.g., "US", "GB", "IN").
    Enforced as uppercase to guard against casing inconsistencies from raw sources.
    """

    city: Optional[str] = None
    region: Optional[str] = None
    country: Optional[str] = None  # ISO-3166 alpha-2, e.g. "US"

    @field_validator("country", mode="before")
    @classmethod
    def uppercase_country(cls, v: Optional[str]) -> Optional[str]:
        """Normalize country codes to uppercase before storage."""
        if v is not None:
            return v.strip().upper()
        return v


class Links(BaseModel):
    """
    Represents a candidate's professional online presence.
    All fields are optional since no single source is guaranteed to supply them.
    """

    linkedin: Optional[str] = None
    github: Optional[str] = None
    portfolio: Optional[str] = None
    other: List[str] = Field(default_factory=list)


class SkillItem(BaseModel):
    """
    Represents a single skill entry with a confidence score and source provenance.

    Attributes:
        name        : Canonical skill label (e.g., "Python", "React").
        confidence  : Float in [0.0, 1.0] reflecting extraction reliability.
        sources     : List of source identifiers that mentioned this skill
                      (e.g., ["ats", "recruiter_notes"]).
    """

    name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    sources: List[str] = Field(default_factory=list)

    @field_validator("name", mode="before")
    @classmethod
    def strip_name(cls, v: str) -> str:
        return v.strip()


class ExperienceItem(BaseModel):
    """
    Represents a single professional experience record.

    Dates must be in YYYY-MM format after normalization, or the string "Present"
    for an ongoing role. Raw values are cleaned by the normalization engine before
    population here.

    Attributes:
        company : Employer name.
        title   : Job title held.
        start   : Start date in YYYY-MM format.
        end     : End date in YYYY-MM format, or None if current/unknown.
        summary : Optional free-text description of responsibilities.
    """

    company: str
    title: str
    start: str  # YYYY-MM post-normalization
    end: Optional[str] = None  # YYYY-MM or None for "Present" / unknown
    summary: Optional[str] = None

    @field_validator("company", "title", mode="before")
    @classmethod
    def strip_string_fields(cls, v: str) -> str:
        return v.strip()


class EducationItem(BaseModel):
    """
    Represents a single education record for a candidate.

    Attributes:
        institution : Name of the university, college, or institution.
        degree      : Degree type (e.g., "Bachelor of Science").
        field_of_study: Major or area of specialization.
        start       : Start date in YYYY-MM format.
        end         : End date in YYYY-MM format, or None if ongoing/unknown.
    """

    institution: str
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    start: Optional[str] = None  # YYYY-MM post-normalization
    end: Optional[str] = None    # YYYY-MM post-normalization

    @field_validator("institution", mode="before")
    @classmethod
    def strip_institution(cls, v: str) -> str:
        return v.strip()


class ProvenanceItem(BaseModel):
    """
    An immutable audit record for a single field resolution decision.

    Every time a field value is written or overwritten during the merge phase,
    a ProvenanceItem is appended to the CanonicalProfile.provenance list.
    This ensures complete traceability for downstream auditing or dispute resolution.

    Attributes:
        field   : Dot-notation field path on the CanonicalProfile (e.g., "emails[0]").
        source  : Identifier of the originating data source (e.g., "ats", "recruiter_notes").
        method  : Resolution method applied (e.g., "direct_json", "llm_extraction",
                  "confidence_override", "initial_write").
        value   : String representation of the value that was written.
    """

    field: str
    source: str
    method: str
    value: str


class CanonicalProfile(BaseModel):
    """
    The central, unified representation of a single candidate identity.

    This model is the output of the merge resolution phase and the input to
    the projection engine. It accumulates normalized data from all matched
    source records and maintains a full provenance audit trail.

    Attributes:
        candidate_id      : Deterministic unique identifier assigned post-merge.
        full_name         : Resolved display name.
        emails            : Deduplicated list of normalized email addresses.
        phones            : Deduplicated list of E.164-formatted phone numbers.
        location          : Resolved Location object.
        links             : Resolved Links object.
        skills            : Merged and deduplicated list of SkillItems.
        experience        : Chronologically ordered list of ExperienceItems.
        education         : List of EducationItems.
        overall_confidence: Weighted average confidence score across all resolved fields.
                            Calculated during the merge phase.
        provenance        : Append-only audit log of all field resolution decisions.
        raw_sources       : List of source file identifiers that contributed to
                            this profile (e.g., ["ats_input.json", "recruiter_notes.txt"]).
    """

    candidate_id: Optional[str] = None
    full_name: Optional[str] = None
    emails: List[str] = Field(default_factory=list)
    phones: List[str] = Field(default_factory=list)
    location: Optional[Location] = None
    links: Optional[Links] = None
    skills: List[SkillItem] = Field(default_factory=list)
    experience: List[ExperienceItem] = Field(default_factory=list)
    education: List[EducationItem] = Field(default_factory=list)
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    provenance: List[ProvenanceItem] = Field(default_factory=list)
    raw_sources: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def deduplicate_contact_arrays(self) -> "CanonicalProfile":
        """
        Ensure emails and phones lists contain no duplicates post-construction.
        Preserves insertion order (first occurrence wins) using dict.fromkeys trick.
        """
        self.emails = list(dict.fromkeys(self.emails))
        self.phones = list(dict.fromkeys(self.phones))
        return self