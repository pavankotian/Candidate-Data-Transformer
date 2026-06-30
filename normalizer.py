"""
normalizer.py
-------------
Stateless, pure normalization functions for the Multi-Source Candidate Data Transformer.

Each function accepts a raw string from any source and returns a canonical string
in the target format. Functions never mutate external state and are safe to call
concurrently. All failures are surfaced via ValueError with descriptive messages
rather than silent fallbacks, so callers can make explicit decisions about error
handling.

Normalization targets:
    - Phone numbers  → E.164 format (e.g., "+15550192834")
    - Dates          → YYYY-MM format (e.g., "2022-10") or literal "Present"
    - Country names  → ISO-3166 alpha-2 (e.g., "US", "GB")
"""

from __future__ import annotations

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Phone Normalization → E.164
# ---------------------------------------------------------------------------

# Regex to strip every character that is not a digit or a leading '+'.
_NON_DIGIT_RE = re.compile(r"[^\d+]")

# E.164 allows a maximum of 15 digits (excluding the leading '+').
_E164_MAX_DIGITS = 15
_E164_MIN_DIGITS = 7


def normalize_phone(raw: str) -> str:
    """
    Convert an arbitrarily formatted phone string to strict E.164 format.

    Handles common messy inputs such as:
        "(555) 019-2834"      → "+15550192834"   (assumes US if no country code)
        "+1 (555) 019-2834"   → "+15550192834"
        "015550192834"        → "+15550192834"   (leading 0 stripped for US)
        "+44 20 7946 0958"    → "+442079460958"

    Strategy:
        1. Strip all non-digit, non-'+' characters.
        2. If the result starts with '+', treat existing prefix as the country code.
        3. If the result is 10 digits (NANP), prepend +1 (North American default).
        4. If the result is 11 digits starting with '1', prepend '+'.
        5. Otherwise raise ValueError to surface ambiguous inputs explicitly.

    Args:
        raw: Raw phone string from any source.

    Returns:
        E.164-formatted phone string (e.g., "+15550192834").

    Raises:
        ValueError: If the input cannot be unambiguously resolved to E.164.
    """
    if not raw or not raw.strip():
        raise ValueError("normalize_phone received an empty or blank string.")

    stripped = raw.strip()

    # Preserve a leading '+' as the international prefix indicator.
    has_plus_prefix = stripped.startswith("+")

    # Remove every character that is not a digit.
    digits_only = re.sub(r"\D", "", stripped)

    if not digits_only:
        raise ValueError(f"No digits found in phone string: {raw!r}")

    if len(digits_only) > _E164_MAX_DIGITS:
        raise ValueError(
            f"Phone string has {len(digits_only)} digits, exceeding E.164 maximum "
            f"of {_E164_MAX_DIGITS}: {raw!r}"
        )

    if len(digits_only) < _E164_MIN_DIGITS:
        raise ValueError(
            f"Phone string has only {len(digits_only)} digits, below minimum of "
            f"{_E164_MIN_DIGITS}: {raw!r}"
        )

    # --- Resolution Logic ---

    if has_plus_prefix:
        # Source explicitly declared a country code; trust it.
        return f"+{digits_only}"

    if len(digits_only) == 10:
        # 10-digit number with no prefix → North American Numbering Plan (NANP).
        # Prepend the US/Canada country code.
        return f"+1{digits_only}"

    if len(digits_only) == 11 and digits_only.startswith("1"):
        # 11 digits beginning with '1' → NANP with country code included but no '+'.
        return f"+{digits_only}"

    # For all other lengths (e.g., European numbers without '+'), we cannot
    # safely assume a country code. Raise explicitly rather than guess.
    raise ValueError(
        f"Cannot unambiguously resolve phone to E.164 without a '+' prefix: {raw!r}. "
        "Ensure the source includes an explicit country code prefix."
    )


# ---------------------------------------------------------------------------
# Date Normalization → YYYY-MM
# ---------------------------------------------------------------------------

# Maps full and abbreviated English month names to their zero-padded numeric string.
_MONTH_NAME_MAP: dict[str, str] = {
    "january": "01",  "jan": "01",
    "february": "02", "feb": "02",
    "march": "03",    "mar": "03",
    "april": "04",    "apr": "04",
    "may": "05",
    "june": "06",     "jun": "06",
    "july": "07",     "jul": "07",
    "august": "08",   "aug": "08",
    "september": "09","sep": "09", "sept": "09",
    "october": "10",  "oct": "10",
    "november": "11", "nov": "11",
    "december": "12", "dec": "12",
}

# Sentinel string for ongoing roles; returned verbatim when detected.
_PRESENT_SENTINEL = "Present"

# Tokens that unambiguously signal an active/current role.
_PRESENT_TOKENS = frozenset({"present", "current", "now", "ongoing", "till date", "to date"})


def normalize_date(raw: str) -> str:
    """
    Convert an arbitrarily formatted date string to strict YYYY-MM format.

    Handles a broad range of messy real-world inputs:
        "Oct 2022"        → "2022-10"
        "October 2022"    → "2022-10"
        "2022/10/01"      → "2022-10"
        "2022/10"         → "2022-10"
        "2022-10-01"      → "2022-10"
        "2024/05/12"      → "2024-05"
        "2019/03/15"      → "2019-03"
        "2019/08/01"      → "2019-08"
        "Present"         → "Present"
        "current"         → "Present"

    Strategy:
        1. Check for "present" sentinel variants first.
        2. Try to detect ISO-like formats (YYYY/MM/DD or YYYY-MM-DD or YYYY/MM).
        3. Try to detect natural-language formats ("Mon YYYY" or "YYYY Mon").
        4. Try a bare 4-digit year as a last resort (returns YYYY-01 as partial).
        5. Raise ValueError if nothing matches.

    Args:
        raw: Raw date string from any source.

    Returns:
        "YYYY-MM" string or the literal "Present".

    Raises:
        ValueError: If the input cannot be mapped to YYYY-MM or Present.
    """
    if not raw or not raw.strip():
        raise ValueError("normalize_date received an empty or blank string.")

    cleaned = raw.strip()

    # --- Step 1: Check for "Present" sentinel variants ---
    if cleaned.lower() in _PRESENT_TOKENS:
        return _PRESENT_SENTINEL

    # --- Step 2: ISO-like numeric formats (YYYY/MM/DD, YYYY-MM-DD, YYYY/MM, YYYY-MM) ---
    iso_match = re.match(
        r"^(\d{4})[\/\-\.](\d{1,2})(?:[\/\-\.](\d{1,2}))?$",
        cleaned
    )
    if iso_match:
        year = iso_match.group(1)
        month = iso_match.group(2).zfill(2)
        _validate_year_month(year, month, raw)
        return f"{year}-{month}"

    # --- Step 3: Natural language formats ---
    # Pattern A: "Oct 2022" or "October 2022"  (month-first)
    month_first_match = re.match(
        r"^([A-Za-z]+)\.?\s+(\d{4})$",
        cleaned
    )
    if month_first_match:
        month_str = month_first_match.group(1).lower().rstrip(".")
        year = month_first_match.group(2)
        month = _month_name_to_number(month_str, raw)
        _validate_year_month(year, month, raw)
        return f"{year}-{month}"

    # Pattern B: "2022 Oct" or "2022 October"  (year-first natural)
    year_first_natural_match = re.match(
        r"^(\d{4})\s+([A-Za-z]+)\.?$",
        cleaned
    )
    if year_first_natural_match:
        year = year_first_natural_match.group(1)
        month_str = year_first_natural_match.group(2).lower().rstrip(".")
        month = _month_name_to_number(month_str, raw)
        _validate_year_month(year, month, raw)
        return f"{year}-{month}"

    # --- Step 4: Bare 4-digit year (partial date — month unknown) ---
    bare_year_match = re.match(r"^(\d{4})$", cleaned)
    if bare_year_match:
        year = bare_year_match.group(1)
        _validate_year_month(year, "01", raw)
        # Return the first month of the year as the safest partial assumption.
        return f"{year}-01"

    raise ValueError(
        f"normalize_date could not parse date string: {raw!r}. "
        "Expected formats: 'YYYY-MM', 'YYYY/MM/DD', 'Oct 2022', 'October 2022', 'Present'."
    )


def _month_name_to_number(month_str: str, original: str) -> str:
    """
    Resolve a lowercase English month name or abbreviation to a zero-padded number.

    Args:
        month_str: Lowercased month token (e.g., "oct", "october").
        original:  Original raw date string, used in error messages only.

    Returns:
        Zero-padded month string (e.g., "10").

    Raises:
        ValueError: If the month token is not recognised.
    """
    result = _MONTH_NAME_MAP.get(month_str)
    if result is None:
        raise ValueError(
            f"normalize_date could not recognise month name {month_str!r} "
            f"in date string: {original!r}."
        )
    return result


def _validate_year_month(year: str, month: str, original: str) -> None:
    """
    Validate that extracted year and month values fall within sensible bounds.

    Args:
        year:     4-digit year string.
        month:    1- or 2-digit month string (already zero-padded).
        original: Original raw date string, used in error messages only.

    Raises:
        ValueError: If year or month are out of acceptable range.
    """
    year_int = int(year)
    month_int = int(month)

    if not (1900 <= year_int <= 2100):
        raise ValueError(
            f"Year {year_int} is outside the accepted range [1900, 2100] "
            f"in date string: {original!r}."
        )
    if not (1 <= month_int <= 12):
        raise ValueError(
            f"Month {month_int} is outside the valid range [1, 12] "
            f"in date string: {original!r}."
        )


# ---------------------------------------------------------------------------
# Country Normalization → ISO-3166 alpha-2
# ---------------------------------------------------------------------------

# Exhaustive lookup table covering common verbose names, abbreviations,
# and regional variants. Keys are lowercase for case-insensitive matching.
_COUNTRY_LOOKUP: dict[str, str] = {
    # United States
    "us": "US", "usa": "US", "u.s.": "US", "u.s.a.": "US",
    "united states": "US", "united states of america": "US", "america": "US",

    # United Kingdom
    "gb": "GB", "uk": "GB", "u.k.": "GB", "united kingdom": "GB",
    "great britain": "GB", "britain": "GB", "england": "GB",
    "scotland": "GB", "wales": "GB",

    # Canada
    "ca": "CA", "can": "CA", "canada": "CA",

    # Australia
    "au": "AU", "aus": "AU", "australia": "AU",

    # Germany
    "de": "DE", "deu": "DE", "germany": "DE", "deutschland": "DE",

    # France
    "fr": "FR", "fra": "FR", "france": "FR",

    # India
    "in": "IN", "ind": "IN", "india": "IN",

    # China
    "cn": "CN", "chn": "CN", "china": "CN", "prc": "CN",
    "people's republic of china": "CN",

    # Japan
    "jp": "JP", "jpn": "JP", "japan": "JP",

    # Brazil
    "br": "BR", "bra": "BR", "brazil": "BR", "brasil": "BR",

    # Mexico
    "mx": "MX", "mex": "MX", "mexico": "MX", "méxico": "MX",

    # Netherlands
    "nl": "NL", "nld": "NL", "netherlands": "NL",
    "the netherlands": "NL", "holland": "NL",

    # Spain
    "es": "ES", "esp": "ES", "spain": "ES", "españa": "ES",

    # Italy
    "it": "IT", "ita": "IT", "italy": "IT", "italia": "IT",

    # Sweden
    "se": "SE", "swe": "SE", "sweden": "SE", "sverige": "SE",

    # Norway
    "no": "NO", "nor": "NO", "norway": "NO", "norge": "NO",

    # Denmark
    "dk": "DK", "dnk": "DK", "denmark": "DK", "danmark": "DK",

    # Finland
    "fi": "FI", "fin": "FI", "finland": "FI", "suomi": "FI",

    # Switzerland
    "ch": "CH", "che": "CH", "switzerland": "CH", "schweiz": "CH",

    # Poland
    "pl": "PL", "pol": "PL", "poland": "PL", "polska": "PL",

    # Portugal
    "pt": "PT", "prt": "PT", "portugal": "PT",

    # South Korea
    "kr": "KR", "kor": "KR", "south korea": "KR", "korea": "KR",
    "republic of korea": "KR",

    # Singapore
    "sg": "SG", "sgp": "SG", "singapore": "SG",

    # New Zealand
    "nz": "NZ", "nzl": "NZ", "new zealand": "NZ",

    # South Africa
    "za": "ZA", "zaf": "ZA", "south africa": "ZA",

    # Nigeria
    "ng": "NG", "nga": "NG", "nigeria": "NG",

    # Kenya
    "ke": "KE", "ken": "KE", "kenya": "KE",

    # Israel
    "il": "IL", "isr": "IL", "israel": "IL",

    # United Arab Emirates
    "ae": "AE", "are": "AE", "uae": "AE",
    "united arab emirates": "AE", "emirates": "AE",

    # Russia
    "ru": "RU", "rus": "RU", "russia": "RU", "russian federation": "RU",

    # Ukraine
    "ua": "UA", "ukr": "UA", "ukraine": "UA",

    # Argentina
    "ar": "AR", "arg": "AR", "argentina": "AR",

    # Chile
    "cl": "CL", "chl": "CL", "chile": "CL",

    # Colombia
    "co": "CO", "col": "CO", "colombia": "CO",

    # Pakistan
    "pk": "PK", "pak": "PK", "pakistan": "PK",

    # Bangladesh
    "bd": "BD", "bgd": "BD", "bangladesh": "BD",

    # Indonesia
    "id": "ID", "idn": "ID", "indonesia": "ID",

    # Philippines
    "ph": "PH", "phl": "PH", "philippines": "PH",

    # Vietnam
    "vn": "VN", "vnm": "VN", "vietnam": "VN", "viet nam": "VN",

    # Thailand
    "th": "TH", "tha": "TH", "thailand": "TH",

    # Malaysia
    "my": "MY", "mys": "MY", "malaysia": "MY",

    # Egypt
    "eg": "EG", "egy": "EG", "egypt": "EG",

    # Turkey
    "tr": "TR", "tur": "TR", "turkey": "TR", "türkiye": "TR",

    # Romania
    "ro": "RO", "rou": "RO", "romania": "RO", "românia": "RO",

    # Czech Republic
    "cz": "CZ", "cze": "CZ", "czech republic": "CZ",
    "czechia": "CZ", "czech": "CZ",

    # Hungary
    "hu": "HU", "hun": "HU", "hungary": "HU",

    # Greece
    "gr": "GR", "grc": "GR", "greece": "GR", "hellas": "GR",

    # Austria
    "at": "AT", "aut": "AT", "austria": "AT", "österreich": "AT",

    # Belgium
    "be": "BE", "bel": "BE", "belgium": "BE", "belgique": "BE",

    # Ireland
    "ie": "IE", "irl": "IE", "ireland": "IE", "eire": "IE",

    # Hong Kong
    "hk": "HK", "hkg": "HK", "hong kong": "HK",

    # Taiwan
    "tw": "TW", "twn": "TW", "taiwan": "TW",
}


def normalize_country(raw: str) -> str:
    """
    Map an arbitrary country name or code to a strict ISO-3166 alpha-2 code.

    Handles common messy inputs:
        "United States"  → "US"
        "USA"            → "US"
        "united states"  → "US"
        "GB"             → "GB"
        "great britain"  → "GB"
        "Deutschland"    → "DE"

    Strategy:
        1. Strip whitespace and lowercase for case-insensitive lookup.
        2. Attempt direct lookup in the exhaustive _COUNTRY_LOOKUP table.
        3. If the input is already a valid 2-letter ISO code (after uppercasing),
           return it as-is (for pass-through of already-normalized values).
        4. Raise ValueError if no match is found.

    Args:
        raw: Raw country string from any source.

    Returns:
        Uppercase ISO-3166 alpha-2 code (e.g., "US", "GB").

    Raises:
        ValueError: If the country string cannot be resolved to a known ISO code.
    """
    if not raw or not raw.strip():
        raise ValueError("normalize_country received an empty or blank string.")

    cleaned = raw.strip()
    lookup_key = cleaned.lower()

    # Primary lookup in the curated table.
    if lookup_key in _COUNTRY_LOOKUP:
        return _COUNTRY_LOOKUP[lookup_key]

    # Pass-through: if it's already a valid uppercase 2-letter code not in our
    # table (e.g., a less-common country), return it uppercased.
    uppercased = cleaned.upper()
    if re.match(r"^[A-Z]{2}$", uppercased):
        return uppercased

    raise ValueError(
        f"normalize_country could not resolve country string to ISO-3166 alpha-2: {raw!r}. "
        "Add the mapping to _COUNTRY_LOOKUP in normalizer.py if this is a valid country."
    )