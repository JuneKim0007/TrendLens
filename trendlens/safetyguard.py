# safetyguard.py
# validates user-facing strings before they touch SQL or yfinance (performance concerned.)
# rejects injection patterns, enforces format constraints

import re
from datetime import datetime
#ticker validation used before making a db call.
_TICKER_PATTERN = re.compile(r'^[A-Z0-9.\-^]{1,10}$')
_COMPANY_KEY_PATTERN = re.compile(r'^[A-Za-z0-9_\-]{1,50}$')
#sanitizer but this won't be an issue since this api would be most likely tested on a local db.
_SQL_BLACKLIST = re.compile(
    r'\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|EXEC|UNION|SELECT|GRANT|REVOKE'
    r'|TRUNCATE|REPLACE|MERGE|ATTACH|DETACH|PRAGMA|LOAD_EXTENSION)\b',
    re.IGNORECASE)
_INJECTION_CHARS = re.compile(r"[;'\"\/\*]|--")


def _has_injection(value: str) -> bool:
    return bool(_SQL_BLACKLIST.search(value)) or bool(_INJECTION_CHARS.search(value))


def validate_ticker(ticker: str) -> str | None:
    if not isinstance(ticker, str):
        print(f"VALIDATION: ticker must be a string, got {type(ticker)}")
        return None
    cleaned = ticker.strip().upper()
    if _has_injection(cleaned):
        print(f"VALIDATION: rejected ticker '{ticker}' — suspicious pattern")
        return None
    if not _TICKER_PATTERN.match(cleaned):
        print(f"VALIDATION: rejected ticker '{ticker}' — invalid format")
        return None
    return cleaned


def validate_company_key(key: str) -> str | None:
    if not isinstance(key, str):
        return None
    cleaned = key.strip()
    if _has_injection(cleaned):
        return None
    if not _COMPANY_KEY_PATTERN.match(cleaned):
        return None
    return cleaned


def validate_date(date_str: str) -> str | None:
    if date_str is None:
        return None
    if not isinstance(date_str, str):
        return None
    if _has_injection(date_str):
        return None
    try:
        return datetime.strptime(date_str.strip(), "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        return None


def validate_date_range(start: str, end: str) -> tuple[str, str] | None:
    s, e = validate_date(start), validate_date(end)
    if s is None or e is None:
        return None
    if datetime.strptime(s, "%Y-%m-%d") >= datetime.strptime(e, "%Y-%m-%d"):
        return None
    return (s, e)