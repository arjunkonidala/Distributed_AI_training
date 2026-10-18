"""Onset time normalizer and validator for CareFlow.
Hard constrains symptom onset in data and database to strict 24-hour format:
    YYYY-MM-DD HH:MM
Provides approximate datetime resolution for natural language onset expressions
(e.g., '2 hours ago', 'this morning', 'yesterday 3pm', '3 days ago').
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

try:
    from dateutil import parser as date_parser
    HAVE_DATEUTIL = True
except ImportError:
    HAVE_DATEUTIL = False

# Strict 24-hour regex pattern: YYYY-MM-DD HH:MM (00:00 to 23:59)
STRICT_24H_REGEX = re.compile(r"^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01]) (?:[01]\d|2[0-3]):[0-5]\d$")


def is_strict_24h(val: Optional[str]) -> bool:
    """Checks whether the string strictly conforms to 'YYYY-MM-DD HH:MM' in 24-hour time."""
    if not val or not isinstance(val, str):
        return False
    return bool(STRICT_24H_REGEX.match(val.strip()))


def normalize_onset_24h(raw_val: Any, ref_time: Optional[datetime] = None) -> Optional[str]:
    """Converts any natural language or approximate onset expression into a strict 24-hour
    'YYYY-MM-DD HH:MM' timestamp.
    """
    if raw_val is None:
        return None

    text = str(raw_val).strip()
    if not text or text.lower() in ["none", "null", "unknown", "n/a", "not specified"]:
        return None

    # 1. Already strictly valid
    if is_strict_24h(text):
        return text

    now = ref_time or datetime.now()
    lower = text.lower()

    # 2. Relative minutes or hours ago: e.g. "2 hours ago", "30 mins ago", "1 hr ago"
    m_hour = re.search(r"(\d+)\s*(?:hour|hr|hours|hrs)\s*ago", lower)
    if m_hour:
        hours = int(m_hour.group(1))
        dt = now - timedelta(hours=hours)
        return dt.strftime("%Y-%m-%d %H:%M")

    m_min = re.search(r"(\d+)\s*(?:min|minute|mins|minutes)\s*ago", lower)
    if m_min:
        mins = int(m_min.group(1))
        dt = now - timedelta(minutes=mins)
        return dt.strftime("%Y-%m-%d %H:%M")

    # 3. "Now", "just now", "current"
    if any(k in lower for k in ["now", "just now", "current", "starting today"]):
        return now.strftime("%Y-%m-%d %H:%M")

    # 4. "This morning" (~08:00), "this afternoon" (~14:00), "this evening" (~19:00), "tonight" (~21:00)
    if "this morning" in lower:
        return now.strftime("%Y-%m-%d 08:00")
    if "this afternoon" in lower:
        return now.strftime("%Y-%m-%d 14:00")
    if "this evening" in lower:
        return now.strftime("%Y-%m-%d 19:00")
    if "tonight" in lower or "last night" in lower:
        target_day = now if "tonight" in lower else now - timedelta(days=1)
        return target_day.strftime("%Y-%m-%d 21:00")

    # 5. "Yesterday" with optional time
    if "yesterday" in lower:
        y_day = now - timedelta(days=1)
        # Check if time mentioned e.g. "yesterday 3pm", "yesterday morning"
        if "morning" in lower:
            return y_day.strftime("%Y-%m-%d 08:00")
        elif "evening" in lower or "night" in lower:
            return y_day.strftime("%Y-%m-%d 20:00")
        elif "afternoon" in lower:
            return y_day.strftime("%Y-%m-%d 14:00")
        
        m_time = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", lower)
        if m_time:
            hour = int(m_time.group(1))
            minute = int(m_time.group(2) or 0)
            meridiem = m_time.group(3)
            if meridiem == "pm" and hour < 12:
                hour += 12
            elif meridiem == "am" and hour == 12:
                hour = 0
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                return f"{y_day.strftime('%Y-%m-%d')} {hour:02d}:{minute:02d}"
        return y_day.strftime("%Y-%m-%d 12:00")

    # 6. Relative days or weeks ago: e.g. "3 days ago", "2 weeks ago"
    m_days = re.search(r"(\d+)\s*(?:day|days)\s*ago", lower)
    if m_days:
        days = int(m_days.group(1))
        dt = now - timedelta(days=days)
        return dt.strftime("%Y-%m-%d 12:00")

    m_weeks = re.search(r"(\d+)\s*(?:week|weeks)\s*ago", lower)
    if m_weeks:
        weeks = int(m_weeks.group(1))
        dt = now - timedelta(days=weeks * 7)
        return dt.strftime("%Y-%m-%d 12:00")

    # 7. Preceding events e.g. "after eating noodles", "after lunch"
    if "after lunch" in lower:
        return now.strftime("%Y-%m-%d 13:30")
    if "after breakfast" in lower:
        return now.strftime("%Y-%m-%d 08:30")
    if "after dinner" in lower:
        return now.strftime("%Y-%m-%d 20:30")
    if "after" in lower or "since" in lower:
        # Approximate 1 hour ago
        dt = now - timedelta(hours=1)
        return dt.strftime("%Y-%m-%d %H:%M")

    # 8. Try general date parsing via python-dateutil
    if HAVE_DATEUTIL:
        try:
            parsed = date_parser.parse(text, default=now)
            # Ensure year is reasonable (default to current year if omitted)
            if parsed.year < 2000 or parsed.year > now.year + 1:
                parsed = parsed.replace(year=now.year)
            return parsed.strftime("%Y-%m-%d %H:%M")
        except Exception:
            pass

    # 9. Fallback: default to current date and 24-hour time
    return now.strftime("%Y-%m-%d %H:%M")
