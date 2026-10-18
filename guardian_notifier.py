"""Guardian email alerts for CareFlow triage events.

When the Triage Agent handles an emergency turn, the patient's guardian (stored on the patient profile)
is emailed over SMTP. Sending happens in a background thread so the triage reply is never delayed, and
every outcome (sent / failed / skipped) is written to the `guardian_alerts` audit table.

Configuration (environment / .env):
    SMTP_HOST, SMTP_PORT (default 587), SMTP_USERNAME, SMTP_PASSWORD,
    SMTP_FROM (default SMTP_USERNAME), SMTP_USE_SSL ("true" for port 465, default STARTTLS),
    GUARDIAN_ALERT_COOLDOWN_MINUTES (default 30): at most one alert per conversation in this window.
"""
from __future__ import annotations

import os
import re
import smtplib
import sys
import threading
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from typing import Optional

from dotenv import load_dotenv

import database as db

load_dotenv()

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def is_valid_email(value: Optional[str]) -> bool:
    return bool(value and EMAIL_RE.match(value.strip()))


def smtp_configured() -> bool:
    return bool(os.environ.get("SMTP_HOST") and (os.environ.get("SMTP_FROM") or os.environ.get("SMTP_USERNAME")))


def _cooldown_minutes() -> int:
    try:
        return max(0, int(os.environ.get("GUARDIAN_ALERT_COOLDOWN_MINUTES", "30")))
    except ValueError:
        return 30


def _recently_alerted(username: str, session_id: str) -> bool:
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=_cooldown_minutes())
    for alert in db.get_guardian_alerts(username=username, session_id=session_id, limit=20):
        if alert["status"] in ("sent", "queued") and datetime.fromisoformat(alert["created_at"]) >= cutoff:
            return True
    return False


def build_alert_email(patient_name: str, guardian_name: Optional[str], guardian_email: str, patient_message: str) -> EmailMessage:
    sender = os.environ.get("SMTP_FROM") or os.environ.get("SMTP_USERNAME", "")
    when = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    excerpt = patient_message.strip()
    if len(excerpt) > 500:
        excerpt = excerpt[:500].rstrip() + "..."

    msg = EmailMessage()
    msg["Subject"] = f"CareFlow urgent alert: {patient_name} may need help"
    msg["From"] = sender
    msg["To"] = guardian_email
    msg.set_content(
        f"Dear {guardian_name or 'Guardian'},\n\n"
        f"You are listed as the guardian for {patient_name} in CareFlow. At {when}, CareFlow's triage "
        f"assistant flagged a possible medical emergency in their conversation.\n\n"
        f"What they reported:\n\"{excerpt}\"\n\n"
        f"CareFlow has advised them to call emergency services (112 / 911) or go to the nearest emergency "
        f"department. Please contact {patient_name} as soon as possible. If you cannot reach them and believe "
        f"they are in danger, call your local emergency number.\n\n"
        f"This is an automated message from CareFlow. CareFlow does not provide a diagnosis.\n"
    )
    return msg


def send_email(msg: EmailMessage) -> None:
    host = os.environ["SMTP_HOST"]
    port = int(os.environ.get("SMTP_PORT", "587"))
    username = os.environ.get("SMTP_USERNAME")
    password = os.environ.get("SMTP_PASSWORD")
    if os.environ.get("SMTP_USE_SSL", "").lower() in ("1", "true", "yes"):
        server = smtplib.SMTP_SSL(host, port, timeout=20)
    else:
        server = smtplib.SMTP(host, port, timeout=20)
        server.starttls()
    with server:
        if username and password:
            server.login(username, password)
        server.send_message(msg)


def _deliver(username: str, session_id: str, guardian_email: str, msg: EmailMessage, excerpt: str) -> None:
    try:
        send_email(msg)
        db.log_guardian_alert(username, session_id, guardian_email, "sent", excerpt=excerpt)
        print(f"[Guardian Alert] Email sent to guardian of {username}")
    except Exception as e:
        db.log_guardian_alert(username, session_id, guardian_email, "failed", detail=str(e), excerpt=excerpt)
        print(f"[Guardian Alert] Email to guardian of {username} failed: {e}", file=sys.stderr)


def notify_guardian_of_triage(username: str, session_id: str, patient_message: str, background: bool = True) -> str:
    """Emails the patient's guardian about an emergency triage event.

    Returns the alert status for this turn: 'queued' (sending in background), 'sent', 'failed',
    'no_guardian', 'not_configured' or 'cooldown'.
    """
    profile = db.get_profile(username) or {}
    guardian_email = (profile.get("guardian_email") or "").strip()
    if not is_valid_email(guardian_email):
        return "no_guardian"
    if not smtp_configured():
        db.log_guardian_alert(username, session_id, guardian_email, "skipped", detail="SMTP not configured", excerpt=patient_message)
        return "not_configured"
    if _recently_alerted(username, session_id):
        return "cooldown"

    msg = build_alert_email(profile.get("name") or username, profile.get("guardian_name"), guardian_email, patient_message)
    if background:
        # Logged first so a second emergency turn in the same chat sees the cooldown before the send finishes
        db.log_guardian_alert(username, session_id, guardian_email, "queued", excerpt=patient_message)
        threading.Thread(target=_deliver, args=(username, session_id, guardian_email, msg, patient_message), daemon=True).start()
        return "queued"
    _deliver(username, session_id, guardian_email, msg, patient_message)
    return db.get_guardian_alerts(username=username, session_id=session_id, limit=1)[0]["status"]
