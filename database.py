"""Database layer for CareFlow with PostgreSQL support and SQLite (careflow.db) fallback.
Handles patient profiles, conversations, complete chat history (chat_messages), and disease-centric medical conditions.
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

from onset_normalizer import normalize_onset_24h

load_dotenv()

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    HAVE_PSYCOPG2 = True
except ImportError:
    HAVE_PSYCOPG2 = False

ROOT = Path(__file__).parent
SQLITE_PATH = ROOT / "careflow.db"
POSTGRES_URL = os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection():
    if POSTGRES_URL and HAVE_PSYCOPG2:
        try:
            return psycopg2.connect(POSTGRES_URL), "postgres"
        except Exception as e:
            print(f"[DB] Postgres connection failed ({e}), falling back to SQLite at {SQLITE_PATH}")
    
    conn = sqlite3.connect(str(SQLITE_PATH))
    conn.row_factory = sqlite3.Row
    return conn, "sqlite"


def init_db():
    conn, db_type = get_connection()
    cur = conn.cursor()
    
    if db_type == "postgres":
        cur.execute("""
            CREATE TABLE IF NOT EXISTS patient_profiles (
                username VARCHAR(100) PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                date_of_birth VARCHAR(50),
                height VARCHAR(50),
                weight VARCHAR(50),
                created_at VARCHAR(100) NOT NULL,
                updated_at VARCHAR(100) NOT NULL
            );
            CREATE TABLE IF NOT EXISTS conversations (
                id VARCHAR(100) PRIMARY KEY,
                username VARCHAR(100) NOT NULL REFERENCES patient_profiles(username) ON DELETE CASCADE,
                title VARCHAR(255) NOT NULL,
                created_at VARCHAR(100) NOT NULL,
                updated_at VARCHAR(100) NOT NULL,
                follow_up_tracker TEXT DEFAULT '{}',
                pending_treatment_index INT DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS chat_messages (
                id VARCHAR(100) PRIMARY KEY,
                session_id VARCHAR(100) NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                sender VARCHAR(50) NOT NULL,
                message TEXT NOT NULL,
                categories TEXT DEFAULT '[]',
                created_at VARCHAR(100) NOT NULL
            );
            CREATE TABLE IF NOT EXISTS medical_conditions (
                id SERIAL PRIMARY KEY,
                username VARCHAR(100) NOT NULL REFERENCES patient_profiles(username) ON DELETE CASCADE,
                disease_name VARCHAR(255) NOT NULL,
                chief_complaint TEXT,
                onset TEXT,
                severity_score INT,
                location TEXT,
                treatment_name TEXT,
                treatment_status TEXT,
                report_type TEXT,
                raw_data TEXT NOT NULL,
                created_at VARCHAR(100) NOT NULL,
                updated_at VARCHAR(100) NOT NULL,
                UNIQUE(username, disease_name)
            );
            CREATE TABLE IF NOT EXISTS appointments (
                id VARCHAR(100) PRIMARY KEY,
                username VARCHAR(100) NOT NULL REFERENCES patient_profiles(username) ON DELETE CASCADE,
                doctor_type VARCHAR(100) NOT NULL,
                preferred_date VARCHAR(50) NOT NULL,
                preferred_time VARCHAR(50) NOT NULL,
                reason TEXT,
                status VARCHAR(50) DEFAULT 'scheduled',
                created_at VARCHAR(100) NOT NULL
            );
            CREATE TABLE IF NOT EXISTS guardrail_events (
                id VARCHAR(100) PRIMARY KEY,
                session_id VARCHAR(100) NOT NULL,
                username VARCHAR(100),
                stage VARCHAR(20) NOT NULL,
                rule VARCHAR(100) NOT NULL,
                action VARCHAR(50) NOT NULL,
                excerpt TEXT,
                created_at VARCHAR(100) NOT NULL
            );
            CREATE TABLE IF NOT EXISTS patient_memories (
                id VARCHAR(100) PRIMARY KEY,
                username VARCHAR(100) NOT NULL REFERENCES patient_profiles(username) ON DELETE CASCADE,
                memory_type VARCHAR(50) NOT NULL,
                content TEXT NOT NULL,
                confidence_score REAL DEFAULT 1.0,
                status VARCHAR(30) DEFAULT 'active',
                source_session_id VARCHAR(100),
                created_at VARCHAR(100) NOT NULL,
                updated_at VARCHAR(100) NOT NULL
            );
            CREATE TABLE IF NOT EXISTS patient_baseline_summaries (
                username VARCHAR(100) PRIMARY KEY REFERENCES patient_profiles(username) ON DELETE CASCADE,
                chronic_conditions TEXT DEFAULT '[]',
                active_medications TEXT DEFAULT '[]',
                allergies TEXT DEFAULT '[]',
                general_summary TEXT NOT NULL,
                last_consolidated_at VARCHAR(100) NOT NULL,
                version INT DEFAULT 1
            );
        """)
    else:
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS patient_profiles (
                username TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                date_of_birth TEXT,
                height TEXT,
                weight TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                follow_up_tracker TEXT DEFAULT '{}',
                pending_treatment_index INTEGER DEFAULT 0,
                FOREIGN KEY (username) REFERENCES patient_profiles(username) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS chat_messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                sender TEXT NOT NULL,
                message TEXT NOT NULL,
                categories TEXT DEFAULT '[]',
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES conversations(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS medical_conditions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                disease_name TEXT NOT NULL,
                chief_complaint TEXT,
                onset TEXT,
                severity_score INTEGER,
                location TEXT,
                treatment_name TEXT,
                treatment_status TEXT,
                report_type TEXT,
                raw_data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(username, disease_name),
                FOREIGN KEY (username) REFERENCES patient_profiles(username) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS appointments (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                doctor_type TEXT NOT NULL,
                preferred_date TEXT NOT NULL,
                preferred_time TEXT NOT NULL,
                reason TEXT,
                status TEXT DEFAULT 'scheduled',
                created_at TEXT NOT NULL,
                FOREIGN KEY (username) REFERENCES patient_profiles(username) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS guardrail_events (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                username TEXT,
                stage TEXT NOT NULL,
                rule TEXT NOT NULL,
                action TEXT NOT NULL,
                excerpt TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS patient_memories (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                confidence_score REAL DEFAULT 1.0,
                status TEXT DEFAULT 'active',
                source_session_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (username) REFERENCES patient_profiles(username) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS patient_baseline_summaries (
                username TEXT PRIMARY KEY,
                chronic_conditions TEXT DEFAULT '[]',
                active_medications TEXT DEFAULT '[]',
                allergies TEXT DEFAULT '[]',
                general_summary TEXT NOT NULL,
                last_consolidated_at TEXT NOT NULL,
                version INTEGER DEFAULT 1,
                FOREIGN KEY (username) REFERENCES patient_profiles(username) ON DELETE CASCADE
            );
        """)
    
    conn.commit()

    # Automatically normalize any legacy onset rows to strict 24-hour YYYY-MM-DD HH:MM format
    try:
        cur.execute("SELECT id, onset FROM medical_conditions WHERE onset IS NOT NULL")
        for row in cur.fetchall():
            rid = row[0] if not isinstance(row, sqlite3.Row) else row["id"]
            raw_on = row[1] if not isinstance(row, sqlite3.Row) else row["onset"]
            norm_on = normalize_onset_24h(raw_on)
            if norm_on and norm_on != raw_on:
                ph = "%s" if db_type == "postgres" else "?"
                cur.execute(f"UPDATE medical_conditions SET onset = {ph} WHERE id = {ph}", (norm_on, rid))
        conn.commit()
    except Exception as e:
        print(f"[DB] Onset migration note: {e}")

    conn.close()
    
    # Migrate from persistent.txt if it still exists
    migrate_from_persistent_file()


def migrate_from_persistent_file():
    persistent_file = ROOT / "persistent.txt"
    if not persistent_file.exists():
        return
    
    try:
        data = json.loads(persistent_file.read_text(encoding="utf-8"))
        profiles = data.get("profiles", {})
        for username, rec in profiles.items():
            prof = rec.get("profile", {})
            if not get_profile(username):
                create_profile(
                    username=username,
                    name=prof.get("name") or username,
                    date_of_birth=prof.get("date_of_birth"),
                    height=prof.get("height"),
                    weight=prof.get("weight"),
                )
            
            # Migrate conversations and chat messages
            convs = rec.get("conversations", {})
            for conv_id, conv in convs.items():
                if not get_conversation(username, conv_id):
                    create_conversation(username=username, title=conv.get("title", "Previous chat"), conv_id=conv_id)
                for msg in conv.get("messages", []):
                    save_chat_message(
                        session_id=conv_id,
                        sender=msg.get("role", "user"),
                        message=msg.get("text", ""),
                        categories=msg.get("categories") or ([msg.get("category")] if msg.get("category") else []),
                    )
            
            # Migrate forms to medical_conditions if any
            forms = rec.get("forms", {})
            symptoms = forms.get("symptoms", {})
            if symptoms and symptoms.get("chief_complaint"):
                disease = symptoms.get("chief_complaint", "General Condition")
                upsert_condition(username, disease, {
                    "disease_name": disease,
                    "symptoms": symptoms,
                    "treatments": forms.get("treatment", [{}])[0] if isinstance(forms.get("treatment"), list) and forms.get("treatment") else {},
                    "reports": forms.get("reports", {}),
                })
        print(f"[DB] Successfully migrated records from persistent.txt to database")
    except Exception as e:
        print(f"[DB] Note during persistent.txt migration: {e}")
    finally:
        try:
            if persistent_file.exists():
                persistent_file.unlink()
                print("[DB] Removed persistent.txt as requested.")
        except Exception:
            pass


def get_profile(username: str) -> Optional[Dict[str, Any]]:
    conn, _ = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM patient_profiles WHERE LOWER(username) = LOWER(%s)" if _ == "postgres" else "SELECT * FROM patient_profiles WHERE LOWER(username) = LOWER(?)", (username,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    if isinstance(row, sqlite3.Row):
        return dict(row)
    # Postgres RealDictRow or tuple
    cols = ["username", "name", "date_of_birth", "height", "weight", "created_at", "updated_at"]
    return {cols[i]: row[i] for i in range(len(cols))} if isinstance(row, tuple) else dict(row)


def create_profile(username: str, name: str, date_of_birth: Optional[str] = None, height: Optional[str] = None, weight: Optional[str] = None) -> Dict[str, Any]:
    conn, db_type = get_connection()
    cur = conn.cursor()
    now = now_iso()
    placeholder = "%s" if db_type == "postgres" else "?"
    cur.execute(
        f"INSERT INTO patient_profiles (username, name, date_of_birth, height, weight, created_at, updated_at) VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})",
        (username.strip(), name.strip(), date_of_birth or None, height or None, weight or None, now, now)
    )
    conn.commit()
    conn.close()
    # Create initial conversation
    create_conversation(username, "New chat")
    return get_profile(username) # type: ignore


def get_conversations(username: str) -> List[Dict[str, Any]]:
    conn, db_type = get_connection()
    cur = conn.cursor()
    placeholder = "%s" if db_type == "postgres" else "?"
    cur.execute(f"SELECT * FROM conversations WHERE LOWER(username) = LOWER({placeholder}) ORDER BY created_at ASC", (username,))
    rows = cur.fetchall()
    conn.close()
    convs = []
    for row in rows:
        d = dict(row) if isinstance(row, sqlite3.Row) else dict(row) if not isinstance(row, tuple) else {
            "id": row[0], "username": row[1], "title": row[2], "created_at": row[3], "updated_at": row[4],
            "follow_up_tracker": row[5], "pending_treatment_index": row[6]
        }
        if isinstance(d.get("follow_up_tracker"), str):
            try:
                d["follow_up_tracker"] = json.loads(d["follow_up_tracker"])
            except Exception:
                d["follow_up_tracker"] = {}
        convs.append(d)
    return convs


def create_conversation(username: str, title: str = "New chat", conv_id: Optional[str] = None) -> Dict[str, Any]:
    conn, db_type = get_connection()
    cur = conn.cursor()
    cid = conv_id or uuid.uuid4().hex
    now = now_iso()
    placeholder = "%s" if db_type == "postgres" else "?"
    cur.execute(
        f"INSERT INTO conversations (id, username, title, created_at, updated_at, follow_up_tracker, pending_treatment_index) VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, 0)",
        (cid, username, title, now, now, json.dumps({}))
    )
    conn.commit()
    conn.close()
    return get_conversation(username, cid) # type: ignore


def get_conversation(username: str, conv_id: str) -> Optional[Dict[str, Any]]:
    conn, db_type = get_connection()
    cur = conn.cursor()
    placeholder = "%s" if db_type == "postgres" else "?"
    cur.execute(f"SELECT * FROM conversations WHERE id = {placeholder} AND LOWER(username) = LOWER({placeholder})", (conv_id, username))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row) if isinstance(row, sqlite3.Row) else dict(row) if not isinstance(row, tuple) else {
        "id": row[0], "username": row[1], "title": row[2], "created_at": row[3], "updated_at": row[4],
        "follow_up_tracker": row[5], "pending_treatment_index": row[6]
    }
    if isinstance(d.get("follow_up_tracker"), str):
        try:
            d["follow_up_tracker"] = json.loads(d["follow_up_tracker"])
        except Exception:
            d["follow_up_tracker"] = {}
    
    # Load messages
    d["messages"] = get_chat_messages(conv_id)
    return d


def update_conversation_title(conv_id: str, title: str):
    conn, db_type = get_connection()
    cur = conn.cursor()
    placeholder = "%s" if db_type == "postgres" else "?"
    cur.execute(f"UPDATE conversations SET title = {placeholder}, updated_at = {placeholder} WHERE id = {placeholder}", (title, now_iso(), conv_id))
    conn.commit()
    conn.close()


def update_conversation_tracker(conv_id: str, tracker: Dict[str, Any]):
    conn, db_type = get_connection()
    cur = conn.cursor()
    placeholder = "%s" if db_type == "postgres" else "?"
    cur.execute(f"UPDATE conversations SET follow_up_tracker = {placeholder}, updated_at = {placeholder} WHERE id = {placeholder}", (json.dumps(tracker), now_iso(), conv_id))
    conn.commit()
    conn.close()


def save_chat_message(session_id: str, sender: str, message: str, categories: Optional[List[str]] = None) -> Dict[str, Any]:
    """Saves user message or assistant response immediately to PostgreSQL/SQLite chat_messages table."""
    conn, db_type = get_connection()
    cur = conn.cursor()
    mid = uuid.uuid4().hex
    now = now_iso()
    cats_json = json.dumps(categories or [])
    placeholder = "%s" if db_type == "postgres" else "?"
    cur.execute(
        f"INSERT INTO chat_messages (id, session_id, sender, message, categories, created_at) VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})",
        (mid, session_id, sender, message, cats_json, now)
    )
    # Touch conversation updated_at
    cur.execute(f"UPDATE conversations SET updated_at = {placeholder} WHERE id = {placeholder}", (now, session_id))
    conn.commit()
    conn.close()
    return {
        "id": mid,
        "session_id": session_id,
        "role": sender,
        "text": message,
        "categories": categories or [],
        "created_at": now
    }


def get_chat_messages(session_id: str) -> List[Dict[str, Any]]:
    conn, db_type = get_connection()
    cur = conn.cursor()
    placeholder = "%s" if db_type == "postgres" else "?"
    cur.execute(f"SELECT * FROM chat_messages WHERE session_id = {placeholder} ORDER BY created_at ASC", (session_id,))
    rows = cur.fetchall()
    conn.close()
    messages = []
    for row in rows:
        d = dict(row) if isinstance(row, sqlite3.Row) else dict(row) if not isinstance(row, tuple) else {
            "id": row[0], "session_id": row[1], "sender": row[2], "message": row[3], "categories": row[4], "created_at": row[5]
        }
        cats = []
        if isinstance(d.get("categories"), str):
            try:
                cats = json.loads(d["categories"])
            except Exception:
                cats = []
        messages.append({
            "id": d.get("id"),
            "role": d.get("sender"),
            "text": d.get("message"),
            "categories": cats,
            "created_at": d.get("created_at")
        })
    return messages


def get_conditions(username: str) -> List[Dict[str, Any]]:
    conn, db_type = get_connection()
    cur = conn.cursor()
    placeholder = "%s" if db_type == "postgres" else "?"
    cur.execute(f"SELECT * FROM medical_conditions WHERE LOWER(username) = LOWER({placeholder}) ORDER BY updated_at DESC", (username,))
    rows = cur.fetchall()
    conn.close()
    conditions = []
    for row in rows:
        d = dict(row) if isinstance(row, sqlite3.Row) else dict(row) if not isinstance(row, tuple) else {
            "id": row[0], "username": row[1], "disease_name": row[2], "chief_complaint": row[3],
            "onset": row[4], "severity_score": row[5], "location": row[6],
            "treatment_name": row[7], "treatment_status": row[8], "report_type": row[9],
            "raw_data": row[10], "created_at": row[11], "updated_at": row[12]
        }
        if isinstance(d.get("raw_data"), str):
            try:
                d["data"] = json.loads(d["raw_data"])
            except Exception:
                d["data"] = {}
        conditions.append(d)
    return conditions


def get_condition_by_name(username: str, disease_name: str) -> Optional[Dict[str, Any]]:
    conn, db_type = get_connection()
    cur = conn.cursor()
    placeholder = "%s" if db_type == "postgres" else "?"
    cur.execute(f"SELECT * FROM medical_conditions WHERE LOWER(username) = LOWER({placeholder}) AND LOWER(disease_name) = LOWER({placeholder})", (username, disease_name))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row) if isinstance(row, sqlite3.Row) else dict(row) if not isinstance(row, tuple) else {
        "id": row[0], "username": row[1], "disease_name": row[2], "chief_complaint": row[3],
        "onset": row[4], "severity_score": row[5], "location": row[6],
        "treatment_name": row[7], "treatment_status": row[8], "report_type": row[9],
        "raw_data": row[10], "created_at": row[11], "updated_at": row[12]
    }
    if isinstance(d.get("raw_data"), str):
        try:
            d["data"] = json.loads(d["raw_data"])
        except Exception:
            d["data"] = {}
    return d


def upsert_condition(username: str, disease_name: str, condition_data: Dict[str, Any]) -> Dict[str, Any]:
    existing = get_condition_by_name(username, disease_name)
    merged_data = existing["data"] if existing and "data" in existing else {}
    
    # Deep merge incoming condition_data
    for key, val in condition_data.items():
        if isinstance(val, dict) and isinstance(merged_data.get(key), dict):
            for k2, v2 in val.items():
                if v2 is not None and v2 != "":
                    merged_data[key][k2] = v2
        elif val is not None and val != "":
            merged_data[key] = val
    
    merged_data["disease_name"] = disease_name
    symptoms = merged_data.get("symptoms", {})
    treatments = merged_data.get("treatments", {})
    reports = merged_data.get("reports", {})
    
    chief_complaint = symptoms.get("chief_complaint") or disease_name
    onset = normalize_onset_24h(symptoms.get("onset"))
    if onset:
        if "symptoms" not in merged_data:
            merged_data["symptoms"] = {}
        merged_data["symptoms"]["onset"] = onset
    severity_score = symptoms.get("severity_score")
    location = symptoms.get("location")
    treatment_name = treatments.get("intervention_name")
    treatment_status = treatments.get("status")
    report_type = reports.get("report_type")
    
    now = now_iso()
    raw_json = json.dumps(merged_data)
    
    conn, db_type = get_connection()
    cur = conn.cursor()
    placeholder = "%s" if db_type == "postgres" else "?"
    
    if existing:
        cur.execute(
            f"""UPDATE medical_conditions 
               SET chief_complaint = {placeholder}, onset = {placeholder}, severity_score = {placeholder}, 
                   location = {placeholder}, treatment_name = {placeholder}, treatment_status = {placeholder}, 
                   report_type = {placeholder}, raw_data = {placeholder}, updated_at = {placeholder}
               WHERE LOWER(username) = LOWER({placeholder}) AND LOWER(disease_name) = LOWER({placeholder})""",
            (chief_complaint, onset, severity_score, location, treatment_name, treatment_status, report_type, raw_json, now, username, disease_name)
        )
    else:
        cur.execute(
            f"""INSERT INTO medical_conditions 
               (username, disease_name, chief_complaint, onset, severity_score, location, treatment_name, treatment_status, report_type, raw_data, created_at, updated_at)
               VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})""",
            (username, disease_name, chief_complaint, onset, severity_score, location, treatment_name, treatment_status, report_type, raw_json, now, now)
        )
    
    conn.commit()
    conn.close()
    return get_condition_by_name(username, disease_name) or {"data": merged_data}


def book_appointment(username: str, doctor_type: str, preferred_date: str, preferred_time: str, reason: str) -> Dict[str, Any]:
    conn, db_type = get_connection()
    cur = conn.cursor()
    aid = uuid.uuid4().hex
    now = now_iso()
    placeholder = "%s" if db_type == "postgres" else "?"
    cur.execute(
        f"""INSERT INTO appointments (id, username, doctor_type, preferred_date, preferred_time, reason, status, created_at)
           VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, 'confirmed', {placeholder})""",
        (aid, username, doctor_type, preferred_date, preferred_time, reason, now)
    )
    conn.commit()
    conn.close()
    return {
        "appointment_id": aid,
        "username": username,
        "doctor_type": doctor_type,
        "preferred_date": preferred_date,
        "preferred_time": preferred_time,
        "reason": reason,
        "status": "confirmed"
    }


def get_appointments(username: str) -> List[Dict[str, Any]]:
    """Retrieves all appointments for a patient in descending order."""
    conn, db_type = get_connection()
    cur = conn.cursor()
    placeholder = "%s" if db_type == "postgres" else "?"
    cols = ["id", "username", "doctor_type", "preferred_date", "preferred_time", "reason", "status", "created_at"]
    cur.execute(
        f"SELECT {', '.join(cols)} FROM appointments WHERE LOWER(username) = LOWER({placeholder}) ORDER BY created_at DESC",
        (username,)
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(row) if isinstance(row, sqlite3.Row) else {cols[i]: row[i] for i in range(len(cols))} for row in rows]


def get_patient_aggregate_forms(username: str) -> Dict[str, Any]:
    """Constructs the UI-friendly forms structure (symptoms, reports, treatment) from medical_conditions."""
    conditions = get_conditions(username)
    
    symptoms_form: Dict[str, Any] = {}
    reports_form: Dict[str, Any] = {}
    treatment_list: List[Dict[str, Any]] = []
    
    for cond in conditions:
        data = cond.get("data", {})
        sym = data.get("symptoms", {})
        rep = data.get("reports", {})
        tr = data.get("treatments", {})
        
        # Merge symptoms (newest condition fields take priority)
        for k, v in sym.items():
            if k == "trigger_event" and isinstance(v, dict):
                desc = v.get("event_description")
                if desc and "trigger_event" not in symptoms_form:
                    symptoms_form["trigger_event"] = desc
                cat = v.get("event_category")
                if cat and "trigger_category" not in symptoms_form:
                    symptoms_form["trigger_category"] = cat
            elif v is not None and v != "" and v != []:
                if k not in symptoms_form:
                    symptoms_form[k] = v
        if not symptoms_form.get("chief_complaint") and cond.get("disease_name"):
            symptoms_form["chief_complaint"] = cond.get("disease_name")
            
        # Merge reports
        for k, v in rep.items():
            if v is not None and v != "" and k not in reports_form:
                reports_form[k] = v
                
        # Treatments
        if isinstance(tr, dict) and any(v for v in tr.values() if v is not None and v != ""):
            if not any(t.get("intervention_name") == tr.get("intervention_name") for t in treatment_list if tr.get("intervention_name")):
                treatment_list.append(tr)
        elif isinstance(tr, list):
            for t in tr:
                if isinstance(t, dict) and any(v for v in t.values() if v is not None and v != ""):
                    if not any(item.get("intervention_name") == t.get("intervention_name") for item in treatment_list if t.get("intervention_name")):
                        treatment_list.append(t)
    
    return {
        "symptoms": symptoms_form,
        "reports": reports_form,
        "treatment": treatment_list
    }


def get_full_patient_record(username: str, active_conv_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    profile = get_profile(username)
    if not profile:
        return None
    
    convs = get_conversations(username)
    if not convs:
        create_conversation(username, "New chat")
        convs = get_conversations(username)
        
    selected_id = active_conv_id or (convs[0]["id"] if convs else None)
    active_conv = next((c for c in convs if c["id"] == selected_id), convs[0] if convs else None)
    if active_conv:
        active_conv["messages"] = get_chat_messages(active_conv["id"])
    
    forms = get_patient_aggregate_forms(username)
    
    return {
        "profile": profile,
        "forms": forms,
        "conditions": [c.get("data", {}) for c in get_conditions(username)],
        "conversations": [{"id": c["id"], "title": c["title"], "created_at": c["created_at"]} for c in convs],
        "active_conversation_id": active_conv["id"] if active_conv else None,
        "conversation": active_conv or {"id": None, "messages": []},
        "baseline": get_patient_baseline(username),
        "memories": get_patient_memories(username, status="active"),
    }


def log_guardrail_event(session_id: str, username: str, stage: str, rule: str, action: str, excerpt: str = "") -> None:
    """Audit trail of every guardrail decision (input/output stage) for safety review and observability."""
    conn, db_type = get_connection()
    cur = conn.cursor()
    placeholder = "%s" if db_type == "postgres" else "?"
    cur.execute(
        f"INSERT INTO guardrail_events (id, session_id, username, stage, rule, action, excerpt, created_at) VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})",
        (uuid.uuid4().hex, session_id, username, stage, rule, action, (excerpt or "")[:300], now_iso())
    )
    conn.commit()
    conn.close()


def get_guardrail_events(username: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
    conn, db_type = get_connection()
    cur = conn.cursor()
    placeholder = "%s" if db_type == "postgres" else "?"
    cols = ["id", "session_id", "username", "stage", "rule", "action", "excerpt", "created_at"]
    if username:
        cur.execute(f"SELECT {', '.join(cols)} FROM guardrail_events WHERE LOWER(username) = LOWER({placeholder}) ORDER BY created_at DESC LIMIT {placeholder}", (username, limit))
    else:
        cur.execute(f"SELECT {', '.join(cols)} FROM guardrail_events ORDER BY created_at DESC LIMIT {placeholder}", (limit,))
    rows = cur.fetchall()
    conn.close()
    return [dict(row) if isinstance(row, sqlite3.Row) else {cols[i]: row[i] for i in range(len(cols))} for row in rows]


# ------------------ LONG-TERM MEMORY (LTM) LAYER ------------------

def save_patient_memory(
    username: str,
    memory_type: str,
    content: str,
    confidence_score: float = 1.0,
    status: str = "active",
    source_session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Saves an atomic clinical memory record written by the consolidation agent."""
    conn, db_type = get_connection()
    cur = conn.cursor()
    mem_id = uuid.uuid4().hex
    now = now_iso()
    placeholder = "%s" if db_type == "postgres" else "?"
    cur.execute(
        f"""
        INSERT INTO patient_memories 
        (id, username, memory_type, content, confidence_score, status, source_session_id, created_at, updated_at)
        VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
        """,
        (mem_id, username, memory_type, content, confidence_score, status, source_session_id, now, now)
    )
    conn.commit()
    conn.close()
    return {
        "id": mem_id,
        "username": username,
        "memory_type": memory_type,
        "content": content,
        "confidence_score": confidence_score,
        "status": status,
        "source_session_id": source_session_id,
        "created_at": now,
        "updated_at": now
    }


def get_patient_memories(username: str, status: Optional[str] = "active", limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves long-term memories for a patient, optionally filtered by status (active/resolved/contradicted)."""
    conn, db_type = get_connection()
    cur = conn.cursor()
    placeholder = "%s" if db_type == "postgres" else "?"
    cols = ["id", "username", "memory_type", "content", "confidence_score", "status", "source_session_id", "created_at", "updated_at"]
    
    if status:
        cur.execute(
            f"SELECT {', '.join(cols)} FROM patient_memories WHERE LOWER(username) = LOWER({placeholder}) AND status = {placeholder} ORDER BY updated_at DESC LIMIT {placeholder}",
            (username, status, limit)
        )
    else:
        cur.execute(
            f"SELECT {', '.join(cols)} FROM patient_memories WHERE LOWER(username) = LOWER({placeholder}) ORDER BY updated_at DESC LIMIT {placeholder}",
            (username, limit)
        )
    rows = cur.fetchall()
    conn.close()
    return [dict(row) if isinstance(row, sqlite3.Row) else {cols[i]: row[i] for i in range(len(cols))} for row in rows]


def update_patient_memory_status(memory_id: str, status: str) -> None:
    """Updates status of a memory item (e.g. 'resolved', 'contradicted', 'archived')."""
    conn, db_type = get_connection()
    cur = conn.cursor()
    placeholder = "%s" if db_type == "postgres" else "?"
    now = now_iso()
    cur.execute(
        f"UPDATE patient_memories SET status = {placeholder}, updated_at = {placeholder} WHERE id = {placeholder}",
        (status, now, memory_id)
    )
    conn.commit()
    conn.close()


def upsert_patient_baseline(
    username: str,
    chronic_conditions: List[str],
    active_medications: List[Dict[str, Any]],
    allergies: List[str],
    general_summary: str
) -> Dict[str, Any]:
    """Updates or inserts the synthesized long-term patient clinical baseline profile."""
    conn, db_type = get_connection()
    cur = conn.cursor()
    placeholder = "%s" if db_type == "postgres" else "?"
    now = now_iso()
    
    cur.execute(
        f"SELECT version FROM patient_baseline_summaries WHERE LOWER(username) = LOWER({placeholder})",
        (username,)
    )
    row = cur.fetchone()
    
    chronic_json = json.dumps(chronic_conditions or [])
    meds_json = json.dumps(active_medications or [])
    allergies_json = json.dumps(allergies or [])
    
    if row:
        new_version = (row[0] if not isinstance(row, sqlite3.Row) else row["version"]) + 1
        cur.execute(
            f"""
            UPDATE patient_baseline_summaries
            SET chronic_conditions = {placeholder}, active_medications = {placeholder},
                allergies = {placeholder}, general_summary = {placeholder},
                last_consolidated_at = {placeholder}, version = {placeholder}
            WHERE LOWER(username) = LOWER({placeholder})
            """,
            (chronic_json, meds_json, allergies_json, general_summary, now, new_version, username)
        )
    else:
        new_version = 1
        cur.execute(
            f"""
            INSERT INTO patient_baseline_summaries
            (username, chronic_conditions, active_medications, allergies, general_summary, last_consolidated_at, version)
            VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
            """,
            (username, chronic_json, meds_json, allergies_json, general_summary, now, new_version)
        )
    conn.commit()
    conn.close()
    return {
        "username": username,
        "chronic_conditions": chronic_conditions,
        "active_medications": active_medications,
        "allergies": allergies,
        "general_summary": general_summary,
        "last_consolidated_at": now,
        "version": new_version
    }


def get_patient_baseline(username: str) -> Optional[Dict[str, Any]]:
    """Retrieves the consolidated clinical baseline for a patient."""
    conn, db_type = get_connection()
    cur = conn.cursor()
    placeholder = "%s" if db_type == "postgres" else "?"
    cols = ["username", "chronic_conditions", "active_medications", "allergies", "general_summary", "last_consolidated_at", "version"]
    cur.execute(
        f"SELECT {', '.join(cols)} FROM patient_baseline_summaries WHERE LOWER(username) = LOWER({placeholder})",
        (username,)
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row) if isinstance(row, sqlite3.Row) else {cols[i]: row[i] for i in range(len(cols))}
    try:
        d["chronic_conditions"] = json.loads(d.get("chronic_conditions") or "[]")
    except Exception:
        d["chronic_conditions"] = []
    try:
        d["active_medications"] = json.loads(d.get("active_medications") or "[]")
    except Exception:
        d["active_medications"] = []
    try:
        d["allergies"] = json.loads(d.get("allergies") or "[]")
    except Exception:
        d["allergies"] = []
    return d
