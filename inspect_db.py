"""Database inspection utility for CareFlow.
Run `python3 inspect_db.py` to view all stored patient records, chat messages, and condition forms.
"""
import json
import database as db

def main():
    conn, db_type = db.get_connection()
    print(f"=== CareFlow Database Inspection (Engine: {db_type.upper()}) ===\n")
    
    # 1. Patient Profiles
    cur = conn.cursor()
    cur.execute("SELECT username, name, date_of_birth, height, weight, created_at FROM patient_profiles")
    profiles = cur.fetchall()
    print(f"--- 1. PATIENT PROFILES ({len(profiles)} records) ---")
    for p in profiles:
        print(f"  • Username: {p[0]} | Name: {p[1]} | DOB: {p[2]} | Height: {p[3]} | Weight: {p[4]}")
    print()

    # 2. Conversations
    cur.execute("SELECT id, username, title, updated_at FROM conversations ORDER BY updated_at DESC")
    convs = cur.fetchall()
    print(f"--- 2. CONVERSATIONS ({len(convs)} threads) ---")
    for c in convs:
        print(f"  • ID: {c[0][:8]}... | User: {c[1]} | Title: {c[2]} | Last Updated: {c[3]}")
    print()

    # 3. Medical Conditions (Structured Intake)
    cur.execute("SELECT username, disease_name, chief_complaint, onset, severity_score, location, treatment_name, treatment_status, report_type, raw_data FROM medical_conditions")
    conditions = cur.fetchall()
    print(f"--- 3. STRUCTURED MEDICAL CONDITIONS ({len(conditions)} conditions) ---")
    for cond in conditions:
        raw = json.loads(cond[9]) if cond[9] else {}
        sym = raw.get("symptoms", {})
        trig = sym.get("trigger_event", {})
        trig_str = f" | Trigger: {trig.get('event_description')} ({trig.get('event_category', 'general')})" if trig and trig.get("event_description") else ""
        qual_str = f" | Quality: {sym.get('character_quality')}" if sym.get("character_quality") else ""
        assoc_str = f" | Associated: {', '.join(sym.get('associated_symptoms', []))}" if sym.get("associated_symptoms") else ""

        print(f"  • User: {cond[0]} | Condition: {cond[1]}")
        print(f"    - Complaint: {cond[2]} | Onset: {cond[3]} | Severity: {cond[4]}/10 | Location: {cond[5]}{qual_str}{trig_str}{assoc_str}")
        print(f"    - Treatment: {cond[6]} (Status: {cond[7]}) | Report: {cond[8]}")
    print()

    # 4. Chat Messages History
    cur.execute("SELECT session_id, sender, message, categories, created_at FROM chat_messages ORDER BY created_at ASC")
    messages = cur.fetchall()
    print(f"--- 4. COMPLETE CHAT MESSAGES ({len(messages)} messages stored) ---")
    for m in messages:
        sender_label = "USER" if m[1] == "user" else "ASSISTANT"
        print(f"  [{m[4][:19]}] [{sender_label}] {m[2]}")
    print()

    conn.close()

if __name__ == "__main__":
    main()
