"""Database clear utility for CareFlow.
Run `python3 clear_db.py` to reset/clear all patient records, chat messages, and intake forms.
"""
import database as db

def clear_database():
    conn, db_type = db.get_connection()
    cur = conn.cursor()
    
    if db_type == "postgres":
        cur.execute("""
            TRUNCATE TABLE chat_messages CASCADE;
            TRUNCATE TABLE medical_conditions CASCADE;
            TRUNCATE TABLE appointments CASCADE;
            TRUNCATE TABLE conversations CASCADE;
            TRUNCATE TABLE patient_profiles CASCADE;
        """)
    else:
        cur.executescript("""
            DELETE FROM chat_messages;
            DELETE FROM medical_conditions;
            DELETE FROM appointments;
            DELETE FROM conversations;
            DELETE FROM patient_profiles;
        """)
    
    conn.commit()
    conn.close()
    print(f"[CareFlow] Successfully cleared all database records from {db_type.upper()}.")

if __name__ == "__main__":
    clear_database()
