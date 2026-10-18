import sqlite3
import json

conn = sqlite3.connect('careflow.db')
conn.row_factory = sqlite3.Row
cur = conn.cursor()

cur.execute("SELECT follow_up_tracker FROM conversations WHERE username='kt'")
conv = cur.fetchone()
print(f"Tracker: {conv['follow_up_tracker']}")
conn.close()
