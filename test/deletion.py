import sqlite3

conn = sqlite3.connect("road_safety.db")
cursor = conn.cursor()

cursor.execute("SELECT id FROM inputs ORDER BY id DESC LIMIT 1")
last_ids = cursor.fetchall()

for (row_id,) in last_ids:
    cursor.execute("DELETE FROM inputs WHERE id = ?", (row_id,))

conn.commit()
conn.close()

print("✅ Rows deleted successfully from road_safety.db!")
