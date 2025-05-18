import sqlite3

# Connect to your database
conn = sqlite3.connect("road_safety.db")
cursor = conn.cursor()

# 1. Find the last 10 ids (assuming you have an 'id' column auto-increment)
cursor.execute("SELECT id FROM inputs ORDER BY id DESC LIMIT 1")
last_ids = cursor.fetchall()

# 2. Delete rows with those ids
for (row_id,) in last_ids:
    cursor.execute("DELETE FROM inputs WHERE id = ?", (row_id,))

conn.commit()
conn.close()

print("✅ Rows deleted successfully from road_safety.db!")
