import sqlite3

conn = sqlite3.connect('road_safety.db')
cursor = conn.cursor()

# Add label column (only if not already there)
cursor.execute("ALTER TABLE inputs ADD COLUMN label INTEGER")

conn.commit()
conn.close()

print("✅ 'label' column added to inputs table!")
