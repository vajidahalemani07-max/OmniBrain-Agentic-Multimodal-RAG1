import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def create_database():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product TEXT NOT NULL,
            category TEXT,
            revenue REAL,
            year INTEGER
        )
    """)

    conn.commit()
    conn.close()


def insert_sample_data():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM sales")
    count = cursor.fetchone()[0]

    if count == 0:
        data = [
            ("Laptop", "Electronics", 50000, 2024),
            ("Mobile", "Electronics", 30000, 2024),
            ("Laptop", "Electronics", 60000, 2025),
            ("Mobile", "Electronics", 40000, 2025),
            ("Tablet", "Electronics", 25000, 2025),
            ("Chair", "Furniture", 10000, 2025)
        ]

        cursor.executemany("""
            INSERT INTO sales
            (product, category, revenue, year)
            VALUES (?, ?, ?, ?)
        """, data)

        conn.commit()

    conn.close()


if __name__ == "__main__":
    create_database()
    insert_sample_data()

    print("Database created successfully.")
    print(f"Database location: {DB_PATH}")