from sql.database import get_connection


def get_schema():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type='table'
        AND name NOT LIKE 'sqlite_%'
    """)

    tables = cursor.fetchall()

    schema = {}

    for table in tables:
        table_name = table[0]

        cursor.execute(f'PRAGMA table_info("{table_name}")')

        columns = cursor.fetchall()

        schema[table_name] = [
            {
                "name": column[1],
                "type": column[2]
            }
            for column in columns
        ]

    conn.close()

    return schema


def format_schema():
    schema = get_schema()

    result = []

    for table_name, columns in schema.items():

        result.append(f"Table: {table_name}")

        for column in columns:
            result.append(
                f"  - {column['name']} ({column['type']})"
            )

    return "\n".join(result)


if __name__ == "__main__":
    print(format_schema())