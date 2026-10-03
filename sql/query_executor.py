from sql.database import get_connection


def execute_query(query):
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(query)

        rows = cursor.fetchall()

        if cursor.description:
            columns = [
                description[0]
                for description in cursor.description
            ]
        else:
            columns = []

        conn.commit()

        return {
            "success": True,
            "columns": columns,
            "rows": rows,
            "error": None
        }

    except Exception as e:

        conn.rollback()

        return {
            "success": False,
            "columns": [],
            "rows": [],
            "error": str(e)
        }

    finally:
        conn.close()