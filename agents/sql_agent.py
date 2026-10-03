from sql.query_executor import execute_query
from sql.schema import format_schema


def generate_sql(question):
    """
    Converts common user questions into SQL.
    """

    question = question.lower().strip()

    if "all" in question and "sales" in question:
        return "SELECT * FROM sales"

    if "total revenue" in question:
        return "SELECT SUM(revenue) AS total_revenue FROM sales"

    if "average revenue" in question:
        return "SELECT AVG(revenue) AS average_revenue FROM sales"

    if "highest revenue" in question:
        return """
        SELECT *
        FROM sales
        ORDER BY revenue DESC
        LIMIT 1
        """

    if "lowest revenue" in question:
        return """
        SELECT *
        FROM sales
        ORDER BY revenue ASC
        LIMIT 1
        """

    if "2025" in question and "revenue" in question:
        return """
        SELECT SUM(revenue) AS total_revenue
        FROM sales
        WHERE year = 2025
        """

    if "2024" in question and "revenue" in question:
        return """
        SELECT SUM(revenue) AS total_revenue
        FROM sales
        WHERE year = 2024
        """

    if "products" in question:
        return """
        SELECT product, revenue
        FROM sales
        """

    return None


def run_sql_agent(question):

    sql_query = generate_sql(question)

    if sql_query is None:
        return {
            "success": False,
            "question": question,
            "sql": None,
            "result": None,
            "message": "I could not generate SQL for this question.",
            "schema": format_schema()
        }

    result = execute_query(sql_query)

    return {
        "success": result["success"],
        "question": question,
        "sql": sql_query,
        "result": result,
        "message": "SQL query executed successfully."
        if result["success"]
        else "SQL query failed."
    }


if __name__ == "__main__":

    question = input("Ask your SQL question: ")

    response = run_sql_agent(question)

    print("\nQuestion:")
    print(response["question"])

    print("\nSQL:")
    print(response["sql"])

    print("\nResult:")
    print(response["result"])