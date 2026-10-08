import os
import re

def supervisor(state):
    """
    Deterministic supervisor classifier with prioritized routing:
    1. SQL Agent: Queries asking for SQL execution, databases, tabular aggregations, structured metrics.
    2. Vision Agent: Queries asking for page analysis, charts, diagrams, or visual inspection.
    3. Search Agent: General textual RAG retrieval.
    """
    question = state.get("question", "").lower().strip()

    # 1. SQL INTENTS
    sql_triggers = [
        "sql", "database", "sqlite", "query", "select ", "group by", 
        "average quarterly", "calculate", "schema", "table query"
    ]
    if any(k in question for k in sql_triggers):
        print("[Supervisor] Explicit routing -> SQL Agent", flush=True)
        return {"next_agent": "sql"}

    # 2. VISION INTENTS
    vision_triggers = [
        "vision", "chart", "table", "graph", "plot", "page", "image", "diagram", "figure"
    ]
    if any(k in question for k in vision_triggers):
        print("[Supervisor] Explicit routing -> Vision Agent", flush=True)
        return {"next_agent": "vision"}

    # 3. SEARCH / RAG (DEFAULT)
    print("[Supervisor] Semantic routing -> Search Agent", flush=True)
    return {"next_agent": "search"}