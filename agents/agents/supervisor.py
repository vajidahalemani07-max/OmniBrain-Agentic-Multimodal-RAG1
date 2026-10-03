from graph.state import AgentState


def supervisor(state: AgentState):
    question = state["question"].lower()

    # Multiple agents needed
    if (
        ("revenue" in question or "profit" in question)
        and ("stock" in question or "share price" in question)
    ):
        return {
            "next_agent": "multi",
            "selected_agents": ["search", "sql"]
        }

    # Stock-related questions → SQL
    elif "stock" in question or "share price" in question:
        return {
            "next_agent": "sql",
            "selected_agents": ["sql"]
        }

    # Chart/image/table questions → Vision
    elif (
        "chart" in question
        or "graph" in question
        or "image" in question
        or "table" in question
    ):
        return {
            "next_agent": "vision",
            "selected_agents": ["vision"]
        }

    # Other questions → Search
    else:
        return {
            "next_agent": "search",
            "selected_agents": ["search"]
        }