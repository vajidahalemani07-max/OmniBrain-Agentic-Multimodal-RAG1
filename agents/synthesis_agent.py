def synthesis_agent(state: dict):
    results = []
    if state.get("search_results"):
        results.extend(state["search_results"])
    if state.get("sql_results"):
        results.extend(state["sql_results"])
    if state.get("vision_results"):
        results.extend(state["vision_results"])

    final_answer = "\n\n".join(str(r) for r in results) if results else "No sufficient data retrieved."
    return {"final_answer": final_answer}