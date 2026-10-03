from graph.state import AgentState


def synthesis_agent(state: AgentState):

    search_results = state.get("search_results", [])
    vision_results = state.get("vision_results", [])
    sql_results = state.get("sql_results", [])

    answer_parts = []

    if search_results:
        answer_parts.append(
            "Information from Search Agent:\n"
            + "\n".join(search_results)
        )

    if vision_results:
        answer_parts.append(
            "Information from Vision Agent:\n"
            + "\n".join(vision_results)
        )

    if sql_results:
        answer_parts.append(
            "Information from SQL Agent:\n"
            + "\n".join(sql_results)
        )

    if answer_parts:
        final_answer = "\n\n".join(answer_parts)
    else:
        final_answer = "No information was retrieved."

    print("Synthesis Agent executed")

    return {
        "final_answer": final_answer
    }