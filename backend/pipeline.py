import os
import sys
from graph.workflow import graph

def run_pipeline(question: str):
    """
    Executes the LangGraph multi-agent workflow for a given question.
    """
    # 1. Initialize the state for the workflow
    # Agar query me 'vision:' hai, toh next_agent fallback 'vision' rakhenge, warna 'search'
    initial_state = {
        "question": question,
        "search_results": [],
        "vision_results": [],
        "sql_results": [],
        "next_agent": "vision" if "vision:" in question.lower() else "search",
        "final_answer": ""
    }

    try:
        # 2. Invoke LangGraph execution
        final_state = graph.invoke(initial_state)

        # 3. Extract safe final answer
        final_answer = final_state.get("final_answer", "")
        if not final_answer:
            # Fallback agar final_answer empty aaye
            if final_state.get("vision_results"):
                final_answer = "\n\n".join(final_state["vision_results"])
            elif final_state.get("search_results"):
                final_answer = "\n\n".join(final_state["search_results"])
            else:
                final_answer = "No relevant information found."

        # 4. Extract proper routing agent name
        detected_route = final_state.get("next_agent")
        if not detected_route or detected_route == "UNKNOWN":
            detected_route = "vision" if "vision:" in question.lower() else "search"

        return {
            "answer": final_answer,
            "route": detected_route
        }

    except Exception as e:
        print(f"Pipeline Execution Error: {str(e)}", flush=True)
        return {
            "answer": f"Backend Error: {str(e)}",
            "route": "error"
        }