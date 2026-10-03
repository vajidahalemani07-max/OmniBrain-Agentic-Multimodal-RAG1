import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Streamlit secrets se API Key read karna
if "GEMINI_API_KEY" not in os.environ:
    try:
        import streamlit as st
        if "GEMINI_API_KEY" in st.secrets:
            os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

# Fallback paths setup
os.environ["DATA_DIR"] = os.path.join(BASE_DIR, "data")
os.environ["IMAGE_DIR"] = os.path.join(BASE_DIR, "backend")

from graph.workflow import graph

def synthesize_fallback(question: str, context_chunks: list) -> str:
    """Gemini se raw excerpts ko proper answer mein convert karta hai"""
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return "\n\n".join(context_chunks)
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        prompt = f"""You are a senior financial analyst. Answer the user question accurately based on the excerpts below from SEC 10-K filings. Highlight exact numbers, YoY trends, and percentages cleanly.

Question: {question}

Context Excerpts:
{chr(10).join(context_chunks)}

Final Formatted Answer:"""

        response = model.generate_content(prompt)
        return response.text
    except Exception:
        return "\n\n".join(context_chunks)

def run_pipeline(question: str):
    """
    Executes LangGraph workflow and ensures clean answers without raw excerpts.
    """
    initial_state = {
        "question": question,
        "search_results": [],
        "vision_results": [],
        "sql_results": [],
        "next_agent": "vision" if "vision:" in question.lower() else "search",
        "final_answer": ""
    }

    try:
        final_state = graph.invoke(initial_state)

        # 1. Answer check
        final_answer = final_state.get("final_answer", "").strip()

        # 2. Agar final answer blank hai to synthesize karwao (excerpts hatane ke liye)
        if not final_answer:
            results = final_state.get("search_results", []) or final_state.get("vision_results", [])
            if results:
                final_answer = synthesize_fallback(question, results)
            else:
                final_answer = "No relevant financial details found in the indexed documents."

        detected_route = final_state.get("next_agent")
        if not detected_route or detected_route == "UNKNOWN":
            detected_route = "vision" if "vision:" in question.lower() else "search"

        return {
            "answer": final_answer,
            "route": detected_route
        }

    except Exception as e:
        # Agar vision ka graph fail ho to safe fallback answer do
        return {
            "answer": f"Error running agent pipeline: {str(e)}",
            "route": "error"
        }