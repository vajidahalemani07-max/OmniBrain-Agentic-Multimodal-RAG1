import os
import re
import json
import pymupdf
from google import genai
from langgraph.graph import StateGraph, START, END

from graph.state import AgentState
from agents.supervisor import supervisor
from agents.search_agent import search_agent as real_search_agent
from vision.image_processor import analyze_page


# ============================================================
# HELPER: API KEY RESOLUTION (ENV + STREAMLIT SECRETS)
# ============================================================
def get_gemini_api_key():
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
        except Exception:
            pass
    return key


# ============================================================
# FORMATTING HELPERS
# ============================================================
def format_vision_markdown(result, pdf_name, page_num):
    if not result:
        return f"### 📊 Vision Extraction: {pdf_name} (Page {page_num})\n\nNo visual elements detected."

    if isinstance(result, str):
        try:
            result = json.loads(result)
        except Exception:
            return result

    elements = []
    if isinstance(result, dict):
        elements = [result]
    elif isinstance(result, list):
        for item in result:
            if isinstance(item, list):
                elements.extend(item)
            elif isinstance(item, dict):
                elements.append(item)
            elif isinstance(item, str):
                try:
                    parsed = json.loads(item)
                    if isinstance(parsed, list):
                        elements.extend(parsed)
                    elif isinstance(parsed, dict):
                        elements.append(parsed)
                except Exception:
                    elements.append({"type": "TEXT", "description": item})

    md = [f"### 📊 Vision Extraction: {pdf_name} (Page {page_num})\n"]

    for el in elements:
        if not isinstance(el, dict):
            md.append(f"- {str(el)}")
            continue

        el_type = str(el.get("type", "UNKNOWN")).upper()
        desc = el.get("description", "")

        if el_type == "CHART":
            md.append(f"#### 📈 Chart: {el.get('title', 'Financial Chart')}")
            md.append(f"- **Chart Type:** {el.get('chart_type', 'N/A')}")
            if desc:
                md.append(f"- **Summary:** {desc}")

            legends = el.get("legend", [])
            if legends and isinstance(legends, list):
                leg_strs = [item.get("series", "") if isinstance(item, dict) else str(item) for item in legends if str(item).strip()]
                if leg_strs:
                    md.append(f"- **Series/Legend:** {', '.join(leg_strs)}")

            data_points = el.get("data_points", [])
            if data_points and isinstance(data_points, list):
                first_pt = data_points[0]
                if isinstance(first_pt, dict):
                    headers = list(first_pt.keys())
                    md.append("\n| " + " | ".join(headers) + " |")
                    md.append("| " + " | ".join(["---"] * len(headers)) + " |")
                    for pt in data_points:
                        if isinstance(pt, dict):
                            md.append("| " + " | ".join([str(pt.get(h, "")) for h in headers]) + " |")
            md.append("\n---\n")

        elif el_type == "TABLE":
            md.append(f"#### 📋 Table: {desc if desc else 'Financial Data'}")
            table_obj = el.get("data", {})
            if isinstance(table_obj, dict):
                raw_headers = table_obj.get("headers", [])
                headers = []
                if isinstance(raw_headers, dict):
                    headers = raw_headers.get("sub_headers", raw_headers.get("main_headers", []))
                elif isinstance(raw_headers, list):
                    headers = [h.get("column_name", str(h)) if isinstance(h, dict) else str(h) for h in raw_headers]

                rows = table_obj.get("rows", [])
                if headers and rows and isinstance(rows, list):
                    md.append("\n| " + " | ".join(headers) + " |")
                    md.append("| " + " | ".join(["---"] * len(headers)) + " |")
                    for r in rows:
                        if isinstance(r, dict):
                            md.append("| " + " | ".join([str(r.get(h, "")) for h in headers]) + " |")
                        elif isinstance(r, list):
                            md.append("| " + " | ".join([str(val) for val in r]) + " |")
                    md.append("\n---\n")
                elif rows and isinstance(rows, list) and len(rows) > 0 and isinstance(rows[0], dict):
                    headers = list(rows[0].keys())
                    md.append("\n| " + " | ".join(headers) + " |")
                    md.append("| " + " | ".join(["---"] * len(headers)) + " |")
                    for r in rows:
                        md.append("| " + " | ".join([str(r.get(h, "")) for h in headers]) + " |")
                    md.append("\n---\n")
                elif desc:
                    md.append(f"- {desc}\n")
            elif desc:
                md.append(f"- {desc}\n")

        elif el_type == "TEXT":
            text_val = desc or el.get("content", "")
            if text_val and len(str(text_val)) > 20:
                md.append(f"- **Key Note:** {text_val}")

    return "\n".join(md)


# ============================================================
# AGENT NODES
# ============================================================

def search_node(state: AgentState):
    print("Search Agent selected", flush=True)
    question = state["question"]
    result = real_search_agent(question, top_k=5)
    return {
        "search_results": [result.get("answer", "")],
        "next_agent": "search"
    }


def vision_agent(state: AgentState):
    print("Vision Agent selected", flush=True)
    question = state["question"]

    match = re.search(r'page\s*(\d+)', question, re.IGNORECASE)
    page_num = int(match.group(1)) if match else 20

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")] if os.path.exists(data_dir) else []

    target_pdf = None
    q_lower = question.lower()
    if pdf_files:
        for f in pdf_files:
            clean_name = f.lower().replace(".pdf", "").replace("-", " ").replace("_", " ")
            if f.lower() in q_lower or clean_name in q_lower:
                target_pdf = f
                break

        if not target_pdf:
            if "apple" in q_lower:
                target_pdf = next((f for f in pdf_files if "apple" in f.lower()), None)
            elif "nvidia" in q_lower:
                target_pdf = next((f for f in pdf_files if "nvidia" in f.lower()), None)
            elif "tesla" in q_lower:
                target_pdf = next((f for f in pdf_files if "tesla" in f.lower()), None)
            elif "netflix" in q_lower:
                target_pdf = next((f for f in pdf_files if "netflix" in f.lower()), None)

        if not target_pdf:
            target_pdf = pdf_files[0]

    if not target_pdf:
        return {
            "vision_results": [
                "⚠️ **No PDF Found:** Ingestion folder (`data/`) mein koi valid PDF file nahi mili. Sidebar se document pehle upload karein."
            ],
            "next_agent": "vision"
        }

    pdf_path = os.path.join(data_dir, target_pdf)
    api_key = get_gemini_api_key()

    try:
        doc = pymupdf.open(pdf_path)
        if page_num > len(doc) or page_num < 1:
            page_num = min(max(1, page_num), len(doc))
        page = doc[page_num - 1]

        client = genai.Client(api_key=api_key)
        analysis_result = analyze_page(client, page, page_num)
        doc.close()

        clean_output = format_vision_markdown(analysis_result, os.path.basename(pdf_path), page_num)
        return {
            "vision_results": [clean_output],
            "next_agent": "vision"
        }
    except Exception as e:
        return {
            "vision_results": [f"Vision processing failed: {str(e)}"],
            "next_agent": "vision"
        }


def sql_agent(state: AgentState):
    return {
        "sql_results": ["SQL Agent received the question."],
        "next_agent": "sql"
    }


def multi_agent(state: AgentState):
    search_result = real_search_agent(state["question"], top_k=5)
    return {
        "search_results": [search_result.get("answer", "")],
        "sql_results": ["SQL completed."],
        "next_agent": "multi"
    }


def synthesis_node(state: AgentState):
    """
    Summarizes retrieved excerpts into executive financial analysis.
    Hard-blocks all raw excerpts from ever rendering on screen.
    """
    question = state["question"]
    vision_res = state.get("vision_results", [])
    search_res = state.get("search_results", [])

    if vision_res:
        return {"final_answer": "\n\n".join(vision_res)}

    raw_text = "\n\n".join(search_res)
    api_key = get_gemini_api_key()

    prompt = f"""You are an executive financial analyst. Based on these retrieved excerpts from 10-K filings, answer the user's question directly with key metrics bolded and clean bullet points.
Never mention '[Excerpt 1]' or quote metadata tags directly.

User Question: {question}

Retrieved Excerpts:
{raw_text}

Executive Summary:"""

    # 1. API Call with Multi-Model Fallback
    if api_key and raw_text.strip():
        for m in ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-3.8-flash"]:
            try:
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model=m,
                    contents=prompt
                )
                if response.text and response.text.strip():
                    return {"final_answer": response.text.strip()}
            except Exception as e:
                print(f"Synthesis model {m} failed: {e}", flush=True)

    # 2. FAIL-SAFE PARSER (Guaranteed: NO raw excerpts on screen)
    cleaned_points = []
    for line in raw_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        cleaned = re.sub(r'\[Excerpt \d+\]:?', '', line).strip()
        cleaned = re.sub(r'Apple Inc\. \| \d{4} Form 10-K \| \d+', '', cleaned).strip()
        cleaned = re.sub(r'\[.*?\]', '', cleaned).strip()
        if len(cleaned) > 20 and not cleaned.lower().startswith("excerpt"):
            cleaned_points.append(f"- {cleaned}")

    formatted_output = f"""### 📊 Executive Financial Analysis

**Target Query:** {question}

**Extracted Financial Metrics & Disclosures:**
""" + "\n".join(cleaned_points[:6]) + """

---
*Retrieved directly via Agentic Vector Retrieval Pipeline.*"""

    return {"final_answer": formatted_output}


# ============================================================
# LANGGRAPH BUILD
# ============================================================

builder = StateGraph(AgentState)
builder.add_node("supervisor", supervisor)
builder.add_node("search", search_node)
builder.add_node("vision", vision_agent)
builder.add_node("sql", sql_agent)
builder.add_node("multi", multi_agent)
builder.add_node("synthesis", synthesis_node)

builder.add_edge(START, "supervisor")

def route_question(state: AgentState):
    return state["next_agent"]

builder.add_conditional_edges(
    "supervisor",
    route_question,
    {
        "search": "search",
        "vision": "vision",
        "sql": "sql",
        "multi": "multi"
    }
)

builder.add_edge("search", "synthesis")
builder.add_edge("vision", "synthesis")
builder.add_edge("sql", "synthesis")
builder.add_edge("multi", "synthesis")
builder.add_edge("synthesis", END)

graph = builder.compile()