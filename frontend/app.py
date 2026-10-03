import os
import sys
import shutil
import subprocess
import requests
import streamlit as st

# Root directory path resolve karo
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Streamlit secrets se API Key read karo
if "GEMINI_API_KEY" in st.secrets:
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Direct Pipeline Import (Single-process mode)
try:
    from backend.pipeline import run_pipeline
except Exception as e:
    run_pipeline = None

# ============================================================
# PAGE CONFIGURATION & STYLING
# ============================================================
st.set_page_config(
    page_title="OmniBrain - Multimodal Financial RAG",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stChatMessage { border-radius: 8px; padding: 12px; margin-bottom: 10px; }
    .badge-supervisor { background-color: #1e3a8a; color: #93c5fd; padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }
    .badge-agent { background-color: #065f46; color: #a7f3d0; padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# SIDEBAR: DOCUMENT INGESTION
# ============================================================
with st.sidebar:
    st.title("📁 Ingest Document")
    st.caption("Upload Financial 10-K Reports & Sync Vector Store")

    uploaded_files = st.file_uploader(
        "Upload Financial PDF",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload SEC Form 10-K or financial filings"
    )

    if st.button("Index to Vector Store", use_container_width=True, type="primary"):
        if uploaded_files:
            with st.spinner("Processing & indexing documents..."):
                success = False
                # Try HTTP endpoint if available
                try:
                    files_payload = [
                        ("files", (f.name, f.getvalue(), "application/pdf"))
                        for f in uploaded_files
                    ]
                    response = requests.post(f"{BACKEND_URL}/upload", files=files_payload, timeout=5)
                    if response.status_code == 200:
                        success = True
                except Exception:
                    pass

                # Fallback: Direct local ingestion
                if not success:
                    try:
                        saved = []
                        for file in uploaded_files:
                            file_path = os.path.join(DATA_DIR, file.name)
                            with open(file_path, "wb") as buffer:
                                buffer.write(file.getvalue())
                            saved.append(file.name)

                        proc = subprocess.run(
                            [sys.executable, "-m", "rag.ingest"],
                            cwd=BASE_DIR,
                            capture_output=True,
                            text=True
                        )
                        if proc.returncode == 0:
                            success = True
                        else:
                            st.error(f"Ingestion failed: {proc.stderr}")
                    except Exception as e:
                        st.error(f"Direct Ingestion Error: {str(e)}")

                if success:
                    st.success(f"✅ Successfully indexed {len(uploaded_files)} document(s)!")
                    st.rerun()
        else:
            st.warning("Please select at least one PDF file first.")

    st.divider()
    st.markdown("""
    **Architecture:**
    - 🧭 **LangGraph Supervisor**
    - 🔍 **Search Agent (RAG)**
    - 👁️ **Vision Agent (Gemini Flash)**
    - ⚡ **Direct Engine (In-Memory)**
    """)

# ============================================================
# MAIN CHAT INTERFACE
# ============================================================
st.title("🧠 OmniBrain: Multimodal Financial RAG")
st.caption("LangGraph Supervisor • Multimodal Store • Vision & RAG Agents")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("route"):
            st.markdown(
                f"<span class='badge-supervisor'>[Supervisor] Routed to: {msg['route'].upper()} Agent</span>",
                unsafe_allow_html=True
            )
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about financial trends, visual charts, or stock metrics..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing request and orchestrating agents..."):
            answer = None
            route = "search"

            # 1. First preference: Direct in-memory Python pipeline execution
            if run_pipeline is not None:
                try:
                    result = run_pipeline(prompt)
                    answer = result.get("answer", "No answer generated.")
                    route = result.get("route", "search")
                except Exception as direct_err:
                    st.warning(f"Direct execution error: {direct_err}. Trying API fallback...")

            # 2. Secondary fallback: HTTP endpoint
            if not answer:
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/query",
                        json={"question": prompt},
                        timeout=60
                    )
                    if response.status_code == 200:
                        result = response.json()
                        answer = result.get("answer", "No answer generated.")
                        route = result.get("route", "search")
                except Exception:
                    pass

            if not answer:
                answer = "Error: Could not execute query via direct pipeline or backend service."
                route = "error"

            st.markdown(
                f"<span class='badge-supervisor'>[Supervisor] Routed to: {route.upper()} Agent</span>",
                unsafe_allow_html=True
            )
            st.markdown(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "route": route
            })