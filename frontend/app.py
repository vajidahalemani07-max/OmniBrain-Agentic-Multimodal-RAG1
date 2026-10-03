import os
import requests
import streamlit as st

# ============================================================
# PAGE CONFIGURATION & STYLING
# ============================================================
st.set_page_config(
    page_title="OmniBrain - Multimodal Financial RAG",
    page_icon="🧠",
    layout="wide"
)

# Custom minimal dark mode CSS for neat layout
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stChatMessage {
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }
    .badge-supervisor {
        background-color: #1e3a8a;
        color: #93c5fd;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
    }
    .badge-agent {
        background-color: #065f46;
        color: #a7f3d0;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

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
                try:
                    # Backend /upload endpoint ko bhejenge
                    files_payload = [
                        ("files", (f.name, f.getvalue(), "application/pdf"))
                        for f in uploaded_files
                    ]
                    response = requests.post(f"{BACKEND_URL}/upload", files=files_payload)

                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"✅ Successfully indexed {len(data.get('indexed_files', []))} document(s)!")
                        st.rerun()
                    else:
                        st.error(f"Upload failed: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Backend server offline! Ensure `python app.py` is running on port 8000.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please select at least one PDF file first.")

    st.divider()
    st.markdown("""
    **Architecture:**
    - 🧭 **LangGraph Supervisor**
    - 🔍 **Search Agent (RAG)**
    - 👁️ **Vision Agent (Gemini Flash)**
    - ⚡ **FastAPI Backend**
    """)

# ============================================================
# MAIN CHAT INTERFACE
# ============================================================
st.title("🧠 OmniBrain: Multimodal Financial RAG")
st.caption("LangGraph Supervisor • Multimodal Store • Vision & RAG Agents")

# Chat history initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("route"):
            st.markdown(
                f"<span class='badge-supervisor'>[Supervisor] Routed to: {msg['route'].upper()} Agent</span>",
                unsafe_allow_html=True
            )
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("Ask about financial trends, visual charts, or stock metrics..."):
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process via Backend API
    with st.chat_message("assistant"):
        with st.spinner("Analyzing request and orchestrating agents..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/query",
                    json={"question": prompt},
                    timeout=120
                )

                if response.status_code == 200:
                    result = response.json()
                    answer = result.get("answer", "No response generated.")
                    route = result.get("route", "unknown")

                    # Display Routing Badge
                    st.markdown(
                        f"<span class='badge-supervisor'>[Supervisor] Routed to: {route.upper()} Agent</span>",
                        unsafe_allow_html=True
                    )

                    # Display Formatted Answer
                    st.markdown(answer)

                    # Save to state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "route": route
                    })
                else:
                    err_msg = f"Server Error ({response.status_code}): {response.text}"
                    st.error(err_msg)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})

            except requests.exceptions.ConnectionError:
                err_msg = "Could not connect to backend server. Make sure `python app.py` is running on port 8000."
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
            except requests.exceptions.Timeout:
                err_msg = "Request timed out while waiting for Vision/RAG analysis."
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
            except Exception as e:
                err_msg = f"Unexpected error: {str(e)}"
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})