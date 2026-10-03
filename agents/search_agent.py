import os
import sys

# Project root path add kar rahe hain
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from rag.retriever import retrieve_documents

def search_agent(question: str, top_k: int = 5):
    """
    Real RAG Search Agent jo direct retrieve_documents() ko call karta hai.
    """
    try:
        results = retrieve_documents(question, top_k=top_k)
        
        if not results:
            return {"answer": "No relevant documents or chunks found in vector store."}
        
        # Extracted text chunks ko compile kar rahe hain
        context_parts = []
        for i, item in enumerate(results, 1):
            text_content = item.get("text", "").strip()
            if text_content:
                context_parts.append(f"[Excerpt {i}]:\n{text_content}")
                
        compiled_context = "\n\n".join(context_parts)
        return {"answer": compiled_context}
        
    except Exception as e:
        return {"answer": f"Error during document retrieval: {str(e)}"}