import os
import pickle
import numpy as np

# Directory of this file (rag folder)
RAG_DIR = os.path.dirname(os.path.abspath(__file__))
# Absolute path to vector_store.pkl (always points to project_root/rag/vector_store.pkl)
VECTOR_STORE_PATH = os.path.join(RAG_DIR, "vector_store.pkl")


def create_vector_store(chunks, embeddings):
    """
    Create and save the vector store.

    Each chunk contains:
    - company
    - source
    - text
    """

    vector_store = {
        "chunks": chunks,
        "embeddings": np.array(embeddings)
    }

    with open(VECTOR_STORE_PATH, "wb") as f:
        pickle.dump(vector_store, f)

    print("=" * 60)
    print("VECTOR STORE CREATED")
    print("=" * 60)

    print(f"Chunks stored: {len(chunks)}")
    print(f"Embedding shape: {np.array(embeddings).shape}")
    print(f"Saved to: {VECTOR_STORE_PATH}")

    # Show company information
    companies = {}

    for chunk in chunks:
        if isinstance(chunk, dict):
            company = chunk.get("company", "Unknown")
            if company not in companies:
                companies[company] = 0
            companies[company] += 1

    if companies:
        print("\nChunks by company:")
        for company, count in companies.items():
            print(f"{company}: {count}")


def load_vector_store():
    """
    Load the saved vector store.
    """

    if not os.path.exists(VECTOR_STORE_PATH):
        raise FileNotFoundError(
            f"Vector store not found at {VECTOR_STORE_PATH}. "
            "Run: python -m rag.ingest"
        )

    with open(VECTOR_STORE_PATH, "rb") as f:
        vector_store = pickle.load(f)

    return vector_store


if __name__ == "__main__":

    # Test data
    chunks = [
        {
            "company": "Apple",
            "source": "apple_10k_2024.pdf",
            "text": "Apple reported strong revenue in fiscal year 2024."
        },
        {
            "company": "Apple",
            "source": "apple_10k_2024.pdf",
            "text": "Apple provides iPhone, Mac and services products."
        },
        {
            "company": "Apple",
            "source": "apple_10k_2024.pdf",
            "text": "Apple annual reports contain financial information."
        }
    ]

    embeddings = np.random.rand(3, 384)

    create_vector_store(
        chunks,
        embeddings
    )

    store = load_vector_store()

    print("\nTesting vector store...")
    print("Number of chunks:", len(store["chunks"]))
    print("Embedding shape:", store["embeddings"].shape)
    print("\nFirst chunk:")
    print(store["chunks"][0])
    print("\nVector store test successful!")