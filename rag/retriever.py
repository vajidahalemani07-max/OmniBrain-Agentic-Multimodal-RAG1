import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from rag.vector_store import load_vector_store
from rag.embeddings import model


# ============================================================
# COMPANY DETECTION
# ============================================================

def detect_company(query):
    """
    Detect company name from the user's question.
    """

    query_lower = query.lower()

    if "apple" in query_lower:
        return "Apple"

    if "netflix" in query_lower:
        return "Netflix"

    if "nvidia" in query_lower:
        return "NVIDIA"

    if "tesla" in query_lower:
        return "Tesla"

    return None


# ============================================================
# DOCUMENT RETRIEVAL
# ============================================================

def retrieve_documents(query, top_k=5):
    """
    Retrieve the most relevant PDF chunks.

    If a company is mentioned in the question,
    search only that company's documents.
    """

    # Load vector store
    vector_store = load_vector_store()

    chunks = vector_store["chunks"]
    embeddings = vector_store["embeddings"]

    # Detect company
    company = detect_company(query)

    print(f"Detected company: {company}")

    # --------------------------------------------------------
    # Filter chunks by company
    # --------------------------------------------------------

    candidate_indexes = []

    for i, chunk in enumerate(chunks):

        if isinstance(chunk, dict):

            chunk_company = chunk.get("company", "")

            if company is None or chunk_company == company:
                candidate_indexes.append(i)

        else:
            # Compatibility with old vector stores
            if company is None:
                candidate_indexes.append(i)

    if not candidate_indexes:

        return []

    # --------------------------------------------------------
    # Create query embedding
    # --------------------------------------------------------

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    # Only compare against filtered documents
    filtered_embeddings = embeddings[candidate_indexes]

    scores = cosine_similarity(
        query_embedding,
        filtered_embeddings
    )[0]

    # --------------------------------------------------------
    # Get top results
    # --------------------------------------------------------

    top_positions = np.argsort(scores)[::-1][:top_k]

    results = []

    for position in top_positions:

        original_index = candidate_indexes[position]

        chunk = chunks[original_index]

        if isinstance(chunk, dict):

            results.append({
                "company": chunk.get("company", "Unknown"),
                "source": chunk.get("source", "Unknown"),
                "text": chunk.get("text", ""),
                "score": float(scores[position])
            })

        else:

            results.append({
                "company": "Unknown",
                "source": "Unknown",
                "text": chunk,
                "score": float(scores[position])
            })

    return results


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("RETRIEVER TEST")
    print("=" * 60)

    questions = [
        "What was Apple's revenue?",
        "What was NVIDIA's revenue?",
        "What was Tesla's revenue?",
        "What was Netflix's revenue?"
    ]

    for question in questions:

        print("\n" + "=" * 60)
        print("Question:")
        print(question)
        print("=" * 60)

        results = retrieve_documents(
            question,
            top_k=3
        )

        print("\nTop relevant chunks:")

        for i, result in enumerate(results, 1):

            print("\n" + "-" * 60)
            print(f"Result {i}")
            print(f"Company: {result['company']}")
            print(f"Source: {result['source']}")
            print(f"Similarity Score: {result['score']:.4f}")
            print("-" * 60)

            print(result["text"][:1000])

    print("\n" + "=" * 60)
    print("RETRIEVER TEST COMPLETED")
    print("=" * 60)