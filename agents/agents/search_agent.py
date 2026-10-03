from rag.retriever import retrieve_documents


def search_agent(question, top_k=5):
    """
    Search Agent.

    Takes a user question, detects the company,
    searches the company PDF documents using RAG,
    and returns relevant information.
    """

    print("Search Agent executed")

    # Retrieve relevant documents
    results = retrieve_documents(
        question,
        top_k=top_k
    )

    # No results
    if not results:
        return {
            "agent": "search",
            "question": question,
            "answer": "No relevant information found."
        }

    information = []

    # Format retrieved results
    for i, result in enumerate(results, 1):

        company = result.get(
            "company",
            "Unknown"
        )

        source = result.get(
            "source",
            "Unknown"
        )

        score = result.get(
            "score",
            0.0
        )

        text = result.get(
            "text",
            ""
        )

        information.append(
            f"Result {i}\n"
            f"Company: {company}\n"
            f"Source: {source}\n"
            f"Similarity Score: {score:.4f}\n"
            f"Information:\n{text}"
        )

    return {
        "agent": "search",
        "question": question,
        "answer": "\n\n".join(information)
    }


# ============================================================
# TEST SEARCH AGENT
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SEARCH AGENT TEST")
    print("=" * 60)

    questions = [
        "What was Apple's revenue?",
        "What was NVIDIA's revenue?",
        "What was Netflix's revenue?",
        "What was Tesla's revenue?"
    ]

    for question in questions:

        print("\n" + "=" * 60)
        print("Question:")
        print(question)
        print("=" * 60)

        result = search_agent(
            question,
            top_k=3
        )

        print("\nSearch Agent Result:")
        print("-" * 60)

        print(
            result["answer"][:5000]
        )

    print("\n" + "=" * 60)
    print("SEARCH AGENT TEST COMPLETED")
    print("=" * 60)