from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


def create_embeddings(texts):
    """
    Convert text into numerical embeddings.
    """

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    return embeddings


if __name__ == "__main__":

    sample_texts = [
        "Apple reported its financial results.",
        "Apple revenue increased during the year.",
        "Tesla reported vehicle deliveries."
    ]

    embeddings = create_embeddings(sample_texts)

    print("=" * 60)
    print("EMBEDDING TEST")
    print("=" * 60)

    print("Number of texts:", len(sample_texts))
    print("Embedding shape:", embeddings.shape)
    print("Embedding dimension:", embeddings.shape[1])