from rag.pdf_loader import load_pdf


def chunk_text(text, chunk_size=800, overlap=100):
    """
    Split text into overlapping chunks.
    """

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start += chunk_size - overlap

    return chunks


if __name__ == "__main__":

    print("=" * 60)
    print("TEXT CHUNKER TEST")
    print("=" * 60)

    sample_text = """
    Apple reported its annual financial results for fiscal year 2024.
    The company generated significant revenue from its products and services.
    """

    chunks = chunk_text(sample_text)

    print("Number of chunks:", len(chunks))

    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(chunk)