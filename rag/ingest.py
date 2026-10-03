import os
import numpy as np

from rag.pdf_loader import load_pdf
from rag.chunker import chunk_text
from rag.embeddings import create_embeddings
from rag.vector_store import create_vector_store


DATA_DIR = "data"


def get_company_name(filename):
    """
    Get company name from PDF filename.
    """

    filename_lower = filename.lower()

    if "apple" in filename_lower:
        return "Apple"

    elif "netflix" in filename_lower:
        return "Netflix"

    elif "nvidia" in filename_lower:
        return "NVIDIA"

    elif "tesla" in filename_lower:
        return "Tesla"

    return "Unknown"


def ingest_documents():

    all_chunks = []

    print("=" * 60)
    print("DOCUMENT INGESTION")
    print("=" * 60)

    pdf_files = [
        file for file in os.listdir(DATA_DIR)
        if file.lower().endswith(".pdf")
    ]

    print(f"PDF files found: {len(pdf_files)}")

    for pdf_file in pdf_files:

        pdf_path = os.path.join(DATA_DIR, pdf_file)

        company = get_company_name(pdf_file)

        print("\n" + "-" * 60)
        print(f"Processing: {pdf_file}")
        print(f"Company: {company}")

        text = load_pdf(pdf_path)

        print(f"Characters: {len(text)}")

        chunks = chunk_text(text)

        print(f"Chunks: {len(chunks)}")

        # Add company information to every chunk
        for chunk in chunks:

            all_chunks.append({
                "company": company,
                "source": pdf_file,
                "text": chunk
            })

    print("\n" + "=" * 60)
    print(f"TOTAL CHUNKS: {len(all_chunks)}")
    print("=" * 60)

    # Extract only text for embedding
    texts = [
        chunk["text"]
        for chunk in all_chunks
    ]

    print("\nGenerating embeddings...")

    embeddings = create_embeddings(texts)

    embeddings = np.array(embeddings)

    print(f"Embedding shape: {embeddings.shape}")

    print("\nSaving vector store...")

    create_vector_store(
        all_chunks,
        embeddings
    )

    print("\n" + "=" * 60)
    print("INGESTION COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    ingest_documents()