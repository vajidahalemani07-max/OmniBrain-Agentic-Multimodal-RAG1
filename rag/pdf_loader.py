from pathlib import Path
import fitz

DATA_DIR = Path("data")


def load_pdf(pdf_path):
    doc = fitz.open(str(pdf_path))
    pages = []

    for page in doc:
        text = page.get_text()
        if text:
            pages.append(text)

    doc.close()
    return "\n".join(pages)


if __name__ == "__main__":
    pdf_files = list(DATA_DIR.glob("*.pdf"))

    print("=" * 60)
    print("PDF LOADER TEST")
    print("=" * 60)

    print(f"PDF files found: {len(pdf_files)}")

    for pdf in pdf_files:
        print(f"\nReading: {pdf.name}")

        text = load_pdf(pdf)

        print(f"Characters extracted: {len(text)}")
        print("First 300 characters:")
        print(text[:300])
        print("-" * 60)