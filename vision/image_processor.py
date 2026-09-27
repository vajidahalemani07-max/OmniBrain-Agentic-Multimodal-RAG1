from google import genai
import os
import json
import time
import pymupdf

from vision.image_extractor import render_page

MODEL = "gemini-3.6-flash"

PROMPT = """
Analyze this entire financial-report page for a multimodal RAG pipeline.

Identify every distinct element on the page.

Allowed element types:
- TEXT
- TABLE
- CHART
- IMAGE
- DIAGRAM

For each element return:
- type
- approximate location
- description

For TABLE:
- extract visible data
- preserve rows and columns
- preserve headers and units

For CHART:
- identify chart type
- identify title
- identify axes and labels
- identify legend
- extract visible data points where possible
- preserve units

For IMAGE or DIAGRAM:
- describe what is shown
- extract important visible labels

For TEXT:
- describe or extract readable text.

Do not invent information.

Return ONLY valid JSON.
"""


def analyze_page(client, page, page_number):

    # Render PDF page, send it to gemini and return the analysis
    image_path = render_page(page,page_number)
    # Upload image
    image = client.files.upload(
        file=image_path
    )

    # Retry temporary Gemini failures
    for attempt in range(3):

        try:
            print(
                f"Gemini attempt {attempt + 1}...",
                flush=True
            )

            response = client.models.generate_content(
                model=MODEL,
                contents=[
                    image,
                    PROMPT
                ]
            )

            text = response.text

            # Remove markdown JSON fences
            text = text.replace("```json", "")
            text = text.replace("```", "")
            text = text.strip()

            try:
                return json.loads(text)

            except json.JSONDecodeError:
                return {
                    "raw_response": text
                }

        except Exception as e:

            print(
                f"Gemini error: {e}",
                flush=True
            )

            if attempt < 2:
                time.sleep(10)

    return {
        "error": "Gemini unavailable after 3 attempts"
    }

def analyze_pdf(
    pdf_path,
    document_id,
    start_page=1,
    end_page=None
):
#analyze selected pages of a pdf
    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"]
    )

    doc = pymupdf.open(pdf_path)

    if end_page is None:
        end_page = len(doc)

    results = []

    for page_index in range(start_page - 1,end_page):

        page_number = page_index + 1

        print(f"\nProcessing page {page_number}...",flush=True)

        page_result = analyze_page(
            client,
            doc[page_index],
            page_number
        )

        results.append({
            "document_id": document_id,
            "page": page_number,
            "elements": page_result
        })

        print(
            f"Page {page_number} complete.",
            flush=True
        )

    doc.close()

    return results