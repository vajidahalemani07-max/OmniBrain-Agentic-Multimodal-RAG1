import os
import json
import pymupdf
from google import genai

from vision.image_processor import analyze_pdf
from vision.chart_analyzer import extract_charts


DEFAULT_PDF = "data/Tesla_10K_2023.pdf"
MODEL = "gemini-3.6-flash"


def vision_agent(state):
    """
    Vision agent for financial PDF questions.

    It:
    1. Gets the question from AgentState.
    2. Uses the supplied PDF path if available.
    3. Finds candidate pages using PDF text.
    4. Sends candidate pages through the existing Vision pipeline.
    5. Extracts charts from those pages.
    6. Uses Gemini to select the chart relevant to the question.
    7. Returns structured vision_results.
    """

    question = state.get("question", "")

    pdf_path = state.get("pdf_path", DEFAULT_PDF)

    document_id = state.get(
        "document_id",
        os.path.basename(pdf_path)
    )

    page = state.get("page")

    if page is None:
        page = state.get("page_number")

    if not os.path.exists(pdf_path):
        return [{
            "document": document_id,
            "page": page,
            "question": question,
            "answer": "",
            "chart_data": "",
            "analysis": f"PDF not found: {pdf_path}"
        }]

    # ---------------------------------------------------------
    # CASE 1: Specific page supplied
    # ---------------------------------------------------------

    if page is not None:

        page_results = analyze_pdf(
            pdf_path=pdf_path,
            document_id=document_id,
            start_page=page,
            end_page=page
        )

        return select_relevant_chart(
            page_results,
            question,
            document_id
        )

    # ---------------------------------------------------------
    # CASE 2: No specific page supplied
    # Search PDF text for candidate pages.
    # ---------------------------------------------------------

    doc = pymupdf.open(pdf_path)

    candidate_pages = []

    question_words = [
        word.lower().strip(".,?!")
        for word in question.split()
        if len(word) > 3
    ]

    for page_index, pdf_page in enumerate(doc):

        text = pdf_page.get_text().lower()

        if any(word in text for word in question_words):
            candidate_pages.append(page_index + 1)

    page_count = len(doc)
    doc.close()

    # If text search doesn't find anything,
    # check the first few pages as a fallback.
    if not candidate_pages:

        candidate_pages = list(
            range(1, min(6, page_count + 1))
        )

    all_page_results = []

    for candidate_page in candidate_pages:

        page_results = analyze_pdf(
            pdf_path=pdf_path,
            document_id=document_id,
            start_page=candidate_page,
            end_page=candidate_page
        )

        all_page_results.extend(page_results)

    # ---------------------------------------------------------
    # Select the chart that actually answers the question.
    # ---------------------------------------------------------

    return select_relevant_chart(
        all_page_results,
        question,
        document_id
    )


def select_relevant_chart(
    page_results,
    question,
    document_id
):
    """
    Select the chart that is most relevant to the user's question.
    """

    candidates = []

    for page_result in page_results:

        page_number = page_result.get("page")

        elements = page_result.get("elements", [])

        charts = extract_charts(elements)

        for chart in charts:

            candidates.append({
                "page": page_number,
                "chart": chart
            })

    # No charts found.
    if not candidates:

        return [{
            "document": document_id,
            "page": None,
            "question": question,
            "answer": "",
            "chart_data": "",
            "analysis": "No relevant chart was found."
        }]

    # If there is only one chart, use it directly.
    if len(candidates) == 1:

        return format_result(
            candidates[0],
            question,
            document_id
        )

    # ---------------------------------------------------------
    # Ask Gemini which chart answers the question.
    # ---------------------------------------------------------

    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"]
    )

    chart_descriptions = []

    for index, candidate in enumerate(candidates):

        chart_descriptions.append({
            "index": index,
            "page": candidate["page"],
            "chart": candidate["chart"]
        })

    selection_prompt = f"""
You are selecting the most relevant chart from a financial report.

User question:
{question}

Below are candidate charts extracted from the PDF:

{json.dumps(chart_descriptions, indent=2)}

Choose the ONE chart that best answers the user's question.

Do not choose a chart merely because it is a chart.
Choose the chart whose title, labels, data, or description actually
relates to the question.

Return ONLY valid JSON in this exact format:

{{
    "selected_index": 0,
    "reason": "brief explanation"
}}

The selected_index must be one of the candidate indexes.
"""

    for attempt in range(3):

        try:

            print(
                f"Chart selection attempt {attempt + 1}...",
                flush=True
            )

            response = client.models.generate_content(
                model=MODEL,
                contents=selection_prompt
            )

            text = response.text.strip()

            text = text.replace("```json", "")
            text = text.replace("```", "")
            text = text.strip()

            selection = json.loads(text)

            selected_index = selection.get(
                "selected_index"
            )

            if (
                isinstance(selected_index, int)
                and 0 <= selected_index < len(candidates)
            ):

                selected = candidates[selected_index]

                result = format_result(
                    selected,
                    question,
                    document_id
                )

                result[0]["analysis"] = (
                    f"{result[0]['analysis']} "
                    f"Chart selection reason: "
                    f"{selection.get('reason', '')}"
                )

                return result

        except Exception as e:

            print(
                f"Chart selection error: {e}",
                flush=True
            )

    # ---------------------------------------------------------
    # Fallback if Gemini selection fails.
    # ---------------------------------------------------------

    return format_result(
        candidates[0],
        question,
        document_id
    )


def format_result(
    candidate,
    question,
    document_id
):
    """
    Convert one selected chart into the required
    vision_results format.
    """

    chart = candidate["chart"]

    description = chart.get(
        "description",
        ""
    )

    return [{
        "document": document_id,
        "page": candidate["page"],
        "question": question,
        "answer": description,
        "chart_data": chart,
        "analysis": description
    }]