import io
import json
import pymupdf
from PIL import Image

def analyze_page(client, page, page_num: int):
    # 1. Render PDF page to image
    pix = page.get_pixmap(dpi=150)
    img_bytes = pix.tobytes("png")
    image = Image.open(io.BytesIO(img_bytes))

    prompt = """Analyze this financial document page in detail.
Extract any charts, tables, or key narrative points.
Return the output strictly as a JSON array of objects with this schema:
[
  {
    "type": "CHART" | "TABLE" | "TEXT",
    "title": "string (optional)",
    "chart_type": "string (optional)",
    "description": "string",
    "legend": [{"series": "string", "color": "string"}],
    "data_points": [{"key": "value"}],
    "data": {
      "headers": ["col1", "col2"],
      "rows": [["val1", "val2"]]
    }
  }
]
Do not wrap in backticks or markdown, just return valid JSON.
"""

    # 2. Call Gemini
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[image, prompt]
    )

    text_resp = response.text.strip()
    if text_resp.startswith("```json"):
        text_resp = text_resp[7:]
    if text_resp.startswith("```"):
        text_resp = text_resp[3:]
    if text_resp.endswith("```"):
        text_resp = text_resp[:-3]
    text_resp = text_resp.strip()

    # 3. Recursive Unpack to ensure flat list of dicts
    def flatten_elements(raw):
        out = []
        if isinstance(raw, list):
            for item in raw:
                out.extend(flatten_elements(item))
        elif isinstance(raw, dict):
            # Check for wrapper dicts like {"data": [...]}
            found_sublist = False
            for k in ["elements", "data", "results", "items"]:
                if k in raw and isinstance(raw[k], list):
                    out.extend(flatten_elements(raw[k]))
                    found_sublist = True
                    break
            if not found_sublist:
                out.append(raw)
        elif isinstance(raw, str):
            try:
                sub = json.loads(raw)
                out.extend(flatten_elements(sub))
            except Exception:
                out.append({"type": "TEXT", "description": raw})
        return out

    try:
        parsed = json.loads(text_resp)
        return flatten_elements(parsed)
    except Exception:
        return [{"type": "TEXT", "description": text_resp}]