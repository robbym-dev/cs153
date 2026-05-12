"""Vision extraction: marked-up plan PDF → list of scope items.

Renders one PDF page to a 300 DPI PNG and asks Claude Opus 4.7 (adaptive
thinking + high effort) to read the sidebar of scope codes / quantities /
units. The model output goes through `_clean()` which:

  * fixes single-letter prefix OCR slips (W9 → WS9)
  * drops zero / negative quantities (legend placeholders)
  * coerces qty to float and skips malformed rows with a logged warning

Public surface:
    extract_page(pdf_path, page_number) -> list[{"code","quantity","unit"}]
    render_page_png(pdf_path, page_number, dpi=300) -> bytes
    extract_items_from_image(png_bytes) -> list[{"code","quantity","unit"}]
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
from pathlib import Path

from anthropic import Anthropic
from pdf2image import convert_from_path

logger = logging.getLogger(__name__)

MODEL = "claude-opus-4-7"
PROMPT = (
    "This is a marked-up construction elevation drawing. On the right side there is "
    "a sidebar showing scope item codes (like WS1, WS5, WS8) with quantities and units. "
    "Extract every item from the sidebar. "
    "Scope codes follow the pattern WS followed by a number (WS1, WS5, WS10, WS19) "
    "or R followed by a number (R1, R2, R5). There are no single-letter prefix codes — "
    "if you see what looks like W9, it is WS9. "
    "Return ONLY a JSON array where each element has: "
    "code (string), quantity (number), unit (string — one of EA, FT, SQ FT, LF). "
    "No other text."
)

# Single-letter prefix variants the model occasionally emits.
SINGLE_LETTER_PREFIX_FIXUPS = {"W": "WS"}
_SINGLE_LETTER_CODE_RE = re.compile(r"^([A-Z])(\d+)$")


def _normalize_code(code: str) -> str:
    m = _SINGLE_LETTER_CODE_RE.match(code)
    if m and m.group(1) in SINGLE_LETTER_PREFIX_FIXUPS:
        return SINGLE_LETTER_PREFIX_FIXUPS[m.group(1)] + m.group(2)
    return code


def _clean(raw_items: list[dict]) -> list[dict]:
    cleaned: list[dict] = []
    for raw in raw_items:
        try:
            code = _normalize_code(str(raw["code"]).strip())
            qty = float(raw["quantity"])
            unit = str(raw["unit"]).strip()
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("malformed extraction item %r (%s) — skipping", raw, exc)
            continue
        if not code or not unit:
            logger.warning("extraction item missing code/unit %r — skipping", raw)
            continue
        if qty <= 0:
            continue
        cleaned.append({"code": code, "quantity": qty, "unit": unit})
    return cleaned


def render_page_png(pdf_path: Path | str, page_number: int, dpi: int = 300) -> bytes:
    if page_number < 1:
        raise ValueError(f"page_number must be >= 1, got {page_number}")
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    images = convert_from_path(
        str(pdf_path), dpi=dpi, first_page=page_number, last_page=page_number
    )
    if not images:
        raise ValueError(f"no page {page_number} in {pdf_path}")
    buf = io.BytesIO()
    images[0].save(buf, format="PNG")
    return buf.getvalue()


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


def extract_items_from_image(png_bytes: bytes) -> list[dict]:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set in the environment")
    client = Anthropic()
    b64 = base64.standard_b64encode(png_bytes).decode("ascii")
    message = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive", "display": "summarized"},
        output_config={"effort": "high"},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
    )
    text = ""
    for block in message.content:
        if block.type == "text":
            text = block.text
            break
    text = _strip_code_fences(text)
    try:
        raw_items = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"vision API returned non-JSON output: {text[:200]!r}"
        ) from exc
    if not isinstance(raw_items, list):
        raise ValueError(
            f"expected JSON array from vision API, got {type(raw_items).__name__}"
        )
    return _clean(raw_items)


def extract_page(pdf_path: Path | str, page_number: int) -> list[dict]:
    """Render one PDF page and run vision extraction; return cleaned items."""
    png = render_page_png(pdf_path, page_number)
    return extract_items_from_image(png)
