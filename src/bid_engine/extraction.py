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
from pdf2image import convert_from_path, pdfinfo_from_path

logger = logging.getLogger(__name__)

MODEL = "claude-opus-4-7"

# Claude vision rejects images whose long edge exceeds this many pixels.
MAX_IMAGE_PIXELS = 8000
_PDF_POINTS_PER_INCH = 72
PROMPT = (
    "This is a construction elevation or plan drawing. Look for any keynote codes, "
    "scope item codes, or annotation labels with associated quantities. These could "
    "follow any pattern — examples include WS1, E03, R02, F05, M1, or any letters "
    "followed by numbers. Also look for any sidebar or legend showing codes with "
    "quantities and units (EA, FT, SQ FT, LF, SF, LOC, LS).\n"
    "Extract every code-quantity pair you can find on this page. If there is a "
    "keynote legend or work scope notes section, extract each code with its full "
    "description.\n"
    "If a code looks like a single letter followed by digits (e.g. W9), it may be a "
    "misread of a two-letter code (e.g. WS9) — flag it but include it as-read.\n"
    "Return ONLY a JSON array where each element has: code (string), quantity "
    "(number), unit (string). If a code appears in the keynotes but has no quantity "
    "on this page, include it with quantity 0. No other text."
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
        # The prompt asks for keynote-only codes to be returned with qty 0 so
        # downstream code can see the full keynote dictionary. Only drop
        # negative quantities (which are unambiguously bogus).
        if qty < 0:
            continue
        cleaned.append({"code": code, "quantity": qty, "unit": unit})
    return cleaned


def _safe_dpi(pdf_path: Path, requested_dpi: int, max_pixels: int) -> int:
    """Clamp DPI so the rendered image's long edge stays under `max_pixels`.

    Large architectural sheets (Arch D 24x36, Arch E 36x48) exceed Claude's
    8000-pixel limit at 300 DPI. We honor the caller's requested DPI when
    safe, and otherwise back off to the largest DPI that fits.
    """
    info = pdfinfo_from_path(str(pdf_path))
    raw_size = info.get("Page size", "")  # e.g. "2592 x 1728 pts" or "792 x 612 pts (letter)"
    # Pull the first two numbers; ignore any trailing format hint like "(letter)".
    numbers = re.findall(r"\d+(?:\.\d+)?", raw_size)
    if len(numbers) < 2:
        logger.warning("could not parse page size %r — using requested DPI %d",
                       raw_size, requested_dpi)
        return requested_dpi
    long_inches = max(float(numbers[0]), float(numbers[1])) / _PDF_POINTS_PER_INCH
    if long_inches <= 0:
        return requested_dpi
    max_dpi = int(max_pixels / long_inches)
    if max_dpi < requested_dpi:
        logger.info(
            "page is %.1f\" long edge; clamping DPI %d → %d to stay under %d-px limit",
            long_inches, requested_dpi, max_dpi, max_pixels,
        )
        return max_dpi
    return requested_dpi


def render_page_png(
    pdf_path: Path | str,
    page_number: int,
    dpi: int = 300,
    *,
    clamp_dpi: bool = True,
) -> bytes:
    """Render one PDF page to PNG bytes.

    `clamp_dpi=True` (default) auto-reduces DPI so the rendered long edge stays
    under Claude's 8000-pixel image limit. Set `clamp_dpi=False` when the caller
    will crop the result before sending to the API (e.g. sliding window) — the
    individual crops will fit even when the full render does not.
    """
    if page_number < 1:
        raise ValueError(f"page_number must be >= 1, got {page_number}")
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    effective_dpi = _safe_dpi(pdf_path, dpi, MAX_IMAGE_PIXELS) if clamp_dpi else dpi
    images = convert_from_path(
        str(pdf_path),
        dpi=effective_dpi,
        first_page=page_number,
        last_page=page_number,
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
