"""Standalone test: extract sidebar scope items from one page of a marked-up plan PDF."""

import argparse
import base64
import io
import json
import os
import sys

from anthropic import Anthropic
from pdf2image import convert_from_path

MODEL = "claude-opus-4-7"
PROMPT = (
    "This is a marked-up construction elevation drawing. On the right side there is "
    "a sidebar showing scope item codes (like WS1, WS5, WS8) with quantities and units. "
    "Extract every item from the sidebar. Return ONLY a JSON array where each element has: "
    "code (string), quantity (number), unit (string — one of EA, FT, SQ FT, LF). "
    "No other text."
)


def render_page_png(pdf_path: str, page_number: int, dpi: int = 300) -> bytes:
    images = convert_from_path(
        pdf_path, dpi=dpi, first_page=page_number, last_page=page_number
    )
    if not images:
        raise ValueError(f"No page {page_number} in {pdf_path}")
    buf = io.BytesIO()
    images[0].save(buf, format="PNG")
    return buf.getvalue()


def extract_items(png_bytes: bytes) -> list[dict]:
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
        if block.type == "thinking":
            continue
        if block.type == "text":
            text = block.text
            break
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", help="Path to the marked-up plan PDF")
    parser.add_argument("page", type=int, help="1-indexed page number to extract")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set", file=sys.stderr)
        return 2

    png_bytes = render_page_png(args.pdf, args.page)
    items = extract_items(png_bytes)

    for item in items:
        print(f"{item['code']}\t{item['quantity']}\t{item['unit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
