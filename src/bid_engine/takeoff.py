"""Multi-step takeoff agent — decompose raw-plan reading into focused sub-calls.

The single-call extraction (`bid_engine.extraction`) asks Claude to do five
distinct cognitive jobs in one prompt: find the legend, recognize codes,
identify quantity columns, count placed instances, measure geometric regions.
Empirically that produces hallucinated quantities on raw engineer drawings
(see the park_ave_original baseline: 0/42 match rate, identical "default"
values copy-pasted across four pages).

This module decomposes the task into independent calls, each with a tight
prompt and the model's full attention on one sub-problem:

    extract_scale       — "What is the drawing scale on this sheet?"
    extract_dimensions  — "List every annotated dimension on this page."
    extract_keynotes    — "Find the keynote legend; return code → description."
    (TODO step 2d)      — per-code count / measure with scale + dimensions in context
    (TODO aggregate)    — combine across pages into ScopeItem objects

Current state: 2a / 2b / 2c are implemented. The orchestrator and per-code
quantity probe are stubbed pending validation of the steps above.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anthropic import Anthropic

from bid_engine.extraction import render_page_png

logger = logging.getLogger(__name__)

MODEL = "claude-opus-4-7"
MAX_TOKENS = 8000


@dataclass(frozen=True)
class PageProbe:
    """Captured outputs from steps 2a-2c on one page."""
    page: int
    scale: Any
    dimensions: list[dict]
    keynotes: list[dict]


# ---------------------------------------------------------------------------
# Step 2a — drawing scale
# ---------------------------------------------------------------------------

SCALE_PROMPT = (
    "This is a construction elevation drawing. Find the drawing scale.\n"
    "The scale is usually printed in the title block (typically lower-right corner) "
    "or directly under each individual drawing, in the form '1/8\" = 1'-0\"' or "
    "'1/4\" = 1'-0\"' or similar.\n"
    "Return ONLY a JSON object with these keys:\n"
    "  raw: the scale exactly as printed (string)\n"
    "  drawing_inches: the drawing-side measurement converted to decimal inches "
    "(e.g. 0.125 for 1/8\", 0.25 for 1/4\")\n"
    "  real_feet: the real-world measurement in feet (typically 1.0)\n"
    "  notes: brief note on where you found it or any ambiguity (e.g. multiple "
    "drawings at different scales)\n"
    "If the page has multiple drawings at different scales, return a JSON array "
    "of these objects instead. No other text."
)


# ---------------------------------------------------------------------------
# Step 2b — annotated dimensions
# ---------------------------------------------------------------------------

DIMENSIONS_PROMPT = (
    "This is a construction elevation drawing. List every annotated dimension you "
    "can see on the page.\n"
    "Dimensions are typically shown with extension lines and tick marks (or arrows) "
    "with a number floating above or beside them, in feet-and-inches notation "
    "(e.g. 8'-6\") or just feet (24'-0\"). They appear between grid lines, between "
    "features (windows, doors, parapets, columns), or labeling the size of "
    "specific elements.\n"
    "Return ONLY a JSON array. Each element has:\n"
    "  raw: the dimension exactly as printed (e.g. \"8'-6\\\"\")\n"
    "  value_feet: dimension converted to decimal feet (e.g. 8'-6\" -> 8.5)\n"
    "  label: what the dimension is annotating, e.g. "
    "\"grid A to grid B\", \"window opening width\", \"floor-to-floor height\"\n"
    "Skip dimensions that are part of the title block or scale legend. "
    "No other text."
)


# ---------------------------------------------------------------------------
# Step 2c — keynote legend
# ---------------------------------------------------------------------------

KEYNOTES_PROMPT = (
    "This is a construction elevation drawing. Find the keynote legend or work "
    "scope notes section. The legend lists each code with its full description, "
    "e.g. 'WS1: Clean cast stone bands, repoint and cover all horizontal & "
    "vertical joints'.\n"
    "Codes can be any alphanumeric pattern (WS1, R02, E03, F05, M1, etc.).\n"
    "Return ONLY a JSON array where each element has:\n"
    "  code: the code exactly as printed\n"
    "  description: the full description text, including any sub-references\n"
    "Skip the title block, the drawing index, dimension labels, and the symbol "
    "legend (which uses graphic symbols, not codes). If no keynote legend is "
    "visible on this page, return an empty array. No other text."
)


# ---------------------------------------------------------------------------
# Wire helpers
# ---------------------------------------------------------------------------


def _call_with_image(prompt: str, png_bytes: bytes, *, max_tokens: int = MAX_TOKENS) -> str:
    """Single API round-trip: image + prompt → text response."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set in the environment")
    client = Anthropic()
    b64 = base64.standard_b64encode(png_bytes).decode("ascii")
    msg = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
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
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    for block in msg.content:
        if block.type == "text":
            return block.text.strip()
    return ""


def _parse_json(text: str) -> Any:
    """Strip code fences and parse JSON. Raises ValueError on malformed output."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"model returned non-JSON output: {text[:300]!r}"
        ) from exc


# ---------------------------------------------------------------------------
# Public step functions
# ---------------------------------------------------------------------------


def extract_scale(png_bytes: bytes) -> Any:
    """Step 2a: drawing scale as object or list of objects."""
    return _parse_json(_call_with_image(SCALE_PROMPT, png_bytes))


def extract_dimensions(png_bytes: bytes) -> list[dict]:
    """Step 2b: list of annotated dimensions."""
    result = _parse_json(_call_with_image(DIMENSIONS_PROMPT, png_bytes))
    if not isinstance(result, list):
        raise ValueError(f"extract_dimensions: expected JSON array, got {type(result).__name__}")
    return result


def extract_keynotes(png_bytes: bytes) -> list[dict]:
    """Step 2c: list of keynote (code, description) pairs."""
    result = _parse_json(_call_with_image(KEYNOTES_PROMPT, png_bytes))
    if not isinstance(result, list):
        raise ValueError(f"extract_keynotes: expected JSON array, got {type(result).__name__}")
    return result


def probe_page(pdf_path: Path | str, page: int) -> PageProbe:
    """Run steps 2a-2c on one page. Used for iteration / validation; the
    full orchestrator (steps 2d + aggregation) is not implemented yet.
    """
    png = render_page_png(pdf_path, page)
    logger.info("rendered page %d (%d KB)", page, len(png) // 1024)
    scale = extract_scale(png)
    logger.info("step 2a (scale) → %s", scale)
    dimensions = extract_dimensions(png)
    logger.info("step 2b (dimensions) → %d items", len(dimensions))
    keynotes = extract_keynotes(png)
    logger.info("step 2c (keynotes) → %d items", len(keynotes))
    return PageProbe(page=page, scale=scale, dimensions=dimensions, keynotes=keynotes)


# ---------------------------------------------------------------------------
# CLI probe
# ---------------------------------------------------------------------------


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", help="Path to the plan PDF")
    parser.add_argument("page", type=int, help="1-indexed page number")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    probe = probe_page(args.pdf, args.page)

    print("\n===== SCALE =====")
    print(json.dumps(probe.scale, indent=2))
    print("\n===== DIMENSIONS =====")
    print(json.dumps(probe.dimensions, indent=2))
    print(f"({len(probe.dimensions)} dimensions extracted)")
    print("\n===== KEYNOTES =====")
    print(json.dumps(probe.keynotes, indent=2))
    print(f"({len(probe.keynotes)} keynotes extracted)")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
