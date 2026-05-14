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
import io
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from PIL import Image

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
# Keynote-description quantity parser
# ---------------------------------------------------------------------------
#
# Engineers sometimes write quantity hints directly into keynote descriptions:
#   "PATCH HOLES IN FACE BRICK (APPROX. QUANTITY: EIGHT (8) BRICKS)"
#   "REPOINT JOINTS IN DELINEATED AREAS (±100 LF)"
#   "(BASE BID QUANTITY: TWENTY-ONE (21) LINTELS, ALTERNATE 2 QUANTITY: FOUR (4) LINTELS)"
# This parser pulls those out as structured data without an API call.


@dataclass(frozen=True)
class ParsedQuantity:
    code: str
    quantity: float
    unit: str
    variant: str    # "base_bid" | "alternate_<N>"
    raw: str        # the matched substring


# Variant markers — require a trailing ":" so we don't trigger on the bare word
# "QUANTITY" in unrelated contexts.
_VARIANT_RE = re.compile(
    r"(BASE\s+BID\s+QUANTITY|ALTERNATE\s+(\d+)\s+QUANTITY|"
    r"APPROX\.?\s+QUANTITY|QUANTITY)(?=\s*:)",
    re.IGNORECASE,
)

# "(21) LINTELS" — numeric in parens followed by a noun
_PAREN_NUM_NOUN_RE = re.compile(
    r"\(\s*(\d+(?:\.\d+)?)\s*\)\s*([A-Z]+)",
    re.IGNORECASE,
)

# "100 LF", "±100 LF", "26 SQ FT" — bare number with explicit unit / noun
_NUM_UNIT_RE = re.compile(
    r"±?\s*(\d+(?:\.\d+)?)\s*"
    r"(LF|FT|LIN\s+FT|LINEAR\s+FT|SF|SQ\s*FT|SQFT|EA|LS|"
    r"BRICKS?|LINTELS?|COURSES?|DOORS?|WINDOWS?|OPENINGS?|JOINTS?)\b",
    re.IGNORECASE,
)

# Noun → unit hints. Nouns that map naturally to EA (countable items) or
# to a linear/area dimension based on what the work entails.
_NOUN_TO_UNIT = {
    "BRICK": "EA", "BRICKS": "EA",
    "LINTEL": "EA", "LINTELS": "EA",
    "COURSE": "EA", "COURSES": "EA",
    "DOOR": "EA", "DOORS": "EA",
    "WINDOW": "EA", "WINDOWS": "EA",
    "OPENING": "EA", "OPENINGS": "EA",
    "JOINT": "LF", "JOINTS": "LF",
}

_UNIT_TOKEN_TO_CANON = {
    "LF": "LF",
    "FT": "LF",
    "LIN FT": "LF",
    "LINEAR FT": "LF",
    "SF": "SF",
    "SQ FT": "SF",
    "SQFT": "SF",
    "EA": "EA",
    "LS": "LS",
}


def _canon_unit(token: str) -> str:
    t = " ".join(token.strip().upper().split())  # collapse whitespace
    if t in _UNIT_TOKEN_TO_CANON:
        return _UNIT_TOKEN_TO_CANON[t]
    if t in _NOUN_TO_UNIT:
        return _NOUN_TO_UNIT[t]
    return t


def parse_keynote_quantities(code: str, description: str) -> list[ParsedQuantity]:
    """Extract structured (code, quantity, unit, variant) records from a
    keynote description.

    If "BASE BID QUANTITY:" / "ALTERNATE N QUANTITY:" / "QUANTITY:" /
    "APPROX. QUANTITY:" markers are present, split on them and one record
    per segment. Otherwise scan for a bare "<number> <unit>" pattern.
    Returns [] when no quantity can be found.
    """
    if not description:
        return []

    markers = list(_VARIANT_RE.finditer(description))

    if not markers:
        results: list[ParsedQuantity] = []
        for m in _NUM_UNIT_RE.finditer(description):
            qty = float(m.group(1))
            unit = _canon_unit(m.group(2))
            results.append(ParsedQuantity(code, qty, unit, "base_bid", m.group(0)))
        return results

    results = []
    for i, m in enumerate(markers):
        end = markers[i + 1].start() if i + 1 < len(markers) else len(description)
        segment = description[m.start():end]
        if m.group(1).upper().startswith("ALTERNATE"):
            variant = f"alternate_{m.group(2)}"
        else:
            variant = "base_bid"

        # Prefer "(N) <noun>" inside the segment
        paren_match = _PAREN_NUM_NOUN_RE.search(segment)
        if paren_match:
            qty = float(paren_match.group(1))
            unit = _canon_unit(paren_match.group(2))
            results.append(ParsedQuantity(code, qty, unit, variant, paren_match.group(0)))
            continue

        # Fall back to bare "<number> <unit>"
        num_unit = _NUM_UNIT_RE.search(segment)
        if num_unit:
            qty = float(num_unit.group(1))
            unit = _canon_unit(num_unit.group(2))
            results.append(ParsedQuantity(code, qty, unit, variant, num_unit.group(0)))
    return results


# ---------------------------------------------------------------------------
# Step 2d — per-code targeted count/measure
# ---------------------------------------------------------------------------


COUNT_OR_MEASURE_PROMPT_TEMPLATE = """\
This is a construction elevation drawing. Drawing scale: {scale_raw}.
The keynote '{code}' is defined as:
  "{description}"

Find every callout of '{code}' on this elevation (typically a circular label
with a leader line pointing to the relevant feature). Do NOT include the
appearance of '{code}' in the keynote legend itself, only placed callouts on
the drawing.

Decide whether this work is best measured as:
  - EA (each / discrete items, e.g. doors, lintels, brick patches, grilles)
  - LF (linear feet, e.g. joints, bands, sills, trim)
  - SF (square feet, e.g. brick areas, wall regions)
based on what the keynote describes.

Then count or measure accordingly, using the scale and any visible
dimensions to estimate. If the description embeds an approximate quantity
(e.g. "±100 LF"), that hint is helpful but you should still verify by
looking at the drawing.

Return ONLY a JSON object:
  {{"unit": "EA" or "LF" or "SF",
    "quantity": <number>,
    "callout_count": <integer, number of placed callouts of this code>,
    "notes": "<brief note on how you arrived at the number>"}}

If you cannot see any callouts of '{code}' on this elevation, return
quantity 0 and callout_count 0.
"""


@dataclass(frozen=True)
class ProbedQuantity:
    code: str
    quantity: float
    unit: str
    callout_count: int
    notes: str


def _scale_raw(scale: Any) -> str:
    if isinstance(scale, dict):
        return str(scale.get("raw", "unspecified"))
    if isinstance(scale, list) and scale:
        return ", ".join(str(s.get("raw", "?")) for s in scale)
    return "unspecified"


def probe_code_quantity(
    png_bytes: bytes,
    code: str,
    description: str,
    scale: Any,
) -> ProbedQuantity:
    """Step 2d: focused per-code call asking the model to count or measure."""
    prompt = COUNT_OR_MEASURE_PROMPT_TEMPLATE.format(
        scale_raw=_scale_raw(scale), code=code, description=description
    )
    response = _parse_json(_call_with_image(prompt, png_bytes))
    if not isinstance(response, dict):
        raise ValueError(f"probe_code_quantity: expected JSON object, got {type(response).__name__}")
    return ProbedQuantity(
        code=code,
        quantity=float(response.get("quantity", 0)),
        unit=str(response.get("unit", "EA")).upper(),
        callout_count=int(response.get("callout_count", 0)),
        notes=str(response.get("notes", "")),
    )


# ---------------------------------------------------------------------------
# Per-page orchestrator (steps 2a-2d)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PageTakeoff:
    page: int
    scale: Any
    dimensions: list[dict]
    keynotes: list[dict]
    parsed: dict[str, list[ParsedQuantity]]   # code -> records from descriptions
    probed: dict[str, ProbedQuantity]         # code -> record from per-code probe


def takeoff_page(pdf_path: Path | str, page: int) -> PageTakeoff:
    """Run the full multi-step takeoff on one page: 2a/2b/2c + parser + 2d
    for any code that doesn't have an embedded quantity in its description.
    """
    png = render_page_png(pdf_path, page)
    logger.info("rendered page %d (%d KB)", page, len(png) // 1024)
    scale = extract_scale(png)
    logger.info("step 2a (scale) → %s", _scale_raw(scale))
    dimensions = extract_dimensions(png)
    logger.info("step 2b (dimensions) → %d items", len(dimensions))
    keynotes = extract_keynotes(png)
    logger.info("step 2c (keynotes) → %d codes", len(keynotes))

    parsed: dict[str, list[ParsedQuantity]] = {}
    probed: dict[str, ProbedQuantity] = {}
    for note in keynotes:
        code = note.get("code", "")
        desc = note.get("description", "")
        if not code:
            continue
        pqs = parse_keynote_quantities(code, desc)
        if pqs:
            parsed[code] = pqs
            logger.info(
                "  %s: parser found %d quantity record(s) embedded in description",
                code, len(pqs),
            )
        else:
            logger.info("  %s: no embedded quantity → step 2d probe", code)
            try:
                probed[code] = probe_code_quantity(png, code, desc, scale)
            except Exception as exc:  # noqa: BLE001 — keep going on per-code failure
                logger.error("  %s: probe failed: %s", code, exc)

    return PageTakeoff(
        page=page,
        scale=scale,
        dimensions=dimensions,
        keynotes=keynotes,
        parsed=parsed,
        probed=probed,
    )


# ---------------------------------------------------------------------------
# Unit conversion — domain-reasoning follow-up call
# ---------------------------------------------------------------------------
#
# Step 2d's per-page probe sometimes returns EA for codes that, by construction
# convention, should be measured in LF or SF (e.g. "scrape and paint steel
# lintel" → measured as SF of paint area, not EA of lintels). This follow-up
# asks the model — using only the keynote description, NOT a pre-existing unit
# catalog — whether EA is the right unit, and if not, what the LF/SF estimate
# would be for the count it returned.


UNIT_CONVERSION_PROMPT_TEMPLATE = (
    "In construction estimating, the scope item {code}: {description} was "
    "counted as {count} instances.\n\n"
    "Based on standard construction measurement practices, what unit should "
    "this work be measured in — EA (each, for discrete items like doors or "
    "grilles), LF (linear feet, for joints, bands, trim, repointing along "
    "edges), or SF (square feet, for area work like brick replacement, "
    "painting surfaces)?\n\n"
    "If the correct unit is LF or SF, estimate the total quantity for "
    "{count} instances on a 3-story masonry building.\n\n"
    "Return ONLY JSON: "
    '{{"correct_unit": string, "quantity": number, "reasoning": string}}.'
)


@dataclass(frozen=True)
class UnitConversion:
    code: str
    original_unit: str
    original_quantity: float
    correct_unit: str
    converted_quantity: float
    reasoning: str


def _call_text_only(prompt: str, *, max_tokens: int = 4000) -> str:
    """Text-only API call (no image). Used by the unit-conversion follow-up."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set in the environment")
    client = Anthropic()
    msg = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        thinking={"type": "adaptive", "display": "summarized"},
        output_config={"effort": "medium"},
        messages=[{"role": "user", "content": prompt}],
    )
    for block in msg.content:
        if block.type == "text":
            return block.text.strip()
    return ""


def convert_unit_if_needed(
    code: str,
    description: str,
    quantity: float,
    current_unit: str,
) -> UnitConversion | None:
    """If `current_unit` is EA and `quantity` > 0, ask the model whether EA is
    the right unit given the keynote description, and if not, get the LF/SF
    estimate. Returns None when no conversion is appropriate (already non-EA,
    zero quantity, or the model agrees EA is correct).

    The decision is grounded in the description and construction convention —
    NO project-specific unit catalog is consulted.
    """
    if current_unit.upper() != "EA":
        return None
    if quantity <= 0:
        return None

    prompt = UNIT_CONVERSION_PROMPT_TEMPLATE.format(
        code=code, description=description, count=int(quantity),
    )
    response = _parse_json(_call_text_only(prompt))
    if not isinstance(response, dict):
        raise ValueError(
            f"unit conversion: expected JSON object, got {type(response).__name__}"
        )
    correct_unit = str(response.get("correct_unit", "EA")).upper()
    if correct_unit == "EA":
        logger.info("  %s: model agrees EA is correct", code)
        return None
    new_qty = float(response.get("quantity", quantity))
    reasoning = str(response.get("reasoning", ""))
    logger.info("  %s: EA → %s (%.2f → %.2f)", code, correct_unit, quantity, new_qty)
    return UnitConversion(
        code=code,
        original_unit="EA",
        original_quantity=quantity,
        correct_unit=correct_unit,
        converted_quantity=new_qty,
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# Sliding-window probe — split the elevation into 4 quadrants
# ---------------------------------------------------------------------------
#
# Step 2d on the full page is unreliable for small/numerous callouts: at the
# clamped 190 DPI (Arch D), individual WS5 grille circles or WS8 hatched
# segments can be a few pixels across and easy to miss. Re-rendering the page
# at full DPI and probing 4 quadrants gives the model more detail per code
# without ever sending a too-large image.


QUADRANT_NAMES = ("TL", "TR", "BL", "BR")

SLIDING_WINDOW_NOTE = (
    "\n\nThis image is the {quadrant} quadrant (of TL/TR/BL/BR) of a larger "
    "elevation drawing rendered at higher resolution. Count only what is "
    "visible in this quadrant — do not infer from the rest of the elevation. "
    "The keynote legend itself is likely NOT in this quadrant; rely on the "
    "code description above."
)


def crop_quadrants(png_bytes: bytes) -> dict[str, bytes]:
    """Split a rendered page into 4 non-overlapping quadrants."""
    img = Image.open(io.BytesIO(png_bytes))
    w, h = img.size
    mid_w, mid_h = w // 2, h // 2
    boxes = {
        "TL": (0, 0, mid_w, mid_h),
        "TR": (mid_w, 0, w, mid_h),
        "BL": (0, mid_h, mid_w, h),
        "BR": (mid_w, mid_h, w, h),
    }
    out: dict[str, bytes] = {}
    for name, box in boxes.items():
        buf = io.BytesIO()
        img.crop(box).save(buf, format="PNG")
        out[name] = buf.getvalue()
    return out


def probe_code_sliding_window(
    quadrant_pngs: dict[str, bytes],
    code: str,
    description: str,
    scale: Any,
) -> ProbedQuantity:
    """Step 2d via sliding window. Sum callout_count across the 4 quadrants;
    pick the unit with the largest accumulated non-EA quantity, else EA.
    """
    total_callouts = 0
    qty_by_unit: dict[str, float] = defaultdict(float)
    notes_parts: list[str] = []
    for name in QUADRANT_NAMES:
        png = quadrant_pngs.get(name)
        if png is None:
            continue
        prompt = (
            COUNT_OR_MEASURE_PROMPT_TEMPLATE.format(
                scale_raw=_scale_raw(scale), code=code, description=description,
            )
            + SLIDING_WINDOW_NOTE.format(quadrant=name)
        )
        try:
            response = _parse_json(_call_with_image(prompt, png))
        except Exception as exc:  # noqa: BLE001
            logger.error("  %s: quadrant %s failed: %s", code, name, exc)
            continue
        if not isinstance(response, dict):
            logger.error("  %s: quadrant %s returned non-object", code, name)
            continue
        cc = int(response.get("callout_count", 0))
        qty = float(response.get("quantity", 0))
        unit = str(response.get("unit", "EA")).upper()
        total_callouts += cc
        qty_by_unit[unit] += qty
        if cc > 0 or qty > 0:
            notes_parts.append(f"{name}: cc={cc} qty={qty:g} {unit}")

    non_ea = {u: q for u, q in qty_by_unit.items() if u != "EA" and q > 0}
    if non_ea:
        final_unit = max(non_ea, key=non_ea.get)
        final_qty = non_ea[final_unit]
    else:
        final_unit = "EA"
        final_qty = float(total_callouts)

    return ProbedQuantity(
        code=code,
        quantity=final_qty,
        unit=final_unit,
        callout_count=total_callouts,
        notes=" | ".join(notes_parts) if notes_parts else "no callouts in any quadrant",
    )


# ---------------------------------------------------------------------------
# v2 orchestrator — parser + (sliding-window | step-2d) + unit conversion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PageTakeoffV2:
    page: int
    scale: Any
    keynotes: list[dict]
    parsed: dict[str, list[ParsedQuantity]]
    probed: dict[str, ProbedQuantity]
    converted: dict[str, UnitConversion]  # code -> conversion (only when applied)


def takeoff_page_v2(
    pdf_path: Path | str,
    page: int,
    *,
    sliding_window: bool = True,
    convert_units: bool = True,
    sliding_window_dpi: int = 300,
) -> PageTakeoffV2:
    """v2 of the per-page takeoff:

      - 2a (scale) + 2c (keynotes)  [2b dimensions skipped — returned [] on PA]
      - Parser pass on each keynote description
      - For codes without a parsed quantity: sliding-window probe (if enabled)
        else standard step-2d on the full page
      - Unit conversion follow-up for any (parsed or probed) EA quantity
    """
    base_png = render_page_png(pdf_path, page)  # clamped DPI for steps 2a/2c
    scale = extract_scale(base_png)
    logger.info("step 2a (scale) → %s", _scale_raw(scale))
    keynotes = extract_keynotes(base_png)
    logger.info("step 2c (keynotes) → %d codes", len(keynotes))

    if sliding_window:
        hires = render_page_png(pdf_path, page, dpi=sliding_window_dpi, clamp_dpi=False)
        quadrants = crop_quadrants(hires)
        logger.info(
            "sliding window: rendered at %d DPI, %d quadrants",
            sliding_window_dpi, len(quadrants),
        )
    else:
        quadrants = None

    parsed: dict[str, list[ParsedQuantity]] = {}
    probed: dict[str, ProbedQuantity] = {}

    for note in keynotes:
        code = note.get("code", "")
        desc = note.get("description", "")
        if not code:
            continue
        pqs = parse_keynote_quantities(code, desc)
        if pqs:
            parsed[code] = pqs
            logger.info("  %s: parser found %d record(s)", code, len(pqs))
            continue
        try:
            if quadrants is not None:
                probed[code] = probe_code_sliding_window(quadrants, code, desc, scale)
            else:
                probed[code] = probe_code_quantity(base_png, code, desc, scale)
        except Exception as exc:  # noqa: BLE001
            logger.error("  %s: probe failed: %s", code, exc)

    # Unit conversion: pass over both parsed (base_bid only) and probed
    converted: dict[str, UnitConversion] = {}
    if convert_units:
        candidates: list[tuple[str, str, float, str]] = []
        for code, pqs in parsed.items():
            for pq in pqs:
                if pq.variant == "base_bid":
                    candidates.append(
                        (code, _description_for(code, keynotes), pq.quantity, pq.unit)
                    )
                    break  # only one base_bid per code
        for code, prq in probed.items():
            candidates.append((code, _description_for(code, keynotes), prq.quantity, prq.unit))
        logger.info("unit conversion: evaluating %d candidate(s)", len(candidates))
        for code, desc, qty, unit in candidates:
            try:
                conv = convert_unit_if_needed(code, desc, qty, unit)
            except Exception as exc:  # noqa: BLE001
                logger.error("  %s: unit conversion failed: %s", code, exc)
                continue
            if conv is not None:
                converted[code] = conv

    return PageTakeoffV2(
        page=page,
        scale=scale,
        keynotes=keynotes,
        parsed=parsed,
        probed=probed,
        converted=converted,
    )


def _description_for(code: str, keynotes: list[dict]) -> str:
    for kn in keynotes:
        if kn.get("code") == code:
            return kn.get("description", "")
    return ""


# ---------------------------------------------------------------------------
# CLI probe
# ---------------------------------------------------------------------------


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", help="Path to the plan PDF")
    parser.add_argument("page", type=int, help="1-indexed page number")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the v1 pipeline (2a-2d + description-quantity parser). "
             "Default: 2a-2c only.",
    )
    parser.add_argument(
        "--v2",
        action="store_true",
        help="Run the v2 pipeline (parser + sliding-window probe + unit "
             "conversion follow-up).",
    )
    parser.add_argument(
        "--no-sliding-window",
        action="store_true",
        help="With --v2, fall back to full-page step 2d (skip quadrant split).",
    )
    parser.add_argument(
        "--no-unit-convert",
        action="store_true",
        help="With --v2, skip the unit-conversion follow-up call.",
    )
    parser.add_argument(
        "--json-out",
        help="Optional path to write structured results as JSON.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.v2:
        v2 = takeoff_page_v2(
            args.pdf,
            args.page,
            sliding_window=not args.no_sliding_window,
            convert_units=not args.no_unit_convert,
        )
        print("\n===== SCALE =====")
        print(json.dumps(v2.scale, indent=2))
        print(f"\n===== KEYNOTES ({len(v2.keynotes)}) =====")
        for kn in v2.keynotes:
            print(f"  {kn.get('code'):8} {kn.get('description', '')[:80]}")
        print(f"\n===== PARSED ({sum(len(v) for v in v2.parsed.values())}) =====")
        for code, pqs in sorted(v2.parsed.items()):
            for pq in pqs:
                print(f"  {code:6} {pq.quantity:>8.2f} {pq.unit:4} ({pq.variant})")
        print(f"\n===== PROBED ({len(v2.probed)}) =====")
        for code, prq in sorted(v2.probed.items()):
            print(
                f"  {code:6} {prq.quantity:>8.2f} {prq.unit:4} "
                f"(callouts={prq.callout_count})  {prq.notes[:80]}"
            )
        print(f"\n===== UNIT CONVERSIONS ({len(v2.converted)}) =====")
        for code, conv in sorted(v2.converted.items()):
            print(
                f"  {code:6} {conv.original_quantity:>6.1f} {conv.original_unit} → "
                f"{conv.converted_quantity:>8.2f} {conv.correct_unit}   {conv.reasoning[:80]}"
            )
        if args.json_out:
            payload = {
                "page": v2.page,
                "scale": v2.scale,
                "keynotes": v2.keynotes,
                "parsed": {c: [asdict(p) for p in pqs] for c, pqs in v2.parsed.items()},
                "probed": {c: asdict(p) for c, p in v2.probed.items()},
                "converted": {c: asdict(p) for c, p in v2.converted.items()},
            }
            Path(args.json_out).write_text(json.dumps(payload, indent=2))
            print(f"\nWrote JSON: {args.json_out}")
        return 0

    if not args.full:
        probe = probe_page(args.pdf, args.page)
        print("\n===== SCALE =====")
        print(json.dumps(probe.scale, indent=2))
        print("\n===== DIMENSIONS =====")
        print(json.dumps(probe.dimensions, indent=2))
        print(f"({len(probe.dimensions)} dimensions extracted)")
        print("\n===== KEYNOTES =====")
        print(json.dumps(probe.keynotes, indent=2))
        print(f"({len(probe.keynotes)} keynotes extracted)")
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(asdict(probe), indent=2))
        return 0

    takeoff = takeoff_page(args.pdf, args.page)
    print("\n===== SCALE =====")
    print(json.dumps(takeoff.scale, indent=2))
    print(f"\n===== DIMENSIONS ({len(takeoff.dimensions)}) =====")
    print(json.dumps(takeoff.dimensions, indent=2))
    print(f"\n===== KEYNOTES ({len(takeoff.keynotes)}) =====")
    print(json.dumps(takeoff.keynotes, indent=2))

    print(f"\n===== PARSED FROM DESCRIPTIONS ({sum(len(v) for v in takeoff.parsed.values())}) =====")
    for code, pqs in sorted(takeoff.parsed.items()):
        for pq in pqs:
            print(f"  {code:6} {pq.quantity:>8.2f} {pq.unit:4} ({pq.variant})   raw: {pq.raw!r}")

    print(f"\n===== PROBED PER-CODE ({len(takeoff.probed)}) =====")
    for code, prq in sorted(takeoff.probed.items()):
        print(f"  {code:6} {prq.quantity:>8.2f} {prq.unit:4}   callouts={prq.callout_count}   {prq.notes}")

    if args.json_out:
        payload = {
            "page": takeoff.page,
            "scale": takeoff.scale,
            "dimensions": takeoff.dimensions,
            "keynotes": takeoff.keynotes,
            "parsed": {c: [asdict(p) for p in pqs] for c, pqs in takeoff.parsed.items()},
            "probed": {c: asdict(p) for c, p in takeoff.probed.items()},
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
        print(f"\nWrote JSON: {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
