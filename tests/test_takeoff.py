"""Tests for the keynote-description quantity parser and the sliding-window
quadrant cropper in bid_engine.takeoff.

The four parametrized parser cases mirror the actual keynote patterns observed
on page 8 (A-300) of the Park Avenue original drawings.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from bid_engine.takeoff import ParsedQuantity, crop_quadrants, parse_keynote_quantities


# ---------------------------------------------------------------------------
# Real-world patterns from PA A-300 page 8 keynote legend
# ---------------------------------------------------------------------------


def test_parses_approx_quantity_spelled_out_with_paren_num():
    # WS7: "PATCH HOLES IN FACE BRICK (APPROX. QUANTITY: EIGHT (8) BRICKS)."
    results = parse_keynote_quantities(
        "WS7",
        "PATCH HOLES IN FACE BRICK (APPROX. QUANTITY: EIGHT (8) BRICKS).",
    )
    assert len(results) == 1
    r = results[0]
    assert r.code == "WS7"
    assert r.quantity == 8.0
    assert r.unit == "EA"          # BRICKS → EA
    assert r.variant == "base_bid"


def test_parses_plus_minus_with_explicit_lf():
    # WS8: "RAKE OUT AND REPOINT BRICK MORTAR JOINTS IN DELINEATED AREAS (±100 LF). SEE DETAIL 3/A430 (TYP)."
    results = parse_keynote_quantities(
        "WS8",
        "RAKE OUT AND REPOINT BRICK MORTAR JOINTS IN DELINEATED AREAS "
        "(±100 LF). SEE DETAIL 3/A430 (TYP).",
    )
    assert len(results) == 1
    assert results[0].quantity == 100.0
    assert results[0].unit == "LF"
    assert results[0].variant == "base_bid"


def test_parses_quantity_marker_with_lintels():
    # WS15: "...(QUANTITY: TWO (2) LINTELS)..."
    results = parse_keynote_quantities(
        "WS15",
        "REMOVE LOOSE BRICK AROUND LINTEL. SCRAPE, WIRE BRUSH, PRIME AND PAINT "
        "ALL EXPOSED PORTIONS OF STEEL LINTEL. INSTALL NEW FLASHING AND BRICK "
        "VENEER (QUANTITY: TWO (2) LINTELS). CONTRACTOR TO WORK IN A MANNER "
        "WHICH PROVIDES CONTINUOUS SUPPORT TO MASONRY ABOVE. SEE DETAIL 5/A701.",
    )
    assert len(results) == 1
    assert results[0].quantity == 2.0
    assert results[0].unit == "EA"          # LINTELS → EA
    assert results[0].variant == "base_bid"


def test_parses_base_bid_and_alternate_in_one_description():
    # WS10: "...(BASE BID QUANTITY: TWENTY-ONE (21) LINTELS, ALTERNATE 2 QUANTITY: FOUR (4) LINTELS)."
    results = parse_keynote_quantities(
        "WS10",
        "REPOINT HORIZONTAL JOINT AT LINTELS BETWEEN WINDOWS AND AT CORNERS. "
        "AT SINGLE WINDOWS, REPOINT HORIZONTAL JOINT AT WINDOW CORNERS. "
        "REPOINT THREE COURSES OF BRICK ABOVE LINTEL. "
        "(BASE BID QUANTITY: TWENTY-ONE (21) LINTELS, "
        "ALTERNATE 2 QUANTITY: FOUR (4) LINTELS).",
    )
    assert len(results) == 2
    base = next(r for r in results if r.variant == "base_bid")
    alt = next(r for r in results if r.variant == "alternate_2")
    assert base.quantity == 21.0
    assert base.unit == "EA"
    assert alt.quantity == 4.0
    assert alt.unit == "EA"


# ---------------------------------------------------------------------------
# Negative / robustness cases
# ---------------------------------------------------------------------------


def test_description_with_no_quantity_returns_empty():
    results = parse_keynote_quantities(
        "WS1",
        "CLEAN CAST STONE BANDS, REPOINT AND COVER ALL HORIZONTAL & VERTICAL JOINTS.",
    )
    assert results == []


def test_empty_description():
    assert parse_keynote_quantities("WS1", "") == []


def test_detail_references_not_parsed_as_quantities():
    """`SEE DETAIL 3/A430` and `(TYP)` should not produce phantom quantities."""
    results = parse_keynote_quantities(
        "WS16",
        "REPAIR CRACKED BRICK. SEE DETAIL 2/A430 (TYP).",
    )
    assert results == []


def test_spelled_out_three_courses_does_not_match():
    """Spelled-out numbers (THREE) without numeric form should be ignored."""
    results = parse_keynote_quantities(
        "WS10",
        "REPOINT THREE COURSES OF BRICK ABOVE LINTEL.",
    )
    assert results == []


def test_returns_parsed_quantity_dataclass():
    results = parse_keynote_quantities(
        "WS8",
        "DELINEATED AREAS (±100 LF).",
    )
    assert isinstance(results[0], ParsedQuantity)
    assert results[0].raw  # raw matched substring is populated


@pytest.mark.parametrize(
    "noun, expected_unit",
    [
        ("BRICKS", "EA"),
        ("BRICK", "EA"),
        ("LINTELS", "EA"),
        ("LINTEL", "EA"),
        ("DOORS", "EA"),
        ("WINDOWS", "EA"),
    ],
)
def test_noun_to_unit_mapping(noun, expected_unit):
    desc = f"DO WORK (QUANTITY: ONE (1) {noun})."
    results = parse_keynote_quantities("X1", desc)
    assert results[0].unit == expected_unit


# ---------------------------------------------------------------------------
# Sliding-window quadrant cropper
# ---------------------------------------------------------------------------


def _make_png(width: int, height: int) -> bytes:
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_crop_quadrants_returns_four_named_quadrants():
    png = _make_png(800, 600)
    out = crop_quadrants(png)
    assert set(out.keys()) == {"TL", "TR", "BL", "BR"}


def test_crop_quadrants_each_quarter_size():
    png = _make_png(800, 600)
    out = crop_quadrants(png)
    for name, q_bytes in out.items():
        q = Image.open(io.BytesIO(q_bytes))
        assert q.size == (400, 300), f"{name}: got {q.size}"


def test_crop_quadrants_handles_odd_dimensions():
    """Odd width/height should round down on the midline; no off-by-one error."""
    png = _make_png(801, 601)
    out = crop_quadrants(png)
    tl = Image.open(io.BytesIO(out["TL"]))
    br = Image.open(io.BytesIO(out["BR"]))
    # TL gets floor(w/2)=400, BR gets ceil(w/2)=401. Both quadrants cover full image.
    assert tl.size[0] + br.size[0] == 801
    assert tl.size[1] + br.size[1] == 601
