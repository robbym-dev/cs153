"""Tests for bid_engine.pricing.

The five `test_validates_*` cases compare our pricing engine against Reference's
actual line totals from the Park Avenue Elementary School bid. Each should
land within 5%.
"""

from __future__ import annotations

import pytest

from bid_engine.pricing import (
    DEFAULT_MARKUP,
    DEFAULT_UNIT_COSTS,
    DEFAULT_WAGES,
    Bid,
    BidLineItem,
    BidMarkup,
    PrevailingWage,
    ScopeItem,
    UnitCost,
    price_bid,
    price_scope_item,
    waste_rate_for,
)


# ---------------------------------------------------------------------------
# Calibration cases — pricing engine vs Reference's actual bid (must be < 5%)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code, quantity, unit, reference_total",
    [
        ("WS4", 205.7, "LF", 1520.47),
        ("WS5", 23.0, "EA", 1335.17),
        ("WS1", 170.34, "LF", 1259.10),
        ("WS8", 29.87, "LF", 493.27),
        ("WS15", 126.73, "SF", 3497.05),
    ],
)
def test_validates_within_5pct_of_reference(code, quantity, unit, reference_total):
    """Each line item must price within 5% of Reference's reference figure."""
    scope = ScopeItem(code=code, quantity=quantity, unit=unit, description="")
    unit_cost = DEFAULT_UNIT_COSTS[(code, unit)]
    wage = DEFAULT_WAGES[unit_cost.trade]

    line = price_scope_item(scope, unit_cost, wage)

    delta_pct = abs(line.total_cost - reference_total) / reference_total
    assert delta_pct < 0.05, (
        f"{code} {quantity} {unit}: priced ${line.total_cost:.2f}, "
        f"Reference ${reference_total:.2f}, delta {delta_pct:.2%}"
    )


# ---------------------------------------------------------------------------
# Dataclass invariants
# ---------------------------------------------------------------------------


def test_prevailing_wage_total_is_base_plus_supplements():
    w = PrevailingWage("bricklayer", 56.58, 51.44)
    assert w.total == pytest.approx(108.02)


def test_default_wages_consistent_with_spec():
    assert DEFAULT_WAGES["bricklayer"].total == pytest.approx(108.02)
    assert DEFAULT_WAGES["ironworker"].total == pytest.approx(107.80)
    assert DEFAULT_WAGES["mason_tender"].total == pytest.approx(77.57)


# ---------------------------------------------------------------------------
# Waste rates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "unit, expected",
    [("EA", 0.0), ("LF", 0.10), ("FT", 0.10), ("SF", 0.10), ("SQ FT", 0.10), ("LS", 0.0)],
)
def test_waste_rate_known_units(unit, expected):
    assert waste_rate_for(unit) == expected


def test_waste_rate_unknown_unit_defaults_to_zero(caplog):
    with caplog.at_level("WARNING"):
        assert waste_rate_for("BOGUS") == 0.0
    assert any("unknown unit" in r.message for r in caplog.records)


def test_waste_rate_case_and_whitespace_insensitive():
    assert waste_rate_for("  ea  ") == 0.0
    assert waste_rate_for("lf") == 0.10


# ---------------------------------------------------------------------------
# price_scope_item edge cases
# ---------------------------------------------------------------------------


def test_zero_quantity_yields_zero_cost():
    scope = ScopeItem("WS1", 0.0, "LF")
    line = price_scope_item(scope, DEFAULT_UNIT_COSTS[("WS1", "LF")], DEFAULT_WAGES["bricklayer"])
    assert line.total_cost == 0.0


def test_negative_quantity_raises():
    scope = ScopeItem("WS1", -1.0, "LF")
    with pytest.raises(ValueError, match="quantity must be >= 0"):
        price_scope_item(scope, DEFAULT_UNIT_COSTS[("WS1", "LF")], DEFAULT_WAGES["bricklayer"])


def test_unit_labor_uses_full_wage_total():
    scope = ScopeItem("WS5", 1.0, "EA")
    uc = DEFAULT_UNIT_COSTS[("WS5", "EA")]
    wage = DEFAULT_WAGES[uc.trade]
    line = price_scope_item(scope, uc, wage)
    assert line.unit_labor == pytest.approx(uc.hours_per_unit * wage.total)


def test_waste_applied_to_lf():
    scope = ScopeItem("WS1", 100.0, "LF")
    uc = DEFAULT_UNIT_COSTS[("WS1", "LF")]
    wage = DEFAULT_WAGES[uc.trade]
    line = price_scope_item(scope, uc, wage)
    # qty_with_waste = 110.0
    expected_total = 110.0 * (uc.hours_per_unit * wage.total + uc.material_per_unit)
    assert line.total_cost == pytest.approx(expected_total)


def test_no_waste_applied_to_ea():
    scope = ScopeItem("WS5", 10.0, "EA")
    uc = DEFAULT_UNIT_COSTS[("WS5", "EA")]
    wage = DEFAULT_WAGES[uc.trade]
    line = price_scope_item(scope, uc, wage)
    expected_total = 10.0 * (uc.hours_per_unit * wage.total + uc.material_per_unit)
    assert line.total_cost == pytest.approx(expected_total)


# ---------------------------------------------------------------------------
# price_bid — full bid assembly + markups
# ---------------------------------------------------------------------------


def test_price_bid_applies_all_markups_to_subtotal():
    scope = [ScopeItem("WS5", 23.0, "EA")]
    bid = price_bid(scope)

    # Markups apply to subtotal independently.
    assert bid.overhead == pytest.approx(bid.subtotal * 0.20)
    assert bid.tax == pytest.approx(bid.subtotal * 0.085)
    assert bid.bid_bond == pytest.approx(bid.subtotal * 0.015)
    assert bid.contingencies == pytest.approx(bid.subtotal * 0.05)
    assert bid.total == pytest.approx(
        bid.subtotal + bid.overhead + bid.tax + bid.bid_bond + bid.contingencies
    )


def test_price_bid_total_matches_subtotal_times_135pct():
    """Spec markups (20 + 8.5 + 1.5 + 5) = 35% on top of subtotal."""
    scope = [ScopeItem("WS1", 100.0, "LF")]
    bid = price_bid(scope)
    assert bid.total == pytest.approx(bid.subtotal * 1.35)


def test_price_bid_skips_unknown_codes(caplog):
    scope = [
        ScopeItem("WS1", 100.0, "LF"),
        ScopeItem("WS999_UNKNOWN", 5.0, "EA"),
    ]
    with caplog.at_level("ERROR"):
        bid = price_bid(scope)
    assert len(bid.line_items) == 1
    assert bid.line_items[0].scope_item.code == "WS1"
    assert any("no unit cost" in r.message for r in caplog.records)


def test_price_bid_empty_input_returns_zero_bid():
    bid = price_bid([])
    assert bid.subtotal == 0.0
    assert bid.total == 0.0
    assert bid.line_items == ()


def test_price_bid_custom_markup():
    scope = [ScopeItem("WS5", 1.0, "EA")]
    custom = BidMarkup(overhead=0.0, tax=0.0, bid_bond=0.0, contingencies=0.0)
    bid = price_bid(scope, markup=custom)
    assert bid.total == pytest.approx(bid.subtotal)
