"""Pricing engine: convert extracted scope items into a priced bid.

Math (per the project spec, calibrated against Tyler's actual Park Avenue
Elementary School bid):

    qty_with_waste = qty * (1 + waste_rate[unit])
    unit_labor     = unit_cost.hours_per_unit * wage.total
    unit_material  = unit_cost.material_per_unit
    total_cost     = qty_with_waste * (unit_labor + unit_material)

Bid markups apply to the line-item subtotal independently and sum:

    overhead       = subtotal * 20.0%
    tax            = subtotal *  8.5%
    bid_bond       = subtotal *  1.5%
    contingencies  = subtotal *  5.0%
    total          = subtotal + overhead + tax + bid_bond + contingencies
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Mapping

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScopeItem:
    code: str
    quantity: float
    unit: str
    description: str = ""


@dataclass(frozen=True)
class UnitCost:
    code: str
    trade: str
    hours_per_unit: float
    material_per_unit: float


@dataclass
class PrevailingWage:
    trade: str
    base_rate: float
    supplements: float
    total: float = field(init=False)

    def __post_init__(self) -> None:
        self.total = self.base_rate + self.supplements


@dataclass(frozen=True)
class BidLineItem:
    scope_item: ScopeItem
    unit_labor: float
    unit_material: float
    total_cost: float


@dataclass(frozen=True)
class BidMarkup:
    overhead: float = 0.20
    tax: float = 0.085
    bid_bond: float = 0.015
    contingencies: float = 0.05


@dataclass(frozen=True)
class Bid:
    line_items: tuple[BidLineItem, ...]
    subtotal: float
    overhead: float
    tax: float
    bid_bond: float
    contingencies: float
    total: float


# ---------------------------------------------------------------------------
# Defaults — domain data from the project spec
# ---------------------------------------------------------------------------


WASTE_RATES: dict[str, float] = {
    "EA": 0.0,
    "LF": 0.10,
    "FT": 0.10,        # alias for LF
    "SF": 0.10,
    "SQ FT": 0.10,     # alias for SF
    "LS": 0.0,
}

# Orange County, NY (Warwick) prevailing wages — base + supplements per spec.
DEFAULT_WAGES: dict[str, PrevailingWage] = {
    "bricklayer": PrevailingWage("bricklayer", 56.58, 51.44),
    "stone_mason": PrevailingWage("stone_mason", 56.58, 51.44),
    "mason_tender": PrevailingWage("mason_tender", 37.10, 40.47),
    "ironworker": PrevailingWage("ironworker", 52.00, 55.80),
    "carpenter": PrevailingWage("carpenter", 48.25, 44.19),
    "painter": PrevailingWage("painter", 42.85, 37.69),
}

# Calibrated against Tyler's actual line items (see spec).
DEFAULT_UNIT_COSTS: dict[str, UnitCost] = {
    "WS1": UnitCost("WS1", "bricklayer", 0.06, 0.50),
    "WS2": UnitCost("WS2", "stone_mason", 0.18, 13.50),
    "WS4": UnitCost("WS4", "bricklayer", 0.06, 0.50),
    "WS5": UnitCost("WS5", "ironworker", 0.55, 0.00),
    "WS8": UnitCost("WS8", "bricklayer", 0.13, 0.50),
    "WS10": UnitCost("WS10", "bricklayer", 0.05, 0.50),
    "WS15": UnitCost("WS15", "bricklayer", 0.15, 8.50),
}

DEFAULT_MARKUP = BidMarkup()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def waste_rate_for(unit: str) -> float:
    """Lookup waste rate; warn-and-default to 0 for unknown units."""
    key = unit.strip().upper()
    if key in WASTE_RATES:
        return WASTE_RATES[key]
    logger.warning("unknown unit %r — defaulting to 0%% waste", unit)
    return 0.0


# ---------------------------------------------------------------------------
# Pricing functions
# ---------------------------------------------------------------------------


def price_scope_item(
    scope_item: ScopeItem,
    unit_cost: UnitCost,
    wage: PrevailingWage,
) -> BidLineItem:
    """Compute a priced line item for a single ScopeItem."""
    if scope_item.quantity < 0:
        raise ValueError(
            f"scope_item.quantity must be >= 0, got {scope_item.quantity!r} "
            f"for code {scope_item.code!r}"
        )
    if unit_cost.code != scope_item.code:
        logger.warning(
            "unit_cost.code=%r does not match scope_item.code=%r",
            unit_cost.code,
            scope_item.code,
        )
    if unit_cost.trade != wage.trade:
        logger.warning(
            "unit_cost.trade=%r does not match wage.trade=%r — pricing anyway",
            unit_cost.trade,
            wage.trade,
        )

    waste = waste_rate_for(scope_item.unit)
    qty_with_waste = scope_item.quantity * (1 + waste)
    unit_labor = unit_cost.hours_per_unit * wage.total
    unit_material = unit_cost.material_per_unit
    total_cost = qty_with_waste * (unit_labor + unit_material)

    logger.info(
        "priced %s: %.2f %s (×%.2f waste) → labor=$%.2f material=$%.2f total=$%.2f",
        scope_item.code,
        scope_item.quantity,
        scope_item.unit,
        1 + waste,
        qty_with_waste * unit_labor,
        qty_with_waste * unit_material,
        total_cost,
    )
    return BidLineItem(
        scope_item=scope_item,
        unit_labor=unit_labor,
        unit_material=unit_material,
        total_cost=total_cost,
    )


def price_bid(
    scope_items: Iterable[ScopeItem],
    unit_costs: Mapping[str, UnitCost] = DEFAULT_UNIT_COSTS,
    wages: Mapping[str, PrevailingWage] = DEFAULT_WAGES,
    markup: BidMarkup = DEFAULT_MARKUP,
) -> Bid:
    """Price a full set of scope items into a complete Bid."""
    line_items: list[BidLineItem] = []
    skipped: list[str] = []
    for si in scope_items:
        try:
            uc = unit_costs[si.code]
        except KeyError:
            logger.error("no unit cost for code %r — skipping line item", si.code)
            skipped.append(si.code)
            continue
        try:
            wage = wages[uc.trade]
        except KeyError:
            logger.error(
                "no prevailing wage for trade %r (code %r) — skipping",
                uc.trade,
                si.code,
            )
            skipped.append(si.code)
            continue
        line_items.append(price_scope_item(si, uc, wage))

    if skipped:
        logger.warning("priced %d items, skipped %d: %s", len(line_items), len(skipped), skipped)

    subtotal = sum(li.total_cost for li in line_items)
    overhead = subtotal * markup.overhead
    tax = subtotal * markup.tax
    bid_bond = subtotal * markup.bid_bond
    contingencies = subtotal * markup.contingencies
    total = subtotal + overhead + tax + bid_bond + contingencies

    return Bid(
        line_items=tuple(line_items),
        subtotal=subtotal,
        overhead=overhead,
        tax=tax,
        bid_bond=bid_bond,
        contingencies=contingencies,
        total=total,
    )
