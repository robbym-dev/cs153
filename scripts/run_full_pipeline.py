"""Run the end-to-end pipeline: load extractions → price → compare to Tyler.

Reads tests/extractions/page{2,3,6}.txt and tests/baseline_page5.txt
(pages 7 and 8 are byte-identical duplicates of 5 and 6 — same elevation in
different bid packages — so they are excluded). Aggregates by (code, unit),
runs through `price_bid`, and reports the priced bid against Tyler's DIV-07
THERMAL & MOISTURE PROTECTION subtotal of $28,871.
"""

from __future__ import annotations

import logging
import sys
from collections import defaultdict
from pathlib import Path

from bid_engine.pricing import ScopeItem, price_bid

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTION_FILES = {
    2: PROJECT_ROOT / "tests" / "extractions" / "page2.txt",
    3: PROJECT_ROOT / "tests" / "extractions" / "page3.txt",
    5: PROJECT_ROOT / "tests" / "baseline_page5.txt",
    6: PROJECT_ROOT / "tests" / "extractions" / "page6.txt",
}
TYLER_DIV07_SUBTOTAL = 28871.45  # DIV-07 THERMAL & MOISTURE PROTECTION

logging.basicConfig(level=logging.INFO, format="%(message)s")
# Suppress the per-line INFO trace from pricing.py so the report stays readable;
# WARNINGs and ERRORs (unknown unit, missing unit cost) still surface.
logging.getLogger("bid_engine.pricing").setLevel(logging.WARNING)
logger = logging.getLogger("pipeline")


def load_extracted_items() -> dict[tuple[str, str], float]:
    """Aggregate (code, unit) → total quantity across the deduped pages."""
    totals: dict[tuple[str, str], float] = defaultdict(float)
    for page, path in sorted(EXTRACTION_FILES.items()):
        if not path.exists():
            logger.error("missing extraction file: %s (page %d) — skipping", path, page)
            continue
        try:
            content = path.read_text()
        except OSError as exc:
            logger.error("could not read %s: %s — skipping page %d", path, exc, page)
            continue

        kept = 0
        for lineno, raw in enumerate(content.splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                logger.warning(
                    "page %d line %d: expected 3 tab-separated fields, got %d — skipping (%r)",
                    page, lineno, len(parts), raw,
                )
                continue
            code, qty_s, unit = (p.strip() for p in parts)
            if not code or not unit:
                logger.warning("page %d line %d: empty code or unit — skipping (%r)",
                               page, lineno, raw)
                continue
            try:
                qty = float(qty_s)
            except ValueError:
                logger.warning("page %d line %d: non-numeric quantity %r — skipping",
                               page, lineno, qty_s)
                continue
            totals[(code, unit)] += qty
            kept += 1
        logger.info("page %d: loaded %d row(s) from %s", page, kept, path.name)

    return dict(totals)


def report_bid(bid, scope_count: int) -> None:
    sep = "-" * 78
    logger.info("")
    logger.info(sep)
    logger.info("BID BREAKDOWN  (priced %d / %d line items)",
                len(bid.line_items), scope_count)
    logger.info(sep)
    logger.info("%-8s %-6s %10s %12s %12s %14s",
                "CODE", "UNIT", "QTY", "UNIT LABOR", "UNIT MAT'L", "TOTAL COST")
    for li in bid.line_items:
        si = li.scope_item
        logger.info(
            "%-8s %-6s %10.2f %12.2f %12.2f %14.2f",
            si.code, si.unit, si.quantity,
            li.unit_labor, li.unit_material, li.total_cost,
        )

    logger.info("")
    logger.info("%-32s %14.2f", "SUBTOTAL", bid.subtotal)
    logger.info("%-32s %14.2f", "OVERHEAD & PROFIT (20%)", bid.overhead)
    logger.info("%-32s %14.2f", "TAX (8.5%)", bid.tax)
    logger.info("%-32s %14.2f", "BID BOND (1.5%)", bid.bid_bond)
    logger.info("%-32s %14.2f", "CONTINGENCIES (5%)", bid.contingencies)
    logger.info("%-32s %14.2f", "TOTAL BASE BID", bid.total)


def report_comparison(engine_subtotal: float) -> None:
    delta = engine_subtotal - TYLER_DIV07_SUBTOTAL
    pct = (delta / TYLER_DIV07_SUBTOTAL) * 100.0
    sep = "-" * 78
    logger.info("")
    logger.info(sep)
    logger.info("COMPARISON vs Tyler DIV-07 (THERMAL & MOISTURE PROTECTION)")
    logger.info(sep)
    logger.info("  engine subtotal:    $%14.2f", engine_subtotal)
    logger.info("  Tyler DIV-07:       $%14.2f", TYLER_DIV07_SUBTOTAL)
    logger.info("  delta:              $%+14.2f  (%+.2f%%)", delta, pct)


def main() -> int:
    logger.info("=" * 78)
    logger.info("BID-ENGINE FULL PIPELINE")
    logger.info("=" * 78)

    aggregates = load_extracted_items()
    if not aggregates:
        logger.error("no scope items loaded — aborting")
        return 1

    scope_items = [
        ScopeItem(code=code, quantity=qty, unit=unit)
        for (code, unit), qty in sorted(aggregates.items())
    ]
    logger.info("aggregated %d unique (code, unit) keys", len(scope_items))

    try:
        bid = price_bid(scope_items)
    except Exception as exc:
        logger.exception("price_bid raised an unexpected exception: %s", exc)
        return 2

    report_bid(bid, scope_count=len(scope_items))
    report_comparison(bid.subtotal)
    return 0


if __name__ == "__main__":
    sys.exit(main())
