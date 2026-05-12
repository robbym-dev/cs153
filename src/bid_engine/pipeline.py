"""End-to-end pipeline: marked-up plan PDF → priced bid spreadsheet.

`run_pipeline(pdf_path, page_numbers, project_config)` wires together the four
domain modules:

    extraction  →  pricing  →  scope_checker  →  bid_generator

The `extractor` keyword argument is an injection point so tests can supply
cached extraction results without calling the vision API. Defaults to the
real `bid_engine.extraction.extract_page`.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from bid_engine.bid_generator import BidHeader, generate_bid_excel
from bid_engine.extraction import extract_page
from bid_engine.pricing import Bid, ScopeItem, price_bid
from bid_engine.scope_checker import ProjectConfig, ScopeAlert, check_scope

logger = logging.getLogger(__name__)

Extractor = Callable[[Path, int], list[dict]]


@dataclass(frozen=True)
class PipelineResult:
    bid: Bid
    alerts: tuple[ScopeAlert, ...]
    output_path: Path
    scope_items: tuple[ScopeItem, ...]


def _default_extractor(pdf_path: Path, page: int) -> list[dict]:
    return extract_page(pdf_path, page)


def _aggregate(
    pdf_path: Path,
    page_numbers: Sequence[int],
    extract_fn: Extractor,
) -> dict[tuple[str, str], float]:
    """Run extraction on every requested page; aggregate by (code, unit)."""
    totals: dict[tuple[str, str], float] = defaultdict(float)
    succeeded = 0
    for page in page_numbers:
        try:
            items = extract_fn(pdf_path, page)
        except Exception as exc:  # noqa: BLE001 — keep going on per-page failure
            logger.error("page %d extraction failed: %s", page, exc)
            continue
        for item in items:
            try:
                totals[(item["code"], item["unit"])] += float(item["quantity"])
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning(
                    "page %d: malformed extraction item %r (%s) — skipping",
                    page, item, exc,
                )
        if items:
            succeeded += 1
        logger.info("page %d: %d item(s) extracted", page, len(items))
    logger.info(
        "aggregated %d unique (code, unit) keys from %d/%d page(s)",
        len(totals), succeeded, len(page_numbers),
    )
    return totals


def run_pipeline(
    pdf_path: Path | str,
    page_numbers: Sequence[int],
    project_config: ProjectConfig | None = None,
    *,
    output_path: Path | str = "bid.xlsx",
    header: BidHeader | None = None,
    extractor: Extractor | None = None,
) -> PipelineResult:
    """Run extraction → pricing → scope check → Excel write.

    Args:
        pdf_path: Marked-up plan PDF.
        page_numbers: 1-indexed pages to extract.
        project_config: Optional ProjectConfig for scope_checker. If None, the
            checker still runs the three required-item checks (scaffolding,
            fencing, shed); conditional checks (boom, prevailing wage) skip.
        output_path: Where to write the bid spreadsheet.
        header: BidHeader for the spreadsheet's metadata block. Defaults to a
            BidHeader using the PDF's stem as the project name.
        extractor: Override the extraction function (used by tests to bypass
            the vision API). Default: `bid_engine.extraction.extract_page`.

    Returns:
        PipelineResult with bid, alerts, output path, and the aggregated
        scope items the bid was built from.
    """
    pdf_path = Path(pdf_path)
    if not page_numbers:
        raise ValueError("page_numbers must contain at least one page")
    if any(p < 1 for p in page_numbers):
        raise ValueError(f"page numbers must be >= 1, got {list(page_numbers)}")

    extract_fn = extractor or _default_extractor

    # Only check filesystem when using the real extractor — tests inject
    # their own and don't need an actual PDF on disk.
    if extractor is None and not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    totals = _aggregate(pdf_path, page_numbers, extract_fn)
    if not totals:
        raise RuntimeError(
            f"extraction produced no scope items across pages {list(page_numbers)}"
        )

    scope_items = tuple(
        ScopeItem(code=code, quantity=qty, unit=unit)
        for (code, unit), qty in sorted(totals.items())
    )

    bid = price_bid(scope_items)
    alerts = tuple(check_scope(scope_items, project_config))
    final_header = header or BidHeader(project_name=pdf_path.stem)
    final_output = generate_bid_excel(bid, final_header, output_path)

    logger.info(
        "pipeline complete: %d line items, $%.2f total, %d alert(s), → %s",
        len(bid.line_items), bid.total, len(alerts), final_output,
    )
    return PipelineResult(
        bid=bid,
        alerts=alerts,
        output_path=final_output,
        scope_items=scope_items,
    )
