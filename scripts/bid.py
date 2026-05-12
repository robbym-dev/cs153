"""bid — extract, price, scope-check, and write a construction bid spreadsheet.

Example:
    python scripts/bid.py plan.pdf --pages 2,3,5,6 --state NY --stories 3 \\
        --name "Park Avenue Elementary School" --address "10 Park Ave, Warwick NY" \\
        --date 2026-05-12 --output bid.xlsx

The PDF must be a marked-up plan set with a sidebar of scope codes (WSn / Rn)
on each elevation page. Quantities are aggregated across pages, priced against
the calibrated unit-cost catalog, run through the scope checker, and written
out as an Excel file in Tyler's format. Scope alerts (missing scaffolding,
fencing, etc.) are printed to stdout after the bid summary.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from bid_engine.bid_generator import BidHeader
from bid_engine.pipeline import run_pipeline
from bid_engine.pricing import DEFAULT_WAGES
from bid_engine.scope_checker import ProjectConfig


def _parse_pages(arg: str) -> list[int]:
    pages: list[int] = []
    for chunk in arg.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            page = int(chunk)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid page number {chunk!r}") from exc
        if page < 1:
            raise argparse.ArgumentTypeError(f"page numbers must be >= 1, got {page}")
        pages.append(page)
    if not pages:
        raise argparse.ArgumentTypeError("--pages must list at least one page")
    return pages


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="bid",
        description="Run the bid-engine pipeline on a marked-up plan PDF.",
    )
    p.add_argument("pdf", help="path to the marked-up plan PDF")
    p.add_argument(
        "--pages",
        type=_parse_pages,
        required=True,
        help="comma-separated 1-indexed page numbers (e.g. 2,3,5,6)",
    )
    p.add_argument("--state", default="", help="project state code (e.g. NY)")
    p.add_argument("--stories", type=int, default=1, help="number of stories (>=1)")
    p.add_argument("--output", default="bid.xlsx", help="output .xlsx path")
    p.add_argument("--name", default="", help="project name (defaults to PDF stem)")
    p.add_argument("--address", default="", help="project address")
    p.add_argument("--date", default="", help="bid date (free-form; YYYY-MM-DD recommended)")
    p.add_argument(
        "--quiet", action="store_true", help="suppress per-page info logs"
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"error: PDF not found: {pdf_path}", file=sys.stderr)
        return 1

    if args.stories < 1:
        print(f"error: --stories must be >= 1, got {args.stories}", file=sys.stderr)
        return 1

    cfg = ProjectConfig(
        state=args.state,
        stories=args.stories,
        wage_rates=DEFAULT_WAGES if args.state.strip().upper() == "NY" else None,
    )
    header = BidHeader(
        project_name=args.name or pdf_path.stem,
        address=args.address,
        date=args.date,
        scope="COMPLETE",
    )

    try:
        result = run_pipeline(
            pdf_path,
            args.pages,
            project_config=cfg,
            output_path=args.output,
            header=header,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (RuntimeError, ValueError) as exc:
        print(f"error: pipeline failed: {exc}", file=sys.stderr)
        return 2

    bid = result.bid
    print()
    print(f"Bid generated: {result.output_path}")
    print(f"  line items:   {len(bid.line_items)}")
    print(f"  subtotal:     ${bid.subtotal:>12,.2f}")
    print(f"  overhead:     ${bid.overhead:>12,.2f}")
    print(f"  tax:          ${bid.tax:>12,.2f}")
    print(f"  bid bond:     ${bid.bid_bond:>12,.2f}")
    print(f"  contingencies:${bid.contingencies:>12,.2f}")
    print(f"  TOTAL:        ${bid.total:>12,.2f}")

    if result.alerts:
        print(f"\nScope alerts ({len(result.alerts)}):")
        for a in result.alerts:
            print(f"  [{a.severity.upper():8}] {a.item_id}")
            print(f"             {a.description}")
            print(f"             → {a.suggested_action}")
        # Critical alerts are advisory, not a failure — contractor decides.
    else:
        print("\nNo scope alerts — bid scope looks complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
