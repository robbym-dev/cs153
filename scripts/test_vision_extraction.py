"""Standalone test: extract sidebar scope items from one page of a marked-up plan PDF.

Thin CLI wrapper around bid_engine.extraction. Prints tab-separated
(code, quantity, unit) — one row per item.
"""

import argparse
import os
import sys

from bid_engine.extraction import extract_page


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", help="Path to the marked-up plan PDF")
    parser.add_argument("page", type=int, help="1-indexed page number to extract")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set", file=sys.stderr)
        return 2

    items = extract_page(args.pdf, args.page)
    for item in items:
        print(f"{item['code']}\t{item['quantity']}\t{item['unit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
