"""Compare AI-extracted sidebar quantities to Tyler's bid spreadsheet.

Aggregates by (code, unit) since the same code can appear on multiple
elevations / locations within one page and across pages. Pages 5 and 6 are
loaded; pages 7 and 8 produced byte-identical output to 5 and 6 and are
excluded as duplicates.
"""

import re
from collections import defaultdict
from pathlib import Path

import openpyxl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTION_FILES = {
    2: PROJECT_ROOT / "tests" / "extractions" / "page2.txt",
    3: PROJECT_ROOT / "tests" / "extractions" / "page3.txt",
    5: PROJECT_ROOT / "tests" / "baseline_page5.txt",
    6: PROJECT_ROOT / "tests" / "extractions" / "page6.txt",
}
SPREADSHEET = PROJECT_ROOT / "test_data" / "Park_Avenue_Elementary_School.xlsx"
TOLERANCE = 0.05  # per spec — quantities matched within 0.05 in manual validation

CODE_RE = re.compile(r"^([A-Z]+\d+)\s*:")


def normalize_unit(u: str) -> str:
    u = u.strip().upper().replace(".", "")
    if u in ("LF", "FT", "LIN FT", "LINEAR FT"):
        return "LF"
    if u in ("SF", "SQ FT", "SQFT"):
        return "SF"
    if u in ("EA", "EACH"):
        return "EA"
    return u


def load_extraction() -> dict:
    totals = defaultdict(float)
    rows = 0
    for path in EXTRACTION_FILES.values():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 3:
                    continue
                code, qty, unit = parts
                totals[(code.strip(), normalize_unit(unit))] += float(qty)
                rows += 1
    print(f"  loaded {rows} rows across {len(EXTRACTION_FILES)} deduped pages")
    return dict(totals)


def load_spreadsheet() -> dict:
    wb = openpyxl.load_workbook(SPREADSHEET, data_only=True)
    ws = wb["DETAIL"]
    totals = defaultdict(float)
    rows = 0
    for r in range(28, ws.max_row + 1):
        item_num = ws.cell(r, 1).value
        try:
            int(item_num)
        except (TypeError, ValueError):
            continue
        desc = ws.cell(r, 5).value
        qty = ws.cell(r, 6).value
        unit = ws.cell(r, 7).value
        if not desc or qty is None or not unit:
            continue
        m = CODE_RE.match(str(desc).strip())
        if not m:
            continue
        try:
            qty_f = float(qty)
        except (TypeError, ValueError):
            continue
        totals[(m.group(1), normalize_unit(str(unit)))] += qty_f
        rows += 1
    print(f"  loaded {rows} prefixed line items from DETAIL sheet")
    return dict(totals)


def main() -> None:
    print("Loading extractions...")
    extr = load_extraction()
    print("Loading spreadsheet...")
    sheet = load_spreadsheet()

    extr_keys = set(extr.keys())
    sheet_keys = set(sheet.keys())
    common = extr_keys & sheet_keys

    matches, diffs = [], []
    for k in sorted(common):
        e_qty, s_qty = extr[k], sheet[k]
        delta = e_qty - s_qty
        (matches if abs(delta) < TOLERANCE else diffs).append((k, e_qty, s_qty, delta))

    only_sheet = sorted(sheet_keys - extr_keys)
    only_extr = sorted(extr_keys - sheet_keys)

    line = "-" * 78

    def header():
        print(f"  {'CODE':6} {'UNIT':4}   {'EXTRACTED':>10} {'SHEET':>10} {'DELTA':>10}")

    print(f"\n{line}\nMATCHES ({len(matches)}):")
    header()
    for (code, unit), e, s, d in matches:
        print(f"  {code:6} {unit:4}   {e:>10.2f} {s:>10.2f} {d:>+10.3f}")

    print(f"\n{line}\nQUANTITY DIFFERS ({len(diffs)}):")
    header()
    for (code, unit), e, s, d in diffs:
        print(f"  {code:6} {unit:4}   {e:>10.2f} {s:>10.2f} {d:>+10.3f}")

    print(f"\n{line}\nIN SPREADSHEET ONLY ({len(only_sheet)}):")
    print(f"  {'CODE':6} {'UNIT':4}   {'SHEET':>10}")
    for code, unit in only_sheet:
        print(f"  {code:6} {unit:4}   {sheet[(code, unit)]:>10.2f}")

    print(f"\n{line}\nIN EXTRACTION ONLY ({len(only_extr)}):")
    print(f"  {'CODE':6} {'UNIT':4}   {'EXTRACTED':>10}")
    for code, unit in only_extr:
        print(f"  {code:6} {unit:4}   {extr[(code, unit)]:>10.2f}")

    total_keys = len(extr_keys | sheet_keys)
    match_rate = (len(matches) / total_keys) if total_keys else 0.0
    extr_total = sum(extr.values())
    sheet_total = sum(sheet.values())

    print(f"\n{line}\nSUMMARY")
    print(
        f"  unique (code, unit) keys:  extracted={len(extr_keys)}  "
        f"spreadsheet={len(sheet_keys)}  union={total_keys}"
    )
    print(
        f"  match rate (|delta| < {TOLERANCE}): "
        f"{len(matches)}/{total_keys} = {match_rate:.1%}"
    )
    print(
        f"  qty differs: {len(diffs)}    sheet-only: {len(only_sheet)}    "
        f"extr-only: {len(only_extr)}"
    )
    print(
        f"  aggregate quantity sum:  extracted={extr_total:.1f}  "
        f"spreadsheet={sheet_total:.1f}  delta={extr_total - sheet_total:+.1f}"
    )


if __name__ == "__main__":
    main()
