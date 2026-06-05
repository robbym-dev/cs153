# bid-engine

## Overview

Construction estimators spend hours hand-counting items off marked-up plan PDFs and pricing each one against historical unit costs. `bid-engine` automates that pipeline: it takes a marked-up plan PDF, runs each page through Claude Vision to extract coded scope items (e.g. `WS3: 120 LF`), aggregates them into a quantity takeoff, prices the takeoff against a calibrated unit-cost library, runs a scope-completeness check, and emits a contractor-ready Excel bid — converting a multi-hour manual workflow into a single command (or one click in the web UI).

## Install

```sh
make install
```

Requires Python 3.10+ and `poppler` (for `pdf2image`). On macOS:

```sh
brew install poppler
```

Set your Anthropic API key:

```sh
export ANTHROPIC_API_KEY=sk-ant-...
```

## Run

**CLI:**

```sh
python scripts/bid.py plan.pdf --pages 2,3,5,6 --state NY --output bid.xlsx
```

Optional flags: `--stories <N>`, `--name <project name>`, `--address <addr>`, `--date YYYY-MM-DD`, `--quiet`.

**Web UI:** start the FastAPI backend and the Vite frontend in two shells:

```sh
uvicorn bid_engine.api:app --host 127.0.0.1 --port 8765 --reload
```

```sh
cd frontend && npm install && npm run dev
```

Then open http://localhost:5173, drop a PDF, and click **Generate Bid**.

## Architecture

- **Extraction** (`bid_engine/extraction.py`) — sends each rendered page to Claude Opus 4.7 Vision with a structured prompt and parses the response into `(code, quantity, unit)` tuples.
- **Pricing** (`bid_engine/pricing.py`) — joins scope items against a calibrated unit-cost library (NY Orange County prevailing wages applied deterministically) and sums labor + material into a `Bid` with overhead, tax, bond, and contingency markups.
- **Scope checker** (`bid_engine/scope_checker.py`) — runs deterministic rules over the takeoff (missing companions, story-based scaffolding triggers, GC line items) and emits typed `ScopeAlert`s.
- **Bid generator** (`bid_engine/bid_generator.py`) — writes the priced bid to an `openpyxl` workbook matching the estimator's existing template (DETAIL sheet, header row 26, data from row 28).

## Evaluation

Validated on 2 NYC projects (Park Avenue Elementary School, JHS 145 Bronx). On Park Avenue: **22 of 24 line items within ±15%** of the estimator's reference bid, **aggregate pricing delta ≈ 4%**.

## Limitations

- Scope is **DIV-07 only** (waterproofing, dampproofing, building envelope sealants); other CSI divisions are out of scope.
- Unit costs are calibrated from a **single estimator's** historical data — accuracy will degrade for trades or regions outside that calibration window.
- Performance is gated on **clean markup**: raw-plan (unmarked) takeoffs cap around **38% OCR accuracy** in our test set. The tool is designed for the marked-up plan workflow that estimators already produce.

## AI Disclosure

Built using Claude Code for development assistance. Claude Opus 4.7 Vision API used for PDF extraction. All AI usage is in extraction — pricing and scope checking are deterministic.

## Test

```sh
make test
```

85 tests passing.
