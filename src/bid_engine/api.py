"""FastAPI backend for the bid engine web UI.

Two endpoints for running the pipeline:

  - POST /api/generate-bid          — synchronous, returns full JSON.
  - POST /api/generate-bid-stream   — Server-Sent Events; one event per
                                       pipeline stage so the UI can render a
                                       step-by-step progress indicator.

Excel outputs are served as static files from /api/downloads/.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
import uuid
from collections import defaultdict
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from bid_engine.bid_generator import BidHeader, generate_bid_excel
from bid_engine.comparison import load_reference_bid
from bid_engine.extraction import extract_page
from bid_engine.pipeline import PipelineResult, run_pipeline
from bid_engine.pricing import (
    DEFAULT_UNIT_COSTS,
    ScopeItem,
    normalize_unit,
    price_bid,
)
from bid_engine.scope_checker import ProjectConfig, check_scope

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

DOWNLOADS_DIR = Path(tempfile.gettempdir()) / "bid_engine_downloads"
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="BidEngine API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/api/downloads",
    StaticFiles(directory=str(DOWNLOADS_DIR)),
    name="downloads",
)


def _parse_pages(pages: str) -> list[int]:
    out: list[int] = []
    for tok in pages.replace(" ", "").split(","):
        if not tok:
            continue
        try:
            n = int(tok)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"invalid page number {tok!r} — must be a positive integer",
            ) from exc
        if n < 1:
            raise HTTPException(
                status_code=400,
                detail=f"page numbers must be >= 1, got {n}",
            )
        out.append(n)
    if not out:
        raise HTTPException(
            status_code=400, detail="at least one page number is required"
        )
    return out


def _description_for(code: str, unit: str) -> str:
    key = (code, normalize_unit(unit))
    uc = DEFAULT_UNIT_COSTS.get(key)
    return uc.description if uc else ""


def _serialize(result: PipelineResult, download_url: str) -> dict:
    bid = result.bid
    scope_items_json = [
        {
            "code": si.code,
            "quantity": round(si.quantity, 2),
            "unit": si.unit,
            "description": _description_for(si.code, si.unit),
        }
        for si in result.scope_items
    ]
    line_items_json = [
        {
            "code": li.scope_item.code,
            "description": _description_for(li.scope_item.code, li.scope_item.unit),
            "quantity": round(li.scope_item.quantity, 2),
            "unit": li.scope_item.unit,
            "unit_cost": round(li.unit_labor + li.unit_material, 2),
            "unit_labor": round(li.unit_labor, 2),
            "unit_material": round(li.unit_material, 2),
            "total": round(li.total_cost, 2),
        }
        for li in bid.line_items
    ]
    summary_json = {
        "subtotal": round(bid.subtotal, 2),
        "markups": {
            "overhead": round(bid.overhead, 2),
            "tax": round(bid.tax, 2),
            "bid_bond": round(bid.bid_bond, 2),
            "contingencies": round(bid.contingencies, 2),
        },
        "grand_total": round(bid.total, 2),
    }
    # Prevailing-wage alerts are suppressed in the response: pricing.py
    # already applies NY Orange County prevailing wages via DEFAULT_WAGES,
    # so the scope-checker's "no wage_rates supplied" warning is misleading
    # for users of the web UI. The check still runs in the pipeline (for
    # callers that pass their own ProjectConfig.wage_rates and need
    # verification) but is hidden from the bid response.
    alerts_json = [
        {
            "item_id": a.item_id,
            "severity": a.severity,
            "description": a.description,
            "suggested_action": a.suggested_action,
        }
        for a in result.alerts
        if not a.item_id.startswith("prevailing_wage")
    ]
    return {
        "scope_items": scope_items_json,
        "line_items": line_items_json,
        "summary": summary_json,
        "alerts": alerts_json,
        "download_url": download_url,
    }


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _save_upload(pdf: UploadFile) -> tuple[Path, Path, str]:
    if not pdf.filename or not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="upload must be a PDF file")
    job_id = uuid.uuid4().hex[:12]
    upload_dir = DOWNLOADS_DIR / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = upload_dir / "input.pdf"
    with pdf_path.open("wb") as f:
        shutil.copyfileobj(pdf.file, f)
    return pdf_path, upload_dir / "bid.xlsx", job_id


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/generate-bid")
async def generate_bid(
    pdf: UploadFile = File(...),
    pages: str = Form(...),
    state: str = Form("NY"),
    stories: int = Form(3),
) -> JSONResponse:
    page_numbers = _parse_pages(pages)
    pdf_path, output_xlsx, job_id = _save_upload(pdf)
    header = BidHeader(project_name=Path(pdf.filename or "bid").stem)
    config = ProjectConfig(state=state.strip().upper(), stories=int(stories))

    logger.info(
        "job %s: pdf=%s pages=%s state=%s stories=%d",
        job_id, pdf.filename, page_numbers, state, stories,
    )
    try:
        result = await run_in_threadpool(
            run_pipeline,
            pdf_path,
            page_numbers,
            project_config=config,
            output_path=output_xlsx,
            header=header,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("pipeline failed")
        raise HTTPException(
            status_code=500, detail=f"pipeline failure: {exc}"
        ) from exc

    download_url = f"/api/downloads/{job_id}/bid.xlsx"
    return JSONResponse(_serialize(result, download_url))


@app.post("/api/generate-bid-stream")
async def generate_bid_stream(
    pdf: UploadFile = File(...),
    pages: str = Form(...),
    state: str = Form("NY"),
    stories: int = Form(3),
) -> StreamingResponse:
    """SSE-stream pipeline progress. Stages emitted in order:

        extracting (one event per page, with current/total/page)
        pricing
        scope_check
        generating_excel
        done    (carries the full result payload)
        error   (terminal; carries a message)
    """
    page_numbers = _parse_pages(pages)
    pdf_path, output_xlsx, job_id = _save_upload(pdf)
    header = BidHeader(project_name=Path(pdf.filename or "bid").stem)
    config = ProjectConfig(state=state.strip().upper(), stories=int(stories))
    n_pages = len(page_numbers)

    logger.info(
        "job %s (stream): pdf=%s pages=%s state=%s stories=%d",
        job_id, pdf.filename, page_numbers, state, stories,
    )

    async def event_stream() -> AsyncIterator[str]:
        try:
            yield _sse({"stage": "extracting", "current": 0, "total": n_pages})
            totals: dict[tuple[str, str], float] = defaultdict(float)
            for i, page in enumerate(page_numbers, 1):
                yield _sse({
                    "stage": "extracting",
                    "current": i,
                    "total": n_pages,
                    "page": page,
                })
                # yield to the event loop so the client sees the update
                # *before* the long Vision call blocks the threadpool slot.
                await asyncio.sleep(0)
                try:
                    items = await run_in_threadpool(extract_page, pdf_path, page)
                except Exception as exc:
                    logger.error("page %d extraction failed: %s", page, exc)
                    yield _sse({
                        "stage": "extracting",
                        "current": i,
                        "total": n_pages,
                        "page": page,
                        "warning": f"page {page}: {exc}",
                    })
                    continue
                for item in items:
                    try:
                        totals[(item["code"], item["unit"])] += float(item["quantity"])
                    except (KeyError, TypeError, ValueError):
                        continue

            if not totals:
                yield _sse({
                    "stage": "error",
                    "message": (
                        f"extraction produced no scope items across pages "
                        f"{page_numbers}"
                    ),
                })
                return

            scope_items = tuple(
                ScopeItem(code=code, quantity=qty, unit=unit)
                for (code, unit), qty in sorted(totals.items())
            )

            yield _sse({"stage": "pricing"})
            await asyncio.sleep(0)
            bid = await run_in_threadpool(price_bid, scope_items)

            yield _sse({"stage": "scope_check"})
            await asyncio.sleep(0)
            alerts = tuple(await run_in_threadpool(check_scope, scope_items, config))

            yield _sse({"stage": "generating_excel"})
            await asyncio.sleep(0)
            output = await run_in_threadpool(
                generate_bid_excel, bid, header, output_xlsx
            )

            result = PipelineResult(
                bid=bid,
                alerts=alerts,
                output_path=output,
                scope_items=scope_items,
            )
            download_url = f"/api/downloads/{job_id}/bid.xlsx"
            yield _sse({"stage": "done", "result": _serialize(result, download_url)})
        except Exception as exc:
            logger.exception("stream pipeline failed")
            yield _sse({"stage": "error", "message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/api/parse-reference")
async def parse_reference(reference: UploadFile = File(...)) -> JSONResponse:
    """Parse a Tyler-format reference bid XLSX and return per-line qty + total.

    Used by the frontend's Compare panel to score our generated bid against
    a historical reference.
    """
    fname = (reference.filename or "").lower()
    if not (fname.endswith(".xlsx") or fname.endswith(".xls")):
        raise HTTPException(
            status_code=400, detail="upload must be an Excel file (.xlsx)"
        )

    tmp_dir = DOWNLOADS_DIR / f"ref_{uuid.uuid4().hex[:8]}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / (reference.filename or "reference.xlsx")
    try:
        with tmp_path.open("wb") as f:
            shutil.copyfileobj(reference.file, f)
        try:
            ref = await run_in_threadpool(load_reference_bid, tmp_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("reference parse failed")
            raise HTTPException(
                status_code=500, detail=f"could not parse spreadsheet: {exc}"
            ) from exc
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
            tmp_dir.rmdir()
        except OSError:
            pass

    lines = [
        {
            "code": ln.code,
            "unit": ln.unit,
            "qty": round(ln.qty, 2),
            "total": round(ln.total, 2),
        }
        for ln in sorted(ref.values(), key=lambda x: (x.code, x.unit))
    ]
    return JSONResponse({
        "filename": reference.filename,
        "lines": lines,
        "aggregate_total": round(sum(ln.total for ln in ref.values()), 2),
    })
