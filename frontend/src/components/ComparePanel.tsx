import { useMemo, useRef, useState } from "react";
import type { LineItem, ReferenceLine, ReferenceResponse } from "../types";
import { fmtBytes, fmtCurrency, fmtQty } from "../format";
import { Spinner } from "./Spinner";
import { normalizeUnit } from "../units";

interface Props {
  lineItems: LineItem[];
}

interface Row {
  code: string;
  unit: string;
  description: string;
  ourQty: number;
  refQty: number | null;
  ourTotal: number;
  refTotal: number | null;
  source: "matched" | "ours_only" | "ref_only";
}

type Bucket = "good" | "fair" | "poor" | "none";

const WITHIN_GOOD = 15;
const WITHIN_FAIR = 30;

const bucketCellClass: Record<Bucket, string> = {
  good: "bg-emerald-50 text-emerald-900 border-emerald-100",
  fair: "bg-amber-50 text-amber-900 border-amber-100",
  poor: "bg-rose-50 text-rose-900 border-rose-100",
  none: "bg-slate-50 text-slate-500 border-slate-100",
};

function bucketFor(pct: number | null): Bucket {
  if (pct === null) return "none";
  const abs = Math.abs(pct);
  if (abs <= WITHIN_GOOD) return "good";
  if (abs <= WITHIN_FAIR) return "fair";
  return "poor";
}

function pct(ours: number, ref: number | null): number | null {
  if (ref === null || ref === 0) return null;
  return ((ours - ref) / ref) * 100;
}

function fmtPct(p: number | null): string {
  if (p === null) return "—";
  const sign = p > 0 ? "+" : "";
  return `${sign}${p.toFixed(1)}%`;
}

export function ComparePanel({ lineItems }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [refFile, setRefFile] = useState<File | null>(null);
  const [ref, setRef] = useState<ReferenceResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isOver, setIsOver] = useState(false);

  const upload = async (f: File) => {
    if (!/\.xlsx?$/i.test(f.name)) {
      setError("Please upload an Excel (.xlsx) file.");
      return;
    }
    setBusy(true);
    setError(null);
    setRef(null);
    setRefFile(f);
    try {
      const form = new FormData();
      form.append("reference", f);
      const res = await fetch("/api/parse-reference", {
        method: "POST",
        body: form,
      });
      if (!res.ok) {
        const detail = await res
          .json()
          .then((j: { detail?: string }) => j.detail)
          .catch(() => res.statusText);
        throw new Error(detail || `Request failed (${res.status})`);
      }
      const data = (await res.json()) as ReferenceResponse;
      setRef(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
      setRefFile(null);
    } finally {
      setBusy(false);
    }
  };

  const clear = () => {
    setRef(null);
    setRefFile(null);
    setError(null);
  };

  // Build comparison rows: union of our (code, unit) keys and reference's.
  const rows: Row[] = useMemo(() => {
    if (!ref) return [];
    const refByKey = new Map<string, ReferenceLine>();
    for (const r of ref.lines) {
      refByKey.set(`${r.code}|${normalizeUnit(r.unit)}`, r);
    }
    // Our side: aggregate by (code, normalized_unit) so WS10 LF and WS10 SF
    // remain distinct rows (matching the reference's keying).
    const ourMap = new Map<string, Row>();
    for (const li of lineItems) {
      const key = `${li.code}|${normalizeUnit(li.unit)}`;
      const existing = ourMap.get(key);
      if (existing) {
        existing.ourQty += li.quantity;
        existing.ourTotal += li.total;
      } else {
        const refLine = refByKey.get(key) ?? null;
        ourMap.set(key, {
          code: li.code,
          unit: normalizeUnit(li.unit),
          description: li.description,
          ourQty: li.quantity,
          ourTotal: li.total,
          refQty: refLine ? refLine.qty : null,
          refTotal: refLine ? refLine.total : null,
          source: refLine ? "matched" : "ours_only",
        });
      }
    }
    const built: Row[] = Array.from(ourMap.values());
    // Reference-only entries
    const ourKeys = new Set(ourMap.keys());
    for (const r of ref.lines) {
      const key = `${r.code}|${normalizeUnit(r.unit)}`;
      if (ourKeys.has(key)) continue;
      built.push({
        code: r.code,
        unit: r.unit,
        description: "",
        ourQty: 0,
        ourTotal: 0,
        refQty: r.qty,
        refTotal: r.total,
        source: "ref_only",
      });
    }
    built.sort((a, b) => a.code.localeCompare(b.code) || a.unit.localeCompare(b.unit));
    return built;
  }, [ref, lineItems]);

  // Aggregate accuracy — over matched rows where ref total > 0.
  const stats = useMemo(() => {
    const matched = rows.filter((r) => r.source === "matched" && (r.refTotal ?? 0) > 0);
    const withinGood = matched.filter((r) => {
      const p = pct(r.ourTotal, r.refTotal);
      return p !== null && Math.abs(p) <= WITHIN_GOOD;
    }).length;
    const ourSum = matched.reduce((s, r) => s + r.ourTotal, 0);
    const refSum = matched.reduce((s, r) => s + (r.refTotal ?? 0), 0);
    const aggregateDelta = refSum > 0 ? ((ourSum - refSum) / refSum) * 100 : null;
    return {
      matchedCount: matched.length,
      withinGood,
      ourSum,
      refSum,
      aggregateDelta,
    };
  }, [rows]);

  return (
    <section className="panel">
      <div className="panel-header">
        <div>
          <h2 className="text-base font-semibold text-slate-900">
            Compare against reference bid
          </h2>
          <p className="text-sm text-slate-500">
            Optional — upload a historical bid spreadsheet to score this
            AI-generated bid against it line-for-line.
          </p>
        </div>
        {ref && (
          <button onClick={clear} className="btn-secondary">
            Clear
          </button>
        )}
      </div>

      <div className="panel-body space-y-5">
        {!ref && (
          <div
            onDragOver={(e) => {
              e.preventDefault();
              if (!busy) setIsOver(true);
            }}
            onDragLeave={() => setIsOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setIsOver(false);
              if (busy) return;
              const f = e.dataTransfer.files?.[0];
              if (f) upload(f);
            }}
            onClick={() => !busy && inputRef.current?.click()}
            role="button"
            tabIndex={0}
            className={[
              "cursor-pointer rounded-xl border-2 border-dashed transition",
              "px-6 py-8 flex flex-col items-center justify-center gap-2 text-center",
              isOver
                ? "border-accent bg-blue-50/60"
                : "border-slate-300 hover:border-slate-400 hover:bg-slate-50",
              busy ? "opacity-60 cursor-not-allowed" : "",
            ].join(" ")}
          >
            <input
              ref={inputRef}
              type="file"
              accept=".xlsx,.xls,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
              className="hidden"
              disabled={busy}
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) upload(f);
              }}
            />
            {busy ? (
              <>
                <Spinner size={18} className="text-accent" />
                <div className="font-medium text-slate-700">
                  Parsing {refFile?.name}…
                </div>
                <div className="text-xs text-slate-500">
                  {refFile && fmtBytes(refFile.size)}
                </div>
              </>
            ) : (
              <>
                <div className="h-10 w-10 rounded-full bg-slate-100 flex items-center justify-center">
                  <svg
                    viewBox="0 0 24 24"
                    className="h-5 w-5 text-emerald-600"
                    fill="currentColor"
                  >
                    <path d="M6 2a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6H6zm7 1.5L18.5 9H14a1 1 0 0 1-1-1V3.5zM7.6 12.4l1.6 2.4 1.6-2.4h1.7l-2.4 3.4 2.5 3.6h-1.7L9.2 17l-1.7 2.4H5.8l2.5-3.6-2.4-3.4h1.7z" />
                  </svg>
                </div>
                <div className="font-medium text-slate-700">
                  Drop a reference bid .xlsx here
                </div>
                <div className="text-sm text-slate-500">
                  or click to browse — Tyler-format DETAIL sheet expected
                </div>
              </>
            )}
          </div>
        )}

        {error && (
          <div className="rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800">
            <strong className="font-semibold">Error:</strong> {error}
          </div>
        )}

        {ref && (
          <>
            <AccuracyStrip
              filename={ref.filename}
              matched={stats.matchedCount}
              within={stats.withinGood}
              aggregateDelta={stats.aggregateDelta}
              ourSum={stats.ourSum}
              refSum={stats.refSum}
            />
            <CompareTable rows={rows} />
            <LegendBar />
          </>
        )}
      </div>
    </section>
  );
}

function AccuracyStrip({
  filename,
  matched,
  within,
  aggregateDelta,
  ourSum,
  refSum,
}: {
  filename: string;
  matched: number;
  within: number;
  aggregateDelta: number | null;
  ourSum: number;
  refSum: number;
}) {
  const aggBucket = bucketFor(aggregateDelta);
  const goodPct = matched > 0 ? (within / matched) * 100 : 0;
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50/60 px-5 py-4">
      <div className="flex items-baseline justify-between mb-3">
        <div className="text-sm">
          <span className="font-semibold text-slate-900">{within}</span>
          <span className="text-slate-600">
            {" "}/ {matched} line items within ±{WITHIN_GOOD}%
          </span>
          <span className="text-slate-400 mx-2">|</span>
          <span className="text-slate-600">Aggregate delta: </span>
          <span
            className={[
              "font-semibold",
              aggBucket === "good"
                ? "text-emerald-700"
                : aggBucket === "fair"
                  ? "text-amber-700"
                  : aggBucket === "poor"
                    ? "text-rose-700"
                    : "text-slate-700",
            ].join(" ")}
          >
            {fmtPct(aggregateDelta)}
          </span>
        </div>
        <div className="text-xs text-slate-500 truncate max-w-xs" title={filename}>
          vs. {filename}
        </div>
      </div>
      <div className="h-1.5 w-full rounded-full bg-slate-200 overflow-hidden">
        <div
          className="h-full bg-emerald-500 transition-all"
          style={{ width: `${goodPct}%` }}
        />
      </div>
      <div className="mt-3 flex items-center gap-6 text-xs text-slate-500 tabular-nums">
        <span>
          Ours: <span className="text-slate-900 font-medium">{fmtCurrency(ourSum)}</span>
        </span>
        <span>
          Reference:{" "}
          <span className="text-slate-900 font-medium">{fmtCurrency(refSum)}</span>
        </span>
        <span>
          Δ:{" "}
          <span className="text-slate-900 font-medium">
            {fmtCurrency(ourSum - refSum)}
          </span>
        </span>
      </div>
    </div>
  );
}

function CompareTable({ rows }: { rows: Row[] }) {
  return (
    <div className="overflow-x-auto rounded-lg border border-slate-200">
      <table className="min-w-full text-sm">
        <thead className="bg-slate-50 text-left">
          <tr className="text-xs font-medium uppercase tracking-wider text-slate-500">
            <th className="px-4 py-2.5 w-20">Code</th>
            <th className="px-4 py-2.5 w-14">Unit</th>
            <th className="px-4 py-2.5 text-right w-24">Our Qty</th>
            <th className="px-4 py-2.5 text-right w-24">Ref Qty</th>
            <th className="px-4 py-2.5 text-right w-20">Δ Qty</th>
            <th className="px-4 py-2.5 text-right w-28">Our Total</th>
            <th className="px-4 py-2.5 text-right w-28">Ref Total</th>
            <th className="px-4 py-2.5 text-right w-20">Δ Total</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100">
          {rows.map((r) => {
            const qtyPct = pct(r.ourQty, r.refQty);
            const totalPct = pct(r.ourTotal, r.refTotal);
            const qtyBucket =
              r.source === "matched" ? bucketFor(qtyPct) : "none";
            const totalBucket =
              r.source === "matched" ? bucketFor(totalPct) : "none";
            return (
              <tr key={`${r.code}-${r.unit}`} className="hover:bg-slate-50/60">
                <td className="px-4 py-2.5 font-mono font-medium text-slate-900">
                  {r.code}
                </td>
                <td className="px-4 py-2.5 text-slate-700">{r.unit}</td>
                <td className="px-4 py-2.5 text-right tabular-nums text-slate-700">
                  {r.source === "ref_only" ? "—" : fmtQty(r.ourQty)}
                </td>
                <td className="px-4 py-2.5 text-right tabular-nums text-slate-700">
                  {r.refQty === null ? "—" : fmtQty(r.refQty)}
                </td>
                <td
                  className={`px-4 py-2.5 text-right tabular-nums text-xs font-semibold border-l ${bucketCellClass[qtyBucket]}`}
                >
                  {qtyBucket === "none" ? "—" : fmtPct(qtyPct)}
                </td>
                <td className="px-4 py-2.5 text-right tabular-nums text-slate-700">
                  {r.source === "ref_only" ? "—" : fmtCurrency(r.ourTotal)}
                </td>
                <td className="px-4 py-2.5 text-right tabular-nums text-slate-700">
                  {r.refTotal === null ? "—" : fmtCurrency(r.refTotal)}
                </td>
                <td
                  className={`px-4 py-2.5 text-right tabular-nums text-xs font-semibold border-l ${bucketCellClass[totalBucket]}`}
                >
                  {totalBucket === "none" ? "—" : fmtPct(totalPct)}
                </td>
              </tr>
            );
          })}
          {rows.length === 0 && (
            <tr>
              <td colSpan={8} className="px-4 py-8 text-center text-slate-500">
                No comparable rows.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

function LegendBar() {
  return (
    <div className="flex items-center gap-4 text-xs text-slate-500">
      <span className="inline-flex items-center gap-1.5">
        <span className="h-3 w-3 rounded bg-emerald-200 border border-emerald-300" />
        ≤ {WITHIN_GOOD}%
      </span>
      <span className="inline-flex items-center gap-1.5">
        <span className="h-3 w-3 rounded bg-amber-200 border border-amber-300" />
        ≤ {WITHIN_FAIR}%
      </span>
      <span className="inline-flex items-center gap-1.5">
        <span className="h-3 w-3 rounded bg-rose-200 border border-rose-300" />
        &gt; {WITHIN_FAIR}%
      </span>
      <span className="inline-flex items-center gap-1.5">
        <span className="h-3 w-3 rounded bg-slate-200 border border-slate-300" />
        unmatched
      </span>
    </div>
  );
}
