import { useState } from "react";
import type { BidSummary, LineItem } from "../types";
import { fmtCurrency, fmtQty } from "../format";
import { Spinner } from "./Spinner";

interface Props {
  lineItems: LineItem[];
  summary: BidSummary;
  downloadUrl: string;
  downloadName?: string;
}

const markupLabel: Record<keyof BidSummary["markups"], string> = {
  overhead: "Overhead & Profit (20%)",
  tax: "Sales Tax (8.5%)",
  bid_bond: "Bid Bond (1.5%)",
  contingencies: "Contingencies (5%)",
};

export function BidSummaryPanel({
  lineItems,
  summary,
  downloadUrl,
  downloadName = "bid.xlsx",
}: Props) {
  const visibleLineItems = lineItems.filter((li) => li.quantity > 0);
  return (
    <section className="panel">
      <div className="panel-header">
        <div>
          <h2 className="text-base font-semibold text-slate-900">
            Bid Summary
          </h2>
          <p className="text-sm text-slate-500">
            Priced at NY Orange County prevailing wages, with waste &
            markups applied.
          </p>
        </div>
        <DownloadButton url={downloadUrl} filename={downloadName} />
      </div>

      <div className="overflow-x-auto border-b border-slate-200">
        <table className="min-w-full text-sm">
          <thead className="bg-slate-50 text-left">
            <tr className="text-xs font-medium uppercase tracking-wider text-slate-500">
              <th className="px-6 py-3 w-20">Code</th>
              <th className="px-6 py-3">Description</th>
              <th className="px-6 py-3 text-right w-24">Qty</th>
              <th className="px-6 py-3 w-16">Unit</th>
              <th className="px-6 py-3 text-right w-32">Unit Cost</th>
              <th className="px-6 py-3 text-right w-36">Total</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {visibleLineItems.map((li, i) => (
              <tr
                key={`${li.code}-${li.unit}-${i}`}
                className="hover:bg-slate-50/60"
              >
                <td className="px-6 py-3 font-mono font-medium text-slate-900">
                  {li.code}
                </td>
                <td className="px-6 py-3 text-slate-700">
                  {li.description || (
                    <span className="text-slate-400 italic">—</span>
                  )}
                </td>
                <td className="px-6 py-3 text-right tabular-nums text-slate-700">
                  {fmtQty(li.quantity)}
                </td>
                <td className="px-6 py-3 text-slate-700">{li.unit}</td>
                <td className="px-6 py-3 text-right tabular-nums text-slate-700">
                  {fmtCurrency(li.unit_cost)}
                </td>
                <td className="px-6 py-3 text-right tabular-nums font-medium text-slate-900">
                  {fmtCurrency(li.total)}
                </td>
              </tr>
            ))}
            {visibleLineItems.length === 0 && (
              <tr>
                <td
                  colSpan={6}
                  className="px-6 py-8 text-center text-slate-500"
                >
                  No priced line items with quantity &gt; 0.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="panel-body bg-slate-50/40">
        <div className="ml-auto max-w-md space-y-2">
          <Row label="Subtotal" value={summary.subtotal} />
          {(Object.keys(summary.markups) as Array<keyof BidSummary["markups"]>).map(
            (k) => (
              <Row key={k} label={markupLabel[k]} value={summary.markups[k]} muted />
            )
          )}
          <div className="border-t border-slate-300 pt-3 mt-3 flex items-baseline justify-between">
            <span className="text-sm font-semibold uppercase tracking-wider text-slate-700">
              Total Base Bid
            </span>
            <span className="text-2xl font-semibold tabular-nums text-slate-900">
              {fmtCurrency(summary.grand_total)}
            </span>
          </div>
        </div>
        <div className="mt-5 rounded-md border border-slate-200 bg-white px-4 py-3 text-xs leading-relaxed text-slate-600">
          <span className="font-semibold text-slate-700">Scope coverage:</span>{" "}
          this total covers <em>direct scope items only</em>. Scaffolding,
          general conditions, and site protection are flagged below in
          Scope Check and should be priced separately before submission.
        </div>
      </div>
    </section>
  );
}

function Row({
  label,
  value,
  muted,
}: {
  label: string;
  value: number;
  muted?: boolean;
}) {
  return (
    <div className="flex items-baseline justify-between text-sm">
      <span className={muted ? "text-slate-500" : "text-slate-700"}>
        {label}
      </span>
      <span
        className={`tabular-nums ${
          muted ? "text-slate-600" : "font-medium text-slate-900"
        }`}
      >
        {fmtCurrency(value)}
      </span>
    </div>
  );
}

function DownloadButton({
  url,
  filename,
}: {
  url: string;
  filename: string;
}) {
  const [pending, setPending] = useState(false);

  const handle = async () => {
    setPending(true);
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`download failed: ${res.status}`);
      const blob = await res.blob();
      const a = document.createElement("a");
      const objectUrl = URL.createObjectURL(blob);
      a.href = objectUrl;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(objectUrl);
    } catch (e) {
      alert(String(e));
    } finally {
      setPending(false);
    }
  };

  return (
    <button onClick={handle} className="btn-primary" disabled={pending}>
      {pending ? (
        <Spinner size={14} />
      ) : (
        <svg
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2.2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="7 10 12 15 17 10" />
          <line x1="12" y1="15" x2="12" y2="3" />
        </svg>
      )}
      Download Excel
    </button>
  );
}
