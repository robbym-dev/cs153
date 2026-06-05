import type { ScopeItem } from "../types";
import { fmtQty } from "../format";

interface Props {
  items: ScopeItem[];
}

const isDisplayable = (it: ScopeItem): boolean =>
  it.quantity > 0 && !it.unit.trim().toUpperCase().startsWith("DESC:");

export function ScopeItemsPanel({ items }: Props) {
  const visible = items.filter(isDisplayable);
  const hidden = items.length - visible.length;
  return (
    <section className="panel">
      <div className="panel-header">
        <div>
          <h2 className="text-base font-semibold text-slate-900">
            Extracted Scope Items
          </h2>
          <p className="text-sm text-slate-500">
            Real-quantity line items pulled from your marked-up plan.
          </p>
        </div>
        <span
          className="text-xs font-medium text-slate-500 bg-slate-100 px-2 py-1 rounded"
          title={
            hidden > 0
              ? `${hidden} zero-quantity / placeholder row${hidden === 1 ? "" : "s"} hidden`
              : undefined
          }
        >
          {visible.length} {visible.length === 1 ? "item" : "items"}
        </span>
      </div>
      <div className="overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead className="bg-slate-50 text-left">
            <tr className="text-xs font-medium uppercase tracking-wider text-slate-500">
              <th className="px-6 py-3 w-24">Code</th>
              <th className="px-6 py-3 text-right w-24">Qty</th>
              <th className="px-6 py-3 w-16">Unit</th>
              <th className="px-6 py-3">Description</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {visible.map((it, i) => (
              <tr
                key={`${it.code}-${it.unit}-${i}`}
                className="hover:bg-slate-50/60"
              >
                <td className="px-6 py-3 font-mono font-medium text-slate-900">
                  {it.code}
                </td>
                <td className="px-6 py-3 text-right tabular-nums text-slate-700">
                  {fmtQty(it.quantity)}
                </td>
                <td className="px-6 py-3 text-slate-700">{it.unit}</td>
                <td className="px-6 py-3 text-slate-700">
                  {it.description || (
                    <span className="text-slate-400 italic">
                      no description on file
                    </span>
                  )}
                </td>
              </tr>
            ))}
            {visible.length === 0 && (
              <tr>
                <td colSpan={4} className="px-6 py-8 text-center text-slate-500">
                  No real-quantity scope items extracted.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}
