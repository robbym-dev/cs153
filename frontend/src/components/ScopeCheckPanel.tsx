import type { ScopeAlert } from "../types";

interface Props {
  alerts: ScopeAlert[];
}

const SEVERITY_RANK: Record<ScopeAlert["severity"], number> = {
  critical: 0,
  warning: 1,
  info: 2,
};

const formatId = (id: string) =>
  id
    .split(/[_:]/)
    .map((s) => s.charAt(0).toUpperCase() + s.slice(1))
    .join(" ");

export function ScopeCheckPanel({ alerts }: Props) {
  const sorted = [...alerts].sort(
    (a, b) => SEVERITY_RANK[a.severity] - SEVERITY_RANK[b.severity]
  );
  const critical = sorted.filter((a) => a.severity === "critical").length;
  const warning = sorted.filter((a) => a.severity === "warning").length;

  return (
    <section className="panel">
      <div className="panel-header">
        <div>
          <h2 className="text-base font-semibold text-slate-900">
            Scope Completeness Check
          </h2>
          <p className="text-sm text-slate-500">
            Required items per project type & jurisdiction.
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs">
          {critical > 0 && (
            <span className="inline-flex items-center gap-1 rounded-full bg-rose-50 px-2.5 py-1 font-medium text-rose-700 border border-rose-100">
              {critical} critical
            </span>
          )}
          {warning > 0 && (
            <span className="inline-flex items-center gap-1 rounded-full bg-amber-50 px-2.5 py-1 font-medium text-amber-700 border border-amber-100">
              {warning} warning{warning === 1 ? "" : "s"}
            </span>
          )}
          {critical === 0 && warning === 0 && (
            <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 px-2.5 py-1 font-medium text-emerald-700 border border-emerald-100">
              All checks passed
            </span>
          )}
        </div>
      </div>
      <div className="panel-body space-y-3">
        {sorted.length === 0 ? (
          <AllClear />
        ) : (
          sorted.map((alert) => (
            <AlertCard key={alert.item_id} alert={alert} />
          ))
        )}
      </div>
    </section>
  );
}

function AllClear() {
  return (
    <div className="rounded-lg border border-emerald-200 bg-emerald-50 px-4 py-4 flex items-center gap-3">
      <div className="h-9 w-9 rounded-full bg-emerald-600 text-white flex items-center justify-center">
        <svg
          width="18"
          height="18"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2.6"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <polyline points="20 6 9 17 4 12" />
        </svg>
      </div>
      <div>
        <div className="font-medium text-emerald-900">All scope checks pass.</div>
        <div className="text-sm text-emerald-800/80">
          Required items present; jurisdiction-specific rules satisfied.
        </div>
      </div>
    </div>
  );
}

function AlertCard({ alert }: { alert: ScopeAlert }) {
  const style = severityStyle(alert.severity);
  return (
    <div
      className={`rounded-lg border ${style.border} ${style.bg} px-4 py-3 flex items-start gap-3`}
    >
      <div
        className={`h-8 w-8 rounded-full ${style.icon} text-white flex items-center justify-center flex-shrink-0`}
      >
        {alert.severity === "critical" ? (
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.4"
          >
            <line x1="12" y1="8" x2="12" y2="13" />
            <line x1="12" y1="17" x2="12" y2="17" />
            <circle cx="12" cy="12" r="10" />
          </svg>
        ) : (
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.4"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
            <line x1="12" y1="9" x2="12" y2="13" />
            <line x1="12" y1="17" x2="12.01" y2="17" />
          </svg>
        )}
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2 flex-wrap">
          <span className={`font-semibold ${style.title}`}>
            {formatId(alert.item_id)}
          </span>
          <span
            className={`text-[10px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded ${style.badge}`}
          >
            {alert.severity}
          </span>
        </div>
        <div className={`text-sm mt-0.5 ${style.body}`}>{alert.description}</div>
        <div className="text-xs mt-2 text-slate-600">
          <span className="font-medium">Suggested:</span>{" "}
          {alert.suggested_action}
        </div>
      </div>
    </div>
  );
}

function severityStyle(s: ScopeAlert["severity"]) {
  switch (s) {
    case "critical":
      return {
        border: "border-rose-200",
        bg: "bg-rose-50",
        icon: "bg-rose-600",
        title: "text-rose-900",
        body: "text-rose-900/85",
        badge: "bg-rose-200/70 text-rose-900",
      };
    case "warning":
      return {
        border: "border-amber-200",
        bg: "bg-amber-50",
        icon: "bg-amber-500",
        title: "text-amber-900",
        body: "text-amber-900/85",
        badge: "bg-amber-200/70 text-amber-900",
      };
    default:
      return {
        border: "border-slate-200",
        bg: "bg-slate-50",
        icon: "bg-slate-500",
        title: "text-slate-900",
        body: "text-slate-700",
        badge: "bg-slate-200 text-slate-700",
      };
  }
}
