import type { Stage } from "../types";
import { Spinner } from "./Spinner";

interface StepDef {
  stage: Exclude<Stage, "done" | "error">;
  label: string;
}

const STEPS: StepDef[] = [
  { stage: "extracting", label: "Extracting scope items" },
  { stage: "pricing", label: "Pricing line items" },
  { stage: "scope_check", label: "Checking scope completeness" },
  { stage: "generating_excel", label: "Generating Excel workbook" },
];

const ORDER: Stage[] = STEPS.map((s) => s.stage);

type StepState = "pending" | "active" | "done";

function stateFor(stepIdx: number, current: Stage): StepState {
  if (current === "done") return "done";
  if (current === "error") return stepIdx === 0 ? "active" : "pending";
  const currentIdx = ORDER.indexOf(current);
  if (currentIdx === -1) return "pending";
  if (stepIdx < currentIdx) return "done";
  if (stepIdx === currentIdx) return "active";
  return "pending";
}

interface Props {
  current: Stage;
  extractCurrent?: number;
  extractTotal?: number;
}

export function ProgressPanel({ current, extractCurrent, extractTotal }: Props) {
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50/60 px-5 py-4">
      <ol className="space-y-3">
        {STEPS.map((step, i) => {
          const state = stateFor(i, current);
          const sub =
            step.stage === "extracting" && state !== "pending"
              ? extractTotal && extractTotal > 0
                ? state === "done"
                  ? `${extractTotal} page${extractTotal === 1 ? "" : "s"} processed`
                  : `Page ${Math.max(1, extractCurrent ?? 1)} of ${extractTotal}`
                : null
              : null;
          return (
            <li key={step.stage} className="flex items-start gap-3">
              <StepIcon state={state} />
              <div className="min-w-0 flex-1 pt-0.5">
                <div
                  className={[
                    "text-sm font-medium transition-colors",
                    state === "pending"
                      ? "text-slate-400"
                      : state === "active"
                        ? "text-slate-900"
                        : "text-slate-700",
                  ].join(" ")}
                >
                  {step.label}
                </div>
                {sub && (
                  <div className="text-xs text-slate-500 mt-0.5 tabular-nums">
                    {sub}
                  </div>
                )}
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}

function StepIcon({ state }: { state: StepState }) {
  if (state === "done") {
    return (
      <div className="h-6 w-6 rounded-full bg-emerald-600 text-white flex items-center justify-center flex-shrink-0">
        <svg
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="3"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <polyline points="20 6 9 17 4 12" />
        </svg>
      </div>
    );
  }
  if (state === "active") {
    return (
      <div className="h-6 w-6 rounded-full bg-white border border-accent text-accent flex items-center justify-center flex-shrink-0">
        <Spinner size={12} />
      </div>
    );
  }
  return (
    <div className="h-6 w-6 rounded-full bg-white border border-slate-300 flex items-center justify-center flex-shrink-0">
      <div className="h-1.5 w-1.5 rounded-full bg-slate-300" />
    </div>
  );
}
