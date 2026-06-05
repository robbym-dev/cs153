import { useRef, useState } from "react";
import { Header } from "./components/Header";
import { UploadZone } from "./components/UploadZone";
import { InputForm } from "./components/InputForm";
import { ProgressPanel } from "./components/ProgressPanel";
import { ScopeItemsPanel } from "./components/ScopeItemsPanel";
import { BidSummaryPanel } from "./components/BidSummaryPanel";
import { ScopeCheckPanel } from "./components/ScopeCheckPanel";
import { ComparePanel } from "./components/ComparePanel";
import { postSSE } from "./sse";
import type { BidResponse, Stage } from "./types";

interface ProgressState {
  stage: Stage;
  extractCurrent?: number;
  extractTotal?: number;
}

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [pages, setPages] = useState("2,3,5,6");
  const [state, setState] = useState("NY");
  const [stories, setStories] = useState(3);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<ProgressState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<BidResponse | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const canSubmit = !!file && pages.trim().length > 0 && !busy;

  const onGenerate = async () => {
    if (!file) return;
    setBusy(true);
    setError(null);
    setResult(null);
    setProgress({ stage: "extracting", extractCurrent: 0 });
    abortRef.current = new AbortController();
    try {
      const form = new FormData();
      form.append("pdf", file);
      form.append("pages", pages);
      form.append("state", state);
      form.append("stories", String(stories));

      await postSSE(
        "/api/generate-bid-stream",
        form,
        (evt) => {
          if (evt.stage === "done" && evt.result) {
            setResult(evt.result);
            setProgress({ stage: "done" });
            return;
          }
          if (evt.stage === "error") {
            setError(evt.message || "Pipeline error.");
            setProgress(null);
            return;
          }
          setProgress((prev) => ({
            stage: evt.stage,
            extractCurrent:
              evt.stage === "extracting" ? evt.current : prev?.extractCurrent,
            extractTotal:
              evt.stage === "extracting" ? evt.total : prev?.extractTotal,
          }));
        },
        abortRef.current.signal
      );
    } catch (e: unknown) {
      if ((e as Error).name !== "AbortError") {
        setError(e instanceof Error ? e.message : String(e));
      }
      setProgress(null);
    } finally {
      setBusy(false);
    }
  };

  const onCancel = () => {
    abortRef.current?.abort();
  };

  const downloadName = file
    ? `${file.name.replace(/\.pdf$/i, "")}_bid.xlsx`
    : "bid.xlsx";

  return (
    <div className="min-h-full flex flex-col bg-slate-50">
      <Header />

      <main className="flex-1">
        <div className="mx-auto max-w-6xl px-6 py-8 space-y-6">
          <section className="panel">
            <div className="panel-body space-y-6">
              <UploadZone file={file} onFile={setFile} disabled={busy} />
              <InputForm
                pages={pages}
                state={state}
                stories={stories}
                disabled={busy}
                onPages={setPages}
                onState={setState}
                onStories={setStories}
              />

              {busy && progress ? (
                <div className="pt-2 border-t border-slate-100 space-y-4">
                  <ProgressPanel
                    current={progress.stage}
                    extractCurrent={progress.extractCurrent}
                    extractTotal={progress.extractTotal}
                  />
                  <div className="flex items-center justify-between">
                    <p className="text-xs text-slate-500">
                      Extraction calls Claude Vision per page — typically
                      30–60s each.
                    </p>
                    <button onClick={onCancel} className="btn-secondary">
                      Cancel
                    </button>
                  </div>
                </div>
              ) : (
                <div className="flex items-center justify-between gap-4 pt-2 border-t border-slate-100">
                  <div className="text-sm text-slate-500">
                    Ready when you are. We'll extract, price, and check
                    completeness in one pass.
                  </div>
                  <button
                    onClick={onGenerate}
                    disabled={!canSubmit}
                    className="btn-primary"
                  >
                    Generate Bid
                  </button>
                </div>
              )}

              {error && (
                <div className="rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800">
                  <strong className="font-semibold">Error:</strong> {error}
                </div>
              )}
            </div>
          </section>

          {result && (
            <div className="space-y-6">
              <BidSummaryPanel
                lineItems={result.line_items}
                summary={result.summary}
                downloadUrl={result.download_url}
                downloadName={downloadName}
              />
              <ComparePanel lineItems={result.line_items} />
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <ScopeItemsPanel items={result.scope_items} />
                <ScopeCheckPanel alerts={result.alerts} />
              </div>
            </div>
          )}
        </div>
      </main>

      <footer className="border-t border-slate-200 bg-white">
        <div className="mx-auto max-w-6xl px-6 py-4 text-xs text-slate-500 flex items-center justify-between">
          <span>BidEngine · CS153 project</span>
          <span>Prices calibrated to NY Orange County prevailing wages</span>
        </div>
      </footer>
    </div>
  );
}
