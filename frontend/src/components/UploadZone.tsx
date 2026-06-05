import { useCallback, useRef, useState } from "react";
import { fmtBytes } from "../format";

interface Props {
  file: File | null;
  onFile: (f: File | null) => void;
  disabled?: boolean;
}

export function UploadZone({ file, onFile, disabled }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isOver, setIsOver] = useState(false);

  const accept = useCallback(
    (f: File | null) => {
      if (!f) return;
      if (!f.name.toLowerCase().endsWith(".pdf")) {
        alert("Please upload a PDF file.");
        return;
      }
      onFile(f);
    },
    [onFile]
  );

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsOver(false);
    if (disabled) return;
    const f = e.dataTransfer.files?.[0] ?? null;
    accept(f);
  };

  return (
    <div className="space-y-2">
      <div className="label">Plan PDF</div>
      <div
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setIsOver(true);
        }}
        onDragLeave={() => setIsOver(false)}
        onDrop={onDrop}
        onClick={() => !disabled && inputRef.current?.click()}
        role="button"
        tabIndex={0}
        className={[
          "relative cursor-pointer rounded-xl border-2 border-dashed transition",
          "px-6 py-10 flex flex-col items-center justify-center gap-3 text-center",
          isOver
            ? "border-accent bg-blue-50/60"
            : "border-slate-300 hover:border-slate-400 hover:bg-slate-50",
          disabled ? "opacity-50 cursor-not-allowed" : "",
          file ? "bg-slate-50" : "",
        ].join(" ")}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,application/pdf"
          className="hidden"
          disabled={disabled}
          onChange={(e) => accept(e.target.files?.[0] ?? null)}
        />
        {file ? (
          <div className="flex items-center gap-4">
            <div className="h-12 w-12 rounded-lg bg-white border border-slate-200 flex items-center justify-center">
              <svg
                viewBox="0 0 24 24"
                className="h-6 w-6 text-rose-600"
                fill="currentColor"
              >
                <path d="M6 2a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6H6zm7 1.5L18.5 9H14a1 1 0 0 1-1-1V3.5zM8 13h8v1.5H8V13zm0 3h8v1.5H8V16zm0 3h5v1.5H8V19z" />
              </svg>
            </div>
            <div className="text-left">
              <div className="font-medium text-slate-900">{file.name}</div>
              <div className="text-sm text-slate-500">
                {fmtBytes(file.size)} ·{" "}
                <button
                  type="button"
                  className="text-accent hover:underline"
                  onClick={(e) => {
                    e.stopPropagation();
                    onFile(null);
                  }}
                >
                  Remove
                </button>
              </div>
            </div>
          </div>
        ) : (
          <>
            <div className="h-12 w-12 rounded-full bg-slate-100 flex items-center justify-center">
              <svg
                viewBox="0 0 24 24"
                className="h-6 w-6 text-slate-500"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
            </div>
            <div>
              <div className="font-medium text-slate-700">
                Drop a marked-up plan PDF here
              </div>
              <div className="text-sm text-slate-500">
                or click to browse — single PDF, no size limit
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
