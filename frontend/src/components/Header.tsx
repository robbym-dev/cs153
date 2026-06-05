export function Header() {
  return (
    <header className="bg-ink-900 text-white">
      <div className="mx-auto max-w-6xl px-6 py-5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="h-9 w-9 rounded-md bg-white flex items-center justify-center">
            <svg viewBox="0 0 32 32" className="h-6 w-6">
              <path
                d="M9 8h7.5a4.5 4.5 0 0 1 2.4 8.32A5 5 0 0 1 17 24H9V8zm3 3v4h4a2 2 0 1 0 0-4h-4zm0 7v4h5a2 2 0 1 0 0-4h-5z"
                fill="#0B1220"
              />
            </svg>
          </div>
          <div>
            <div className="flex items-baseline gap-2">
              <h1 className="text-xl font-semibold tracking-tight">
                BidEngine
              </h1>
              <span className="text-xs font-medium uppercase tracking-wider text-slate-400">
                v0.1
              </span>
            </div>
            <p className="text-sm text-slate-300">
              AI-assisted scope extraction & priced bid generation for
              construction plans.
            </p>
          </div>
        </div>
        <div className="hidden sm:flex items-center gap-4 text-xs text-slate-400">
          <span className="inline-flex items-center gap-1.5">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
            Connected
          </span>
        </div>
      </div>
    </header>
  );
}
