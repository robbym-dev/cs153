interface Props {
  pages: string;
  state: string;
  stories: number;
  disabled?: boolean;
  onPages: (s: string) => void;
  onState: (s: string) => void;
  onStories: (n: number) => void;
}

const STATES = [
  "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
  "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
  "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
  "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
  "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC",
];

export function InputForm({
  pages,
  state,
  stories,
  disabled,
  onPages,
  onState,
  onStories,
}: Props) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
      <div className="space-y-2 sm:col-span-1">
        <label htmlFor="pages" className="label block">
          Page numbers
        </label>
        <input
          id="pages"
          type="text"
          inputMode="numeric"
          autoComplete="off"
          placeholder="e.g. 2,3,5,6"
          value={pages}
          disabled={disabled}
          onChange={(e) => onPages(e.target.value)}
          className="input"
        />
        <p className="text-xs text-slate-500">
          Comma-separated. Each page is extracted independently.
        </p>
      </div>

      <div className="space-y-2">
        <label htmlFor="state" className="label block">
          State
        </label>
        <select
          id="state"
          value={state}
          disabled={disabled}
          onChange={(e) => onState(e.target.value)}
          className="input pr-8 appearance-none bg-no-repeat bg-[right_0.6rem_center]"
          style={{
            backgroundImage:
              "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%23475569' stroke-width='2'><polyline points='6 9 12 15 18 9'/></svg>\")",
          }}
        >
          {STATES.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
        <p className="text-xs text-slate-500">
          Drives prevailing-wage checks.
        </p>
      </div>

      <div className="space-y-2">
        <label htmlFor="stories" className="label block">
          Stories
        </label>
        <input
          id="stories"
          type="number"
          min={1}
          max={20}
          value={stories}
          disabled={disabled}
          onChange={(e) =>
            onStories(Math.max(1, Math.min(20, Number(e.target.value) || 1)))
          }
          className="input"
        />
        <p className="text-xs text-slate-500">
          Multi-story → boom lift recommendation.
        </p>
      </div>
    </div>
  );
}
