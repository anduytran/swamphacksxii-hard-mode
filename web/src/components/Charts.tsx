// LastScene chart kit: small dependency-free charts that match the dashboard's tokens
// (styles.css → "Insights"). Every chart has a hover tooltip with the exact value and a text
// alternative. See README → "Adding a visualization".
//
// Colours: series use the validated slots --series-1 (periwinkle) and --series-2 (deep gold),
// one pair per theme. To add a third series colour, validate it against both surfaces first:
//   node <dataviz skill>/scripts/validate_palette.js "#5F5BC0,#B07D12,#NEW" --mode light --surface "#F4F5F9"
//   node <dataviz skill>/scripts/validate_palette.js "#7A76CC,#BF8714,#NEW" --mode dark --surface "#1E1D3D"
// and add --series-3 / .s3 in styles.css. Never reuse the status colours for data.
import { useMemo, useState, type PointerEvent } from "react";

// ---------- one tooltip element shared by every chart ----------

let tipEl: HTMLDivElement | null = null;

export function ChartTooltip() {
  return <div ref={(el) => void (tipEl = el)} className="tooltip" hidden />;
}

/** Pointer handlers that show `text` in the shared tooltip next to the cursor. */
export function tooltip(text: string) {
  return {
    onPointerEnter: () => {
      if (!tipEl) return;
      tipEl.textContent = text;
      tipEl.hidden = false;
    },
    onPointerMove: (e: PointerEvent) => {
      if (!tipEl) return;
      const x = Math.min(e.clientX + 12, window.innerWidth - tipEl.offsetWidth - 8);
      tipEl.style.transform = `translate(${x}px, ${e.clientY + 12}px)`;
    },
    onPointerLeave: () => {
      if (tipEl) tipEl.hidden = true;
    },
  };
}

// ---------- horizontal bars: one series, ranked ----------

export interface BarRow {
  label: string;
  value: number;
  key: string;
}

/** With onSelect, each row is a 44 px button that drills down into that row. */
export function BarList({ rows, unit = "", empty, onSelect }: { rows: BarRow[]; unit?: string; empty: string; onSelect?: (row: BarRow) => void }) {
  if (!rows.length) {
    return (
      <div className="bar-list">
        <p className="empty">{empty}</p>
      </div>
    );
  }
  const max = Math.max(...rows.map((r) => r.value), 1);
  return (
    <div className={onSelect ? "bar-list interactive" : "bar-list"}>
      {rows.map((r) => {
        const inner = (
          <>
            <span className="bar-label">{r.label}</span>
            <span className="bar-track" {...tooltip(`${r.label}: ${r.value} ${unit}`.trim())}>
              <span className="bar s1" style={{ width: `${(r.value / max) * 100}%` }} />
            </span>
            <span className="bar-value">{r.value}</span>
          </>
        );
        return onSelect ? (
          <button key={r.key} type="button" className="bar-row" aria-label={`${r.label}: ${r.value} ${unit}. Show details`} onClick={() => onSelect(r)}>
            {inner}
          </button>
        ) : (
          <div key={r.key} className="bar-row">
            {inner}
          </div>
        );
      })}
    </div>
  );
}

// ---------- grouped columns over shared categories (1–2 series) ----------

export interface Series {
  name: string;
  values: number[];
}

/** Series i is drawn with --series-(i+1); pair the chart with a .legend in the markup. */
export function Columns({ categories, series, label }: { categories: string[]; series: Series[]; label: string }) {
  const max = Math.max(1, ...series.flatMap((s) => s.values));
  const description =
    `${label}: ` + categories.map((c, i) => `${c}: ${series.map((s) => `${s.values[i]} ${s.name.toLowerCase()}`).join(", ")}`).join("; ");
  return (
    <div className="column-chart" role="img" aria-label={description} style={{ gridTemplateColumns: `repeat(${categories.length}, minmax(0, 1fr))` }}>
      {categories.map((category, i) => (
        <div key={category} className="col-group">
          <div className="col-bars">
            {series.map((s, si) => (
              <span key={s.name} className="col-hit" {...tooltip(`${category}: ${s.values[i]} ${s.name.toLowerCase()}`)}>
                <span className={`col s${si + 1}`} style={{ height: s.values[i] ? `${(s.values[i] / max) * 100}%` : "0" }} />
              </span>
            ))}
          </div>
          <span className="col-label">{category}</span>
        </div>
      ))}
    </div>
  );
}

// ---------- the always-available text view of a chart ----------

type Cell = string | number;

/** Column headers are buttons that sort the rows (numbers numerically), with aria-sort. */
export function DataTable({ headers, rows }: { headers: string[]; rows: Cell[][] }) {
  const [sort, setSort] = useState<{ col: number; dir: "ascending" | "descending" } | null>(null);
  // Rows keep their original index as a key, so sorting moves rows instead of rewriting them.
  const sorted = useMemo(() => {
    const indexed = rows.map((cells, key) => ({ cells, key }));
    if (!sort) return indexed;
    const cmp = (a: Cell, b: Cell) =>
      typeof a === "number" && typeof b === "number" ? a - b : String(a).localeCompare(String(b), undefined, { numeric: true });
    return indexed.sort((a, b) => cmp(a.cells[sort.col], b.cells[sort.col]) * (sort.dir === "ascending" ? 1 : -1));
  }, [rows, sort]);
  return (
    <table>
      <thead>
        <tr>
          {headers.map((h, i) =>
            h ? (
              <th key={h} scope="col" aria-sort={sort?.col === i ? sort.dir : "none"}>
                <button
                  type="button"
                  className="sort-btn"
                  onClick={() => setSort((s) => ({ col: i, dir: s?.col === i && s.dir === "ascending" ? "descending" : "ascending" }))}
                >
                  {h}
                </button>
              </th>
            ) : (
              <th key={`blank-${i}`} scope="col" />
            ),
          )}
        </tr>
      </thead>
      <tbody>
        {sorted.map(({ cells, key }) => (
          <tr key={key}>
            {cells.map((v, j) => (
              <td key={j}>{v}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
