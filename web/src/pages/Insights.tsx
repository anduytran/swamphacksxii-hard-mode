// Insights: patterns a caregiver should know about, over a chosen window (24 h – 3 days).
import { useState, type ReactNode } from "react";
import { BarList, Columns, DataTable, HeatGrid, type BarRow } from "../components/Charts";
import { EmptyState, repeat, Sk } from "../components/bits";
import { Icon, type IconName } from "../components/Icons";
import { useApi } from "../lib/api";
import { useApp, useLoadStatus } from "../lib/app";
import { bucketByHour, capitalize, clockTime, timeAgo } from "../lib/format";
import { getCached, useResource } from "../lib/resource";
import type { Insights, MissingTimes, Question } from "../lib/types";

const RANGES = [
  { hours: 24, label: "24 hours" },
  { hours: 48, label: "2 days" },
  { hours: 72, label: "3 days" },
];
// Activity buckets: 3-hour blocks of the viewer's local day.
const BUCKETS = ["12–3a", "3–6a", "6–9a", "9a–12", "12–3p", "3–6p", "6–9p", "9p–12"];

// The chosen range survives switching pages; the item drawer reads the latest data.
let rangeHours = 72;
export const latestInsights = () => getCached<Insights>(`insights:${rangeHours}`);

export default function InsightsPage() {
  const api = useApi();
  const { openItem } = useApp();
  const [hours, setHours] = useState(rangeHours);
  const res = useResource(`insights:${hours}`, () => api<Insights>(`/api/v1/dashboard/insights?hours=${hours}`));
  const summary = useResource("summary", () => api<{ summary: string | null }>("/api/v1/dashboard/summary"));
  useLoadStatus(res.loading, res.error, res.reload);
  const data = res.data;

  const choose = (h: number) => {
    rangeHours = h;
    setHours(h);
  };
  const rows = (list: { name: string; count: number }[]): BarRow[] => list.map((r) => ({ label: capitalize(r.name), value: r.count, key: r.name }));
  const onSelect = (row: BarRow) => openItem(row.key);
  const ofKind = (kind: "memory" | "question") => bucketByHour((data?.activity ?? []).filter((a) => a.kind === kind), BUCKETS.length);
  const series = [
    { name: "Things put down", values: ofKind("memory") },
    { name: "Questions asked", values: ofKind("question") },
  ];

  return (
    <section className="page">
      <div className="page-head">
        <div>
          <p className="eyebrow">Last {data?.window_hours ?? hours} hours</p>
          <h1 className="display">How the week is going.</h1>
        </div>
        <div className="segmented" role="group" aria-label="Time range">
          {RANGES.map((r) => (
            <button key={r.hours} type="button" aria-pressed={hours === r.hours} onClick={() => choose(r.hours)}>
              {r.label}
            </button>
          ))}
        </div>
      </div>

      <div className="tiles">
        <Tile icon="layers" label="Things put down" value={data?.memories_logged} />
        <Tile icon="message" label="Questions asked" value={data?.questions_asked} />
        <Tile icon="question" label="Asked about but never logged" value={data?.unanswered_count} alert={(data?.unanswered_count ?? 0) > 0} />
        <Tile icon="items" label="Items tracked" value={data?.items_tracked} />
      </div>

      {summary.data?.summary && (
        <section className="card summary" aria-labelledby="summary-h">
          <p className="eyebrow" id="summary-h">
            <Icon name="sparkle" />
            The last 3 days, in brief
          </p>
          <p className="summary-text">{summary.data.summary}</p>
          <p className="summary-by">Written by Gemini from the last 3 days of activity.</p>
        </section>
      )}

      {data && <Highlights data={data} />}

      <div className="two-col">
        <section className="card">
          <h2>Most asked about</h2>
          {data ? <BarList rows={rows(data.most_asked)} unit="times asked" empty="No questions about specific items yet." onSelect={onSelect} /> : <SkeletonBars />}
        </section>
        <section className="card">
          <h2>Most often put down</h2>
          {data ? <BarList rows={rows(data.most_logged)} unit="times put down" empty="No items logged yet." onSelect={onSelect} /> : <SkeletonBars />}
        </section>
      </div>

      <section className="card">
        <div className="card-head">
          <h2>Activity by time of day</h2>
          <div className="legend" aria-hidden="true">
            <span>
              <i className="swatch s1" />
              Things put down
            </span>
            <span>
              <i className="swatch s2" />
              Questions asked
            </span>
          </div>
        </div>
        {data ? <Columns categories={BUCKETS} series={series} label="Activity by time of day" /> : <Sk cls="block h200" />}
        <details>
          <summary>Show as table</summary>
          <DataTable headers={["", ...series.map((s) => s.name)]} rows={BUCKETS.map((c, i) => [c, ...series.map((s) => s.values[i])])} />
        </details>
      </section>

      <MissingGridCard />

      {/*
        Add new visualizations here: a .card with an <h2> and a chart from components/Charts.tsx
        (BarList, Columns, DataTable). See README → "Adding a visualization".
      */}

      <section className="card">
        <h2>Recent questions</h2>
        <ol className="questions">
          {!data ? (
            repeat(3, (i) => (
              <li key={i} aria-hidden="true">
                <Sk cls="line w60 h20" />
                <Sk cls="line w80" />
              </li>
            ))
          ) : !data.recent_questions.length ? (
            <EmptyState as="li" text="No questions asked yet." />
          ) : (
            data.recent_questions.map((q) => <QuestionRow key={q.timestamp + q.question} q={q} />)
          )}
        </ol>
      </section>
    </section>
  );
}

function Tile({ icon, label, value, alert }: { icon: IconName; label: string; value: number | undefined; alert?: boolean }) {
  return (
    <div className={alert ? "tile alert" : "tile"}>
      <span className="tile-icon">
        <Icon name={icon} />
      </span>
      <span className="tile-value">{value ?? <Sk cls="line w40 h40" />}</span>
      <span className="tile-label">{label}</span>
    </div>
  );
}

function SkeletonBars() {
  return (
    <div className="bar-list">
      {repeat(5, (i) => (
        <div key={i} className="bar-row" aria-hidden="true">
          <Sk cls="line w70" />
          <Sk cls="line h12" />
          <Sk cls="line w100" />
        </div>
      ))}
    </div>
  );
}

/** "Worth a conversation": what the numbers suggest doing. Item names in bold. */
function Highlights({ data }: { data: Insights }) {
  const notes: ReactNode[] = [];
  const neverLogged = [...new Set(data.unanswered.map((q) => q.item).filter((i): i is string => Boolean(i)))];
  // The most-asked item that *was* logged: keeps the advice distinct from the never-logged notes.
  const top = data.most_asked.find((r) => !neverLogged.includes(r.name));
  if (top && top.count >= 2) {
    notes.push(
      <p key="top">
        Questions about the <strong>{top.name}</strong> came up {top.count} times. A fixed spot for it could help.
      </p>,
    );
  }
  for (const item of neverLogged.slice(0, 3)) {
    notes.push(
      <p key={item}>
        The <strong>{item}</strong> was asked about but never logged. Encourage saying it out loud when putting it down.
      </p>,
    );
  }
  if (data.questions_asked && !data.unanswered_count) notes.push(<p key="nice">Every question was about an item that had been logged first. Nice.</p>);
  if (!data.memories_logged && !data.questions_asked) notes.push(<p key="none">No activity yet. Insights appear once the wearable starts logging.</p>);
  if (!notes.length) return null;
  return (
    <section className="card notice" aria-labelledby="notice-h">
      <span className="notice-icon">
        <Icon name="bulb" />
      </span>
      <div>
        <h2 id="notice-h">Worth a conversation</h2>
        {notes}
      </div>
    </section>
  );
}

function QuestionRow({ q }: { q: Question }) {
  return (
    <li>
      <div className="q-head">
        <strong>“{q.question}”</strong>
        <span className={`badge ${q.answerable ? "good" : "warning"}`}>{q.answerable ? "✓ Logged earlier" : "! Never logged"}</span>
      </div>
      <p className="q-answer">{q.answer}</p>
      <time dateTime={q.timestamp}>
        {clockTime(q.timestamp)} · {timeAgo(q.timestamp)}
      </time>
    </li>
  );
}

// ---------- When things go missing: hour of day x day of week ----------

const GRID_RANGES = [
  { days: 7, label: "1 week" },
  { days: 28, label: "4 weeks" },
  { days: 90, label: "3 months" },
];
const WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const WEEKDAY_NAMES = ["Mondays", "Tuesdays", "Wednesdays", "Thursdays", "Fridays", "Saturdays", "Sundays"];
const HOURS = Array.from({ length: 24 }, (_, h) => h);
const hourName = (h: number) => `${h % 12 || 12}${h < 12 ? "am" : "pm"}`;
const BLOCKS = ["12–3am", "3–6am", "6–9am", "9am–12", "12–3pm", "3–6pm", "6–9pm", "9pm–12"];
let gridDays = 28; // survives switching pages, like the range above

type Measure = "asked" | "put_down";

/** Sums each UTC hour into the viewer's local weekday (Monday first) and hour. */
export function binMissing(hours: MissingTimes["hours"], measure: Measure): number[][] {
  const grid = WEEKDAYS.map(() => HOURS.map(() => 0));
  for (const h of hours) {
    const at = new Date(h.at);
    grid[(at.getDay() + 6) % 7][at.getHours()] += h[measure];
  }
  return grid;
}

/** The finding in one sentence: the busiest weekday and 3-hour block. */
export function missingPeak(grid: number[][], measure: Measure): string | null {
  const total = grid.flat().reduce((a, b) => a + b, 0);
  if (!total) return null;
  const byDay = grid.map((r) => r.reduce((a, b) => a + b, 0));
  const byBlock = BLOCKS.map((_, b) => grid.reduce((sum, r) => sum + r.slice(b * 3, b * 3 + 3).reduce((a, c) => a + c, 0), 0));
  const day = byDay.indexOf(Math.max(...byDay));
  const block = byBlock.indexOf(Math.max(...byBlock));
  const what = measure === "asked" ? "Questions about lost things come" : "Things get put down";
  return `${what} most on ${WEEKDAY_NAMES[day]} and between ${BLOCKS[block]} (${Math.round((byBlock[block] / total) * 100)}% of the total).`;
}

function MissingGridCard() {
  const api = useApi();
  const [days, setDays] = useState(gridDays);
  const [measure, setMeasure] = useState<Measure>("asked");
  const res = useResource(`missing:${days}`, () => api<MissingTimes>(`/api/v1/dashboard/missing-times?days=${days}`));
  const grid = res.data ? binMissing(res.data.hours, measure) : null;
  const noun = measure === "asked" ? "question" : "put-down";
  const peak = grid && missingPeak(grid, measure);

  return (
    <section className="card" aria-labelledby="missing-h">
      <div className="card-head">
        <h2 id="missing-h">When things go missing</h2>
        <div className="segmented" role="group" aria-label="How far back">
          {GRID_RANGES.map((r) => (
            <button
              key={r.days}
              type="button"
              aria-pressed={days === r.days}
              onClick={() => {
                gridDays = r.days;
                setDays(r.days);
              }}
            >
              {r.label}
            </button>
          ))}
        </div>
      </div>
      <div className="missing-sub">
        <div className="segmented small" role="group" aria-label="What to show">
          <button type="button" aria-pressed={measure === "asked"} onClick={() => setMeasure("asked")}>
            Questions asked
          </button>
          <button type="button" aria-pressed={measure === "put_down"} onClick={() => setMeasure("put_down")}>
            Things put down
          </button>
        </div>
        <p className="missing-peak">{grid ? peak ?? `No ${noun}s in this period yet.` : "\u00a0"}</p>
      </div>
      {grid ? (
        <HeatGrid
          rows={WEEKDAYS}
          columns={HOURS.map(String)}
          columnTicks={["12am", "3am", "6am", "9am", "12pm", "3pm", "6pm", "9pm"]}
          values={grid}
          label={`${measure === "asked" ? "Questions asked" : "Things put down"} by day and hour, last ${days} days`}
          cellLabel={(d, h, v) => `${WEEKDAY_NAMES[d]}, ${hourName(h)}–${hourName((h + 1) % 24)}: ${v} ${noun}${v === 1 ? "" : "s"}`}
        />
      ) : (
        <Sk cls="block h200" />
      )}
      <div className="heat-legend" aria-hidden="true">
        <span>None</span>
        <i className="heat-cell h0" />
        <span>Fewer</span>
        {[1, 2, 3, 4].map((l) => (
          <i key={l} className={`heat-cell h${l}`} />
        ))}
        <span>More</span>
      </div>
      <p className="hint">
        In your time zone. The last 3 days come from the memories themselves; before that, only anonymous hourly counts are kept (no words
        or photos), for up to 90 days.
      </p>
      {grid && (
        <details>
          <summary>Show as table</summary>
          <DataTable headers={["Day", ...BLOCKS]} rows={WEEKDAYS.map((d, i) => [d, ...BLOCKS.map((_, b) => grid[i].slice(b * 3, b * 3 + 3).reduce((a, c) => a + c, 0))])} />
        </details>
      )}
    </section>
  );
}
