// Drill-down for one item: photo with its box, where and when, where it's been, and the
// questions about it. Opened from Items cards and Insights bars.
import { useEffect, useState } from "react";
import { useApi } from "../lib/api";
import { useApp } from "../lib/app";
import { capitalize, clockTime, timeAgo } from "../lib/format";
import { getCached, setCached } from "../lib/resource";
import type { Insights, Item, ItemsResponse, Question } from "../lib/types";
import { latestInsights } from "../pages/Insights";
import { ItemChips, Sk } from "./bits";
import { Dialog } from "./Dialog";
import { Icon } from "./Icons";
import { Photo } from "./Photo";

export function ItemDrawer() {
  const { drawerItem, closeItem } = useApp();
  const api = useApi();
  const [state, setState] = useState<{ item?: Item | null; error?: string }>({});
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    if (!drawerItem) return;
    let cancelled = false;
    setState({});
    (async () => {
      let data = getCached<ItemsResponse>("items");
      if (!data) {
        data = await api<ItemsResponse>("/api/v1/dashboard/items");
        setCached("items", data);
      }
      if (!cancelled) setState({ item: data.items.find((i) => i.name === drawerItem) ?? null });
    })().catch((err: Error) => !cancelled && setState({ error: err.message }));
    return () => {
      cancelled = true;
    };
  }, [drawerItem, attempt, api]);

  return (
    <Dialog open={drawerItem !== null} onClose={closeItem} className="drawer" labelledBy="drawer-title">
      <div className="drawer-head">
        <div>
          <p className="eyebrow">Item</p>
          <h2 id="drawer-title">{capitalize(drawerItem)}</h2>
        </div>
        <button type="button" className="icon-btn" data-close aria-label="Close details">
          <Icon name="close" />
        </button>
      </div>
      <div className="drawer-body">
        {state.error ? (
          <>
            <p className="error">Couldn't load this item: {state.error}</p>
            <button type="button" className="pill" onClick={() => setAttempt((a) => a + 1)}>
              Try again
            </button>
          </>
        ) : state.item === undefined ? (
          <>
            <div className="photo sk detail-photo" aria-hidden="true" />
            <Sk cls="line w60 h24" />
            <Sk cls="line w80" />
            <Sk cls="line w40" />
          </>
        ) : (
          drawerItem && <Detail name={drawerItem} item={state.item} questions={questionsAbout(drawerItem, latestInsights())} onClose={closeItem} />
        )}
      </div>
    </Dialog>
  );
}

const questionsAbout = (name: string, insights: Insights | undefined) => (insights?.recent_questions ?? []).filter((q) => q.item === name);

function Detail({ name, item, questions, onClose }: { name: string; item: Item | null; questions: Question[]; onClose(): void }) {
  const questionList = questions.length > 0 && (
    <section className="detail-section">
      <h3>Questions about it</h3>
      <ul className="detail-list">
        {questions.map((q) => (
          <li key={q.timestamp}>
            <strong>“{q.question}”</strong>
            <span className="muted">{q.answer}</span>
            <time className="mono" dateTime={q.timestamp}>
              {clockTime(q.timestamp)} · {timeAgo(q.timestamp)}
            </time>
          </li>
        ))}
      </ul>
    </section>
  );

  if (!item) {
    return (
      <>
        <p className="detail-where">Never logged</p>
        <p className="hint">
          It was asked about but never put down with the wearable. Saying it out loud when putting it down (“I'm putting my glasses on the
          nightstand”) means the next question has an answer.
        </p>
        {questionList}
      </>
    );
  }

  const spots = [{ location: item.location, timestamp: item.last_seen, now: true }, ...item.previous.map((p) => ({ ...p, now: false }))];
  return (
    <>
      <Photo
        className="detail-photo"
        url={item.image_url}
        alt={`The ${name} on the ${item.location || "spot shown"}`}
        boxes={item.detection ? [item.detection] : []}
      />
      <p className="detail-where">{capitalize(item.location) || "Location not said"}</p>
      <p className="when mono">
        {timeAgo(item.last_seen)} · {clockTime(item.last_seen)}
      </p>
      {item.transcript && <p className="quote">“{item.transcript}”</p>}
      <div className="chips">
        <ItemChips item={item} />
      </div>
      <section className="detail-section">
        <h3>Where it's been</h3>
        <ol className="trail">
          {spots.map((s) => (
            <li key={`${s.timestamp}-${s.now}`} className={s.now ? "now" : ""}>
              <strong>{capitalize(s.location) || "Somewhere"}</strong>
              <span className="muted">{s.now ? `now · ${timeAgo(s.timestamp)}` : timeAgo(s.timestamp)}</span>
            </li>
          ))}
        </ol>
      </section>
      {questionList}
      <a className="pill" href="#/map" onClick={onClose}>
        Show on the home map
      </a>
    </>
  );
}
