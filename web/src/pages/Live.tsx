// Live: each voice command as it happens, for testing and the demo. What speech-to-text heard, what was
// decided (and by the model or the rules), the photo picked, what was said back, and each step's time.
import { useEffect, useRef, useState } from "react";
import { EmptyState } from "../components/bits";
import { publicGet, useApi } from "../lib/api";
import { clockTime } from "../lib/format";
import type { Health, VoiceEvent } from "../lib/types";

const POLL_MS = 2000;
const HEALTH_MS = 10000;
const KEEP = 100;

export default function LivePage() {
  const api = useApi();
  const [events, setEvents] = useState<VoiceEvent[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [paused, setPaused] = useState(false);
  const [health, setHealth] = useState<Health | null>(null);
  const newest = useRef(0);

  useEffect(() => {
    if (paused) return;
    let stop = false;
    let timer: number | undefined;
    const poll = async () => {
      try {
        const res = await api<{ events: VoiceEvent[] }>(`/api/v1/dashboard/live?after=${newest.current}`);
        if (stop) return;
        if (res.events.length) {
          newest.current = Math.max(newest.current, ...res.events.map((e) => e.id));
          setEvents((old) => [...res.events, ...old].slice(0, KEEP));
        }
        setError(null);
      } catch (e) {
        if (!stop) setError(e instanceof Error ? e.message : String(e));
      }
      if (!stop) timer = window.setTimeout(poll, document.hidden ? POLL_MS * 5 : POLL_MS);
    };
    poll();
    return () => {
      stop = true;
      window.clearTimeout(timer);
    };
  }, [api, paused]);

  useEffect(() => {
    let stop = false;
    const load = () => publicGet<Health>("/healthz").then((h) => !stop && setHealth(h)).catch(() => {});
    load();
    const t = window.setInterval(load, HEALTH_MS);
    return () => {
      stop = true;
      window.clearInterval(t);
    };
  }, []);

  return (
    <section className="page live">
      <div className="page-head">
        <div>
          <p className="eyebrow">
            <span className={`live-dot${paused || error ? " off" : ""}`} aria-hidden="true" />
            {paused ? "Paused" : error ? "Reconnecting…" : "Updating every 2 seconds"}
          </p>
          <h1 className="display">Live.</h1>
        </div>
        <div className="live-actions">
          <button type="button" className="pill" onClick={() => setPaused((p) => !p)} aria-pressed={paused}>
            {paused ? "Resume" : "Pause"}
          </button>
          <button type="button" className="pill quiet" onClick={() => setEvents([])} disabled={!events.length}>
            Clear
          </button>
        </div>
      </div>

      {error && <p className="live-error" role="status">Couldn't update: {error}</p>}
      {health && <Usage health={health} />}

      {events.length === 0 ? (
        <EmptyState text='Nothing yet. Say "Last Scene" to the wearable, or press a button; each command shows up here within a few seconds.' />
      ) : (
        <ol className="live-list" aria-live="polite">
          {events.map((e) => (
            <EventCard key={e.id} e={e} />
          ))}
        </ol>
      )}
    </section>
  );
}

const OUTCOME: Record<string, { label: string; tone: string }> = {
  saved: { label: "Saved", tone: "good" },
  answered: { label: "Answered", tone: "good" },
  found: { label: "Marked found", tone: "good" },
  ignored: { label: "Ignored: no wake phrase", tone: "quiet" },
  "nothing to do": { label: "Nothing to do", tone: "quiet" },
  error: { label: "Error", tone: "attention" },
};

const STEP_LABEL: Record<string, string> = {
  transcribe: "speech-to-text",
  decide: "decide",
  photos: "photo check",
  save: "save",
  embed: "embed",
  retrieve: "search",
  answer: "answer",
  speak: "speech",
};

function EventCard({ e }: { e: VoiceEvent }) {
  const outcome = OUTCOME[e.outcome] ?? { label: e.outcome, tone: "quiet" };
  const steps = Object.entries(e.steps ?? {});
  const served = Object.entries(e.served ?? {});
  return (
    <li className="card live-event">
      <div className="live-head">
        <span className={`chip live-outcome ${outcome.tone}`}>{outcome.label}</span>
        <span className="chip">{e.kind}</span>
        {e.intent && e.intent !== "ignored" && <span className="chip">{e.intent}</span>}
        <time className="live-time mono" dateTime={e.created_at}>
          {clockTime(e.created_at)} · {(e.total_ms / 1000).toFixed(1)} s
        </time>
      </div>

      <div className="live-body">
        {e.image_url && <img className="live-photo" src={e.image_url} alt={`Saved photo ${(e.saved_photo ?? 0) + 1}`} loading="lazy" />}
        <dl className="live-facts">
          <dt>Heard</dt>
          <dd>{e.heard ? `“${e.heard}”` : <span className="muted">nothing</span>}</dd>
          {e.decided_by && (
            <>
              <dt>Decided by</dt>
              <dd>
                {e.decided_by === "model" ? "the command model" : "the rules (model slow or down)"}
                {e.command_error && <span className="muted"> · {e.command_error}</span>}
              </dd>
            </>
          )}
          {e.item && (
            <>
              <dt>Item</dt>
              <dd>
                {e.item}
                {e.item_from && <span className="muted"> · from {e.item_from}</span>}
              </dd>
            </>
          )}
          {e.photos > 0 && (
            <>
              <dt>Photos</dt>
              <dd>
                {e.photos} sent
                {e.best_photo !== undefined && ` · best is #${e.best_photo + 1}`}
                {e.in_view !== undefined && (e.in_view ? " · item in view" : " · item not in view")}
                {e.seen_where && ` (${e.seen_where})`}
                {e.photo_error && <span className="muted"> · check failed: {e.photo_error}</span>}
              </dd>
            </>
          )}
          {e.answer && (
            <>
              <dt>Said back</dt>
              <dd>
                “{e.answer}”{e.answered_by && <span className="muted"> · {e.answered_by}</span>}
              </dd>
            </>
          )}
          {e.error && (
            <>
              <dt>Error</dt>
              <dd className="live-err">{e.error}</dd>
            </>
          )}
        </dl>
      </div>

      {steps.length > 0 && (
        <div className="live-steps" aria-label="Time per step">
          {steps.map(([k, ms]) => (
            <span key={k} className="live-step">
              <span className="muted">{STEP_LABEL[k] ?? k}</span> <span className="mono">{ms} ms</span>
            </span>
          ))}
        </div>
      )}
      {served.length > 0 && (
        <p className="live-served muted">
          {served.map(([k, who]) => `${STEP_LABEL[k] ?? k}: ${who}`).join(" · ")}
        </p>
      )}
    </li>
  );
}

function Usage({ health }: { health: Health }) {
  const s = health.speech;
  const models = Object.entries(health.model_usage ?? {}).filter(([, u]) => u.used_today > 0);
  return (
    <div className="card live-usage">
      <div className="live-usage-row">
        {s && (
          <>
            <span>
              <span className="muted">Speech-to-text</span> {s.listening} ·{" "}
              <span className="mono">{s.audio_seconds_this_month ?? 0}</span> of {s.monthly_audio_seconds_cap ?? "?"} s this month
            </span>
            <span>
              <span className="muted">Voice</span> {s.provider} · <span className="mono">{s.characters_this_month ?? 0}</span> of{" "}
              {s.monthly_character_cap ?? "?"} characters
            </span>
          </>
        )}
        {models.map(([name, u]) => (
          <span key={name}>
            <span className="muted">{name.replace(/^gemini:/, "")}</span> <span className="mono">{u.used_today}</span>/{u.limit_per_day} today
          </span>
        ))}
        {!health.ok && <span className="live-err">Server not healthy: {health.missing_settings?.join(", ")}</span>}
      </div>
    </div>
  );
}
