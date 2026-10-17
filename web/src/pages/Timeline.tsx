// Timeline: every memory in the last 24 hours, newest first.
import { AutoChips, EmptyState, PlaceChip, repeat, Saw, SkeletonScene } from "../components/bits";
import { Photo } from "../components/Photo";
import { useApi } from "../lib/api";
import { useLoadStatus } from "../lib/app";
import { capitalize, timeAgo } from "../lib/format";
import { useResource } from "../lib/resource";
import type { HistoryEntry } from "../lib/types";

export default function TimelinePage() {
  const api = useApi();
  const res = useResource("history", () => api<HistoryEntry[]>("/api/v1/dashboard/history"));
  useLoadStatus(res.loading, res.error, res.reload);
  const entries = res.data;

  return (
    <section className="page">
      <div className="page-head">
        <div>
          <p className="eyebrow">Last 24 hours · newest first</p>
          <h1 className="display">The day, scene by scene.</h1>
        </div>
      </div>
      <ol className="timeline">
        {!entries ? (
          repeat(4, (i) => <SkeletonScene key={i} />)
        ) : !entries.length ? (
          <EmptyState
            as="li"
            title="Nothing recorded today"
            text="Every memory from the last 24 hours shows here, newest first, with its photo."
            checklist="Hold REMEMBER and say where something is going"
            cta={{ label: "Check again", onClick: res.reload }}
          />
        ) : (
          entries.map((e) => <Scene key={e.id} entry={e} />)
        )}
      </ol>
    </section>
  );
}

function Scene({ entry: e }: { entry: HistoryEntry }) {
  const auto = e.source === "auto";
  return (
    <li className="scene">
      <div className="scene-time">
        <time className="mono" dateTime={e.timestamp}>
          {new Date(e.timestamp).toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" })}
        </time>
        <span className="ago mono">{timeAgo(e.timestamp)}</span>
      </div>
      <div className="scene-body">
        <Photo
          url={e.image_url}
          alt={auto ? `Camera frame: ${e.summary || "automatic sighting"}.` : `Photo taken when they said: ${e.transcript}`}
          boxes={auto ? e.detections : []}
        />
        <div className="scene-text">
          <p className="quote">{e.transcript ? `“${e.transcript}”` : capitalize(e.summary)}</p>
          <Saw scene={e.scene} />
          <div className="chips">
            {(e.items ?? []).map((i) => (
              <PlaceChip key={i.name} name={i.name} location={i.location} />
            ))}
            {auto && <AutoChips boxes={e.detections} />}
          </div>
        </div>
      </div>
    </li>
  );
}
