// Search: the wearable's retrieval step, made visible.
import { useState, type FormEvent } from "react";
import { EmptyState, repeat, SkeletonResult } from "../components/bits";
import { Icon } from "../components/Icons";
import { Photo } from "../components/Photo";
import { useApi } from "../lib/api";
import { useApp } from "../lib/app";
import { clockTime, timeAgo } from "../lib/format";
import type { SearchChunk, SearchResult } from "../lib/types";

const KIND_LABELS: Record<SearchChunk["kind"], string> = { speech: "Said", scene: "Camera saw", item: "Item", text: "Readable text" };
const SUGGESTIONS = ["Where are my keys?", "Where did I leave my glasses?", "Did I take my pills?", "What's on the kitchen counter?"];

export default function SearchPage() {
  const api = useApi();
  const { setStatus } = useApp();
  const [query, setQuery] = useState("");
  const [busy, setBusy] = useState(false);
  const [results, setResults] = useState<SearchResult[] | null>(null);

  const run = async (q: string) => {
    q = q.trim();
    if (!q) return;
    setQuery(q);
    setBusy(true);
    setResults([]);
    setStatus({ text: "Searching…", state: "loading" });
    try {
      const { results } = await api<{ results: SearchResult[] }>(`/api/v1/dashboard/search?q=${encodeURIComponent(q)}`);
      setResults(results);
      setStatus({ text: `${results.length} memor${results.length === 1 ? "y" : "ies"} found for “${q}”.`, state: "" });
    } catch (err) {
      setResults(null);
      setStatus({ text: `Search failed: ${(err as Error).message}`, state: "error", retry: () => void run(q) });
    } finally {
      setBusy(false);
    }
  };

  const submit = (e: FormEvent) => {
    e.preventDefault();
    void run(query);
  };

  return (
    <section className="page">
      <div className="page-head">
        <div>
          <p className="eyebrow">Vector + keyword search over the last 72 hours</p>
          <h1 className="display">Ask what the wearer would.</h1>
        </div>
      </div>
      <label htmlFor="search-q" className="field-label">
        Ask a question
      </label>
      <form className="search-form" role="search" onSubmit={submit}>
        <Icon name="search" className="i search-icon" />
        <input id="search-q" type="search" placeholder="Where are my keys?" autoComplete="off" required value={query} onChange={(e) => setQuery(e.target.value)} />
        <button type="submit" className="primary" disabled={busy || !query.trim()} aria-busy={busy}>
          Search
        </button>
      </form>
      <div className="suggestions" aria-label="Example questions">
        {SUGGESTIONS.map((s) => (
          <button key={s} type="button" className="suggestion" onClick={() => void run(s)}>
            {s}
          </button>
        ))}
      </div>
      <p className="hint search-hint">Shows the memories the wearable would answer from, and which text chunks matched by meaning or by words.</p>
      <ol className="search-results">
        {busy ? (
          repeat(2, (i) => <SkeletonResult key={i} />)
        ) : results && !results.length ? (
          <EmptyState
            as="li"
            title="No memories matched"
            text="Nothing logged in the last 3 days matches that. Try a single item name, like “keys”."
            cta={{ label: "Try “keys”", onClick: () => void run("keys") }}
          />
        ) : (
          results?.map((r, i) => <Result key={r.memory_id} result={r} rank={i + 1} />)
        )}
      </ol>
    </section>
  );
}

function Result({ result: r, rank }: { result: SearchResult; rank: number }) {
  const said = r.chunks.find((c) => c.kind === "speech")?.content;
  return (
    <li className="result">
      <Photo url={r.image_url} alt={`Photo from ${clockTime(r.timestamp)}${said ? `, when they said: ${said}` : ""}`} />
      <div className="result-body">
        <div className="result-head">
          <span className="result-rank mono">#{rank}</span>
          <time className="mono" dateTime={r.timestamp}>
            {clockTime(r.timestamp)} · {timeAgo(r.timestamp)}
          </time>
          <span className="result-score mono">score {r.score.toFixed(4)}</span>
        </div>
        <ul className="chunks">
          {r.chunks.map((c, i) => (
            <li key={i} className={c.matched ? "chunk matched" : "chunk"}>
              <span className={`kind kind-${c.kind}`}>{KIND_LABELS[c.kind] ?? c.kind}</span>
              <span className="chunk-text">{c.content}</span>
              {c.matched && (
                <span className="why mono">
                  {[c.vector_rank && `meaning #${c.vector_rank}`, c.keyword_rank && `words #${c.keyword_rank}`].filter(Boolean).join(" · ")}
                </span>
              )}
            </li>
          ))}
        </ul>
      </div>
    </li>
  );
}
