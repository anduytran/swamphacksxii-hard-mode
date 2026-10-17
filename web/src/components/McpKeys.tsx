// MCP keys: read-only access to this household for an AI assistant (Gemini) or a script,
// through /mcp. A new key is shown once; the server keeps only its hash. See docs/MCP.md.
import { useCallback, useEffect, useState, type FormEvent } from "react";
import { ApiError, type Api } from "../lib/api";
import { timeAgo } from "../lib/format";

interface McpKey {
  id: string;
  label: string;
  hint: string;
  created_at: string;
  last_used_at: string | null;
}

const errorText = (err: unknown) => (err instanceof ApiError ? err.detail : (err as Error).message);

export function McpKeys({ householdId, api, owner }: { householdId: string; api: Api; owner: boolean }) {
  const base = `/api/v1/households/${householdId}/mcp-keys`;
  const [keys, setKeys] = useState<McpKey[] | null>(null);
  const [label, setLabel] = useState("");
  const [created, setCreated] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    try {
      setKeys((await api<{ keys: McpKey[] }>(base)).keys);
    } catch (err) {
      setError(errorText(err));
    }
  }, [api, base]);
  useEffect(() => void load(), [load]);

  const create = async (e: FormEvent) => {
    e.preventDefault();
    if (!label.trim()) return setError("Enter a name for the key, like “Gemini script”.");
    setBusy(true);
    setError("");
    try {
      const { key } = await api<{ key: string }>(base, { method: "POST", body: { label: label.trim() } });
      setCreated(key);
      setLabel("");
      await load();
    } catch (err) {
      setError(errorText(err));
    } finally {
      setBusy(false);
    }
  };

  const revoke = async (k: McpKey) => {
    if (!window.confirm(`Revoke “${k.label}”? Anything using it stops working.`)) return;
    setError("");
    try {
      await api(`${base}/${k.id}`, { method: "DELETE" });
      setCreated("");
      await load();
    } catch (err) {
      setError(errorText(err));
    }
  };

  return (
    <div className="card setup-card">
      <h2>MCP keys</h2>
      <p className="hint">
        Read-only access to this household's memories, items and places for an AI assistant like Gemini, through
        {" "}<span className="mono">{location.origin}/mcp</span>. A key can't change anything. It's shown once; revoke it if it leaks.
      </p>
      {keys && keys.length > 0 && (
        <ul className="rows">
          {keys.map((k) => (
            <li key={k.id}>
              <strong>{k.label}</strong>
              <span className="mono muted">{k.hint}</span>
              <span className="muted">{k.last_used_at ? `used ${timeAgo(k.last_used_at)}` : "never used"}</span>
              {owner && (
                <button type="button" className="pill quiet" onClick={() => void revoke(k)}>
                  Revoke
                </button>
              )}
            </li>
          ))}
        </ul>
      )}
      {keys && !keys.length && <p className="hint">No keys yet.</p>}
      {owner ? (
        <form className="mcp-key-form" onSubmit={create}>
          <input aria-label="Key name" placeholder="Gemini script" maxLength={80} value={label} onChange={(e) => setLabel(e.target.value)} />
          <button type="submit" className="pill" disabled={busy} aria-busy={busy}>
            Make a key
          </button>
        </form>
      ) : (
        <p className="hint">Only the person who set up the household can make or revoke keys.</p>
      )}
      <p className="invite-code mono" role="status">
        {created && `Copy it now, it won't be shown again: ${created}`}
      </p>
      {error && (
        <p className="form-msg error" role="status">
          {error}
        </p>
      )}
    </div>
  );
}
