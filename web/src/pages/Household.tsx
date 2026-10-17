// Household: setting LastScene up, often for someone else.
import { useEffect, useState } from "react";
import { repeat, Sk } from "../components/bits";
import { FormCard } from "../components/FormCard";
import { McpKeys } from "../components/McpKeys";
import { ApiError, useApi, type Api } from "../lib/api";
import { useApp, useLoadStatus } from "../lib/app";
import { useAuth } from "../lib/auth";
import { timeAgo } from "../lib/format";
import { invalidateAll, useResource } from "../lib/resource";
import type { Household } from "../lib/types";

export default function HouseholdPage() {
  const api = useApi();
  const { setSetup } = useApp();
  const res = useResource("households", () => api<{ households: Household[] }>("/api/v1/households/me"));
  useLoadStatus(res.loading, res.error, res.reload);
  const households = res.data?.households;

  useEffect(() => {
    if (households) setSetup({ household: households.length > 0, wearable: households.some((h) => h.devices.length > 0) });
  }, [households, setSetup]);

  // A change here affects every page (devices, places, sharing): drop all cached data.
  const changed = () => {
    invalidateAll();
    res.reload();
  };

  const title = !households ? "Your household." : !households.length ? "Set up LastScene." : households.length === 1 ? households[0].name : "Your households.";
  return (
    <section className="page">
      <div className="page-head">
        <div>
          <p className="eyebrow">Setup</p>
          <h1 className="display">{title}</h1>
        </div>
      </div>
      {!households ? (
        <div className="two-col">
          {repeat(2, (i) => (
            <div key={i} className="card setup-card is-skeleton" aria-hidden="true">
              <Sk cls="line w40 h24" />
              <Sk cls="line w80" />
              <Sk cls="block h44" />
              <Sk cls="block h44" />
              <Sk cls="line w30 h44" />
            </div>
          ))}
        </div>
      ) : !households.length ? (
        <Start api={api} onDone={changed} />
      ) : (
        households.map((h) => <HouseholdView key={h.id} h={h} api={api} onChange={changed} />)
      )}
    </section>
  );
}

function Start({ api, onDone }: { api: Api; onDone(): void }) {
  const { user } = useAuth();
  const myName = user?.name || user?.email || "";
  return (
    <>
      <p className="lede">LastScene is usually set up by a family member for the person who wears it.</p>
      <div className="two-col">
        <FormCard
          title="Set up for someone"
          hint="Creates a household. You can add the wearable and invite family next."
          fields={[
            { name: "name", label: "Household name", required: true, placeholder: "Mom's household", maxLength: 80 },
            { name: "display_name", label: "Your name", defaultValue: myName, maxLength: 80 },
          ]}
          submitLabel="Create household"
          onSubmit={async (v) => {
            await api("/api/v1/households", { method: "POST", body: { name: v.name, display_name: v.display_name || null } });
            onDone();
          }}
        />
        <FormCard
          title="Join a household"
          hint="Ask the person who set it up for an invite code."
          fields={[
            { name: "code", label: "Invite code", required: true, minLength: 4, maxLength: 32, placeholder: "ABCD-1234-EF56", autoComplete: "off" },
            { name: "display_name", label: "Your name", defaultValue: myName, maxLength: 80 },
          ]}
          submitLabel="Join"
          onSubmit={async (v) => {
            await api("/api/v1/invites/accept", { method: "POST", body: { code: v.code, display_name: v.display_name || null } });
            onDone();
          }}
        />
      </div>
    </>
  );
}

function HouseholdView({ h, api, onChange }: { h: Household; api: Api; onChange(): void }) {
  const owner = h.role === "owner";
  const base = `/api/v1/households/${h.id}`;

  const devices = (
    <div className="card setup-card">
      <h2>Wearable</h2>
      {h.devices.length ? (
        <ul className="rows">
          {h.devices.map((d) => (
            <li key={d.client_id}>
              <strong>{d.label || "Wearable"}</strong>
              <span className="mono muted">{d.client_id}</span>
              <span className="muted">{d.last_seen ? `last heard ${timeAgo(d.last_seen)}` : "not heard from yet"}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="hint">No wearable yet. Turn it on and connect it to Wi-Fi, then enter its Client ID and the pairing code on its label.</p>
      )}
    </div>
  );

  const claim = owner && (
    <FormCard
      title="Add a wearable"
      hint="The device must have connected at least once so LastScene knows its pairing code."
      fields={[
        { name: "client_id", label: "Client ID", required: true, minLength: 4, maxLength: 128, autoComplete: "off", spellCheck: false, placeholder: "from the device label" },
        { name: "pairing_code", label: "Pairing code", required: true, minLength: 4, maxLength: 32, autoComplete: "off", spellCheck: false, placeholder: "ABCD-1234-EF56" },
        { name: "label", label: "Name", placeholder: "Mom's wearable", maxLength: 80 },
      ]}
      submitLabel="Add wearable"
      onSubmit={async (v) => {
        const r = await api<{ memories_placed: number }>(`${base}/devices`, {
          method: "POST",
          body: { client_id: v.client_id, pairing_code: v.pairing_code, label: v.label || null },
        });
        onChange();
        return `Added. ${r.memories_placed} earlier memories were placed on the map.`;
      }}
    />
  );

  const people = <People h={h} api={api} owner={owner} />;
  const sharing = <Sharing h={h} api={api} owner={owner} />;

  return (
    <div className="household">
      <div className="two-col">
        {devices}
        {claim || people}
      </div>
      {claim ? (
        <div className="two-col">
          {people}
          {sharing}
        </div>
      ) : (
        sharing
      )}
      <Places h={h} api={api} />
      <McpKeys householdId={h.id} api={api} owner={owner} />
    </div>
  );
}

function People({ h, api, owner }: { h: Household; api: Api; owner: boolean }) {
  const [busy, setBusy] = useState(false);
  const [invite, setInvite] = useState("");
  const makeInvite = async () => {
    setBusy(true);
    try {
      const { code, expires_in_hours } = await api<{ code: string; expires_in_hours: number }>(`/api/v1/households/${h.id}/invites`, { method: "POST" });
      setInvite(`Share this code: ${code} (works once, for ${expires_in_hours} hours)`);
    } catch (err) {
      setInvite(`Couldn't make a code: ${err instanceof ApiError ? err.detail : (err as Error).message}`);
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="card setup-card">
      <h2>People</h2>
      <ul className="rows">
        {h.members.map((m, i) => (
          <li key={i}>
            <strong>
              {m.display_name}
              {m.is_you ? " (you)" : ""}
            </strong>
            <span className="muted">{m.role === "owner" ? "set this up" : "family"}</span>
          </li>
        ))}
      </ul>
      {owner && (
        <button type="button" className="pill" disabled={busy} aria-busy={busy} onClick={makeInvite}>
          Invite someone
        </button>
      )}
      <p className="invite-code mono" role="status">
        {invite}
      </p>
    </div>
  );
}

function Sharing({ h, api, owner }: { h: Household; api: Api; owner: boolean }) {
  const [on, setOn] = useState(h.share_public);
  const [error, setError] = useState("");
  const toggle = async (checked: boolean) => {
    setOn(checked);
    setError("");
    try {
      await api(`/api/v1/households/${h.id}`, { method: "PATCH", body: { share_public: checked } });
    } catch (err) {
      setOn(!checked);
      setError(err instanceof ApiError ? err.detail : (err as Error).message);
    }
  };
  return (
    <div className="card setup-card">
      <h2>Public map</h2>
      <label className="toggle">
        <input type="checkbox" checked={on} disabled={!owner} onChange={(e) => void toggle(e.target.checked)} />
        <span>Share anonymous counts from our public places</span>
      </label>
      <p className="hint">
        Off by default. Only item names and counts from places marked public, rounded to ~100 m, and only where 3+ households contribute. Private
        places never leave this household. New memories count from the moment this is on.
      </p>
      {!owner && <p className="hint">Only the person who set up the household can change this.</p>}
      {error && (
        <p className="form-msg error" role="status">
          {error}
        </p>
      )}
    </div>
  );
}

function Places({ h, api }: { h: Household; api: Api }) {
  const save = async (placeId: string, patch: Record<string, unknown>) => {
    await api(`/api/v1/households/${h.id}/places/${placeId}`, { method: "PATCH", body: patch });
    invalidateAll(); // place names and kinds show on the maps
  };
  return (
    <div className="card setup-card">
      <h2>Places</h2>
      <p className="hint">
        Found automatically from the wearable's Wi-Fi position. Homes stay private. Mark shops, parks and transit public if you want them to count on
        the anonymous public map.
      </p>
      {h.places.length ? (
        <ul className="rows">
          {h.places.map((p) => (
            <li key={p.id} className="place-row">
              <input
                className="place-name"
                defaultValue={p.name || ""}
                placeholder={p.label}
                aria-label={`Name for ${p.label}`}
                maxLength={80}
                onBlur={(e) => {
                  const name = e.target.value.trim();
                  if (name === e.target.defaultValue) return; // unchanged since the last save
                  e.target.defaultValue = name;
                  void save(p.id, { name: name || null });
                }}
              />
              <select aria-label={`Who can see ${p.label}`} defaultValue={p.kind} onChange={(e) => void save(p.id, { kind: e.target.value })}>
                <option value="private">Private</option>
                <option value="public">Public</option>
              </select>
              <span className="mono muted">{p.visits} memories</span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="hint">No places yet.</p>
      )}
    </div>
  );
}
