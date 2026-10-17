// Items: where each thing was last put. The newest gets the big "latest scene" card.
import { useState } from "react";
import { EmptyState, ItemChips, repeat, Saw, SkeletonFeatured, SkeletonItemCard } from "../components/bits";
import { Icon } from "../components/Icons";
import { Photo } from "../components/Photo";
import { useApi } from "../lib/api";
import { useApp, useLoadStatus } from "../lib/app";
import { capitalize, clockTime, plural, shortAgo, timeAgo } from "../lib/format";
import { useResource } from "../lib/resource";
import type { Item, ItemsResponse } from "../lib/types";

export default function ItemsPage() {
  const api = useApi();
  const { openItem } = useApp();
  const res = useResource("items", () => api<ItemsResponse>("/api/v1/dashboard/items"));
  useLoadStatus(res.loading, res.error, res.reload);
  const [filter, setFilter] = useState("");
  const q = filter.trim().toLowerCase();
  const shown = (item: Item) => !q || item.name.includes(q);
  const items = res.data?.items;

  return (
    <section className="page">
      <div className="page-head">
        <div>
          <p className="eyebrow">
            Last {res.data?.window_hours ?? 72} hours · {plural(items?.length ?? 0, "item")}
          </p>
          <h1 className="display">Where things were left.</h1>
        </div>
        <div className="field search-field">
          <label htmlFor="item-filter">Find an item</label>
          <div className="search">
            <Icon name="search" />
            <input id="item-filter" type="search" placeholder="keys, glasses…" autoComplete="off" value={filter} onChange={(e) => setFilter(e.target.value)} />
          </div>
        </div>
      </div>

      {!items ? (
        <>
          <SkeletonFeatured />
          <ul className="items-grid">{repeat(6, (i) => <SkeletonItemCard key={i} />)}</ul>
        </>
      ) : !items.length ? (
        <ul className="items-grid">
          <EmptyState
            as="li"
            title="No items yet"
            text="When the wearer says something like “I'm putting my keys on the counter”, the item shows up here with a photo."
            checklist="Hold REMEMBER and say where something is going"
            cta={{ label: "Check again", onClick: res.reload }}
          />
        </ul>
      ) : (
        <>
          {shown(items[0]) && <Featured item={items[0]} onOpen={() => openItem(items[0].name)} />}
          <ul className="items-grid">
            {items.slice(1).filter(shown).map((item) => (
              <ItemCard key={item.name} item={item} onOpen={() => openItem(item.name)} />
            ))}
          </ul>
        </>
      )}
    </section>
  );
}

function Featured({ item, onOpen }: { item: Item; onOpen(): void }) {
  return (
    <article className="featured">
      <Photo url={item.image_url} alt={`The ${item.name} on the ${item.location || "spot shown"}`} boxes={item.detection ? [item.detection] : []}>
        <span className="tag">
          <span className="live-dot" />
          Latest scene
        </span>
      </Photo>
      <div className="featured-body">
        <span className="when mono">
          {timeAgo(item.last_seen)} · {clockTime(item.last_seen)}
        </span>
        <h2 className="item-name">
          <button type="button" className="card-link" onClick={onOpen}>
            {capitalize(item.name)}
          </button>
        </h2>
        <p className="item-location">{capitalize(item.location) || "Location not said"}</p>
        <p className="quote">{item.transcript ? `“${item.transcript}”` : item.scene || ""}</p>
        <Saw scene={item.scene} />
        <div className="chips">
          <ItemChips item={item} />
        </div>
      </div>
    </article>
  );
}

function ItemCard({ item, onOpen }: { item: Item; onOpen(): void }) {
  return (
    <li className="item-card">
      <Photo url={item.image_url} alt={`The ${item.name} on the ${item.location || "spot shown"}`} boxes={item.detection ? [item.detection] : []} />
      <div className="item-body">
        <div className="item-title">
          <h3 className="item-name">
            <button type="button" className="card-link" onClick={onOpen}>
              {capitalize(item.name)}
            </button>
          </h3>
          <span className="when mono" title={clockTime(item.last_seen)}>
            {shortAgo(item.last_seen)}
          </span>
        </div>
        <p className="item-location">{capitalize(item.location) || `“${item.transcript}”`}</p>
        <div className="chips">
          <ItemChips item={item} />
        </div>
        {item.previous.length > 0 && (
          <div className="before">
            <span className="eyebrow small">Before that</span>
            <ul>
              {item.previous.map((p) => (
                <li key={p.timestamp}>
                  {capitalize(p.location) || "Somewhere"} · {timeAgo(p.timestamp)}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </li>
  );
}

