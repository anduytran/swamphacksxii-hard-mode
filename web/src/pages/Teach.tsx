// Teach items: photos of the household's own things, each with a box drawn around the item,
// for the vision pipeline (YOLO) to learn from. The pipeline downloads them; nothing here runs a model.
import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import { EmptyState, repeat, Sk } from "../components/bits";
import { Dialog } from "../components/Dialog";
import { FormCard } from "../components/FormCard";
import { Icon } from "../components/Icons";
import { ApiError, useApi, type Api } from "../lib/api";
import { useLoadStatus } from "../lib/app";
import { useResource } from "../lib/resource";
import type { BoxSuggestion, Household, ModelStatus, ReviewCandidate, ReviewQueue, TrainingItem, TrainingList, TrainingSample } from "../lib/types";
import { timeAgo } from "../lib/format";
import { RANGES, rangeOf, spread, type Box } from "../lib/distance";
import { usableBox } from "../lib/box";
import { publishReviewCount } from "../lib/reviewCount";
import { openPhoto, PHOTO_ACCEPT } from "../lib/photo";

const GOOD_ENOUGH = 10; // photos per item before it's "ready"
const MAX_SIDE = 1024; // photos are shrunk to this before upload (the pipeline trains at 640)
const PER_RANGE = 2; // photos per distance before that range counts as covered

export default function TeachPage() {
  const api = useApi();
  const homes = useResource("households", () => api<{ households: Household[] }>("/api/v1/households/me"));
  const households = homes.data?.households;
  const [chosen, setChosen] = useState<string | null>(null);
  const household = households?.find((h) => h.id === chosen) ?? households?.[0];
  const base = household ? `/api/v1/households/${household.id}/training` : null;
  const list = useResource(base && `training:${household!.id}`, () => api<TrainingList>(`${base}/items`));
  const model = useResource(base && `trainmodel:${household!.id}`, () => api<ModelStatus>(`${base}/model`));
  const review = useResource(base && `review:${household!.id}`, () => api<ReviewQueue>(`${base}/review`));
  const changed = () => {
    list.reload();
    model.reload();
    review.reload();
  };
  useEffect(() => {
    if (household && review.data) publishReviewCount(household.id, review.data.count);
  }, [household, review.data]);
  useLoadStatus(homes.loading || list.loading, homes.error ?? list.error, () => (homes.reload(), list.reload()));

  return (
    <section className="page">
      <div className="page-head">
        <div>
          <p className="eyebrow">Setup</p>
          <h1 className="display">Teach the camera your things.</h1>
        </div>
        {households && households.length > 1 && (
          <label className="field household-pick">
            <span className="field-label">Household</span>
            <select value={household?.id} onChange={(e) => setChosen(e.target.value)}>
              {households.map((h) => (
                <option key={h.id} value={h.id}>
                  {h.name}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>
      <p className="lede teach-lede">
        The wearable's camera knows everyday things like keys and glasses. Show it <em>your</em> keys: add an item, then take{" "}
        {GOOD_ENOUGH} or so photos of it and draw a box around it in each. Mix close-ups with photos from a step back and
        from across the room, where the wearable usually sees things. Photos stay until you delete them.
      </p>

      {households && !households.length ? (
        <EmptyState title="Set up a household first" text="Items you teach belong to a household, so everyone in it shares them." cta={{ label: "Set up a household", href: "#/household" }} />
      ) : household && !household.devices.length ? (
        <EmptyState
          title="Add the wearable first"
          text="Items are taught for the wearable's camera, so a household needs its wearable paired before you add photos."
          cta={{ label: "Add the wearable", href: "#/household" }}
        />
      ) : !list.data || !base ? (
        <div className="teach-grid" aria-hidden="true">
          {repeat(2, (i) => (
            <div key={i} className="card teach-card is-skeleton">
              <Sk cls="line w40 h24" />
              <Sk cls="line w60" />
              <Sk cls="block h200" />
            </div>
          ))}
        </div>
      ) : (
        <>
          {review.data && review.data.count > 0 && <NeedsALook queue={review.data} base={base} api={api} onChange={changed} />}
          {model.data && <ModelLine status={model.data} />}
          <div className="teach-grid">
            <FormCard
              title="Add an item"
              hint="Use the word people say out loud. Other words are what else they might call it."
              fields={[
                { name: "name", label: "Item", required: true, maxLength: 40, placeholder: "keys", autoComplete: "off" },
                { name: "aliases", label: "Other words, comma-separated", maxLength: 200, placeholder: "car keys, key ring", autoComplete: "off" },
              ]}
              submitLabel="Add item"
              resetOnSuccess
              onSubmit={async (v) => {
                const aliases = (v.aliases ?? "").split(",").map((a) => a.trim()).filter(Boolean);
                await api(`${base}/items`, { method: "POST", body: { name: v.name, aliases } });
                changed();
                return `Added. Now add photos of ${v.name.trim().toLowerCase()}.`;
              }}
            />
            {list.data.items.map((item) => (
              <ItemCard
                key={item.id}
                item={item}
                base={base}
                api={api}
                maxPhotos={list.data!.limits.samples_per_item}
                needsLook={review.data?.per_item[item.id] ?? 0}
                onChange={changed}
              />
            ))}
          </div>
          {!list.data.items.length && <p className="hint">No items yet. Start with the one that goes missing most.</p>}
          <details className="teach-tech">
            <summary>For the vision pipeline team</summary>
            <p>
              The pipeline reads these with the wearable's token: <code>GET /api/v1/training/vocabulary</code> (class names for{" "}
              <code>set_classes</code>) and <code>GET /api/v1/training/dataset</code> (each photo with its YOLO label).{" "}
              <code>backend/scripts/fetch_training_set.py dataset</code> turns that into a YOLO folder with <code>data.yaml</code>.
            </p>
          </details>
        </>
      )}
    </section>
  );
}

function ItemCard({
  item,
  base,
  api,
  maxPhotos,
  needsLook,
  onChange,
}: {
  item: TrainingItem;
  base: string;
  api: Api;
  maxPhotos: number;
  needsLook: number;
  onChange(): void;
}) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [queue, setQueue] = useState<File[]>([]);
  const [confirming, setConfirming] = useState(false);
  const [error, setError] = useState("");
  const count = item.samples.length;
  const room = maxPhotos - count;

  const remove = async (path: string) => {
    setError("");
    try {
      await api(path, { method: "DELETE" });
      onChange();
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : (err as Error).message);
    }
  };

  return (
    <article className="card teach-card">
      <header className="teach-head">
        <div>
          <h2>{item.name}</h2>
          {item.aliases.length > 0 && <p className="hint">Also: {item.aliases.join(", ")}</p>}
        </div>
        <span className={count >= GOOD_ENOUGH ? "chip good" : count >= 5 ? "chip" : "chip warn"}>
          {count >= GOOD_ENOUGH ? <><Icon name="shield" /> Ready</> : `${count} of ${GOOD_ENOUGH} photos`}
        </span>
      </header>
      <div className="teach-meter" role="meter" aria-label={`${item.name} photos`} aria-valuemin={0} aria-valuemax={GOOD_ENOUGH} aria-valuenow={Math.min(count, GOOD_ENOUGH)}>
        <span style={{ width: `${Math.min(100, (count / GOOD_ENOUGH) * 100)}%` }} />
      </div>
      <DistanceSpread samples={item.samples} />
      {needsLook > 0 && (
        <button type="button" className="needs-look-line" onClick={() => document.getElementById("needs-a-look")?.scrollIntoView({ behavior: "smooth", block: "start" })}>
          <Icon name="sparkle" />
          {needsLook} photo{needsLook === 1 ? "" : "s"} need{needsLook === 1 ? "s" : ""} a look
        </button>
      )}

      {count > 0 ? (
        <ul className="sample-grid">
          {item.samples.map((s, i) => (
            <li key={s.id}>
              <SampleThumb sample={s} alt={`${item.name}, photo ${i + 1}`} />
              <button type="button" className="icon-btn sample-delete" aria-label={`Delete photo ${i + 1} of ${item.name}`} onClick={() => remove(`${base}/samples/${s.id}`)}>
                <Icon name="close" />
              </button>
            </li>
          ))}
        </ul>
      ) : (
        <p className="hint">No photos yet. Different distances, angles, backgrounds and light all help.</p>
      )}

      <div className="teach-actions">
        <input
          ref={fileRef}
          type="file"
          accept={PHOTO_ACCEPT}
          capture="environment"
          multiple
          hidden
          onChange={(e) => {
            const files = [...(e.target.files ?? [])].slice(0, Math.max(0, room));
            e.target.value = "";
            if (files.length) setQueue(files);
          }}
        />
        <button type="button" className="primary" disabled={room <= 0} onClick={() => fileRef.current?.click()}>
          {room <= 0 ? `${maxPhotos} photos is the most` : "Add photos"}
        </button>
        {confirming ? (
          <span className="confirm-row">
            <button type="button" className="pill danger" onClick={() => remove(`${base}/items/${item.id}`)}>
              Delete {item.name}
              {count ? ` and ${count} photo${count === 1 ? "" : "s"}` : ""}
            </button>
            <button type="button" className="pill quiet" onClick={() => setConfirming(false)}>
              Cancel
            </button>
          </span>
        ) : (
          <button type="button" className="pill quiet" onClick={() => setConfirming(true)}>
            Delete item
          </button>
        )}
      </div>
      {error && <p className="form-msg error">{error}</p>}

      {queue.length > 0 && (
        <BoxEditor
          item={item.name}
          files={queue}
          suggest={(photo) => suggestBox(api, `${base}/items/${item.id}/suggest-box`, { image: photo.blob })}
          upload={async (photo, box) => {
            const form = new FormData();
            form.append("image", photo.blob, "photo.jpg");
            form.append("bbox", JSON.stringify(box));
            form.append("width", String(photo.width));
            form.append("height", String(photo.height));
            await api(`${base}/items/${item.id}/samples`, { method: "POST", body: form });
          }}
          onClose={() => {
            setQueue([]);
            onChange();
          }}
        />
      )}
    </article>
  );
}

function SampleThumb({ sample, alt }: { sample: TrainingSample; alt: string }) {
  const [x, y, w, h] = sample.bbox;
  return (
    <figure className="sample" style={{ aspectRatio: `${sample.width} / ${sample.height}` }}>
      <img src={sample.image_url} alt={alt} loading="lazy" />
      <span className="sample-box" style={{ left: `${x * 100}%`, top: `${y * 100}%`, width: `${w * 100}%`, height: `${h * 100}%` }} />
    </figure>
  );
}

interface Prepared {
  blob: Blob;
  url: string;
  width: number;
  height: number;
}

/** Shrink to MAX_SIDE, upright, as JPEG (HEIC included): smaller uploads, a format the server and the
 * training pipeline both read, and the box matches the pixels sent. */
async function prepare(file: File): Promise<Prepared> {
  const bitmap = await openPhoto(file);
  const scale = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(bitmap.width * scale);
  canvas.height = Math.round(bitmap.height * scale);
  canvas.getContext("2d")!.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close();
  const blob = await new Promise<Blob>((ok, fail) => canvas.toBlob((b) => (b ? ok(b) : fail(new Error("Couldn't read that photo"))), "image/jpeg", 0.88));
  return { blob, url: URL.createObjectURL(blob), width: canvas.width, height: canvas.height };
}

const clamp = (v: number) => Math.min(1, Math.max(0, v));

interface Suggested {
  bbox: Box;
  note: string;
}

/** Gemini's box for a photo (an upload, or a memory photo by id), or null: no suggestion is never an error,
 * the caregiver just draws the box (vision off, the hourly limit, quota errors all land here). */
async function suggestBox(api: Api, path: string, from: { image: Blob } | { memory_id: string }): Promise<Suggested | null> {
  const form = new FormData();
  if ("image" in from) form.append("image", from.image, "photo.jpg");
  else form.append("memory_id", from.memory_id);
  try {
    const s = await api<BoxSuggestion>(path, { method: "POST", body: form });
    return usableBox(s.bbox) ? { bbox: s.bbox, note: s.note } : null;
  } catch {
    return null;
  }
}

/** One photo at a time: drag a box around the item, then save (or skip) and go to the next. With `suggest`,
 * a suggested box is pre-drawn while nobody has started drawing, so Save is one tap; dragging still redraws. */
function BoxEditor({
  item,
  files,
  upload,
  suggest,
  onClose,
}: {
  item: string;
  files: File[];
  upload(photo: Prepared, box: Box): Promise<void>;
  suggest?(photo: Prepared): Promise<Suggested | null>;
  onClose(): void;
}) {
  const [index, setIndex] = useState(0);
  const [photo, setPhoto] = useState<Prepared | null>(null);
  const [box, setBox] = useState<Box | null>(null);
  const [suggested, setSuggested] = useState<Suggested | null>(null);
  const [looking, setLooking] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [saved, setSaved] = useState(0);
  const start = useRef<[number, number] | null>(null);
  const frame = useRef<HTMLDivElement>(null);
  const touched = useRef(false); // the caregiver drew on this photo: a late suggestion mustn't replace it
  const suggestRef = useRef(suggest);
  suggestRef.current = suggest;

  useEffect(() => {
    let url = "";
    let gone = false;
    setPhoto(null);
    setBox(null);
    setSuggested(null);
    setLooking(false);
    setError("");
    touched.current = false;
    prepare(files[index]).then(
      (p) => {
        url = p.url;
        if (gone) return URL.revokeObjectURL(url);
        setPhoto(p);
        if (!suggestRef.current) return;
        setLooking(true);
        suggestRef.current(p).then((s) => {
          if (gone) return;
          setLooking(false);
          if (s && !touched.current) {
            setSuggested(s);
            setBox(s.bbox);
          }
        });
      },
      () => setError("Couldn't open that photo. JPEG, PNG, WebP and iPhone HEIC photos work; skip this one or try another."),
    );
    return () => {
      gone = true;
      if (url) URL.revokeObjectURL(url);
    };
  }, [files, index]);
  const own = (b: Box | null) => {
    touched.current = true;
    setSuggested(null);
    setBox(b);
  };

  const at = (e: ReactPointerEvent): [number, number] => {
    const r = frame.current!.getBoundingClientRect();
    return [clamp((e.clientX - r.left) / r.width), clamp((e.clientY - r.top) / r.height)];
  };
  const drag = (e: ReactPointerEvent) => {
    if (!start.current) return;
    const [x0, y0] = start.current;
    const [x1, y1] = at(e);
    setBox([Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0)]);
  };
  const tooSmall = !box || box[2] < 0.02 || box[3] < 0.02;

  const next = () => (index + 1 < files.length ? setIndex(index + 1) : onClose());
  const save = async () => {
    if (!photo || !box || tooSmall) return;
    setBusy(true);
    setError("");
    try {
      await upload(photo, box);
      setSaved((n) => n + 1);
      next();
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : (err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog open onClose={onClose} className="box-dialog" labelledBy="box-title">
      <header className="box-head">
        <div>
          <h2 id="box-title">Draw a box around the {item}</h2>
          <p className="hint">
            Photo {index + 1} of {files.length}
            {saved > 0 && ` · ${saved} saved`}.{" "}
            {suggested
              ? `Suggested box: save if it's tight around the ${item}, or drag to draw your own.`
              : looking
                ? `Looking for the ${item}… or drag from one corner to the other.`
                : `Drag from one corner to the other, tight around the ${item}.`}
          </p>
          {suggested?.note && <p className="hint box-note">Photo note: {suggested.note}</p>}
          <p className="hint box-range" aria-live="polite">
            {box && !tooSmall ? (
              <>
                Counts as <b>{RANGES.find((r) => r.key === rangeOf(box))!.label.toLowerCase()}</b>.
              </>
            ) : (
              " "
            )}
          </p>
        </div>
        <button type="button" className="icon-btn" data-close aria-label="Close">
          <Icon name="close" />
        </button>
      </header>
      <div className="box-stage">
        {photo ? (
          <div
            ref={frame}
            className="box-frame"
            // Exactly the photo's shape (never letterboxed), so the box maps onto its pixels.
            style={{ aspectRatio: `${photo.width} / ${photo.height}`, width: `min(100%, calc(62dvh * ${photo.width / photo.height}))` }}
            onPointerDown={(e) => {
              e.currentTarget.setPointerCapture(e.pointerId);
              start.current = at(e);
              own(null);
            }}
            onPointerMove={drag}
            onPointerUp={(e) => {
              drag(e);
              start.current = null;
            }}
          >
            <img src={photo.url} alt={`Photo ${index + 1} of the ${item}`} draggable={false} />
            {box && (
              <span
                className={suggested ? "sample-box drawing suggested" : "sample-box drawing"}
                style={{ left: `${box[0] * 100}%`, top: `${box[1] * 100}%`, width: `${box[2] * 100}%`, height: `${box[3] * 100}%` }}
              />
            )}
          </div>
        ) : (
          !error && (
            <div className="box-loading" role="status">
              <Sk cls="block h200" />
              <span className="hint">Opening photo…</span>
            </div>
          )
        )}
      </div>
      {error && <p className="form-msg error">{error}</p>}
      <footer className="box-actions">
        <button type="button" className="pill quiet" disabled={!photo} onClick={() => own([0, 0, 1, 1])}>
          Item fills the photo
        </button>
        <span className="box-actions-end">
          <button type="button" className="pill" onClick={next} disabled={busy}>
            Skip
          </button>
          <button type="button" className="primary" onClick={save} disabled={busy || !photo || tooSmall}>
            {busy ? "Saving…" : index + 1 < files.length ? "Save and next" : "Save"}
          </button>
        </span>
      </footer>
    </Dialog>
  );
}

/** How the item's photos spread over distance, and which distance to add next. */
function DistanceSpread({ samples }: { samples: TrainingSample[] }) {
  const { counts, missing } = spread(samples.map((x) => x.bbox), PER_RANGE);
  return (
    <div className="distance">
      <ul className="distance-ranges" aria-label="Photos by distance">
        {RANGES.map((r) => {
          const done = counts[r.key] >= PER_RANGE;
          return (
            <li key={r.key} className={done ? "done" : ""}>
              <span className="distance-count" aria-hidden="true">
                {done ? <Icon name="check" /> : counts[r.key]}
              </span>
              <span>
                {r.label}
                <span className="sr-only">
                  : {counts[r.key]} photo{counts[r.key] === 1 ? "" : "s"}
                  {done ? ", covered" : ""}
                </span>
              </span>
            </li>
          );
        })}
      </ul>
      {samples.length > 0 && (
        <p className="hint distance-next">
          {missing
            ? `Next: ${PER_RANGE - counts[missing.key] === 1 ? "one more" : PER_RANGE - counts[missing.key]} ${missing.label.toLowerCase()}, ${missing.tip}.`
            : "Good spread of distances."}
        </p>
      )}
    </div>
  );
}

/** Whether the camera's model has learned the current photos (trained on the laptop or Vultr). */
function ModelLine({ status }: { status: ModelStatus }) {
  if (!status.photos) return null;
  const { model } = status;
  const text = !model
    ? "The camera hasn't been trained on these photos yet."
    : status.up_to_date
      ? `The camera learned these photos ${model.trained_at ? timeAgo(model.trained_at) : ""}.`
      : `Photos changed since the camera last learned them ${model.trained_at ? timeAgo(model.trained_at) : ""}. It will catch up at the next training run.`;
  return (
    <p className={status.up_to_date ? "model-line good" : "model-line"} role="status">
      <Icon name={status.up_to_date ? "shield" : "sparkle"} />
      {text}
    </p>
  );
}

const cardKey = (c: ReviewCandidate) => `${c.item_id}:${c.memory_id}`;

/** "Needs a look": one photo at a time that the camera was unsure about, or that names an item with no box.
 * Yes and Fix make it a training photo (same limits as Add photos); Not is remembered; Skip is for now only. */
function NeedsALook({ queue, base, api, onChange }: { queue: ReviewQueue; base: string; api: Api; onChange(): void }) {
  const [skipped, setSkipped] = useState<string[]>([]);
  const card = queue.candidates.find((c) => !skipped.includes(cardKey(c)));
  return (
    <section id="needs-a-look" className="card review" aria-labelledby="review-title">
      <header className="teach-head">
        <div>
          <h2 id="review-title">Needs a look</h2>
          <p className="hint">The camera wasn't sure about these. Each answer teaches it.</p>
        </div>
        <span className="chip attention">
          {queue.count} photo{queue.count === 1 ? "" : "s"}
        </span>
      </header>
      {card ? (
        <ReviewCard key={cardKey(card)} card={card} base={base} api={api} onSkip={() => setSkipped((s) => [...s, cardKey(card)])} onChange={onChange} />
      ) : (
        <p className="hint">
          Skipped for now.{" "}
          <button type="button" className="pill quiet" onClick={() => setSkipped([])}>
            Show them again
          </button>
        </p>
      )}
    </section>
  );
}

function ReviewCard({ card, base, api, onSkip, onChange }: { card: ReviewCandidate; base: string; api: Api; onSkip(): void; onChange(): void }) {
  const [box, setBox] = useState<Box | null>(usableBox(card.bbox) ? card.bbox : null);
  const [note, setNote] = useState(card.note);
  const [looking, setLooking] = useState(false);
  // The photo's real pixel size, read when it loads: the box is drawn over it and sent with Yes / Fix.
  const [size, setSize] = useState<{ width: number; height: number } | null>(null);
  const [fixing, setFixing] = useState<File[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const item = card.item_name;

  useEffect(() => {
    // A memory photo has no detector box: ask Gemini once (the server caches its answer for this card).
    if (card.kind !== "memory" || card.bbox || card.suggested) return;
    let gone = false;
    setLooking(true);
    suggestBox(api, `${base}/items/${card.item_id}/suggest-box`, { memory_id: card.memory_id }).then((s) => {
      if (gone) return;
      setLooking(false);
      if (s) {
        setBox(s.bbox);
        setNote(s.note);
      }
    });
    return () => {
      gone = true;
    };
  }, [api, base, card]);

  const decide = async (decision: "yes" | "fix" | "not", bbox: Box | null = null) => {
    const body = { item_id: card.item_id, memory_id: card.memory_id, decision, bbox, width: size?.width, height: size?.height };
    await api(`${base}/review`, { method: "POST", body });
  };
  const run = async (decision: "yes" | "not") => {
    setBusy(true);
    setError("");
    try {
      await decide(decision, decision === "yes" ? box : null);
      onChange();
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : (err as Error).message);
      setBusy(false);
    }
  };
  const fix = async () => {
    setError("");
    try {
      const res = await fetch(card.image_url);
      if (!res.ok) throw new Error("Couldn't open that photo; refresh and try again.");
      const blob = await res.blob();
      setFixing([new File([blob], "photo.jpg", { type: blob.type || "image/jpeg" })]);
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const when = timeAgo(card.captured_at);
  return (
    <div className="review-card">
      <figure className="sample review-photo" style={size ? { aspectRatio: `${size.width} / ${size.height}` } : undefined}>
        <img
          src={card.image_url}
          alt={`Photo that may show the ${item}`}
          onLoad={(e) => setSize({ width: e.currentTarget.naturalWidth, height: e.currentTarget.naturalHeight })}
        />
        {box && (
          <span
            className="sample-box suggested"
            style={{ left: `${box[0] * 100}%`, top: `${box[1] * 100}%`, width: `${box[2] * 100}%`, height: `${box[3] * 100}%` }}
          />
        )}
      </figure>
      <div className="review-ask">
        <h3>Is this the {item}?</h3>
        <p className="hint">
          {card.kind === "detection"
            ? `The camera thought so ${when}${card.confidence != null ? `, ${Math.round(card.confidence * 100)}% sure` : ""}.`
            : `From a memory ${when} that mentions the ${item}.`}{" "}
          {box ? "Check the box." : looking ? "Looking for it…" : "No box yet: draw one if it's there."}
        </p>
        {note && <p className="hint box-note">Photo note: {note}</p>}
        <div className="review-actions">
          {box && (
            <button type="button" className="primary" disabled={busy || !size} onClick={() => run("yes")}>
              Yes
            </button>
          )}
          <button type="button" className={box ? "pill" : "primary"} disabled={busy} onClick={fix}>
            {box ? "Fix the box" : "Draw the box"}
          </button>
          <button type="button" className="pill" disabled={busy} onClick={() => run("not")}>
            Not the {item}
          </button>
          <button type="button" className="pill quiet" disabled={busy} onClick={onSkip}>
            Skip
          </button>
        </div>
        {error && <p className="form-msg error">{error}</p>}
      </div>
      {fixing && (
        <BoxEditor
          item={item}
          files={fixing}
          suggest={async () => (box ? { bbox: box, note } : null)}
          upload={async (_photo, drawn) => {
            await decide("fix", drawn);
            onChange();
          }}
          onClose={() => setFixing(null)}
        />
      )}
    </div>
  );
}
