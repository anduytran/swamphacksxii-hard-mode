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
import type { Household, TrainingItem, TrainingList, TrainingSample } from "../lib/types";

const GOOD_ENOUGH = 10; // photos per item before it's "ready"
const MAX_SIDE = 1024; // photos are shrunk to this before upload (the pipeline trains at 640)

type Box = [number, number, number, number]; // x, y, w, h in 0..1

export default function TeachPage() {
  const api = useApi();
  const homes = useResource("households", () => api<{ households: Household[] }>("/api/v1/households/me"));
  const households = homes.data?.households;
  const [chosen, setChosen] = useState<string | null>(null);
  const household = households?.find((h) => h.id === chosen) ?? households?.[0];
  const base = household ? `/api/v1/households/${household.id}/training` : null;
  const list = useResource(base && `training:${household!.id}`, () => api<TrainingList>(`${base}/items`));
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
        {GOOD_ENOUGH} or so photos of it and draw a box around it in each. Photos stay until you delete them.
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
                list.reload();
                return `Added. Now add photos of ${v.name.trim().toLowerCase()}.`;
              }}
            />
            {list.data.items.map((item) => (
              <ItemCard key={item.id} item={item} base={base} api={api} maxPhotos={list.data!.limits.samples_per_item} onChange={list.reload} />
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

function ItemCard({ item, base, api, maxPhotos, onChange }: { item: TrainingItem; base: string; api: Api; maxPhotos: number; onChange(): void }) {
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
        <p className="hint">No photos yet. Different angles, distances, backgrounds and light all help.</p>
      )}

      <div className="teach-actions">
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
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

/** Shrink to MAX_SIDE, upright (EXIF), as JPEG: smaller uploads, and the box matches the pixels sent. */
async function prepare(file: File): Promise<Prepared> {
  const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
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

/** One photo at a time: drag a box around the item, then save (or skip) and go to the next. */
function BoxEditor({ item, files, upload, onClose }: { item: string; files: File[]; upload(photo: Prepared, box: Box): Promise<void>; onClose(): void }) {
  const [index, setIndex] = useState(0);
  const [photo, setPhoto] = useState<Prepared | null>(null);
  const [box, setBox] = useState<Box | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [saved, setSaved] = useState(0);
  const start = useRef<[number, number] | null>(null);
  const frame = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let url = "";
    setPhoto(null);
    setBox(null);
    setError("");
    prepare(files[index]).then(
      (p) => {
        url = p.url;
        setPhoto(p);
      },
      () => setError("Couldn't open that photo. Try a JPEG or PNG, or skip it."),
    );
    return () => {
      if (url) URL.revokeObjectURL(url);
    };
  }, [files, index]);

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
            {saved > 0 && ` · ${saved} saved`}. Drag from one corner to the other, tight around the {item}.
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
              setBox(null);
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
                className="sample-box drawing"
                style={{ left: `${box[0] * 100}%`, top: `${box[1] * 100}%`, width: `${box[2] * 100}%`, height: `${box[3] * 100}%` }}
              />
            )}
          </div>
        ) : (
          !error && <Sk cls="block h200" />
        )}
      </div>
      {error && <p className="form-msg error">{error}</p>}
      <footer className="box-actions">
        <button type="button" className="pill quiet" disabled={!photo} onClick={() => setBox([0, 0, 1, 1])}>
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
