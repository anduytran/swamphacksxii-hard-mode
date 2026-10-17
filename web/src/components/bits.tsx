// Small shared pieces: chips, empty states, skeletons, and the item chip sets.
import type { ReactNode } from "react";
import { useApp } from "../lib/app";
import { capitalize } from "../lib/format";
import type { Detection, Item } from "../lib/types";

export function Chip({ children, kind }: { children: ReactNode; kind?: "warn" | "auto" | "place" }) {
  return <span className={kind ? `chip ${kind}` : "chip"}>{children}</span>;
}

export function PlaceChip({ name, location }: { name: string; location: string | null }) {
  return (
    <span className="chip place">
      {capitalize(name)} → <b>{location || "somewhere"}</b>
    </span>
  );
}

export function ItemChips({ item }: { item: Item }) {
  return (
    <>
      <Chip>Put down {item.mentions}×</Chip>
      {item.times_asked > 0 && <Chip kind="warn">Asked about {item.times_asked}×</Chip>}
      {item.detection && <AutoChips boxes={[item.detection]} />}
    </>
  );
}

export function AutoChips({ boxes }: { boxes: Detection[] }) {
  return (
    <>
      <Chip kind="auto">Seen automatically</Chip>
      {boxes.some((b) => b.maybe) && <Chip kind="warn">Maybe</Chip>}
    </>
  );
}

/** What Gemini saw in the photo, shown under the wearer's own words. */
export function Saw({ scene }: { scene: string | null }) {
  if (!scene || scene.startsWith("Seen automatically")) return null;
  return <p className="saw">Camera saw: {scene}</p>;
}

// ---------- empty states ----------

type Cta = { label: string; href: string } | { label: string; onClick: () => void };

interface EmptyProps {
  text: string;
  title?: string;
  /** The last setup step's label: shows the household → wearable → this checklist. */
  checklist?: string;
  cta?: Cta;
  as?: "div" | "li";
}

/** Plain: one line. With a title: a card with the setup checklist and one action. */
export function EmptyState({ text, title, checklist, cta, as: Tag = "div" }: EmptyProps) {
  const { setup } = useApp();
  if (!title) return <Tag className="empty">{text}</Tag>;
  const steps: [string, boolean][] | null = checklist
    ? [
        ["Create a household for the person who wears it", setup.household],
        ["Add the wearable with the Client ID and pairing code on its label", setup.wearable],
        [checklist, false],
      ]
    : null;
  // Until setup is done, the action is finishing it.
  const action: Cta | undefined =
    checklist && !(setup.household && setup.wearable)
      ? { label: setup.household ? "Add the wearable" : "Set up a household", href: "#/household" }
      : cta;
  return (
    <Tag className="empty empty-rich">
      <h3>{title}</h3>
      <p>{text}</p>
      {steps && (
        <ol className="checklist" aria-label="Setup steps">
          {steps.map(([label, done]) => (
            <li key={label} className={done ? "done" : ""}>
              <span className="check" aria-hidden="true" />
              <span>
                {label}
                {done && <span className="sr-only"> (done)</span>}
              </span>
            </li>
          ))}
        </ol>
      )}
      {action &&
        ("href" in action ? (
          <a className="primary" href={action.href}>
            {action.label}
          </a>
        ) : (
          <button type="button" className="primary" onClick={action.onClick}>
            {action.label}
          </button>
        ))}
    </Tag>
  );
}

// ---------- loading skeletons (shapes match each page's components) ----------

export const Sk = ({ cls }: { cls: string }) => <span className={`sk ${cls}`} aria-hidden="true" />;

export const repeat = <T,>(n: number, make: (i: number) => T) => Array.from({ length: n }, (_, i) => make(i));

export function SkeletonItemCard() {
  return (
    <li className="item-card is-skeleton" aria-hidden="true">
      <div className="photo sk" />
      <div className="item-body">
        <Sk cls="line w60 h20" />
        <Sk cls="line w40" />
        <Sk cls="line w70" />
      </div>
    </li>
  );
}

export function SkeletonFeatured() {
  return (
    <article className="featured is-skeleton" aria-hidden="true">
      <div className="photo sk" />
      <div className="featured-body">
        <Sk cls="line w30" />
        <Sk cls="line w50 h40" />
        <Sk cls="line w40 h24" />
        <Sk cls="line w90" />
        <Sk cls="line w70" />
      </div>
    </article>
  );
}

export function SkeletonScene() {
  return (
    <li className="scene is-skeleton" aria-hidden="true">
      <div className="scene-time">
        <Sk cls="line w70" />
        <Sk cls="line w50" />
      </div>
      <div className="scene-body">
        <div className="photo sk" />
        <div className="scene-text">
          <Sk cls="line w80 h24" />
          <Sk cls="line w60" />
          <Sk cls="line w30" />
        </div>
      </div>
    </li>
  );
}

export function SkeletonResult() {
  return (
    <li className="result is-skeleton" aria-hidden="true">
      <div className="photo sk" />
      <div className="result-body">
        <Sk cls="line w30" />
        <Sk cls="line w80" />
        <Sk cls="line w60" />
        <Sk cls="line w70" />
      </div>
    </li>
  );
}
