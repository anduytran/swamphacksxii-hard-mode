import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { DirectionView } from "../components/ItemDrawer";

const text = (html: string) => html.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
const now = new Date().toISOString();

describe("DirectionView", () => {
  it("points toward the item with distance, direction, floors and rooms", () => {
    const html = renderToStaticMarkup(
      <DirectionView dir={{ item: "pin", known: true, here: false, meters: 42, bearing: 45, compass: "north-east", floors: 1,
        item_room: "Meeting Room 2360", wearer_room: "Grand Ballroom", accuracy_m: 38, lastscene_at: now, wearer_at: now,
        sentence: "Its LastScene is in meeting room 2360, about 40 meters north-east of you, one floor up." }} />,
    );
    const t = text(html);
    expect(t).toContain("42 m north-east · 1 floor up");
    expect(t).toContain("Wearer: Grand Ballroom → It: Meeting Room 2360");
    expect(t).toContain("±38 m");
    expect(html).toContain('transform="rotate(45)"');
  });

  it("says when it's here, and shows nothing when unknown", () => {
    const here = text(renderToStaticMarkup(<DirectionView dir={{ item: "pin", known: true, here: true, item_room: "Meeting Room 2360" }} />));
    expect(here).toContain("Here, Meeting Room 2360");
    expect(renderToStaticMarkup(<DirectionView dir={{ item: "pin", known: false }} />)).toBe("");
  });
});
