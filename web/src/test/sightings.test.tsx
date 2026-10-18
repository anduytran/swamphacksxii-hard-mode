import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { SightingList } from "../components/ItemDrawer";
import type { Sighting } from "../lib/types";

const now = new Date().toISOString();
const sighting = (over: Partial<Sighting>): Sighting => ({
  id: "s1", captured_at: now, source: "auto", confidence: 0.87, maybe: false, bbox: [0.4, 0.3, 0.2, 0.3],
  image_size: { width: 768, height: 432 }, image_url: "/api/v1/images/ih-s1?exp=1&sig=x", where: "on the desk by the laptop",
  summary: "The pin is on the desk.", nearby: ["laptop", "blue mug"], room: "office", scene: null,
  location: { lat: 29.6465, lng: -82.3475, accuracy_m: 20 }, place: { id: "p1", label: "Reitz Union" }, last_seen: true, ...over,
});
const text = (html: string) => html.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
const render = (list: Sighting[], confirming: string | null = null) =>
  renderToStaticMarkup(<SightingList name="pin" list={list} confirming={confirming} busy={false} onAsk={() => {}} onTrash={() => {}} />);

describe("SightingList", () => {
  it("shows each sighting with raia's place words, the place, and which one is the LastScene", () => {
    const html = render([sighting({}), sighting({ id: "s2", last_seen: false, where: null, summary: null, nearby: [], room: null,
      place: null, scene: "kitchen counter", confidence: 0.42, maybe: true, source: "raia" })]);
    const t = text(html);
    expect(t).toContain("Last 2 sightings");
    expect(t).toContain("On the desk by the laptop");
    expect(t).toContain("Next to laptop, blue mug");
    expect(t).toContain("Reitz Union · office");
    expect(t).toContain("LastScene ·");
    expect(t).toContain("On the kitchen counter");
    expect(t).toContain("Place not known");
    expect(t).toContain("42% (maybe) · raia");
    expect(html).toContain('src="/api/v1/images/ih-s1?exp=1&amp;sig=x"');
    expect(html.match(/<li class="now"/g)?.length).toBe(1);
    expect(html.match(/>Trash<\/button>/g)?.length).toBe(2);
  });

  it("asks before trashing", () => {
    const t = text(render([sighting({}), sighting({ id: "s2", last_seen: false })], "s2"));
    expect(t).toContain("Trash this sighting");
    expect(t).toContain("Cancel");
  });
});
