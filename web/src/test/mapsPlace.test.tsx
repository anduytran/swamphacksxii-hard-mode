import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { MapsNote, MapsPlace } from "../components/MapsPlace";
import { Scene } from "../pages/Timeline";
import type { HistoryEntry, PlaceName } from "../lib/types";

const starbucks: PlaceName = {
  name: "Starbucks, Reitz Union",
  sources: [{ title: "Starbucks", url: "https://maps.google.com/?cid=1" }],
};

const text = (html: string) => html.replace(/<[^>]+>/g, "");

describe("MapsPlace", () => {
  it("puts the linked sources and Google Maps right after the name", () => {
    const html = renderToStaticMarkup(<MapsPlace place={starbucks} />);
    expect(text(html)).toBe("Starbucks, Reitz Union (Starbucks · Google Maps)");
    expect(html).toContain('<a href="https://maps.google.com/?cid=1" target="_blank" rel="noopener noreferrer">Starbucks</a>');
    expect(html).toContain('<span class="google-maps" translate="no">Google Maps</span>');
  });

  it("lists every source", () => {
    const two = { ...starbucks, sources: [...starbucks.sources, { title: "Reitz Union", url: "https://maps.google.com/?cid=2" }] };
    expect(text(renderToStaticMarkup(<MapsPlace place={two} />))).toBe("Starbucks, Reitz Union (Starbucks, Reitz Union · Google Maps)");
  });

  it("says Google Maps data is used", () => {
    expect(text(renderToStaticMarkup(<MapsNote />))).toBe("Place names from Google Maps.");
  });
});

describe("Timeline scene", () => {
  const entry: HistoryEntry = {
    id: "m1",
    device_id: "dev",
    timestamp: new Date().toISOString(),
    source: "manual",
    transcript: "leaving my umbrella here",
    scene: null,
    summary: null,
    detections: [],
    image_url: null,
    items: [],
  };

  it("shows the place name with its attribution", () => {
    const html = renderToStaticMarkup(<Scene entry={{ ...entry, place_name: starbucks }} />);
    expect(text(html)).toContain("At Starbucks, Reitz Union (Starbucks · Google Maps)");
  });

  it("shows nothing from Google Maps without a name", () => {
    expect(renderToStaticMarkup(<Scene entry={entry} />)).not.toContain("Google Maps");
  });
});
