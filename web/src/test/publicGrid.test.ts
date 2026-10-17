import { describe, expect, it } from "vitest";
import { binCells, coreSpots, gridLevel, HEAT_COLORS, rampColor, simulatedLostMap } from "../lib/publicGrid";

const cell = (lat: number, lng: number, left: number, asked = 0, name = "keys") => ({ lat, lng, left, asked, items: [{ name, left, asked }] });

describe("gridLevel", () => {
  it("is the 100 m grid up close and doubles per zoom level out, capped", () => {
    expect(gridLevel(15)).toBe(0);
    expect(gridLevel(13.5)).toBe(0);
    expect(gridLevel(12.5)).toBe(1);
    expect(gridLevel(11.5)).toBe(2);
    expect(gridLevel(3)).toBe(6);
  });
});

describe("binCells", () => {
  it("keeps cells as they are at level 0", () => {
    const out = binCells([cell(29.65, -82.34, 3, 1)], 0);
    expect(out).toEqual([{ ...cell(29.65, -82.34, 3, 1), total: 4, spots: 1 }]);
  });

  it("merges neighbours into bigger cells when zoomed out, summing counts and items", () => {
    const cells = [cell(29.6481, -82.3438, 2, 1, "keys"), cell(29.6484, -82.3441, 1, 0, "keys"), cell(29.6484, -82.3441, 5, 0, "wallet")];
    const [bin, ...rest] = binCells(cells, 3); // 800 m cells
    expect(rest).toHaveLength(0);
    expect(bin.total).toBe(9);
    expect(bin.spots).toBe(3);
    expect(bin.items).toEqual([
      { name: "wallet", left: 5, asked: 0 },
      { name: "keys", left: 3, asked: 1 },
    ]);
  });

  it("keeps distant cells apart", () => {
    expect(binCells([cell(29.6481, -82.3438, 1), cell(29.6386, -82.3681, 1)], 2)).toHaveLength(2);
  });
});

describe("rampColor", () => {
  it("runs from the first stop to the last, clamped", () => {
    expect(rampColor(0)).toEqual(HEAT_COLORS[0]);
    expect(rampColor(1)).toEqual(HEAT_COLORS[HEAT_COLORS.length - 1]);
    expect(rampColor(-1)).toEqual(HEAT_COLORS[0]);
    expect(rampColor(2)).toEqual(HEAT_COLORS[HEAT_COLORS.length - 1]);
  });
});

describe("coreSpots", () => {
  it("frames the middle 80% so a far-off spot doesn't zoom the map out", () => {
    const near = Array.from({ length: 20 }, (_, i) => ({ lat: 29.648 + i * 0.0001, lng: -82.344 + i * 0.0001 }));
    const points = coreSpots([...near, { lat: 30.5, lng: -81 }]);
    expect(points.length).toBeLessThan(21);
    expect(points.every(([lng]) => lng < -82)).toBe(true);
  });

  it("keeps every spot when there are few", () => {
    expect(coreSpots([{ lat: 1, lng: 2 }, { lat: 3, lng: 4 }])).toEqual([
      [2, 1],
      [4, 3],
    ]);
  });
});

describe("simulatedLostMap", () => {
  it("is the same every time, around the UF campus, busiest first", () => {
    const a = simulatedLostMap();
    expect(a).toEqual(simulatedLostMap());
    expect(a.simulated).toBe(true);
    expect(a.cells.length).toBeGreaterThan(50);
    for (const c of a.cells) {
      expect(c.lat).toBeGreaterThan(29.62);
      expect(c.lat).toBeLessThan(29.67);
      expect(c.lng).toBeGreaterThan(-82.39);
      expect(c.lng).toBeLessThan(-82.32);
      expect(c.left + c.asked).toBe(c.items.reduce((s, i) => s + i.left + i.asked, 0));
    }
    const totals = a.cells.map((c) => c.left + c.asked);
    expect(totals).toEqual([...totals].sort((x, y) => y - x));
  });
});
