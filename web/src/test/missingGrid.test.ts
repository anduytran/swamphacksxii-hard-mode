import { describe, expect, it } from "vitest";
import { heatLevel } from "../components/Charts";
import { binMissing, missingPeak } from "../pages/Insights";

// Hours built from local times, so the test passes in any time zone.
const at = (y: number, m: number, d: number, h: number) => new Date(y, m - 1, d, h).toISOString();

describe("binMissing", () => {
  it("places each hour on the viewer's local weekday (Monday first) and hour", () => {
    const grid = binMissing(
      [
        { at: at(2026, 10, 12, 18), asked: 3, put_down: 1 }, // Monday 6pm
        { at: at(2026, 10, 18, 9), asked: 1, put_down: 4 }, // Sunday 9am
        { at: at(2026, 10, 19, 18), asked: 2, put_down: 0 }, // the next Monday 6pm: adds up
      ],
      "asked",
    );
    expect(grid).toHaveLength(7);
    expect(grid[0][18]).toBe(5);
    expect(grid[6][9]).toBe(1);
    expect(grid.flat().reduce((a, b) => a + b, 0)).toBe(6);
  });
});

describe("missingPeak", () => {
  it("names the busiest weekday and 3-hour block", () => {
    const grid = binMissing([{ at: at(2026, 10, 13, 19), asked: 6, put_down: 0 }, { at: at(2026, 10, 15, 8), asked: 2, put_down: 0 }], "asked");
    expect(missingPeak(grid, "asked")).toBe("Questions about lost things come most on Tuesdays and between 6–9pm (75% of the total).");
  });
  it("says nothing when there's nothing", () => {
    expect(missingPeak(binMissing([], "asked"), "asked")).toBeNull();
  });
});

describe("heatLevel", () => {
  it("keeps zero separate and steps up to the busiest cell", () => {
    expect([0, 1, 3, 5, 8].map((v) => heatLevel(v, 8))).toEqual([0, 1, 2, 3, 4]);
    expect(heatLevel(3, 0)).toBe(0);
  });
});
