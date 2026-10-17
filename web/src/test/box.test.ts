import { describe, expect, it } from "vitest";
import { fromBox2d, usableBox } from "../lib/box";

describe("box_2d", () => {
  it("converts [ymin, xmin, ymax, xmax] on 0-1000 to [x, y, w, h] on 0..1", () => {
    expect(fromBox2d([200, 100, 600, 500])).toEqual([0.1, 0.2, 0.4, 0.4]);
    expect(fromBox2d([0, 0, 1000, 1000])).toEqual([0, 0, 1, 1]);
  });

  it("puts reversed corners right and clamps to the photo", () => {
    expect(fromBox2d([600, 500, 200, 100])).toEqual([0.1, 0.2, 0.4, 0.4]);
    expect(fromBox2d([-50, 900, 400, 1200])).toEqual([0.9, 0, 0.1, 0.4]);
  });

  it("gives null for nothing usable", () => {
    expect(fromBox2d([0, 0, 0, 0])).toBeNull();
    expect(fromBox2d([100, 100, 105, 500])).toBeNull();
    expect(fromBox2d([1, 2, 3])).toBeNull();
    expect(fromBox2d([1, 2, 3, NaN])).toBeNull();
  });

  it("only pre-draws boxes the server would accept", () => {
    expect(usableBox([0.1, 0.2, 0.4, 0.4])).toBe(true);
    expect(usableBox([0.9, 0.9, 0.5, 0.5])).toBe(false);
    expect(usableBox([0.1, 0.1, 0.01, 0.5])).toBe(false);
    expect(usableBox(null)).toBe(false);
    expect(usableBox([0.1, 0.1, "0.2", 0.2])).toBe(false);
  });
});
