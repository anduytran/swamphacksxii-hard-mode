import { describe, expect, it } from "vitest";
import { distanceMix, distanceOf, nextDistance } from "../lib/distance";

describe("Teach photo distance", () => {
  it("judges distance from the box's larger side", () => {
    expect(distanceOf([0.3, 0.3, 0.4, 0.2])).toBe("close");
    expect(distanceOf([0, 0, 0.3, 0.1])).toBe("close"); // exactly CLOSE
    expect(distanceOf([0, 0, 0.2, 0.25])).toBe("middle");
    expect(distanceOf([0, 0, 0.15, 0.1])).toBe("middle"); // exactly FAR
    expect(distanceOf([0, 0, 0.12, 0.08])).toBe("far");
    expect(distanceOf([0, 0, 1, 1])).toBe("close"); // "Item fills the photo"
  });

  it("the live-tested pin set has no far photos", () => {
    // larger box side of each of the 10 Pin photos taught on Oct 17
    const sizes = [0.29, 0.44, 0.36, 0.3, 0.29, 0.54, 0.36, 0.42, 0.39, 0.27];
    const mix = distanceMix(sizes.map((s) => [0, 0, s, s * 0.9] as const));
    expect(mix).toEqual({ close: 7, middle: 3, far: 0 });
    expect(nextDistance(mix)).toBe("far");
  });

  it("asks for the range furthest below 3, far first on a tie, and nothing once all are met", () => {
    expect(nextDistance({ close: 0, middle: 0, far: 0 })).toBe("far");
    expect(nextDistance({ close: 0, middle: 2, far: 3 })).toBe("close");
    expect(nextDistance({ close: 5, middle: 1, far: 1 })).toBe("far");
    expect(nextDistance({ close: 5, middle: 1, far: 2 })).toBe("middle");
    expect(nextDistance({ close: 3, middle: 3, far: 3 })).toBeNull();
  });
});
