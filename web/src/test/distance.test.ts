import { describe, expect, it } from "vitest";
import { rangeOf, spread, type Box } from "../lib/distance";

const close: Box = [0, 0, 1, 1];
const step: Box = [0.4, 0.4, 0.15, 0.08];
const far: Box = [0.5, 0.5, 0.04, 0.03];

describe("distance", () => {
  it("judges distance from the box's longer side", () => {
    expect(rangeOf(close)).toBe("close");
    expect(rangeOf([0, 0, 0.3, 0.05])).toBe("close");
    expect(rangeOf(step)).toBe("step");
    expect(rangeOf(far)).toBe("far");
    expect(rangeOf([0, 0, 0.125, 0.1])).toBe("far"); // an eighth of the frame: missed on the Pi
  });

  it("asks for the thinnest range next, the farthest on a tie", () => {
    expect(spread([], 2).missing?.key).toBe("far");
    expect(spread([close, close, far], 2).missing?.key).toBe("step");
    expect(spread([close, step, far], 2).missing?.key).toBe("far");
    const done = spread([close, close, step, step, far, far], 2);
    expect(done.missing).toBeUndefined();
    expect(done.counts).toEqual({ close: 2, step: 2, far: 2 });
  });
});
