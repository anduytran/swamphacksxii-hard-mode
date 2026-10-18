import { describe, expect, it } from "vitest";
import { usableBox } from "../lib/box";

describe("usableBox", () => {
  it("only pre-draws boxes the server would accept", () => {
    expect(usableBox([0.1, 0.2, 0.4, 0.4])).toBe(true);
    expect(usableBox([0.9, 0.9, 0.5, 0.5])).toBe(false);
    expect(usableBox([0.1, 0.1, 0.01, 0.5])).toBe(false);
    expect(usableBox(null)).toBe(false);
    expect(usableBox([0.1, 0.1, "0.2", 0.2])).toBe(false);
  });
});
