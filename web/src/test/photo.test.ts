import { describe, expect, it } from "vitest";
import { looksHeic } from "../lib/photo";

describe("photo", () => {
  it("spots HEIC by type or name", () => {
    expect(looksHeic({ name: "IMG_0001.HEIC", type: "" })).toBe(true);
    expect(looksHeic({ name: "x", type: "image/heif" })).toBe(true);
    expect(looksHeic({ name: "burst", type: "image/heic-sequence" })).toBe(true);
    expect(looksHeic({ name: "keys.jpg", type: "image/jpeg" })).toBe(false);
  });
});
