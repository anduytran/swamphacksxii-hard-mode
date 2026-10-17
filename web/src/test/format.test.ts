import { describe, expect, it } from "vitest";
import { bucketByHour, capitalize, initials, plural, shortAgo, timeAgo } from "../lib/format";

const NOW = Date.parse("2026-10-17T12:00:00Z");
const ago = (minutes: number) => new Date(NOW - minutes * 60000).toISOString();

describe("time formatting", () => {
  it("timeAgo reads naturally", () => {
    expect(timeAgo(ago(0), NOW)).toBe("just now");
    expect(timeAgo(ago(5), NOW)).toBe("5 min ago");
    expect(timeAgo(ago(60), NOW)).toBe("1 hour ago");
    expect(timeAgo(ago(180), NOW)).toBe("3 hours ago");
    expect(timeAgo(ago(60 * 48), NOW)).toBe("2 days ago");
  });

  it("shortAgo is compact", () => {
    expect(shortAgo(ago(0), NOW)).toBe("1m ago");
    expect(shortAgo(ago(90), NOW)).toBe("2h ago");
    expect(shortAgo(ago(60 * 30), NOW)).toBe("1d ago");
  });
});

describe("names", () => {
  it("initials", () => {
    expect(initials("Riley Chen")).toBe("RC");
    expect(initials("sam@example.com")).toBe("S");
    expect(initials("dylan.tran@x.org")).toBe("DT");
    expect(initials("")).toBe("?");
  });

  it("capitalize and plural", () => {
    expect(capitalize("keys")).toBe("Keys");
    expect(capitalize(null)).toBe("");
    expect(plural(1, "item")).toBe("1 item");
    expect(plural(3, "item")).toBe("3 items");
  });
});

describe("bucketByHour", () => {
  it("counts events into 3-hour blocks of the local day", () => {
    const at = (h: number) => ({ timestamp: new Date(2026, 9, 17, h, 30).toISOString() });
    expect(bucketByHour([at(0), at(2), at(9), at(23)], 8)).toEqual([2, 0, 0, 1, 0, 0, 0, 1]);
  });
});
