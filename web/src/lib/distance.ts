// Distance for Teach photos, judged from how big the item's box is in the photo (its longer side, as a
// share of the photo's). The wearable mostly sees things small and from across the room, so a model taught
// only close-ups misses them there: Teach asks for a spread.
export type Box = [number, number, number, number]; // x, y, w, h in 0..1
export type Range = "close" | "step" | "far";

export const RANGES: { key: Range; label: string; tip: string }[] = [
  { key: "close", label: "Close up", tip: "the item fills a good part of the photo" },
  { key: "step", label: "A step back", tip: "from about where you'd stand to pick it up" },
  { key: "far", label: "Across the room", tip: "small in the photo, the way the wearable usually sees it" },
];

export function rangeOf([, , w, h]: Box): Range {
  const side = Math.max(w, h);
  return side >= 0.3 ? "close" : side >= 0.1 ? "step" : "far";
}

/** Photos per range, and the range to ask for next: the thinnest one, on a tie the farthest
 * (the one people skip, and the one the wearable needs). None once each has `perRange`. */
export function spread(boxes: Box[], perRange: number) {
  const counts: Record<Range, number> = { close: 0, step: 0, far: 0 };
  for (const b of boxes) counts[rangeOf(b)]++;
  const missing = [...RANGES].reverse().filter((r) => counts[r.key] < perRange).sort((a, b) => counts[a.key] - counts[b.key])[0];
  return { counts, missing };
}
