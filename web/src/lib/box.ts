// Boxes in the app's format, [x, y, w, h] in 0..1 with a top-left origin.
import type { Box } from "./distance";

/** A box the server will accept as a training box: four numbers, inside the photo, not a sliver. */
export function usableBox(box: unknown): box is Box {
  if (!Array.isArray(box) || box.length !== 4 || !box.every((v) => typeof v === "number" && Number.isFinite(v))) return false;
  const [x, y, w, h] = box as Box;
  return x >= 0 && y >= 0 && w >= 0.02 && h >= 0.02 && x + w <= 1.001 && y + h <= 1.001;
}
