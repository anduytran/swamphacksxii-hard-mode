// Boxes in the app's format, [x, y, w, h] in 0..1 with a top-left origin, and Gemini's detection format,
// box_2d = [ymin, xmin, ymax, xmax] on 0-1000. The server converts Gemini's suggestions (backend/app/box_suggest.py);
// this mirrors it for anything that reads box_2d in the browser, and checks a suggestion before pre-drawing it.
import type { Box } from "./distance";

const unit = (v: number) => Math.min(1, Math.max(0, v));
const r5 = (v: number) => Math.round(v * 1e5) / 1e5;

/** box_2d to [x, y, w, h], clamped to the photo; null when nothing usable is left. */
export function fromBox2d(box2d: number[]): Box | null {
  if (box2d.length !== 4 || box2d.some((v) => !Number.isFinite(v))) return null;
  const [ymin, xmin, ymax, xmax] = box2d.map((v) => unit(v / 1000));
  const [x0, x1] = [Math.min(xmin, xmax), Math.max(xmin, xmax)];
  const [y0, y1] = [Math.min(ymin, ymax), Math.max(ymin, ymax)];
  if (x1 - x0 < 0.01 || y1 - y0 < 0.01) return null;
  return [r5(x0), r5(y0), r5(x1 - x0), r5(y1 - y0)];
}

/** A box the server will accept as a training box: four numbers, inside the photo, not a sliver. */
export function usableBox(box: unknown): box is Box {
  if (!Array.isArray(box) || box.length !== 4 || !box.every((v) => typeof v === "number" && Number.isFinite(v))) return false;
  const [x, y, w, h] = box as Box;
  return x >= 0 && y >= 0 && w >= 0.02 && h >= 0.02 && x + w <= 1.001 && y + h <= 1.001;
}
