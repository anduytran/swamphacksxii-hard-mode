// How far away an item was in a Teach photo, judged from the box drawn around it.
//
// The wearable sees things from across a room, not only up close. Measured on the Pi (Oct 17, 2026): a pin
// model taught with 10 close-up photos (boxes 0.27-0.54 of the photo) found the pin every time when it filled
// about a quarter of the frame, but missed it or scored too low to count (0.17-0.27, under the server's 0.35)
// when it was about an eighth. So the Teach page asks for a mix of close, middle and far photos.

export type Distance = "close" | "middle" | "far";
type Box = readonly [number, number, number, number]; // x, y, w, h in 0..1

/** Box size = its larger side as a share of the photo. At or above CLOSE it's close; under FAR it's far. */
export const CLOSE = 0.3;
export const FAR = 0.15;
/** Photos wanted in each range: 3 x 3 fits inside the 10 photos that make an item "ready". */
export const PER_RANGE = 3;

export const DISTANCES: { key: Distance; label: string; how: string }[] = [
  { key: "close", label: "Close up", how: "fills a big part of the photo" },
  { key: "middle", label: "Arm's length", how: "about a step back" },
  { key: "far", label: "Across the room", how: "small in the photo, the way the wearable often sees it" },
];

export function distanceOf(box: Box): Distance {
  const size = Math.max(box[2], box[3]);
  return size >= CLOSE ? "close" : size >= FAR ? "middle" : "far";
}

export function distanceMix(boxes: readonly Box[]): Record<Distance, number> {
  const mix: Record<Distance, number> = { close: 0, middle: 0, far: 0 };
  for (const b of boxes) mix[distanceOf(b)] += 1;
  return mix;
}

/** The range to ask for next: the one furthest below PER_RANGE (far first on a tie), or null when all are met. */
export function nextDistance(mix: Record<Distance, number>): Distance | null {
  const order: Distance[] = ["far", "middle", "close"];
  const short = order.filter((d) => mix[d] < PER_RANGE);
  if (!short.length) return null;
  return short.reduce((a, b) => (mix[b] < mix[a] ? b : a));
}
