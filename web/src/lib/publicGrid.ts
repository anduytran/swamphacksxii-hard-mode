// The public map's data side: the colour ramp, the zoom-scaled grid, framing, and the
// simulated UF campus data. Pure functions (no deck.gl), so they're unit-tested.
import type { LostCell, LostItem, LostMap } from "./types";

export type RGB = [number, number, number];

// Sequential ramp on the night-indigo map: indigo → plum → coral → gold, from few to many.
// Lightness rises monotonically (OKLab L 0.41 → 0.84, ground 0.21), so it reads as "more";
// the legend under the map (.heat-ramp in styles.css) uses the same stops.
export const HEAT_COLORS: RGB[] = [
  [75, 58, 140], [122, 63, 142], [176, 74, 115], [217, 96, 76], [232, 137, 58], [245, 194, 90],
];

export function rampColor(t: number): RGB {
  const x = Math.min(Math.max(t, 0), 1) * (HEAT_COLORS.length - 1);
  const i = Math.min(Math.floor(x), HEAT_COLORS.length - 2);
  const f = x - i;
  return HEAT_COLORS[i].map((v, k) => Math.round(v + (HEAT_COLORS[i + 1][k] - v) * f)) as RGB;
}

export interface GridCell extends LostCell {
  total: number;
  spots: number; // how many ~100 m public cells this column merges
}

// ~100 m at Gainesville's latitude (the public map's cell size).
export const LAT_STEP = 0.0009;
export const LNG_STEP = 0.00103;

/**
 * Grid level for a zoom: 0 = the ~100 m public cells (zoom 13.5 and closer); each level out
 * doubles the cell size (200 m, 400 m, … 6.4 km), so zooming out merges neighbouring spots
 * into bigger columns instead of shrinking them to specks.
 */
export const gridLevel = (zoom: number) => Math.max(0, Math.min(6, Math.round(13.5 - zoom)));

export function binCells(cells: LostCell[], level: number): GridCell[] {
  if (!level) return cells.map((c) => ({ ...c, total: c.left + c.asked, spots: 1 }));
  const f = 2 ** level;
  const latStep = LAT_STEP * f;
  const lngStep = LNG_STEP * f;
  const bins = new Map<string, Omit<GridCell, "items"> & { items: Map<string, LostItem> }>();
  for (const c of cells) {
    const i = Math.floor(c.lat / latStep);
    const j = Math.floor(c.lng / lngStep);
    const key = `${i},${j}`;
    let bin = bins.get(key);
    if (!bin) {
      bin = { lat: (i + 0.5) * latStep, lng: (j + 0.5) * lngStep, left: 0, asked: 0, total: 0, spots: 0, items: new Map() };
      bins.set(key, bin);
    }
    bin.left += c.left;
    bin.asked += c.asked;
    bin.total += c.left + c.asked;
    bin.spots += 1;
    for (const it of c.items) {
      const item = bin.items.get(it.name) ?? { name: it.name, left: 0, asked: 0 };
      item.left += it.left;
      item.asked += it.asked;
      bin.items.set(it.name, item);
    }
  }
  return [...bins.values()].map((b) => ({ ...b, items: byCount([...b.items.values()]) }));
}

const byCount = (items: LostItem[]) => items.sort((x, y) => y.left + y.asked - (x.left + x.asked));

/**
 * The spots to frame at first: with many spots, the middle 80% by latitude and longitude, so
 * a few far-off ones don't zoom the whole map out until the columns are specks.
 */
export function coreSpots(cells: { lat: number; lng: number }[]): [number, number][] {
  const points = cells.map((c) => [c.lng, c.lat] as [number, number]);
  if (points.length < 10) return points;
  const range = (k: 0 | 1) => {
    const v = points.map((p) => p[k]).sort((a, b) => a - b);
    return [v[Math.floor(v.length * 0.1)], v[Math.ceil(v.length * 0.9) - 1]];
  };
  const [[lng0, lng1], [lat0, lat1]] = [range(0), range(1)];
  return points.filter(([lng, lat]) => lng >= lng0 && lng <= lng1 && lat >= lat0 && lat <= lat1);
}

/** A tilted close-up of one public spot, near enough that the buildings around it read. */
export const spotView = (cell: { lat: number; lng: number }) => ({
  longitude: cell.lng,
  latitude: cell.lat,
  zoom: 15.4,
  pitch: 55,
  bearing: -20,
});

// ---------- Simulated public map (browser only, never saved) ----------

export const UF_CAMPUS = { longitude: -82.3455, latitude: 29.6475 };

// Busy campus spots (approximate positions) and how busy each one is.
const UF_SPOTS: [name: string, lat: number, lng: number, busy: number][] = [
  ["Marston Science Library", 29.6481, -82.3438, 1.0],
  ["Library West", 29.6512, -82.3427, 0.9],
  ["Reitz Union", 29.6463, -82.3478, 1.0],
  ["Turlington Plaza", 29.6493, -82.3437, 0.85],
  ["Ben Hill Griffin Stadium", 29.65, -82.3486, 0.7],
  ["The Hub", 29.648, -82.3455, 0.6],
  ["Broward dining", 29.6457, -82.3416, 0.55],
  ["Shands Hospital", 29.6402, -82.3442, 0.6],
  ["Midtown", 29.6522, -82.3365, 0.8],
  ["Southwest Rec Center", 29.6386, -82.3681, 0.45],
];
const UF_ITEMS = ["umbrella", "water bottle", "keys", "phone charger", "AirPods", "Gator 1 card", "laptop charger",
  "sunglasses", "wallet", "calculator", "jacket", "headphones"];

/**
 * Same shape as /api/v1/public/lost-map, generated with a fixed seed so it looks the same
 * every time: spots scattered around each campus place, snapped to the ~100 m public grid.
 */
export function simulatedLostMap(): LostMap {
  let seed = 20260117;
  const rand = () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const byCell = new Map<string, { lat: number; lng: number; items: Map<string, LostItem> }>();
  for (const [, lat, lng, busy] of UF_SPOTS) {
    for (let i = 0; i < Math.round(14 + busy * 22); i++) {
      const r = Math.pow(rand(), 1.7) * 0.0045; // most within ~250 m, a few ~500 m out
      const a = rand() * Math.PI * 2;
      const cLat = Math.round((lat + r * Math.sin(a)) / LAT_STEP) * LAT_STEP;
      const cLng = Math.round((lng + r * Math.cos(a) * 1.15) / LNG_STEP) * LNG_STEP;
      const count = Math.max(1, Math.round(busy * 26 * Math.exp(-r * 700) * (0.35 + rand())));
      const key = `${cLat.toFixed(4)},${cLng.toFixed(4)}`;
      const cell = byCell.get(key) ?? { lat: +cLat.toFixed(4), lng: +cLng.toFixed(4), items: new Map() };
      for (let n = 0; n < count; n++) {
        const name = UF_ITEMS[Math.floor(Math.pow(rand(), 1.5) * UF_ITEMS.length)]; // a few items dominate
        const item = cell.items.get(name) ?? { name, left: 0, asked: 0 };
        if (rand() < 0.7) item.left += 1;
        else item.asked += 1;
        cell.items.set(name, item);
      }
      byCell.set(key, cell);
    }
  }
  const cells = [...byCell.values()].map((c) => {
    const items = byCount([...c.items.values()]);
    return { lat: c.lat, lng: c.lng, items, left: items.reduce((s, i) => s + i.left, 0), asked: items.reduce((s, i) => s + i.asked, 0) };
  });
  cells.sort((a, b) => b.left + b.asked - (a.left + a.asked));
  return { min_households: 3, simulated: true, cells };
}
