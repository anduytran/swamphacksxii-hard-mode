// deck.gl pieces shared by the map pages. Only imported by the lazily loaded map chunks.
import { WebMercatorViewport } from "@deck.gl/core";
import { MVTLayer, TileLayer } from "@deck.gl/geo-layers";
import { BitmapLayer } from "@deck.gl/layers";

export type RGBA = [number, number, number, number?];

export const MAP_COLORS = {
  roomLine: [62, 58, 52],
  roomFill: [23, 22, 20], // --surface
  lost: [201, 133, 0], // --series-2
  asked: [244, 167, 59], // --amber
  calm: [57, 135, 229], // --series-1
  label: [168, 161, 150],
  text: [243, 239, 232],
  ground: [14, 13, 12],
} satisfies Record<string, [number, number, number]>;

export const GEO_TOOLTIP_STYLE = {
  background: "#f3efe8",
  color: "#0e0d0c",
  fontSize: "13px",
  fontFamily: "Geist, system-ui, sans-serif",
  borderRadius: "8px",
  padding: "8px 10px",
  lineHeight: "1.45",
};

export const NAVY_TINT: [number, number, number] = [46, 56, 72]; // the public map's basemap

// OpenStreetMap's standard tiles (no key; light use with attribution, per the OSM tile
// policy), desaturated and tinted down to sit on the dark theme.
const BASEMAP_URLS = ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"];

export function basemap(tint: [number, number, number] = [70, 66, 60]) {
  return new TileLayer({
    id: "basemap",
    data: BASEMAP_URLS,
    minZoom: 0,
    maxZoom: 19,
    tileSize: 256,
    renderSubLayers: (props) => {
      const [[west, south], [east, north]] = props.tile.boundingBox;
      return new BitmapLayer({
        id: props.id,
        image: props.data as ImageBitmap,
        bounds: [west, south, east, north],
        desaturate: 1,
        tintColor: tint,
      });
    },
  });
}

// 3D buildings: OpenFreeMap vector tiles (OpenStreetMap data, free, no key). The tile URL
// carries a dated version, so it's read from the TileJSON once; no buildings if that fails.
const BUILDINGS_TILEJSON = "https://tiles.openfreemap.org/planet";
let buildingTiles: Promise<string[] | null> | null = null;

export function buildingTileUrls(): Promise<string[] | null> {
  buildingTiles ??= fetch(BUILDINGS_TILEJSON)
    .then((r) => (r.ok ? r.json() : null))
    .then((tilejson: { tiles?: string[] } | null) => tilejson?.tiles ?? null)
    .catch(() => null);
  return buildingTiles;
}

interface BuildingProps {
  hide_3d?: boolean;
  render_height?: number;
}

export function buildingsLayer(tiles: string[]) {
  return new MVTLayer({
    id: "buildings",
    data: tiles,
    // Heights (render_height) only come with z14 tiles: always fetch z14, one level finer
    // than the view, so buildings stand up from zoom 13 inwards.
    minZoom: 14,
    maxZoom: 14,
    zoomOffset: 1,
    loadOptions: { mvt: { layers: ["building"] } },
    extruded: true,
    stroked: false,
    getElevation: (f: { properties: BuildingProps }) => (f.properties.hide_3d ? 0 : f.properties.render_height || 6),
    getFillColor: [74, 68, 60, 240],
    material: { ambient: 0.5, diffuse: 0.55, shininess: 6, specularColor: [30, 28, 24] },
  });
}

export interface GeoView {
  longitude: number;
  latitude: number;
  zoom: number;
  pitch?: number;
  bearing?: number;
}

/** A view that fits the points into a width × height frame. */
export function fitView(points: [number, number][], width: number, height: number, extra: Partial<GeoView> = {}): GeoView {
  const w = width || 800;
  const h = height || 480;
  if (!points.length) return { longitude: -98, latitude: 38, zoom: 3, ...extra };
  if (points.length === 1) return { longitude: points[0][0], latitude: points[0][1], zoom: 15, ...extra };
  const lngs = points.map((p) => p[0]);
  const lats = points.map((p) => p[1]);
  const view = new WebMercatorViewport({ width: w, height: h }).fitBounds(
    [
      [Math.min(...lngs), Math.min(...lats)],
      [Math.max(...lngs), Math.max(...lats)],
    ],
    { padding: Math.min(80, w / 6) },
  );
  return { longitude: view.longitude, latitude: view.latitude, zoom: Math.min(view.zoom, 16.5), ...extra };
}
