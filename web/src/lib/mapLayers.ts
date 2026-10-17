// deck.gl pieces shared by the map pages. Only imported by the lazily loaded map chunks.
import { WebMercatorViewport } from "@deck.gl/core";
import { MVTLayer, TileLayer } from "@deck.gl/geo-layers";
import { BitmapLayer } from "@deck.gl/layers";

export type RGBA = [number, number, number, number?];

// Maps are night maps in both themes: deep indigo ground, pale labels. Keep in step with the
// --map-* tokens in styles.css (the legends under each map use them).
export const MAP_COLORS = {
  roomLine: [70, 68, 120],
  roomFill: [30, 29, 61], // dark --surface #1E1D3D
  hallFill: [24, 23, 51], // corridors: between --map-ground and roomFill
  asked: [232, 104, 126], // --map-asked, rose: needs attention
  calm: [169, 165, 240], // --map-calm, periwinkle
  publicPlace: [245, 178, 60], // --map-public, the amber accent
  trail: [185, 182, 240], // --map-trail
  label: [180, 177, 209],
  text: [244, 245, 249],
  ground: [20, 19, 43], // --map-ground #14132B
} satisfies Record<string, [number, number, number]>;

export const GEO_TOOLTIP_STYLE = {
  background: "#FFFFFF",
  color: "#17162E",
  fontSize: "13px",
  fontFamily: "Geist, system-ui, sans-serif",
  borderRadius: "8px",
  padding: "8px 10px",
  lineHeight: "1.45",
};

export const INDIGO_TINT: [number, number, number] = [62, 58, 112]; // basemap tint: desaturated OSM, night indigo

// OpenStreetMap's standard tiles (no key; light use with attribution, per the OSM tile
// policy), desaturated and tinted indigo to sit under the data.
const BASEMAP_URLS = ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"];

export function basemap(tint: [number, number, number] = INDIGO_TINT) {
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
    getFillColor: [70, 66, 118, 240], // indigo-gray, lit from above
    material: { ambient: 0.5, diffuse: 0.55, shininess: 6, specularColor: [40, 38, 70] },
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
