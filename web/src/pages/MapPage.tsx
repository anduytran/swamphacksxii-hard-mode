// Map: the indoor floor plan, a home or the Reitz Union one floor at a time (where items are
// and get lost), or the household's real places on a map. deck.gl; loaded as its own chunk.
import { OrthographicView } from "@deck.gl/core";
import { PolygonLayer, ScatterplotLayer, TextLayer } from "@deck.gl/layers";
import DeckGL from "@deck.gl/react";
import { useEffect, useMemo, useRef, useState } from "react";
import { DataTable } from "../components/Charts";
import { EmptyState } from "../components/bits";
import { useApi } from "../lib/api";
import { useLoadStatus } from "../lib/app";
import { capitalize, timeAgo } from "../lib/format";
import { basemap, fitView, GEO_TOOLTIP_STYLE, MAP_COLORS } from "../lib/mapLayers";
import { useResource } from "../lib/resource";
import type { HouseMap, MapItem, PlacesResponse, Room } from "../lib/types";
import { useSize } from "../lib/useSize";

type Mode = "house" | "places";
let lastMode: Mode = "house"; // survives switching pages
const HOUSE_VIEW = new OrthographicView({ flipY: true }); // floor-plan units, y down

export default function MapPage() {
  const [mode, setModeState] = useState<Mode>(lastMode);
  const setMode = (m: Mode) => {
    lastMode = m;
    setModeState(m);
  };
  const api = useApi();
  const house = useResource(mode === "house" ? "map" : null, () => api<HouseMap>("/api/v1/dashboard/map"));
  const places = useResource(mode === "places" ? "places" : null, () => api<PlacesResponse>("/api/v1/dashboard/places"));
  const active = mode === "house" ? house : places;
  useLoadStatus(active.loading, active.error, active.reload);

  return (
    <section className="page">
      <div className="page-head">
        <div>
          <p className="eyebrow">{house.data?.plan?.name ?? "Floor"} map · last {house.data?.window_hours ?? 72} hours</p>
          <h1 className="display">Where things end up.</h1>
        </div>
        <div className="map-controls">
          <div className="segmented" role="group" aria-label="Map view">
            <button type="button" aria-pressed={mode === "house"} onClick={() => setMode("house")}>
              Floor plan
            </button>
            <button type="button" aria-pressed={mode === "places"} onClick={() => setMode("places")}>
              Our places
            </button>
          </div>
        </div>
      </div>

      {mode === "house" ? (
        <div className="card map-card">
          <HouseMapView data={house.data} onRetry={house.reload} />
          <div className="legend map-legend">
            <span>
              <i className="swatch room-swatch" />
              Room shade: questions about items last seen there
            </span>
            <span>
              <i className="dot dot-asked" />
              Asked about
            </span>
            <span>
              <i className="dot dot-calm" />
              Not asked about
            </span>
            <span>📷 Room from the photo, not the words</span>
          </div>
          <details>
            <summary>Show as table</summary>
            <DataTable
              headers={["Room", "Put down here", "Questions about items last here", "Items there now"]}
              rows={(house.data?.rooms ?? []).map((r) => [r.name, r.putdowns, r.lost, roomItems(r, house.data?.items ?? []) || "–"])}
            />
          </details>
        </div>
      ) : (
        <div className="card map-card">
          <PlacesMapView data={places.data} onRetry={places.reload} />
          <p className="attribution">
            © <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors
          </p>
          <div className="legend map-legend">
            <span>
              <i className="dot dot-calm" />
              Private place (only your household sees it)
            </span>
            <span>
              <i className="dot dot-public" />
              Public place
            </span>
            <span>
              <i className="dot dot-memory" />A memory
            </span>
          </div>
          <p className="hint places-hint">
            Places are found automatically from the wearable's Wi-Fi position. Name them and choose which are public on the{" "}
            <a href="#/household">Household</a> page.
          </p>
        </div>
      )}
      <p className="hint">Rooms are matched from the spoken location and the camera's description. Drag to pan, scroll to zoom.</p>
    </section>
  );
}

// ---------- House floor plan ----------

function HouseMapView({ data, onRetry }: { data?: HouseMap; onRetry(): void }) {
  const ref = useRef<HTMLDivElement>(null);
  const size = useSize(ref);
  const [zoom, setZoom] = useState(0); // for which room names fit

  // Floors: one frame each, stepped through with the buttons. "Elsewhere" shows on every floor.
  const floors = useMemo(() => {
    const halls = (data?.rooms ?? []).filter((r) => r.kind === "hall");
    return [...new Set((data?.rooms ?? []).map((r) => r.level ?? 0))]
      .sort((a, b) => a - b)
      .map((level) => ({ level, name: halls.find((h) => h.level === level)?.name ?? `Floor ${level}` }));
  }, [data]);
  const levelOf = useMemo(() => new Map((data?.rooms ?? []).map((r) => [r.id, r.level ?? 0])), [data]);
  // Open on the floor where something was put down most recently.
  const [floor, setFloor] = useState<number | null>(null);
  useEffect(() => {
    if (!data) return;
    const latest = [...data.items].filter((i) => i.room !== "unknown").sort((a, b) => b.last_seen.localeCompare(a.last_seen))[0];
    setFloor(latest ? (levelOf.get(latest.room) ?? 0) : (floors[0]?.level ?? 0));
  }, [data]); // eslint-disable-line react-hooks/exhaustive-deps -- once per data set
  const index = Math.max(0, floors.findIndex((f) => f.level === floor));
  const current = floors[index];
  const onFloor = (roomId: string) => roomId === "unknown" || levelOf.get(roomId) === current?.level;
  const view = useMemo<HouseMap | undefined>(
    () => data && current && { ...data, rooms: data.rooms.filter((r) => onFloor(r.id)), items: data.items.filter((i) => onFloor(i.room)) },
    [data, current], // eslint-disable-line react-hooks/exhaustive-deps
  );
  const countOn = (level: number) => data?.items.filter((i) => i.room !== "unknown" && levelOf.get(i.room) === level).length ?? 0;

  // Fit this floor (plus a margin for item names at the edges) to the frame.
  const initialViewState = useMemo(() => {
    if (!view?.rooms.length || !size) return null;
    const xs = view.rooms.flatMap((r) => r.polygon.map((pt) => pt[0]));
    const ys = view.rooms.flatMap((r) => r.polygon.map((pt) => pt[1]));
    const [minX, maxX, minY, maxY] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const fit = Math.min(size.width / ((maxX - minX) * 1.08 + 10), size.height / ((maxY - minY) * 1.04 + 10));
    return { target: [(minX + maxX) / 2, (minY + maxY) / 2, 0] as [number, number, number], zoom: Math.log2(fit), minZoom: Math.log2(fit) - 1, maxZoom: 6 };
  }, [view, size !== null]); // re-fit per floor, not on every resize
  useEffect(() => {
    if (initialViewState) setZoom(initialViewState.zoom);
  }, [initialViewState]);

  const layers = useMemo(() => (view ? houseLayers(view, zoom) : []), [view, zoom]);
  const step = (by: number) => setFloor(floors[Math.min(floors.length - 1, Math.max(0, index + by))].level);

  return (
    <>
      {floors.length > 1 && current && (
        <div className="floor-stepper" role="group" aria-label="Floor">
          <button type="button" className="icon-btn" onClick={() => step(-1)} disabled={index === 0} aria-label={`Down to ${floors[index - 1]?.name ?? "lower floor"}`}>
            <span aria-hidden="true">↓</span>
          </button>
          <p aria-live="polite">
            <strong>{current.name}</strong>
            <span className="floor-count">
              {index + 1} of {floors.length} · {countOn(current.level)} {countOn(current.level) === 1 ? "item" : "items"}
            </span>
          </p>
          <button
            type="button"
            className="icon-btn"
            onClick={() => step(1)}
            disabled={index === floors.length - 1}
            aria-label={`Up to ${floors[index + 1]?.name ?? "higher floor"}`}
          >
            <span aria-hidden="true">↑</span>
          </button>
        </div>
      )}
      <div ref={ref} id="home-map" className={data ? "home-map" : "home-map is-loading"} role="img" aria-label={`Floor plan${current && floors.length > 1 ? ` of ${current.name}` : ""} showing where items were last seen`}>
        {data && !data.items.length ? (
          <EmptyState
            title="Nothing to map yet"
            text="Items appear in their rooms once the wearable logs where they were put down."
            checklist="Log a few items in different rooms"
            cta={{ label: "Check again", onClick: onRetry }}
          />
        ) : (
          initialViewState && (
            <DeckGL
              views={HOUSE_VIEW}
              initialViewState={initialViewState}
              controller
              layers={layers}
              getTooltip={(info) => houseTooltip(data?.items ?? [], info)}
              onViewStateChange={({ viewState }) => {
                const z = Math.round((viewState as { zoom: number }).zoom * 4) / 4; // re-label in quarter steps
                if (z !== zoom) setZoom(z);
                return viewState;
              }}
            />
          )
        )}
      </div>
    </>
  );
}

// Room names wrap at ~12 characters and only show where they fit at the current zoom (the
// rest are in each room's tooltip).
const LABEL_PX = { char: 6.8, line: 13 };

function wrapName(name: string) {
  const lines: string[] = [];
  for (const word of name.toUpperCase().split(" ")) {
    const last = lines.length - 1;
    if (last >= 0 && lines[last].length + word.length < 12) lines[last] += ` ${word}`;
    else lines.push(word);
  }
  return lines;
}

function bounds(r: Room) {
  const xs = r.polygon.map((p) => p[0]);
  const ys = r.polygon.map((p) => p[1]);
  return { x: Math.min(...xs), y: Math.min(...ys), w: Math.max(...xs) - Math.min(...xs), h: Math.max(...ys) - Math.min(...ys) };
}

function houseLayers(data: HouseMap, zoom: number) {
  const scale = 2 ** zoom; // pixels per floor-plan unit
  const named = data.rooms
    .filter((r) => r.kind !== "hall")
    .map((r) => ({ room: r, box: bounds(r), lines: wrapName(r.name) }))
    .filter(({ box, lines }) => {
      const width = Math.max(...lines.map((l) => l.length)) * LABEL_PX.char + 8;
      return width <= box.w * scale && lines.length * LABEL_PX.line + 6 <= box.h * scale;
    });
  const maxLost = Math.max(1, ...data.rooms.map((r) => r.lost));
  const maxAsked = Math.max(1, ...data.items.map((i) => i.times_asked));
  return [
    new PolygonLayer<Room>({
      id: "rooms",
      data: data.rooms,
      getPolygon: (r) => r.polygon,
      getFillColor: (r) =>
        r.lost ? [...MAP_COLORS.asked, 60 + Math.round((r.lost / maxLost) * 150)] : [...(r.kind === "hall" ? MAP_COLORS.hallFill : MAP_COLORS.roomFill), 235],
      getLineColor: MAP_COLORS.roomLine,
      lineWidthUnits: "pixels",
      getLineWidth: 1.5,
      pickable: true,
      updateTriggers: { getFillColor: maxLost },
    }),
    new TextLayer<(typeof named)[number]>({
      id: "room-names",
      data: named,
      getPosition: ({ box }) => [box.x, box.y],
      getPixelOffset: [6, 6],
      getText: ({ lines }) => lines.join("\n"),
      getColor: MAP_COLORS.label,
      getSize: 11,
      lineHeight: LABEL_PX.line / 11,
      getTextAnchor: "start",
      getAlignmentBaseline: "top",
      fontFamily: "Geist Mono, monospace",
      characterSet: "auto",
    }),
    new ScatterplotLayer<MapItem>({
      id: "items",
      data: data.items,
      getPosition: (i) => i.position,
      getRadius: (i) => 6 + (i.times_asked / maxAsked) * 10,
      radiusUnits: "pixels",
      getFillColor: (i) => (i.times_asked ? MAP_COLORS.asked : MAP_COLORS.calm),
      getLineColor: MAP_COLORS.ground,
      stroked: true,
      lineWidthUnits: "pixels",
      getLineWidth: 2,
      pickable: true,
    }),
    new TextLayer<MapItem>({
      id: "item-names",
      data: data.items,
      getPosition: (i) => i.position,
      getText: (i) => i.name,
      getColor: MAP_COLORS.text,
      getSize: 13,
      getPixelOffset: [0, 20],
      fontFamily: "Geist, system-ui, sans-serif",
      characterSet: "auto",
      outlineWidth: 3,
      outlineColor: [...MAP_COLORS.ground, 255],
      fontSettings: { sdf: true },
    }),
  ];
}

// A room's items, a camera on those placed by what the photo showed rather than the words.
function roomItems(room: Room, items: MapItem[]): string {
  const byPhoto = new Set(items.filter((i) => i.room_source === "photo").map((i) => i.name));
  return room.items.map((name) => (byPhoto.has(name) ? `📷 ${name}` : name)).join(", ");
}

function houseTooltip(items: MapItem[], { object, layer }: { object?: unknown; layer?: { id: string } | null }) {
  if (!object || !layer) return null;
  if (layer.id === "items") {
    const i = object as MapItem;
    return {
      text: `${capitalize(i.name)} — ${i.room_name}${i.room_source === "photo" ? " 📷 (room from the photo)" : ""}\n“${i.location || "no place said"}”, ${timeAgo(i.last_seen)}\nPut down ${i.mentions}×, asked about ${i.times_asked}×`,
      style: GEO_TOOLTIP_STYLE,
    };
  }
  if (layer.id === "rooms") {
    const r = object as Room;
    return { text: `${r.name}\n${r.putdowns} put down here · ${r.lost} questions\nNow here: ${roomItems(r, items) || "nothing now"}`, style: GEO_TOOLTIP_STYLE };
  }
  return null;
}

// ---------- The household's places on a real map ----------

type PlacePoint = PlacesResponse["households"][number]["places"][number] & { household: string };
type MemoryPoint = PlacesResponse["memories"][number];

function PlacesMapView({ data, onRetry }: { data?: PlacesResponse; onRetry(): void }) {
  const ref = useRef<HTMLDivElement>(null);
  const size = useSize(ref);
  const places = useMemo<PlacePoint[]>(() => data?.households.flatMap((h) => h.places.map((p) => ({ ...p, household: h.name }))) ?? [], [data]);
  const initialViewState = useMemo(
    () => (size && places.length ? fitView(places.map((p) => [p.lng, p.lat]), size.width, size.height) : null),
    [places, size !== null],
  );
  const layers = useMemo(
    () => [
      basemap(),
      new ScatterplotLayer<PlacePoint>({
        id: "place-areas",
        data: places,
        getPosition: (p) => [p.lng, p.lat],
        getRadius: (p) => p.radius_m,
        radiusUnits: "meters",
        radiusMinPixels: 10,
        getFillColor: (p) => (p.kind === "public" ? [...MAP_COLORS.publicPlace, 90] : [...MAP_COLORS.calm, 70]),
        getLineColor: (p) => (p.kind === "public" ? MAP_COLORS.publicPlace : MAP_COLORS.calm),
        stroked: true,
        lineWidthUnits: "pixels",
        getLineWidth: 2,
        pickable: true,
      }),
      new ScatterplotLayer<MemoryPoint>({
        id: "memories",
        data: data?.memories ?? [],
        getPosition: (m) => [m.lng, m.lat],
        getRadius: 4,
        radiusUnits: "pixels",
        getFillColor: [...MAP_COLORS.text, 220],
        pickable: true,
      }),
      new TextLayer<PlacePoint>({
        id: "place-labels",
        data: places,
        getPosition: (p) => [p.lng, p.lat],
        getText: (p) => p.label,
        getColor: MAP_COLORS.text,
        getSize: 14,
        getPixelOffset: [0, -22],
        fontFamily: "Geist, system-ui, sans-serif",
        characterSet: "auto",
        outlineWidth: 3,
        outlineColor: [...MAP_COLORS.ground, 255],
        fontSettings: { sdf: true },
      }),
    ],
    [places, data],
  );

  return (
    <div ref={ref} id="places-map" className={data ? "home-map geo-map" : "home-map geo-map is-loading"} role="img" aria-label="Map of the household's places">
      {data && !places.length ? (
        <EmptyState
          title="No places yet"
          text="Places appear once the wearable sends Wi-Fi positions with its memories."
          checklist="Log a memory with Wi-Fi turned on"
          cta={{ label: "Check again", onClick: onRetry }}
        />
      ) : (
        initialViewState && (
          <DeckGL
            initialViewState={initialViewState}
            controller
            layers={layers}
            getTooltip={({ object, layer }) => {
              if (!object || !layer) return null;
              if (layer.id === "place-areas") {
                const p = object as PlacePoint;
                return { text: `${p.label} · ${p.kind}\n${p.visits} memories here · ${p.household}`, style: GEO_TOOLTIP_STYLE };
              }
              if (layer.id === "memories") {
                const m = object as MemoryPoint;
                return { text: `“${m.transcript}”\n${m.items || "no item"} · ${timeAgo(m.timestamp)}`, style: GEO_TOOLTIP_STYLE };
              }
              return null;
            }}
          />
        )
      )}
    </div>
  );
}
