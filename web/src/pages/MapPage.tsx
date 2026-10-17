// Map: the house floor plan (where items are and get lost, with animated moves between
// rooms) or the household's real places on a map. deck.gl; loaded as its own chunk.
import { OrthographicView } from "@deck.gl/core";
import { TripsLayer } from "@deck.gl/geo-layers";
import { PathLayer, PolygonLayer, ScatterplotLayer, TextLayer } from "@deck.gl/layers";
import DeckGL from "@deck.gl/react";
import { useEffect, useMemo, useRef, useState } from "react";
import { DataTable } from "../components/Charts";
import { EmptyState } from "../components/bits";
import { useApi } from "../lib/api";
import { useLoadStatus } from "../lib/app";
import { capitalize, prefersReducedMotion, timeAgo } from "../lib/format";
import { basemap, fitView, GEO_TOOLTIP_STYLE, MAP_COLORS } from "../lib/mapLayers";
import { useResource } from "../lib/resource";
import type { HouseMap, MapItem, PlacesResponse, Room } from "../lib/types";
import { useSize } from "../lib/useSize";

type Mode = "house" | "places";
let lastMode: Mode = "house"; // survives switching pages
const LOOP_SECONDS = 10; // one pass through the whole window
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

  const [playing, setPlaying] = useState(() => !prefersReducedMotion());
  const [clock, setClock] = useState("");
  const hasTrips = mode === "house" && (house.data?.trips.length ?? 0) > 0;

  return (
    <section className="page">
      <div className="page-head">
        <div>
          <p className="eyebrow">Home map · last {house.data?.window_hours ?? 72} hours</p>
          <h1 className="display">Where things end up.</h1>
        </div>
        <div className="map-controls">
          <div className="segmented" role="group" aria-label="Map view">
            <button type="button" aria-pressed={mode === "house"} onClick={() => setMode("house")}>
              House
            </button>
            <button type="button" aria-pressed={mode === "places"} onClick={() => setMode("places")}>
              Our places
            </button>
          </div>
          {hasTrips && (
            <button type="button" className="pill" aria-pressed={playing} onClick={() => setPlaying((p) => !p)}>
              {playing ? "Pause" : "Play"}
            </button>
          )}
          {mode === "house" && (
            <span className="mono map-clock" aria-live="off">
              {hasTrips ? clock : ""}
            </span>
          )}
        </div>
      </div>

      {mode === "house" ? (
        <div className="card map-card">
          <HouseMapView data={house.data} playing={playing} onClock={setClock} onRetry={house.reload} />
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
            <span>
              <i className="trail-swatch" />
              Moved between rooms
            </span>
          </div>
          <details>
            <summary>Show as table</summary>
            <DataTable
              headers={["Room", "Put down here", "Questions about items last here", "Items there now"]}
              rows={(house.data?.rooms ?? []).map((r) => [r.name, r.putdowns, r.lost, r.items.join(", ") || "–"])}
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

function HouseMapView({ data, playing, onClock, onRetry }: { data?: HouseMap; playing: boolean; onClock(text: string): void; onRetry(): void }) {
  const ref = useRef<HTMLDivElement>(null);
  const size = useSize(ref);
  const [time, setTime] = useState(0);

  // Animate the trails: one pass through the whole window every LOOP_SECONDS.
  useEffect(() => {
    if (!playing || !data?.trips.length) return;
    const span = Math.max(data.span_hours, 1);
    let last = performance.now();
    let frame = requestAnimationFrame(function tick(now) {
      setTime((t) => (t + ((now - last) / 1000) * (span / LOOP_SECONDS)) % (span + 2));
      last = now;
      frame = requestAnimationFrame(tick);
    });
    return () => cancelAnimationFrame(frame);
  }, [playing, data]);

  useEffect(() => {
    if (!data?.trips.length || !data.start) return onClock("");
    const at = new Date(Date.parse(data.start) + Math.min(time, data.span_hours) * 3.6e6);
    onClock(at.toLocaleString(undefined, { weekday: "short", hour: "numeric", minute: "2-digit" }));
  }, [time, data, onClock]);

  // Fit the floor plan (plus a margin) to the frame, once per data set.
  const initialViewState = useMemo(() => {
    if (!data?.rooms.length || !size) return null;
    const xs = data.rooms.flatMap((r) => r.polygon.map((pt) => pt[0]));
    const ys = data.rooms.flatMap((r) => r.polygon.map((pt) => pt[1]));
    const [minX, maxX, minY, maxY] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const fit = Math.min(size.width / (maxX - minX + 10), size.height / (maxY - minY + 10));
    return { target: [(minX + maxX) / 2, (minY + maxY) / 2, 0] as [number, number, number], zoom: Math.log2(fit), minZoom: 1, maxZoom: 6 };
  }, [data, size !== null]); // fit once per data set, not on every resize

  const layers = useMemo(() => (data ? houseLayers(data, playing, time) : []), [data, playing, time]);

  return (
    <div ref={ref} id="home-map" className={data ? "home-map" : "home-map is-loading"} role="img" aria-label="Floor plan showing where items were last seen">
      {data && !data.items.length ? (
        <EmptyState
          title="Nothing to map yet"
          text="Items appear in their rooms once the wearable logs where they were put down."
          checklist="Log a few items in different rooms"
          cta={{ label: "Check again", onClick: onRetry }}
        />
      ) : (
        initialViewState && (
          <DeckGL views={HOUSE_VIEW} initialViewState={initialViewState} controller layers={layers} getTooltip={houseTooltip} />
        )
      )}
    </div>
  );
}

function houseLayers(data: HouseMap, playing: boolean, time: number) {
  const maxLost = Math.max(1, ...data.rooms.map((r) => r.lost));
  const maxAsked = Math.max(1, ...data.items.map((i) => i.times_asked));
  return [
    new PolygonLayer<Room>({
      id: "rooms",
      data: data.rooms,
      getPolygon: (r) => r.polygon,
      getFillColor: (r) => (r.lost ? [...MAP_COLORS.asked, 40 + Math.round((r.lost / maxLost) * 150)] : [...MAP_COLORS.roomFill, 255]),
      getLineColor: MAP_COLORS.roomLine,
      lineWidthUnits: "pixels",
      getLineWidth: 2,
      pickable: true,
      updateTriggers: { getFillColor: maxLost },
    }),
    new TextLayer<Room>({
      id: "room-names",
      data: data.rooms,
      getPosition: (r) => [r.polygon[0][0] + 2, r.polygon[0][1] + 3],
      getText: (r) => r.name.toUpperCase(),
      getColor: MAP_COLORS.label,
      getSize: 11,
      getTextAnchor: "start",
      getAlignmentBaseline: "top",
      fontFamily: "Geist Mono, monospace",
      characterSet: "auto",
    }),
    playing
      ? new TripsLayer({
          id: "trails",
          data: data.trips,
          getPath: (t) => t.path,
          getTimestamps: (t) => t.timestamps,
          getColor: MAP_COLORS.trail,
          widthUnits: "pixels",
          getWidth: 3,
          capRounded: true,
          jointRounded: true,
          trailLength: Math.max(data.span_hours / 4, 2),
          currentTime: time,
          fadeTrail: true,
        })
      : new PathLayer({
          id: "trails",
          data: data.trips,
          getPath: (t) => t.path,
          getColor: [...MAP_COLORS.trail, 150],
          widthUnits: "pixels",
          getWidth: 2,
          capRounded: true,
          jointRounded: true,
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

function houseTooltip({ object, layer }: { object?: unknown; layer?: { id: string } | null }) {
  if (!object || !layer) return null;
  if (layer.id === "items") {
    const i = object as MapItem;
    return {
      text: `${capitalize(i.name)} — ${i.room_name}\n“${i.location || "no place said"}”, ${timeAgo(i.last_seen)}\nPut down ${i.mentions}×, asked about ${i.times_asked}×`,
      style: GEO_TOOLTIP_STYLE,
    };
  }
  if (layer.id === "rooms") {
    const r = object as Room;
    return { text: `${r.name}\n${r.putdowns} put down here · ${r.lost} questions\nNow here: ${r.items.length ? r.items.join(", ") : "nothing now"}`, style: GEO_TOOLTIP_STYLE };
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
