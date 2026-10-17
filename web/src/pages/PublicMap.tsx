// Public map: anonymous, aggregated, consented. Works without logging in. A 3D grid that
// re-bins as you zoom, over 3D buildings; Live data or Simulated (generated in the browser).
// deck.gl; loaded as its own chunk.
import { FlyToInterpolator } from "@deck.gl/core";
import { ColumnLayer } from "@deck.gl/layers";
import DeckGL from "@deck.gl/react";
import { useEffect, useMemo, useRef, useState } from "react";
import { EmptyState, repeat, Sk } from "../components/bits";
import { Icon } from "../components/Icons";
import { MapsNote, MapsPlace } from "../components/MapsPlace";
import { publicGet } from "../lib/api";
import { useLoadStatus } from "../lib/app";
import { capitalize, prefersReducedMotion } from "../lib/format";
import { basemap, buildingsLayer, buildingTileUrls, fitView, GEO_TOOLTIP_STYLE, type GeoView } from "../lib/mapLayers";
import { binCells, coreSpots, gridLevel, minZoomFor, rampColor, simulatedLostMap, spotView, UF_CAMPUS, type GridCell } from "../lib/publicGrid";
import { useResource } from "../lib/resource";
import type { LostCell, LostMap } from "../lib/types";
import { useSize } from "../lib/useSize";

type Mode = "live" | "sim";
let lastMode: Mode = "live"; // survives switching pages

type ViewState = GeoView & { minZoom?: number; maxZoom?: number; transitionDuration?: number; transitionInterpolator?: FlyToInterpolator };

export default function PublicMapPage() {
  const [mode, setModeState] = useState<Mode>(lastMode);
  const setMode = (m: Mode) => {
    lastMode = m;
    setModeState(m);
  };
  const live = useResource(mode === "live" ? "public:live" : null, () => publicGet<LostMap>("/api/v1/public/lost-map"));
  const sim = useMemo(() => (mode === "sim" ? simulatedLostMap() : undefined), [mode]);
  const data = mode === "sim" ? sim : live.data;
  useLoadStatus(mode === "live" && live.loading, mode === "live" ? live.error : undefined, live.reload);

  const ref = useRef<HTMLDivElement>(null);
  const size = useSize(ref);
  const [tiles, setTiles] = useState<string[] | null>(null);
  useEffect(() => {
    void buildingTileUrls().then(setTiles);
  }, []);

  // A tilted overview of the spots (fit flat, then step back: a tilted camera shows less than
  // fitBounds assumes); with none yet, the UF campus area. Buildings appear from zoom 13.
  const overview = (cells: LostCell[]): ViewState => {
    const zoomed = (v: ViewState): ViewState => ({ ...v, minZoom: minZoomFor(v.zoom), maxZoom: 18 });
    if (!cells.length) return zoomed({ ...UF_CAMPUS, zoom: 13.6, pitch: 52, bearing: -18 });
    const view = fitView(coreSpots(cells), size?.width ?? 800, size?.height ?? 480, { pitch: 52, bearing: -18 });
    return zoomed({ ...view, zoom: Math.min(view.zoom, 14.2) - 0.7 });
  };

  // The camera: set once per data set (mode), then by the list's fly-to buttons.
  const [view, setView] = useState<ViewState | null>(null);
  const [level, setLevel] = useState(0);
  const framed = useRef<Mode | null>(null);
  useEffect(() => {
    if (!data || !size || framed.current === mode) return;
    const next = overview(data.cells);
    setView(framed.current ? { ...next, transitionDuration: 700, transitionInterpolator: new FlyToInterpolator() } : next);
    setLevel(gridLevel(next.zoom));
    framed.current = mode;
  }, [data, size, mode]);

  const flyTo = (cell: LostCell) => {
    setView({ ...spotView(cell), minZoom: view?.minZoom, maxZoom: view?.maxZoom, transitionDuration: 900, transitionInterpolator: new FlyToInterpolator() });
    ref.current?.scrollIntoView({ behavior: prefersReducedMotion() ? "auto" : "smooth", block: "center" });
  };

  const layers = useMemo(() => {
    const f = 2 ** level;
    const binned = binCells(data?.cells ?? [], level);
    const maxTotal = Math.max(1, ...binned.map((c) => c.total));
    return [
      basemap(),
      tiles && buildingsLayer(tiles),
      // 3D grid: one square column per cell (78% of the cell width, so neighbours keep a gap);
      // height and colour both encode how often things were lost or asked about there.
      new ColumnLayer<GridCell>({
        id: "public-cells",
        data: binned,
        diskResolution: 4,
        angle: 45,
        radius: 55 * f,
        extruded: true,
        getPosition: (c) => [c.lng, c.lat],
        getElevation: (c) => (20 + (c.total / maxTotal) * 380) * f, // ~3x the tallest campus buildings at most
        getFillColor: (c) => [...rampColor(c.total / maxTotal), 245],
        material: { ambient: 0.55, diffuse: 0.65, shininess: 24, specularColor: [60, 50, 40] },
        pickable: true,
        autoHighlight: true,
        highlightColor: [255, 255, 255, 70],
        transitions: { getElevation: { duration: 350 } },
      }),
    ];
  }, [data, level, tiles]);

  const cells = data?.cells;
  return (
    <section className="page">
      <div className="page-head">
        <div>
          <p className="eyebrow">Public · anonymous · opt-in</p>
          <h1 className="display">Where people lose things.</h1>
        </div>
        <div className="segmented" role="group" aria-label="Map data">
          <button type="button" aria-pressed={mode === "live"} onClick={() => setMode("live")}>
            Live data
          </button>
          <button type="button" aria-pressed={mode === "sim"} onClick={() => setMode("sim")}>
            Simulated
          </button>
        </div>
      </div>

      <div className="card map-card">
        {mode === "sim" && (
          <p className="sim-banner">
            <Icon name="alert" />
            Simulated reports around the University of Florida campus, for demonstration. Not real data.
          </p>
        )}
        <div
          ref={ref}
          id="public-map"
          className={data ? "home-map geo-map public-map" : "home-map geo-map public-map is-loading"}
          role="img"
          aria-label="3D grid map of public places where items are commonly lost; the list below has the same spots"
        >
          {view && (
            <DeckGL
              initialViewState={view}
              controller
              layers={layers}
              // Re-bin the grid when the zoom crosses a level, so the data scales with the map.
              onViewStateChange={({ viewState }) => {
                const next = gridLevel((viewState as GeoView).zoom);
                if (next !== level) setLevel(next);
                return viewState;
              }}
              getTooltip={({ object }) => {
                const c = object as GridCell | null;
                return (
                  c && {
                    text:
                      `${c.total} times here${c.spots > 1 ? ` (${c.spots} spots, zoom in for detail)` : ""}\n` +
                      c.items
                        .slice(0, 5)
                        .map((i) => `${capitalize(i.name)}: left ${i.left}, asked ${i.asked}`)
                        .join("\n"),
                    style: GEO_TOOLTIP_STYLE,
                  }
                );
              }}
            />
          )}
        </div>
        <p className="attribution">
          Buildings: <a href="https://openfreemap.org">OpenFreeMap</a> · <a href="https://www.openmaptiles.org/">© OpenMapTiles</a> · ©{" "}
          <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors
        </p>
        <div className="legend map-legend">
          <span className="heat-legend">
            Fewer <i className="heat-ramp" aria-hidden="true" /> More · column height and colour: items left or asked about in each area (100 m up close;
            neighbouring areas merge as you zoom out)
          </span>
        </div>
      </div>

      <div className="public-grid">
        <div className="card privacy-card">
          <h2>
            <Icon name="shield" />
            How this stays private
          </h2>
          <ul>
            <li>
              Only places a household marked <strong>public</strong> (shops, parks, transit), and only from households that opted in.
            </li>
            <li>Homes and other private places never appear here.</li>
            <li>
              Positions are rounded to about 100 m, and a spot only shows once <strong>at least {data?.min_households ?? 3} households</strong> have
              contributed.
            </li>
            <li>Only item names and counts: no words, photos, names or devices.</li>
          </ul>
        </div>
        <div className="card">
          <h2>
            Busiest spots{" "}
            {mode === "sim" && (
              <span className="chip attention">Simulated</span>
            )}
          </h2>
          <ol className="public-list">
            {!cells ? (
              repeat(3, (i) => (
                <li key={i} className="public-row" aria-hidden="true">
                  <Sk cls="line w60" />
                  <Sk cls="line w40" />
                </li>
              ))
            ) : !cells.length ? (
              <EmptyState
                as="li"
                title="Nothing public yet"
                text={`A spot appears once ${data?.min_households ?? 3} or more households have shared lost items there. Switch to Simulated to see how a busy map looks.`}
                cta={{ label: "Show simulated data", onClick: () => setMode("sim") }}
              />
            ) : (
              cells.slice(0, 10).map((c) => (
                <li key={`${c.lat},${c.lng}`}>
                  <button
                    type="button"
                    className="public-row"
                    aria-label={`Show spot ${c.lat.toFixed(3)}, ${c.lng.toFixed(3)} on the map: ${c.items
                      .slice(0, 4)
                      .map((i) => i.name)
                      .join(", ")}`}
                    onClick={() => flyTo(c)}
                  >
                    <span className="public-items">
                      {c.items
                        .slice(0, 4)
                        .map((i) => `${i.name} (${i.left + i.asked})`)
                        .join(", ")}
                    </span>
                    <span className="mono public-where">
                      {c.lat.toFixed(3)}, {c.lng.toFixed(3)}
                    </span>
                    <span className="mono public-total">{c.left + c.asked}×</span>
                  </button>
                  {c.place_name && (
                    <p className="public-place">
                      <MapsPlace place={c.place_name} />
                    </p>
                  )}
                </li>
              ))
            )}
          </ol>
          {cells?.slice(0, 10).some((c) => c.place_name) && <MapsNote />}
        </div>
      </div>
    </section>
  );
}
