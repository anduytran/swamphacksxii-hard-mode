// The app: decides between the landing page and a dashboard page, and holds the shell.
import { lazy, Suspense, useEffect, useRef, useState } from "react";
import { ChartTooltip } from "./components/Charts";
import { ItemDrawer } from "./components/ItemDrawer";
import { Header, StatusLine } from "./components/Shell";
import { useApi } from "./lib/api";
import { go, PUBLIC_PAGES, useApp, useHashPage, type Page } from "./lib/app";
import { useAuth } from "./lib/auth";
import { setCached } from "./lib/resource";
import type { Household } from "./lib/types";
import HouseholdPage from "./pages/Household";
import InsightsPage from "./pages/Insights";
import ItemsPage from "./pages/Items";
import { Landing } from "./pages/Landing";
import SearchPage from "./pages/Search";
import TimelinePage from "./pages/Timeline";

// The map pages carry deck.gl, so they load on first visit. A tab left open across a deploy
// asks for chunk names that no longer exist: reload once to pick up the new build.
function lazyPage<T extends { default: () => JSX.Element }>(load: () => Promise<T>) {
  return lazy(() =>
    load().then(
      (mod) => {
        try { sessionStorage.removeItem("chunk-reload"); } catch { /* storage blocked */ }
        return mod;
      },
      (err: unknown) => {
        let reloaded = true;
        try {
          reloaded = sessionStorage.getItem("chunk-reload") === "1";
          sessionStorage.setItem("chunk-reload", "1");
        } catch { /* storage blocked: don't risk a reload loop */ }
        if (!reloaded) location.reload();
        throw err;
      },
    ),
  );
}
const MapPage = lazyPage(() => import("./pages/MapPage"));
const PublicMapPage = lazyPage(() => import("./pages/PublicMap"));

const PAGE_VIEWS: Record<Page, () => JSX.Element> = {
  items: () => <ItemsPage />,
  timeline: () => <TimelinePage />,
  map: () => <MapPage />,
  insights: () => <InsightsPage />,
  search: () => <SearchPage />,
  public: () => <PublicMapPage />,
  household: () => <HouseholdPage />,
};

export function App() {
  const auth = useAuth();
  const page = useHashPage();
  const api = useApi();
  const { setSetup } = useApp();
  const [checked, setChecked] = useState(false);

  useEffect(() => {
    document.body.classList.toggle("signed-in", auth.signedIn);
  }, [auth.signedIn]);

  // After sign-in: note setup progress; someone with no household yet starts at setup.
  useEffect(() => {
    if (auth.status !== "ready") return;
    if (!auth.signedIn) return setChecked(true);
    api<{ households: Household[] }>("/api/v1/households/me")
      .then((res) => {
        setCached("households", res);
        setSetup({ household: res.households.length > 0, wearable: res.households.some((h) => h.devices.length > 0) });
        const here = window.location.hash.replace(/^#\/?/, "");
        if (!res.households.length && !["household", ...PUBLIC_PAGES].includes(here as Page)) go("household");
      })
      .catch(() => {}) // the page itself will show the error
      .finally(() => setChecked(true));
  }, [auth.status, auth.signedIn, api, setSetup]);

  // A different page starts at the top.
  const shown = useRef(page);
  useEffect(() => {
    if (shown.current !== page) window.scrollTo(0, 0);
    shown.current = page;
  }, [page]);

  const canShow = auth.signedIn || PUBLIC_PAGES.includes(page);
  let content = null;
  if (auth.status === "error" || (auth.status === "ready" && !canShow)) content = <Landing />;
  else if (auth.status === "ready" && checked) {
    const View = PAGE_VIEWS[page];
    content = (
      <>
        <StatusLine />
        <Suspense fallback={null}>
          <View key={page} />
        </Suspense>
      </>
    );
  }

  return (
    <>
      <a
        className="skip-link"
        href="#main"
        onClick={(e) => {
          // Move focus without touching the hash, which drives the page routing.
          e.preventDefault();
          document.getElementById("main")?.focus();
        }}
      >
        Skip to content
      </a>
      <div className="shell">
        <Header page={page} />
        <main className="main" id="main" tabIndex={-1}>
          {content}
        </main>
      </div>
      <ItemDrawer />
      <ChartTooltip />
    </>
  );
}
