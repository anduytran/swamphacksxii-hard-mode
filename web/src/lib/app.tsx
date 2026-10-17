// App-wide state shared by the shell and pages: the current page (hash routing), the status
// line, the Refresh button, setup progress, and the item detail drawer.
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

// ---------- routing ----------

export const PAGES = ["items", "map", "timeline", "insights", "search", "public", "household"] as const;
export type Page = (typeof PAGES)[number];
export const PUBLIC_PAGES: Page[] = ["public"]; // pages anyone can open without logging in
const DEFAULT_PAGE: Page = "items";

function pageFromHash(): Page {
  const name = window.location.hash.replace(/^#\/?/, "");
  return (PAGES as readonly string[]).includes(name) ? (name as Page) : DEFAULT_PAGE;
}

export function useHashPage(): Page {
  const [page, setPage] = useState(pageFromHash);
  useEffect(() => {
    const onChange = () => setPage(pageFromHash());
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return page;
}

export const go = (page: Page) => {
  window.location.hash = `#/${page}`;
};

// ---------- status line ----------

export interface Status {
  text: string;
  state: "" | "loading" | "error";
  retry?: () => void;
}

// ---------- shared context ----------

export interface Setup {
  household: boolean;
  wearable: boolean;
}

interface AppState {
  status: Status;
  setStatus(status: Status): void;
  refreshNonce: number;
  refresh(): void;
  setup: Setup;
  setSetup(setup: Setup): void;
  openItem(name: string): void;
  drawerItem: string | null;
  closeItem(): void;
}

const AppContext = createContext<AppState | null>(null);

export function useApp(): AppState {
  const app = useContext(AppContext);
  if (!app) throw new Error("useApp outside AppProvider");
  return app;
}

const EMPTY_STATUS: Status = { text: "", state: "" };

export function AppProvider({ children }: { children: ReactNode }) {
  const [status, setStatusState] = useState<Status>(EMPTY_STATUS);
  const [refreshNonce, setRefreshNonce] = useState(0);
  const [setup, setSetup] = useState<Setup>({ household: false, wearable: false });
  const [drawerItem, setDrawerItem] = useState<string | null>(null);

  const setStatus = useCallback((s: Status) => setStatusState(s.text ? s : EMPTY_STATUS), []);
  const refresh = useCallback(() => setRefreshNonce((n) => n + 1), []);
  const value = useMemo<AppState>(
    () => ({
      status,
      setStatus,
      refreshNonce,
      refresh,
      setup,
      setSetup,
      openItem: setDrawerItem,
      drawerItem,
      closeItem: () => setDrawerItem(null),
    }),
    [status, setStatus, refreshNonce, refresh, setup, drawerItem],
  );
  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

/** Mirrors a page's loading/error state into the status line under the header. */
export function useLoadStatus(loading: boolean, error: Error | undefined, retry: () => void) {
  const { setStatus } = useApp();
  useEffect(() => {
    if (loading) setStatus({ text: "Loading…", state: "loading" });
    else if (error) setStatus({ text: `Couldn't load this page: ${error.message}`, state: "error", retry });
    else setStatus(EMPTY_STATUS);
  }, [loading, error, retry, setStatus]);
  useEffect(() => () => setStatus(EMPTY_STATUS), [setStatus]);
}
