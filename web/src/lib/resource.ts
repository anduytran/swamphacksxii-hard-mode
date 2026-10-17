// Page data with a small shared cache: switching back to a page within a minute shows what
// was loaded without refetching; after that (or on Refresh) it refetches while the old data
// stays on screen. A page with nothing cached yet shows its skeleton.
import { useCallback, useEffect, useRef, useState } from "react";
import { useApp } from "./app";

export const STALE_AFTER_MS = 60_000;

const cache = new Map<string, { data: unknown; at: number }>();

export const getCached = <T,>(key: string) => cache.get(key)?.data as T | undefined;
export const setCached = (key: string, data: unknown) => cache.set(key, { data, at: Date.now() });
/** After a change that affects every page (e.g. a household edit). */
export const invalidateAll = () => cache.clear();

export interface Resource<T> {
  data: T | undefined;
  loading: boolean;
  error: Error | undefined;
  reload(): void;
}

export function useResource<T>(key: string | null, load: () => Promise<T>): Resource<T> {
  const { refreshNonce } = useApp();
  const loadRef = useRef(load);
  loadRef.current = load;
  const [nonce, setNonce] = useState(0);
  const forced = useRef({ nonce, refreshNonce });
  const [state, setState] = useState<{ key: string | null; data: T | undefined; loading: boolean; error: Error | undefined }>(() => ({
    key,
    data: key ? getCached<T>(key) : undefined,
    loading: false,
    error: undefined,
  }));

  useEffect(() => {
    if (!key) return;
    const force = forced.current.nonce !== nonce || forced.current.refreshNonce !== refreshNonce;
    forced.current = { nonce, refreshNonce };
    const hit = cache.get(key);
    if (hit && !force && Date.now() - hit.at < STALE_AFTER_MS) {
      setState({ key, data: hit.data as T, loading: false, error: undefined });
      return;
    }
    let cancelled = false;
    setState((s) => ({ key, data: (hit?.data as T | undefined) ?? (s.key === key ? s.data : undefined), loading: true, error: undefined }));
    loadRef.current().then(
      (data) => {
        if (cancelled) return;
        setCached(key, data);
        setState({ key, data, loading: false, error: undefined });
      },
      (error: Error) => {
        if (!cancelled) setState((s) => ({ ...s, loading: false, error }));
      },
    );
    return () => {
      cancelled = true;
    };
  }, [key, nonce, refreshNonce]);

  const reload = useCallback(() => setNonce((n) => n + 1), []);
  return { data: state.key === key ? state.data : undefined, loading: state.loading, error: state.error, reload };
}
