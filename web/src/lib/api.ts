// Calls to the LastScene API with the signed-in user's Auth0 token.
import { useCallback } from "react";
import { useAuth } from "./auth";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
  /** The server's message without the status code, for showing in forms. */
  get detail() {
    return this.message.replace(/^\d+ /, "");
  }
}

/** `body`: JSON, or FormData for file uploads (the browser sets its multipart header). */
export type Api = <T>(path: string, opts?: { method?: string; body?: unknown }) => Promise<T>;

export function useApi(): Api {
  const { getToken } = useAuth();
  return useCallback(
    async <T,>(path: string, { method = "GET", body }: { method?: string; body?: unknown } = {}) => {
      const token = await getToken();
      const form = body instanceof FormData;
      const res = await fetch(path, {
        method,
        headers: { Authorization: `Bearer ${token}`, ...(body && !form ? { "Content-Type": "application/json" } : {}) },
        body: form ? body : body ? JSON.stringify(body) : undefined,
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new ApiError(res.status, `${res.status} ${detail.detail || res.statusText}`);
      }
      return (await res.json()) as T;
    },
    [getToken],
  ) as Api;
}

/** The public endpoints need no token. */
export async function publicGet<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) throw new ApiError(res.status, `${res.status} ${res.statusText}`);
  return (await res.json()) as T;
}
