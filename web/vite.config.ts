import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// The API runs separately (uvicorn on :8000 locally, a Python function on Vercel). In dev the
// Vite server proxies it; built files are served by Vercel's CDN or, locally, by FastAPI.
const api = "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/api": api, "/config.json": api, "/healthz": api },
  },
  build: {
    // deck.gl is large; it only loads with the map pages (lazy chunks).
    chunkSizeWarningLimit: 1800,
  },
  test: {
    environment: "node",
  },
});
