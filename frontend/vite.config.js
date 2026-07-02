import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

// The dashboard is served by FastAPI: built assets land in serve/static and are
// mounted at /static, so `base` must match. In dev we run Vite's own server and
// proxy the API + WebSocket through to the uvicorn process on :8000.
export default defineConfig({
  plugins: [svelte()],
  base: "/static/",
  build: {
    outDir: "../artificial_society/serve/static",
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/api": "http://localhost:8000",
      "/ws": { target: "ws://localhost:8000", ws: true },
    },
  },
});
