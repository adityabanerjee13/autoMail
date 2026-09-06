import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Builds straight into the directory FastAPI already mounts at /static, so
// there is no copy step and no second server to run in production. base is
// absolute because the shell is served from both /chat and /chat/<id>; a
// relative base would resolve the bundle against the wrong path on a deep link.
export default defineConfig({
  plugins: [react()],
  base: "/static/app/",
  build: {
    outDir: "../src/triage/api/static/app",
    emptyOutDir: true,
  },
  server: {
    // `npm run dev` proxies the API to the running FastAPI process, so the
    // React app can be developed with hot reload against real data.
    proxy: {
      "/api": "http://127.0.0.1:8080",
    },
  },
});
