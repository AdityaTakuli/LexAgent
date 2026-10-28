import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The Python server (ui/server.py) serves the built app and the API on one port.
// During development, `npm run dev` proxies /api to it.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: { "/api": "http://127.0.0.1:8000" },
    fs: { allow: [".."] }, // ../sample_transcript.json, the offline replay
  },
  build: { outDir: "dist", emptyOutDir: true, chunkSizeWarningLimit: 900 },
});
