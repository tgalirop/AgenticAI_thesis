import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    // Local development uses the same /api path as the containerized build.
    proxy: { "/api": "http://localhost:8000" },
  },
});
