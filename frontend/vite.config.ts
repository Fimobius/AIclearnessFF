import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // Vite's default targets Safari 16.4+; macOS 13 (Ventura) ships Safari 16.0, so target lower.
  build: { target: ["es2020", "safari15", "chrome90", "edge90", "firefox90"] },
  server: { port: 5173, proxy: { "/api": "http://127.0.0.1:8000", "/docs": "http://127.0.0.1:8000", "/openapi.json": "http://127.0.0.1:8000" } },
});
