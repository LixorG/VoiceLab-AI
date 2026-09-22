/// <reference types="vitest/config" />
import { fileURLToPath, URL } from 'node:url'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const backend = process.env.VITE_BACKEND_URL ?? 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) } },
  // Served locally from disk (no network cost): one ~540 kB bundle is fine, so don't warn about it.
  build: { chunkSizeWarningLimit: 800 },
  server: {
    host: '127.0.0.1',
    port: 5173,
    proxy: { '/api': { target: backend, changeOrigin: true } },
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    css: false,
    // The heaviest tests (experiments, lazy sections) take ~5 s alone and more when the whole suite runs in
    // parallel on a busy machine: the 5 s default made them fail at random.
    testTimeout: 20_000,
    coverage: {
      provider: 'v8',
      include: ['src/**/*.{ts,tsx}'],
      exclude: ['src/**/*.test.{ts,tsx}', 'src/test/**', 'src/main.tsx', 'src/**/*.d.ts'],
      reporter: ['text-summary', 'html'],
      // Minimums a little below the measured values so regressions fail the check.
      thresholds: { statements: 82, branches: 70, functions: 77, lines: 84 },
    },
  },
})
