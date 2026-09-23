import '@testing-library/jest-dom/vitest'
import { cleanup, configure } from '@testing-library/react'
import { afterEach } from 'vitest'

// The default second of findBy*/waitFor is short for this machine: with the whole suite running in parallel a
// lazy section or a fetch can take longer and the test fails without anything being broken.
configure({ asyncUtilTimeout: 5_000 })

afterEach(() => cleanup())

// jsdom has no media queries: the app asks for the system's light/dark setting, so answer «light» by default.
if (!window.matchMedia) {
  window.matchMedia = ((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
    addListener: () => undefined,
    removeListener: () => undefined,
    dispatchEvent: () => false,
  })) as typeof window.matchMedia
}
