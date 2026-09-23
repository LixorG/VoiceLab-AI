import { create } from 'zustand'
import { createJSONStorage, persist } from 'zustand/middleware'

export type Section = 'generate' | 'library' | 'voices' | 'experiments' | 'projects' | 'models' | 'settings'
export type UiMode = 'simple' | 'advanced'
/** «system» follows the operating system's light/dark setting and changes with it. */
export type Theme = 'dark' | 'light' | 'system'

interface UiState {
  section: Section
  mode: UiMode
  theme: Theme
  sidebarCollapsed: boolean
  setSection: (section: Section) => void
  setMode: (mode: UiMode) => void
  setTheme: (theme: Theme) => void
  toggleTheme: () => void
  toggleSidebar: () => void
}

// Browser storage can throw (private mode, blocked site data); fall back to memory.
const safeStorage = createJSONStorage(() => {
  try {
    window.localStorage.setItem('__voicelab_probe', '1')
    window.localStorage.removeItem('__voicelab_probe')
    return window.localStorage
  } catch {
    const memory = new Map<string, string>()
    return {
      getItem: (k: string) => memory.get(k) ?? null,
      setItem: (k: string, v: string) => void memory.set(k, v),
      removeItem: (k: string) => void memory.delete(k),
    }
  }
})

/** The palette actually painted: «system» asks the operating system (and follows it while it is selected). */
export function resolveTheme(theme: Theme): 'dark' | 'light' {
  if (theme !== 'system') return theme
  return typeof window !== 'undefined' && window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

export const useUiStore = create<UiState>()(
  persist(
    (set) => ({
      section: 'generate',
      mode: 'simple',
      theme: 'dark',
      sidebarCollapsed: false,
      setSection: (section) => set({ section }),
      setMode: (mode) => set({ mode }),
      setTheme: (theme) => set({ theme }),
      toggleTheme: () => set((s) => ({ theme: resolveTheme(s.theme) === 'dark' ? 'light' : 'dark' })),
      toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
    }),
    { name: 'voicelab-ui', storage: safeStorage },
  ),
)
