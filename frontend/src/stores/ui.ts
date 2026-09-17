import { create } from 'zustand'
import { createJSONStorage, persist } from 'zustand/middleware'

export type Section = 'generate' | 'library' | 'voices' | 'experiments' | 'projects' | 'models' | 'settings'
export type UiMode = 'simple' | 'advanced'
export type Theme = 'dark' | 'light'

interface UiState {
  section: Section
  mode: UiMode
  theme: Theme
  sidebarCollapsed: boolean
  setSection: (section: Section) => void
  setMode: (mode: UiMode) => void
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

export const useUiStore = create<UiState>()(
  persist(
    (set) => ({
      section: 'generate',
      mode: 'simple',
      theme: 'dark',
      sidebarCollapsed: false,
      setSection: (section) => set({ section }),
      setMode: (mode) => set({ mode }),
      toggleTheme: () => set((s) => ({ theme: s.theme === 'dark' ? 'light' : 'dark' })),
      toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
    }),
    { name: 'voicelab-ui', storage: safeStorage },
  ),
)
