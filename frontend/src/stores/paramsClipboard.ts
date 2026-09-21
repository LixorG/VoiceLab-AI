import { create } from 'zustand'
import { createJSONStorage, persist } from 'zustand/middleware'

import type { ParamValue } from '@/types/engines'
import type { GenerationRead } from '@/types/generation'
import type { PostProcessConfig } from '@/types/postprocess'

/**
 * Generation settings copied from a result (Experiments, comparison) or from Generar, to paste elsewhere.
 * Voice and text are deliberately left out: pasting changes *how* to generate, not *what* or *with whom*.
 */
export interface CopiedSettings {
  engine: string
  variant: string | null
  params: Record<string, ParamValue>
  emotion: string | null
  intensity: number
  markup: boolean
  normalize: boolean
  postprocess: PostProcessConfig | null
  /** Where it came from, shown on the paste buttons. */
  label: string
  copiedAt: string
}

export function settingsFromGeneration(gen: GenerationRead, label: string): CopiedSettings {
  return {
    engine: gen.engine,
    variant: gen.variant,
    params: { ...gen.params, ...(gen.seed != null && { seed: gen.seed }) },
    emotion: gen.expression?.emotion ?? null,
    intensity: gen.expression?.intensity ?? 50,
    markup: gen.expression?.markup ?? true,
    normalize: gen.expression?.normalize ?? true,
    postprocess: gen.postprocess?.config ?? null,
    label,
    copiedAt: new Date().toISOString(),
  }
}

/** Plain-text form for the system clipboard (readable, and pasteable into notes). */
export function settingsAsText(settings: CopiedSettings): string {
  const { copiedAt: _copiedAt, ...rest } = settings
  return JSON.stringify(rest, null, 2)
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
      getItem: (key: string) => memory.get(key) ?? null,
      setItem: (key: string, value: string) => void memory.set(key, value),
      removeItem: (key: string) => void memory.delete(key),
    }
  }
})

interface ClipboardState {
  copied: CopiedSettings | null
  copy: (settings: CopiedSettings) => Promise<void>
  clear: () => void
}

export const useParamsClipboard = create<ClipboardState>()(
  persist(
    (set) => ({
      copied: null,
      copy: async (settings) => {
        set({ copied: settings })
        try {
          await navigator.clipboard?.writeText(settingsAsText(settings))
        } catch {
          /* the in-app copy is what matters; the system clipboard is a bonus */
        }
      },
      clear: () => set({ copied: null }),
    }),
    { name: 'voicelab-params-clipboard', storage: safeStorage, partialize: (s) => ({ copied: s.copied }) },
  ),
)
