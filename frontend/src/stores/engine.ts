import { create } from 'zustand'
import { createJSONStorage, persist } from 'zustand/middleware'

import { ApiError } from '@/services/api'
import { modelsApi } from '@/services/models'
import type { EngineConfig, EngineSummary, ParamValue, ParameterSpec, PresetId } from '@/types/engines'

type Values = Record<string, ParamValue>

interface EngineState {
  engines: EngineSummary[]
  config: EngineConfig | null
  engineId: string | null
  variantId: string | null
  /** Parameter values per `${engine}:${variant}` so switching engines never mixes parameters. */
  valuesByKey: Record<string, Values>
  activePreset: PresetId | null
  loading: boolean
  error: string | null
  loadEngines: () => Promise<void>
  selectEngine: (engineId: string) => Promise<void>
  selectVariant: (variantId: string) => Promise<void>
  setValue: (paramId: string, value: ParamValue) => void
  applyPreset: (preset: PresetId) => void
  resetDefaults: () => void
}

export const keyOf = (engineId: string | null, variantId: string | null) => `${engineId}:${variantId}`

export function defaultsOf(parameters: ParameterSpec[]): Values {
  return Object.fromEntries(parameters.map((p) => [p.id, p.default]))
}

/** Keep stored values only for parameters that still exist in the schema. */
function mergeValues(parameters: ParameterSpec[], stored: Values | undefined): Values {
  const defaults = defaultsOf(parameters)
  if (!stored) return defaults
  return Object.fromEntries(parameters.map((p) => [p.id, p.id in stored ? stored[p.id] : defaults[p.id]]))
}

const messageOf = (err: unknown) => (err instanceof ApiError ? err.message : String(err))

const storage = createJSONStorage(() => {
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

export const useEngineStore = create<EngineState>()(
  persist(
    (set, get) => {
      const loadConfig = async (engineId: string, variantId: string | null) => {
        set({ loading: true, error: null })
        try {
          const config = await modelsApi.config(engineId, variantId)
          const key = keyOf(engineId, config.variant.id)
          set((s) => ({
            config,
            engineId,
            variantId: config.variant.id,
            activePreset: null,
            valuesByKey: { ...s.valuesByKey, [key]: mergeValues(config.parameters, s.valuesByKey[key]) },
          }))
        } catch (err) {
          set({ error: messageOf(err) })
        } finally {
          set({ loading: false })
        }
      }

      return {
        engines: [],
        config: null,
        engineId: null,
        variantId: null,
        valuesByKey: {},
        activePreset: null,
        loading: false,
        error: null,

        loadEngines: async () => {
          try {
            const list = await modelsApi.list()
            const engines = Array.isArray(list) ? list : []
            set({ engines, error: null })
            const { engineId, variantId } = get()
            const current = engines.find((e) => e.id === engineId)
            const engine = current ?? engines[0]
            if (!engine) return
            const variant = current && engine.variants.some((v) => v.id === variantId) ? variantId : null
            await loadConfig(engine.id, variant)
          } catch (err) {
            set({ error: messageOf(err) })
          }
        },
        selectEngine: (engineId) => loadConfig(engineId, null),
        selectVariant: (variantId) => loadConfig(get().engineId ?? '', variantId),

        setValue: (paramId, value) =>
          set((s) => {
            const key = keyOf(s.engineId, s.variantId)
            return {
              activePreset: null,
              valuesByKey: { ...s.valuesByKey, [key]: { ...s.valuesByKey[key], [paramId]: value } },
            }
          }),

        applyPreset: (preset) =>
          set((s) => {
            const values = s.config?.presets.find((p) => p.id === preset)?.values
            if (!values || !s.config) return {}
            const key = keyOf(s.engineId, s.variantId)
            // Presets never overwrite the seed or free text typed by the user.
            const kept = Object.fromEntries(
              s.config.parameters
                .filter((p) => p.type === 'seed' || p.type === 'text' || p.type === 'select')
                .map((p) => [p.id, s.valuesByKey[key]?.[p.id] ?? p.default]),
            )
            return { activePreset: preset, valuesByKey: { ...s.valuesByKey, [key]: { ...values, ...kept } } }
          }),

        resetDefaults: () =>
          set((s) => {
            if (!s.config) return {}
            return {
              activePreset: null,
              valuesByKey: { ...s.valuesByKey, [keyOf(s.engineId, s.variantId)]: defaultsOf(s.config.parameters) },
            }
          }),
      }
    },
    {
      name: 'voicelab-engine',
      storage,
      partialize: (s) => ({ engineId: s.engineId, variantId: s.variantId, valuesByKey: s.valuesByKey }),
    },
  ),
)
