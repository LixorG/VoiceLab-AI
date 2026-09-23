import { create } from 'zustand'

import { ApiError } from '@/services/api'
import { type GenerationRequest, generationApi, subscribeToJob } from '@/services/generation'
import { t } from '@/i18n/es'
import { keyOf, useEngineStore } from '@/stores/engine'
import { type CopiedSettings, settingsFromGeneration } from '@/stores/paramsClipboard'
import type { EngineRuntimeStatus, GenerationPlan, GenerationRead } from '@/types/generation'
import { DEFAULT_POSTPROCESS, type PostProcessCapabilities, type PostProcessConfig, isPostprocessActive } from '@/types/postprocess'
import { TERMINAL_STATUSES } from '@/types/generation'

interface GenerationState {
  text: string
  referenceId: string | null
  profileId: string | null
  setProfile: (profileId: string | null) => void
  emotion: string | null
  intensity: number
  markup: boolean
  takes: number
  normalize: boolean
  setEmotion: (emotion: string | null) => void
  setIntensity: (intensity: number) => void
  setMarkup: (markup: boolean) => void
  setTakes: (takes: number) => void
  setNormalize: (normalize: boolean) => void
  plan: GenerationPlan | null
  planError: string | null
  loadPlan: () => Promise<void>
  generateVariations: (count: number) => Promise<void>
  postprocess: PostProcessConfig
  setPostprocess: (config: PostProcessConfig) => void
  postprocessCaps: PostProcessCapabilities | null
  loadPostprocessCaps: () => Promise<void>
  /** Re-master a finished generation; resolves to an error message or null. */
  remaster: (id: string, config: PostProcessConfig) => Promise<string | null>
  clearMaster: (id: string) => Promise<string | null>
  /** Generations selected for A/B comparison in the results list. */
  compareIds: string[]
  toggleCompare: (id: string) => void
  clearCompare: () => void
  replaceItem: (gen: GenerationRead) => void
  items: GenerationRead[]
  submitting: 'full' | 'preview' | 'variations' | null
  error: string | null
  runtime: EngineRuntimeStatus | null
  setText: (text: string) => void
  setReference: (id: string | null) => void
  load: () => Promise<void>
  generate: (preview: boolean) => Promise<void>
  repeat: (generation: GenerationRead) => Promise<void>
  /** Apply copied settings (engine, variant, parameters, expression, post-processing); voice and text untouched. */
  applySettings: (settings: CopiedSettings) => Promise<ApplyReport>
  /** The current Generar settings, ready to copy. */
  currentSettings: () => CopiedSettings | null
  cancel: (id: string) => Promise<void>
  remove: (id: string) => Promise<void>
  loadRuntime: () => Promise<void>
  downloadWeights: () => Promise<void>
  clearError: () => void
}

export interface ApplyReport {
  applied: number
  skipped: string[]
}

const messageOf = (err: unknown) => (err instanceof ApiError ? err.message : String(err))
const POLL_MS = 1500
const subscriptions = new Map<string, () => void>()
let runtimeTimer: number | undefined

export const useGenerationStore = create<GenerationState>()((set, get) => {
  const buildRequest = (preview: boolean): GenerationRequest | null => {
    const { engineId, variantId, valuesByKey } = useEngineStore.getState()
    if (!engineId) return null
    const s = get()
    return {
      engine: engineId,
      variant: variantId,
      text: s.text,
      params: valuesByKey[keyOf(engineId, variantId)] ?? {},
      reference_id: s.referenceId,
      profile_id: s.profileId,
      preview,
      emotion: s.emotion,
      intensity: s.intensity,
      markup: s.markup,
      normalize: s.normalize,
      takes: preview ? 1 : s.takes,  // a preview is a quick listen: one take
      postprocess: isPostprocessActive(s.postprocess) ? s.postprocess : null,
    }
  }

  const upsert = (gen: GenerationRead) =>
    set((s) => ({ items: s.items.some((g) => g.id === gen.id) ? s.items.map((g) => (g.id === gen.id ? gen : g)) : [gen, ...s.items] }))

  const refresh = async (id: string) => {
    try {
      upsert(await generationApi.get(id))
    } catch {
      /* deleted meanwhile */
    }
  }

  /** Live progress via SSE, with polling as fallback when the stream is unavailable. */
  const track = (gen: GenerationRead) => {
    if (TERMINAL_STATUSES.includes(gen.status) || subscriptions.has(gen.id)) return
    const poll = async () => {
      await refresh(gen.id)
      const current = get().items.find((g) => g.id === gen.id)
      if (current && !TERMINAL_STATUSES.includes(current.status)) window.setTimeout(() => void poll(), POLL_MS)
      else subscriptions.delete(gen.id)
    }
    let streamed = false
    const unsubscribe = subscribeToJob(
      gen.id,
      (event) => {
        streamed = true
        set((s) => ({
          items: s.items.map((g) =>
            g.id === event.job_id ? { ...g, status: event.status, progress: event.progress, message: event.message, stream_chunks: event.chunks ?? g.stream_chunks } : g,
          ),
        }))
      },
      () => {
        subscriptions.delete(gen.id)
        if (streamed) void refresh(gen.id).then(() => get().loadRuntime())
        else void poll()
      },
    )
    subscriptions.set(gen.id, unsubscribe)
  }

  return {
    text: '',
    referenceId: null,
    profileId: null,
    setProfile: (profileId) => set({ profileId, referenceId: null }),
    emotion: null,
    intensity: 50,
    markup: true,
    takes: 1,
    normalize: true,
    setEmotion: (emotion) => set({ emotion }),
    setIntensity: (intensity) => set({ intensity }),
    setMarkup: (markup) => set({ markup }),
    setTakes: (takes) => set({ takes: Math.max(1, Math.min(5, takes)) }),
    setNormalize: (normalize) => set({ normalize }),
    plan: null,
    planError: null,
    compareIds: [],
    toggleCompare: (id) =>
      set((s) => ({ compareIds: s.compareIds.includes(id) ? s.compareIds.filter((x) => x !== id) : [...s.compareIds, id].slice(-4) })),
    clearCompare: () => set({ compareIds: [] }),
    replaceItem: (gen) => upsert(gen),
    postprocess: DEFAULT_POSTPROCESS,
    setPostprocess: (postprocess) => set({ postprocess }),
    postprocessCaps: null,
    loadPostprocessCaps: async () => {
      if (get().postprocessCaps) return
      try {
        set({ postprocessCaps: await generationApi.postprocessCapabilities() })
      } catch {
        // Without the list every processor stays enabled; the backend still reports what it could not apply.
      }
    },
    remaster: async (id, config) => {
      try {
        upsert(await generationApi.postprocess(id, config))
        return null
      } catch (err) {
        return messageOf(err)
      }
    },
    clearMaster: async (id) => {
      try {
        upsert(await generationApi.clearPostprocess(id))
        return null
      } catch (err) {
        return messageOf(err)
      }
    },

    loadPlan: async () => {
      const body = buildRequest(false)
      if (!body || !body.text.trim()) return set({ plan: null, planError: null })
      try {
        set({ plan: await generationApi.plan(body), planError: null })
      } catch (err) {
        // Missing weights/installation don't block the plan; other errors (markup, references) are shown.
        set({ plan: null, planError: messageOf(err) })
      }
    },

    generateVariations: async (count) => {
      const body = buildRequest(false)
      if (!body) return
      set({ submitting: 'variations', error: null })
      try {
        const accepted = await generationApi.variations(body, count)
        for (const res of [...accepted].reverse()) {
          upsert(res.generation)
          track(res.generation)
        }
      } catch (err) {
        set({ error: messageOf(err) })
      } finally {
        set({ submitting: null })
      }
    },
    items: [],
    submitting: null,
    error: null,
    runtime: null,

    setText: (text) => set({ text }),
    setReference: (referenceId) => set({ referenceId }),

    load: async () => {
      try {
        const items = await generationApi.list()
        set({ items: Array.isArray(items) ? items : [] })
        get().items.forEach(track)
      } catch (err) {
        set({ error: messageOf(err) })
      }
    },

    generate: async (preview) => {
      const body = buildRequest(preview)
      if (!body) return
      set({ submitting: preview ? 'preview' : 'full', error: null })
      try {
        const res = await generationApi.create(body)
        upsert(res.generation)
        track(res.generation)
      } catch (err) {
        set({ error: messageOf(err) })
      } finally {
        set({ submitting: null })
      }
    },

    repeat: async (generation) => {
      await get().applySettings(settingsFromGeneration(generation, generation.label ?? generation.engine))
      set({ text: generation.text, referenceId: generation.reference?.reference_id ?? get().referenceId })
      await get().generate(generation.kind === 'preview')
    },

    applySettings: async (settings) => {
      const engineStore = useEngineStore.getState()
      if (!engineStore.engines.some((e) => e.id === settings.engine)) {
        throw new Error(t.paramsClipboard.unknownEngine(settings.engine))
      }
      if (engineStore.engineId !== settings.engine || engineStore.variantId !== settings.variant) {
        await engineStore.selectEngine(settings.engine)
        if (settings.variant) await useEngineStore.getState().selectVariant(settings.variant)
      }
      const known = new Set(useEngineStore.getState().config?.parameters.map((p) => p.id) ?? [])
      const skipped: string[] = []
      let applied = 0
      for (const [id, value] of Object.entries(settings.params)) {
        if (known.size && !known.has(id)) {
          skipped.push(id)
          continue
        }
        useEngineStore.getState().setValue(id, value)
        applied += 1
      }
      set({
        emotion: settings.emotion,
        intensity: settings.intensity,
        markup: settings.markup,
        normalize: settings.normalize,
        postprocess: settings.postprocess ?? DEFAULT_POSTPROCESS,
      })
      return { applied, skipped }
    },

    currentSettings: () => {
      const { engineId, variantId, valuesByKey } = useEngineStore.getState()
      if (!engineId) return null
      const s = get()
      return {
        engine: engineId,
        variant: variantId,
        params: { ...valuesByKey[keyOf(engineId, variantId)] },
        emotion: s.emotion,
        intensity: s.intensity,
        markup: s.markup,
        normalize: s.normalize,
        postprocess: isPostprocessActive(s.postprocess) ? s.postprocess : null,
        label: t.paramsClipboard.fromGenerate,
        copiedAt: new Date().toISOString(),
      }
    },

    cancel: async (id) => {
      try {
        await generationApi.cancel(id)
        await refresh(id)
      } catch (err) {
        set({ error: messageOf(err) })
      }
    },

    remove: async (id) => {
      subscriptions.get(id)?.()
      subscriptions.delete(id)
      try {
        await generationApi.remove(id)
        set((s) => ({ items: s.items.filter((g) => g.id !== id) }))
      } catch (err) {
        set({ error: messageOf(err) })
      }
    },

    loadRuntime: async () => {
      const { engineId, variantId } = useEngineStore.getState()
      if (!engineId) return
      try {
        const runtime = await generationApi.engineStatus(engineId, variantId)
        set({ runtime })
        window.clearTimeout(runtimeTimer)
        if (runtime.download_state === 'downloading') runtimeTimer = window.setTimeout(() => void get().loadRuntime(), 3000)
      } catch {
        set({ runtime: null })
      }
    },

    downloadWeights: async () => {
      const { engineId, variantId } = useEngineStore.getState()
      if (!engineId) return
      try {
        set({ runtime: await generationApi.downloadWeights(engineId, variantId) })
        void get().loadRuntime()
      } catch (err) {
        set({ error: messageOf(err) })
      }
    },

    clearError: () => set({ error: null }),
  }
})
