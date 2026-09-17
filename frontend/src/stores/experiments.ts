import { create } from 'zustand'

import { ApiError } from '@/services/api'
import { experimentsApi } from '@/services/experiments'
import type { ExperimentArm, ExperimentCreate, ExperimentRead, ExperimentSummary } from '@/types/experiments'
import { type GenerationRead, TERMINAL_STATUSES } from '@/types/generation'

const messageOf = (err: unknown) => (err instanceof ApiError ? err.message : String(err))

interface ExperimentsState {
  items: ExperimentSummary[]
  /** 'new' shows the creation form. */
  selected: string | null
  detail: ExperimentRead | null
  busy: 'create' | 'arms' | 'evaluate' | null
  error: string | null
  load: () => Promise<void>
  open: (id: string | 'new') => Promise<void>
  refresh: () => Promise<void>
  create: (body: ExperimentCreate) => Promise<boolean>
  addArms: (arms: ExperimentArm[]) => Promise<boolean>
  update: (patch: { name?: string; notes?: string }) => Promise<void>
  evaluateAll: () => Promise<void>
  remove: (id: string) => Promise<void>
  replaceGeneration: (gen: GenerationRead) => void
  clearError: () => void
}

export const isRunning = (exp: ExperimentRead | null) => !!exp?.generations.some((g) => !TERMINAL_STATUSES.includes(g.status))

export const useExperimentsStore = create<ExperimentsState>()((set, get) => {
  const setDetail = (detail: ExperimentRead) => {
    set({ detail, selected: detail.id })
    void get().load()
  }

  const run = async (busy: ExperimentsState['busy'], action: () => Promise<void>): Promise<boolean> => {
    set({ busy, error: null })
    try {
      await action()
      return true
    } catch (err) {
      set({ error: messageOf(err) })
      return false
    } finally {
      set({ busy: null })
    }
  }

  return {
    items: [],
    selected: null,
    detail: null,
    busy: null,
    error: null,

    load: async () => {
      try {
        set({ items: await experimentsApi.list() })
      } catch (err) {
        set({ error: messageOf(err) })
      }
    },

    open: async (id) => {
      set({ selected: id, error: null, detail: id === 'new' ? null : get().detail?.id === id ? get().detail : null })
      if (id === 'new') return
      try {
        set({ detail: await experimentsApi.get(id) })
      } catch (err) {
        set({ error: messageOf(err) })
      }
    },

    refresh: async () => {
      const id = get().detail?.id
      if (!id) return
      try {
        const detail = await experimentsApi.get(id)
        if (get().selected === id) set({ detail })
        if (!isRunning(detail)) void get().load()
      } catch {
        // transient: the next poll retries
      }
    },

    create: (body) => run('create', async () => setDetail(await experimentsApi.create(body))),
    addArms: (arms) => run('arms', async () => setDetail(await experimentsApi.addArms(get().detail!.id, arms))),
    evaluateAll: async () => {
      await run('evaluate', async () => set({ detail: await experimentsApi.evaluateAll(get().detail!.id) }))
    },

    update: async (patch) => {
      const id = get().detail?.id
      if (!id) return
      await run(null, async () => setDetail(await experimentsApi.update(id, patch)))
    },

    remove: async (id) => {
      await run(null, async () => {
        await experimentsApi.remove(id)
        set((s) => ({ items: s.items.filter((e) => e.id !== id), ...(s.selected === id ? { selected: null, detail: null } : {}) }))
      })
    },

    replaceGeneration: (gen) =>
      set((s) => (s.detail ? { detail: { ...s.detail, generations: s.detail.generations.map((g) => (g.id === gen.id ? gen : g)) } } : {})),

    clearError: () => set({ error: null }),
  }
})
