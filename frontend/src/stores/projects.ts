import { create } from 'zustand'

import { ApiError } from '@/services/api'
import { projectsApi } from '@/services/projects'
import type { ProjectRead, ProjectSettings, ProjectSummary, SegmentPatch } from '@/types/projects'

const messageOf = (err: unknown) => (err instanceof ApiError ? err.message : String(err))
const segmentOf = (err: unknown) =>
  err instanceof ApiError && err.details && typeof err.details === 'object' ? ((err.details as { segmento?: number }).segmento ?? null) : null

interface ProjectsState {
  items: ProjectSummary[]
  selected: string | null
  detail: ProjectRead | null
  busy: string | null
  error: string | null
  load: () => Promise<void>
  open: (id: string) => Promise<void>
  refresh: () => Promise<void>
  create: (name: string, settings: ProjectSettings) => Promise<boolean>
  rename: (name: string) => Promise<void>
  saveSettings: (settings: ProjectSettings) => Promise<void>
  importScript: (text: string, split: 'paragraphs' | 'sentences') => Promise<boolean>
  addSegment: (text: string) => Promise<void>
  updateSegment: (segmentId: string, patch: SegmentPatch) => Promise<void>
  removeSegment: (segmentId: string) => Promise<void>
  move: (segmentId: string, delta: -1 | 1) => Promise<void>
  generate: (segmentIds: string[] | null, onlyPending?: boolean) => Promise<void>
  remove: (id: string) => Promise<void>
  clearError: () => void
}

export const isProjectRunning = (p: ProjectRead | null) => !!p?.segments.some((s) => s.status === 'queued' || s.status === 'generating')

export const useProjectsStore = create<ProjectsState>()((set, get) => {
  /** Run an action on the open project; errors are shown in Spanish, pointing to the segment when relevant. */
  const act = async (busy: string, action: (id: string) => Promise<ProjectRead>) => {
    const id = get().detail?.id
    if (!id) return false
    set({ busy, error: null })
    try {
      set({ detail: await action(id) })
      void get().load()
      return true
    } catch (err) {
      const segment = segmentOf(err)
      set({ error: segment ? `Segmento ${segment}: ${messageOf(err)}` : messageOf(err) })
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
        set({ items: await projectsApi.list() })
      } catch (err) {
        set({ error: messageOf(err) })
      }
    },

    open: async (id) => {
      set({ selected: id, error: null, detail: get().detail?.id === id ? get().detail : null })
      try {
        set({ detail: await projectsApi.get(id) })
      } catch (err) {
        set({ error: messageOf(err) })
      }
    },

    refresh: async () => {
      const id = get().detail?.id
      if (!id) return
      try {
        const detail = await projectsApi.get(id)
        if (get().selected === id) set({ detail })
        if (!isProjectRunning(detail)) void get().load()
      } catch {
        // transient; the next poll retries
      }
    },

    create: async (name, settings) => {
      set({ busy: 'create', error: null })
      try {
        const detail = await projectsApi.create(name, settings)
        set({ detail, selected: detail.id })
        void get().load()
        return true
      } catch (err) {
        set({ error: messageOf(err) })
        return false
      } finally {
        set({ busy: null })
      }
    },

    rename: async (name) => {
      await act('rename', (id) => projectsApi.update(id, { name }))
    },
    saveSettings: async (settings) => {
      await act('settings', (id) => projectsApi.update(id, { settings }))
    },
    importScript: (text, split) => act('import', (id) => projectsApi.importScript(id, text, split)),
    addSegment: async (text) => {
      await act('add', (id) => projectsApi.addSegment(id, text))
    },
    updateSegment: async (segmentId, patch) => {
      await act(`segment:${segmentId}`, (id) => projectsApi.updateSegment(id, segmentId, patch))
    },
    removeSegment: async (segmentId) => {
      await act(`segment:${segmentId}`, (id) => projectsApi.removeSegment(id, segmentId))
    },
    move: async (segmentId, delta) => {
      const ids = get().detail?.segments.map((s) => s.id) ?? []
      const from = ids.indexOf(segmentId)
      const to = from + delta
      if (from < 0 || to < 0 || to >= ids.length) return
      ;[ids[from], ids[to]] = [ids[to], ids[from]]
      await act('reorder', (id) => projectsApi.reorder(id, ids))
    },
    generate: async (segmentIds, onlyPending = true) => {
      await act(segmentIds?.length === 1 ? `segment:${segmentIds[0]}` : 'generate', (id) => projectsApi.generate(id, segmentIds, onlyPending))
    },

    remove: async (id) => {
      try {
        await projectsApi.remove(id)
        set((s) => ({ items: s.items.filter((p) => p.id !== id), ...(s.selected === id ? { selected: null, detail: null } : {}) }))
      } catch (err) {
        set({ error: messageOf(err) })
      }
    },

    clearError: () => set({ error: null }),
  }
})
