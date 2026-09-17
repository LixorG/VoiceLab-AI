import { create } from 'zustand'

import { t } from '@/i18n/es'
import { ApiError } from '@/services/api'
import { type ReferencePatch, referencesApi, uploadReference } from '@/services/references'
import type { AudioFormats, ReferenceRead } from '@/types/api'

export type UploadPhase = 'uploading' | 'processing' | 'error'

export interface PendingUpload {
  key: string
  name: string
  size: number
  progress: number
  phase: UploadPhase
  error?: string
}

interface ReferencesState {
  /** null = all references; otherwise only those of this voice profile (uploads go to it too). */
  profileId: string | null
  setProfile: (profileId: string | null) => Promise<void>
  items: ReferenceRead[]
  pending: PendingUpload[]
  formats: AudioFormats | null
  loading: boolean
  error: string | null
  notice: string | null
  load: () => Promise<void>
  loadFormats: () => Promise<void>
  uploadFiles: (files: File[]) => Promise<void>
  dismissPending: (key: string) => void
  update: (id: string, patch: ReferencePatch) => Promise<void>
  reanalyze: (id: string) => Promise<void>
  remove: (id: string) => Promise<void>
  clearNotice: () => void
}

const messageOf = (err: unknown) => (err instanceof ApiError ? err.message : String(err))
let counter = 0

export const useReferencesStore = create<ReferencesState>()((set, get) => {
  const patchPending = (key: string, patch: Partial<PendingUpload>) =>
    set((s) => ({ pending: s.pending.map((p) => (p.key === key ? { ...p, ...patch } : p)) }))

  const replaceItem = (ref: ReferenceRead) =>
    set((s) => ({ items: s.items.map((r) => (r.id === ref.id ? { ...ref, is_recommended: r.is_recommended } : r)) }))

  return {
    profileId: null,
    setProfile: async (profileId) => {
      if (profileId === get().profileId && get().items.length) return
      set({ profileId, items: [] })
      await get().load()
    },
    items: [],
    pending: [],
    formats: null,
    loading: false,
    error: null,
    notice: null,

    load: async () => {
      set({ loading: true })
      try {
        const scope = get().profileId
        const items = await referencesApi.list(scope ?? undefined)
        if (scope === get().profileId) set({ items: Array.isArray(items) ? items : [], error: null })
      } catch (err) {
        set({ error: messageOf(err) })
      } finally {
        set({ loading: false })
      }
    },

    loadFormats: async () => {
      try {
        set({ formats: await referencesApi.formats() })
      } catch {
        /* optional: UI falls back to defaults */
      }
    },

    uploadFiles: async (files) => {
      const jobs = files.map((file) => ({ file, key: `up-${++counter}` }))
      set((s) => ({
        pending: [
          ...s.pending,
          ...jobs.map(({ file, key }) => ({ key, name: file.name, size: file.size, progress: 0, phase: 'uploading' as const })),
        ],
      }))
      // Sequential: the backend processes each file on arrival; parallel uploads would only compete for CPU.
      let duplicates = 0
      for (const { file, key } of jobs) {
        try {
          const res = await uploadReference(
            file,
            (fraction) => patchPending(key, { progress: fraction, phase: fraction >= 1 ? 'processing' : 'uploading' }),
            get().profileId ?? undefined,
          )
          if (res.duplicate) duplicates++
          set((s) => ({ pending: s.pending.filter((p) => p.key !== key) }))
        } catch (err) {
          patchPending(key, { phase: 'error', error: messageOf(err) })
        }
      }
      if (duplicates) set({ notice: t.references.duplicates(duplicates) })
      await get().load()
    },

    dismissPending: (key) => set((s) => ({ pending: s.pending.filter((p) => p.key !== key) })),

    update: async (id, patch) => {
      try {
        replaceItem(await referencesApi.update(id, patch))
      } catch (err) {
        set({ error: messageOf(err) })
      }
    },

    reanalyze: async (id) => {
      set((s) => ({ items: s.items.map((r) => (r.id === id ? { ...r, status: 'PROCESSING' } : r)) }))
      try {
        await referencesApi.reanalyze(id)
      } catch (err) {
        set({ error: messageOf(err) })
      }
      await get().load()
    },

    remove: async (id) => {
      try {
        await referencesApi.remove(id)
        await get().load()
      } catch (err) {
        set({ error: messageOf(err) })
      }
    },

    clearNotice: () => set({ notice: null, error: null }),
  }
})
