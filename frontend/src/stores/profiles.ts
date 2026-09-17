import { create } from 'zustand'

import { ApiError } from '@/services/api'
import { profilesApi } from '@/services/profiles'
import type { ParamValue } from '@/types/engines'
import type { EmotionOption, ProfileInput, ProfileRead } from '@/types/profiles'

interface ProfilesState {
  items: ProfileRead[]
  emotions: EmotionOption[]
  loading: boolean
  error: string | null
  load: () => Promise<void>
  loadEmotions: () => Promise<void>
  create: (input: ProfileInput & { name: string }) => Promise<ProfileRead | null>
  update: (id: string, input: ProfileInput) => Promise<void>
  remove: (id: string, deleteReferences: boolean) => Promise<boolean>
  refresh: (id: string) => Promise<void>
  saveRecommended: (id: string, engine: string, variant: string | null, params: Record<string, ParamValue>) => Promise<boolean>
  removeRecommended: (id: string, engine: string) => Promise<void>
  importFile: (file: File) => Promise<ProfileRead | null>
  clearError: () => void
}

const messageOf = (err: unknown) => (err instanceof ApiError ? err.message : String(err))

const sortByName = (items: ProfileRead[]) => [...items].sort((a, b) => a.name.localeCompare(b.name, 'es'))

export const useProfilesStore = create<ProfilesState>()((set, get) => {
  const upsert = (profile: ProfileRead) =>
    set((s) => ({
      items: sortByName(s.items.some((p) => p.id === profile.id) ? s.items.map((p) => (p.id === profile.id ? profile : p)) : [...s.items, profile]),
    }))

  const attempt = async <T>(fn: () => Promise<T>): Promise<T | null> => {
    try {
      set({ error: null })
      return await fn()
    } catch (err) {
      set({ error: messageOf(err) })
      return null
    }
  }

  return {
    items: [],
    emotions: [],
    loading: false,
    error: null,

    load: async () => {
      set({ loading: true })
      const items = await attempt(profilesApi.list)
      set({ loading: false, ...(Array.isArray(items) ? { items: sortByName(items) } : {}) })
    },
    loadEmotions: async () => {
      if (get().emotions.length) return
      const emotions = await attempt(profilesApi.emotions)
      if (Array.isArray(emotions)) set({ emotions })
    },
    create: async (input) => {
      const profile = await attempt(() => profilesApi.create(input))
      if (profile) upsert(profile)
      return profile
    },
    update: async (id, input) => {
      const profile = await attempt(() => profilesApi.update(id, input))
      if (profile) upsert(profile)
    },
    remove: async (id, deleteReferences) => {
      const ok = (await attempt(() => profilesApi.remove(id, deleteReferences))) !== null
      if (ok) set((s) => ({ items: s.items.filter((p) => p.id !== id) }))
      return ok
    },
    refresh: async (id) => {
      const profile = await attempt(() => profilesApi.get(id))
      if (profile) upsert(profile)
    },
    saveRecommended: async (id, engine, variant, params) => {
      const profile = await attempt(() => profilesApi.setRecommended(id, engine, variant, params))
      if (profile) upsert(profile)
      return profile !== null
    },
    removeRecommended: async (id, engine) => {
      const profile = await attempt(() => profilesApi.removeRecommended(id, engine))
      if (profile) upsert(profile)
    },
    importFile: async (file) => {
      const profile = await attempt(() => profilesApi.import(file))
      if (profile) upsert(profile)
      return profile
    },
    clearError: () => set({ error: null }),
  }
})
