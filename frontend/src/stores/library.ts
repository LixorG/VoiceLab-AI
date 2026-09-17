import { create } from 'zustand'

import { t } from '@/i18n/es'
import { ApiError } from '@/services/api'
import { libraryApi } from '@/services/library'
import { EMPTY_QUERY, type LibraryFacets, type LibraryItem, type LibraryQuery } from '@/types/library'

export const PAGE_SIZE = 20

interface LibraryState {
  query: LibraryQuery
  items: LibraryItem[]
  total: number
  facets: LibraryFacets | null
  selected: string[]
  loading: boolean
  error: string | null
  setQuery: (patch: Partial<LibraryQuery>) => void
  resetQuery: () => void
  load: () => Promise<void>
  loadMore: () => Promise<void>
  loadFacets: () => Promise<void>
  toggleFavorite: (id: string) => Promise<void>
  setTags: (id: string, tags: string[]) => Promise<void>
  toggleSelected: (id: string) => void
  clearSelection: () => void
  removeSelected: () => Promise<void>
}

const messageOf = (err: unknown) => (err instanceof ApiError ? err.message : String(err))

export const useLibraryStore = create<LibraryState>()((set, get) => ({
  query: { ...EMPTY_QUERY },
  items: [],
  total: 0,
  facets: null,
  selected: [],
  loading: false,
  error: null,

  setQuery: (patch) => {
    set((s) => ({ query: { ...s.query, ...patch } }))
    void get().load()
  },
  resetQuery: () => {
    set({ query: { ...EMPTY_QUERY } })
    void get().load()
  },

  load: async () => {
    set({ loading: true, error: null })
    try {
      const page = await libraryApi.search(get().query, PAGE_SIZE, 0)
      const ids = new Set(page.items.map((i) => i.id))
      set((s) => ({ items: page.items, total: page.total, selected: s.selected.filter((id) => ids.has(id)) }))
    } catch (err) {
      set({ error: messageOf(err) })
    } finally {
      set({ loading: false })
    }
  },

  loadMore: async () => {
    const { items, query, loading } = get()
    if (loading) return
    set({ loading: true })
    try {
      const page = await libraryApi.search(query, PAGE_SIZE, items.length)
      set((s) => ({ items: [...s.items, ...page.items], total: page.total }))
    } catch (err) {
      set({ error: messageOf(err) })
    } finally {
      set({ loading: false })
    }
  },

  loadFacets: async () => {
    try {
      set({ facets: await libraryApi.facets() })
    } catch (err) {
      set({ error: messageOf(err) })
    }
  },

  toggleFavorite: async (id) => {
    const current = get().items.find((i) => i.id === id)
    if (!current) return
    try {
      const updated = await libraryApi.setFavorite(id, !current.favorite)
      set((s) => ({ items: s.items.map((i) => (i.id === id ? updated : i)) }))
      void get().loadFacets()
    } catch (err) {
      set({ error: messageOf(err) })
    }
  },

  setTags: async (id, tags) => {
    try {
      const updated = await libraryApi.setTags(id, tags)
      set((s) => ({ items: s.items.map((i) => (i.id === id ? updated : i)) }))
      void get().loadFacets()
    } catch (err) {
      set({ error: messageOf(err) })
    }
  },

  toggleSelected: (id) =>
    set((s) => ({ selected: s.selected.includes(id) ? s.selected.filter((x) => x !== id) : [...s.selected, id] })),
  clearSelection: () => set({ selected: [] }),

  removeSelected: async () => {
    const ids = get().selected
    if (ids.length === 0) return
    try {
      const report = await libraryApi.deleteMany(ids)
      set({ selected: [] })
      await get().load()
      void get().loadFacets()
      if (report.failed.length > 0) set({ error: t.library.deleteFailed(report.failed.length) })
    } catch (err) {
      set({ error: messageOf(err) })
    }
  },
}))
