import type { JobStatus } from '@/types/generation'

export interface LibraryItem {
  id: string
  kind: string
  engine: string
  variant: string | null
  text: string
  status: JobStatus
  seed: number | null
  duration_s: number | null
  audio_url: string | null
  favorite: boolean
  tags: string[]
  rating: Record<string, unknown> | null
  profile_id: string | null
  profile_name: string | null
  reference_name: string | null
  created_at: string
}

export interface LibraryPage {
  items: LibraryItem[]
  total: number
  limit: number
  offset: number
}

export interface FacetOption {
  id: string
  label: string
  count: number
}

export interface LibraryFacets {
  tags: { tag: string; count: number }[]
  engines: FacetOption[]
  profiles: FacetOption[]
  total: number
  favorites: number
}

export interface LibraryQuery {
  q: string
  engine: string
  profile_id: string
  status: string
  tag: string
  favorite: boolean
  since: string
  until: string
}

export const EMPTY_QUERY: LibraryQuery = { q: '', engine: '', profile_id: '', status: '', tag: '', favorite: false, since: '', until: '' }

export interface DeleteReport {
  deleted: string[]
  failed: string[]
}
