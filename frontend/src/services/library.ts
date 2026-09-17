import { request } from '@/services/api'
import type { DeleteReport, LibraryFacets, LibraryItem, LibraryPage, LibraryQuery } from '@/types/library'

const json = (body: unknown): RequestInit => ({ body: JSON.stringify(body), headers: { 'Content-Type': 'application/json' } })

/** Only the filters the user actually set travel in the URL, so the query stays readable. */
export function queryString(query: LibraryQuery, limit: number, offset: number): string {
  const params = new URLSearchParams()
  if (query.q.trim()) params.set('q', query.q.trim())
  if (query.engine) params.set('engine', query.engine)
  if (query.profile_id) params.set('profile_id', query.profile_id)
  if (query.status) params.set('status', query.status)
  if (query.tag) params.set('tag', query.tag)
  if (query.favorite) params.set('favorite', 'true')
  if (query.since) params.set('since', query.since)
  if (query.until) params.set('until', query.until)
  params.set('limit', String(limit))
  params.set('offset', String(offset))
  return params.toString()
}

export const libraryApi = {
  search: (query: LibraryQuery, limit: number, offset: number) => request<LibraryPage>(`/library?${queryString(query, limit, offset)}`),
  facets: () => request<LibraryFacets>('/library/facets'),
  setFavorite: (id: string, favorite: boolean) => request<LibraryItem>(`/generation/${id}/favorite`, { method: 'PUT', ...json({ favorite }) }),
  setTags: (id: string, tags: string[]) => request<LibraryItem>(`/generation/${id}/tags`, { method: 'PUT', ...json({ tags }) }),
  deleteMany: (ids: string[]) => request<DeleteReport>('/library/delete', { method: 'POST', ...json({ ids }) }),
}
