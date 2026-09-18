import { request } from '@/services/api'
import type { ProjectRead, ProjectSettings, ProjectSummary, SegmentPatch } from '@/types/projects'

const json = (body: unknown): RequestInit => ({ body: JSON.stringify(body), headers: { 'Content-Type': 'application/json' } })

export const projectsApi = {
  list: () => request<ProjectSummary[]>('/projects'),
  get: (id: string) => request<ProjectRead>(`/projects/${id}`),
  create: (name: string, settings: ProjectSettings) => request<ProjectRead>('/projects', { method: 'POST', ...json({ name, settings }) }),
  update: (id: string, patch: { name?: string; description?: string | null; settings?: ProjectSettings }) =>
    request<ProjectRead>(`/projects/${id}`, { method: 'PATCH', ...json(patch) }),
  remove: (id: string) => request<void>(`/projects/${id}`, { method: 'DELETE' }),
  importScript: (id: string, text: string, split: 'paragraphs' | 'sentences') =>
    request<ProjectRead>(`/projects/${id}/import`, { method: 'POST', ...json({ text, split }) }),
  addSegment: (id: string, text: string, position?: number) =>
    request<ProjectRead>(`/projects/${id}/segments`, { method: 'POST', ...json({ segments: [{ text }], position }) }),
  updateSegment: (id: string, segmentId: string, patch: SegmentPatch) =>
    request<ProjectRead>(`/projects/${id}/segments/${segmentId}`, { method: 'PATCH', ...json(patch) }),
  removeSegment: (id: string, segmentId: string) => request<ProjectRead>(`/projects/${id}/segments/${segmentId}`, { method: 'DELETE' }),
  reorder: (id: string, ids: string[]) => request<ProjectRead>(`/projects/${id}/segments/order`, { method: 'PUT', ...json({ ids }) }),
  generate: (id: string, segmentIds: string[] | null, onlyPending: boolean) =>
    request<ProjectRead>(`/projects/${id}/generate`, { method: 'POST', ...json({ segment_ids: segmentIds, only_pending: onlyPending }) }),
  subtitlesUrl: (id: string, format: 'srt' | 'vtt') => `/api/projects/${id}/subtitles?format=${format}&download=true`,
  exportUrl: (id: string, format: 'wav' | 'mp3' | 'ogg' | 'flac' | 'zip', options: { download?: boolean; partial?: boolean; bust?: number } = {}) =>
    `/api/projects/${id}/export?format=${format}&download=${options.download ?? true}&allow_partial=${options.partial ?? false}${options.bust ? `&t=${options.bust}` : ''}`,
}
