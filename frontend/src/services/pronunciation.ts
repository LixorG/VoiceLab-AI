import { request } from '@/services/api'
import type { NormalizePreview, PronunciationEntry, PronunciationInput } from '@/types/pronunciation'

const json = (body: unknown): RequestInit => ({ body: JSON.stringify(body), headers: { 'Content-Type': 'application/json' } })

export const pronunciationApi = {
  list: (profileId: string | null) =>
    request<PronunciationEntry[]>(profileId ? `/pronunciation?profile_id=${encodeURIComponent(profileId)}` : '/pronunciation'),
  create: (body: PronunciationInput) => request<PronunciationEntry>('/pronunciation', { method: 'POST', ...json(body) }),
  update: (id: string, body: PronunciationInput) => request<PronunciationEntry>(`/pronunciation/${id}`, { method: 'PUT', ...json(body) }),
  remove: (id: string) => request<void>(`/pronunciation/${id}`, { method: 'DELETE' }),
  preview: (body: { text: string; language: string | null; profile_id: string | null }) =>
    request<NormalizePreview>('/pronunciation/preview', { method: 'POST', ...json(body) }),
}
