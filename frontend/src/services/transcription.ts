import { request } from '@/services/api'
import type { ASRStatus, LanguageOption, TranscriptRead } from '@/types/api'

const json = (body: unknown): RequestInit => ({
  body: JSON.stringify(body),
  headers: { 'Content-Type': 'application/json' },
})

export const transcriptionApi = {
  status: () => request<ASRStatus>('/transcription/status'),
  download: () => request<ASRStatus>('/transcription/model/download', { method: 'POST' }),
  unload: () => request<ASRStatus>('/transcription/model/unload', { method: 'POST' }),
  languages: () => request<LanguageOption[]>('/transcription/languages'),
  transcribe: (referenceId: string, body: { scope: 'full' | 'segment'; language: string | null; force?: boolean }) =>
    request<TranscriptRead>(`/transcription/references/${referenceId}`, { method: 'POST', ...json(body) }),
  edit: (transcriptId: string, text: string) =>
    request<TranscriptRead>(`/transcription/transcripts/${transcriptId}`, { method: 'PATCH', ...json({ text }) }),
  revert: (transcriptId: string) =>
    request<TranscriptRead>(`/transcription/transcripts/${transcriptId}/revert`, { method: 'POST' }),
}
