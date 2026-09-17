import { t } from '@/i18n/es'
import { ApiError, isErrorBody, request } from '@/services/api'
import type { AudioFormats, PeaksResponse, ReferenceRead, UploadResponse } from '@/types/api'

export interface ReferencePatch {
  segment_start_s?: number | null
  segment_end_s?: number | null
  emotion_tag?: string | null
  snap_to_words?: boolean
}

const json = (body: unknown): RequestInit => ({
  body: JSON.stringify(body),
  headers: { 'Content-Type': 'application/json' },
})

/** XHR instead of fetch: fetch cannot report upload progress. */
export function uploadReference(
  file: File,
  onProgress: (fraction: number) => void,
  profileId?: string,
): Promise<UploadResponse> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    const form = new FormData()
    form.append('file', file, file.name)
    if (profileId) form.append('profile_id', profileId)

    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total)
    xhr.upload.onload = () => onProgress(1)
    xhr.onerror = () => reject(new ApiError('NETWORK_ERROR', t.errors.network, 0))
    xhr.onload = () => {
      let body: unknown = null
      try {
        body = JSON.parse(xhr.responseText)
      } catch {
        /* non-JSON response */
      }
      if (xhr.status >= 200 && xhr.status < 300) return resolve(body as UploadResponse)
      if (isErrorBody(body)) return reject(new ApiError(body.error_code, body.message, xhr.status, body.details))
      reject(new ApiError('UNEXPECTED_ERROR', t.errors.unexpected, xhr.status))
    }
    xhr.open('POST', '/api/references')
    xhr.setRequestHeader('Accept', 'application/json')
    xhr.send(form)
  })
}

export const referencesApi = {
  list: (profileId?: string) =>
    request<ReferenceRead[]>(`/references${profileId ? `?profile_id=${encodeURIComponent(profileId)}` : ''}`),
  update: (id: string, patch: ReferencePatch) =>
    request<ReferenceRead>(`/references/${id}`, { method: 'PATCH', ...json(patch) }),
  reanalyze: (id: string) => request<ReferenceRead>(`/references/${id}/reanalyze`, { method: 'POST' }),
  remove: (id: string) => request<void>(`/references/${id}`, { method: 'DELETE' }),
  peaks: (id: string, buckets: number) => request<PeaksResponse>(`/audio/references/${id}/peaks?buckets=${buckets}`),
  formats: () => request<AudioFormats>('/audio/formats'),
}
