import { t } from '@/i18n/es'
import { ApiError, isErrorBody, request } from '@/services/api'
import type { ParamValue } from '@/types/engines'
import type { EmotionOption, ProfileInput, ProfileRead } from '@/types/profiles'

const json = (body: unknown): RequestInit => ({
  body: JSON.stringify(body),
  headers: { 'Content-Type': 'application/json' },
})

export const profilesApi = {
  list: () => request<ProfileRead[]>('/voices'),
  get: (id: string) => request<ProfileRead>(`/voices/${id}`),
  create: (input: ProfileInput & { name: string }) => request<ProfileRead>('/voices', { method: 'POST', ...json(input) }),
  update: (id: string, input: ProfileInput) => request<ProfileRead>(`/voices/${id}`, { method: 'PATCH', ...json(input) }),
  remove: (id: string, deleteReferences: boolean) =>
    request<void>(`/voices/${id}?delete_references=${deleteReferences}`, { method: 'DELETE' }),
  setRecommended: (id: string, engine: string, variant: string | null, params: Record<string, ParamValue>) =>
    request<ProfileRead>(`/voices/${id}/settings/${engine}`, { method: 'PUT', ...json({ variant, params }) }),
  removeRecommended: (id: string, engine: string) => request<ProfileRead>(`/voices/${id}/settings/${engine}`, { method: 'DELETE' }),
  emotions: () => request<EmotionOption[]>('/voices/emotions'),
  exportUrl: (id: string) => `/api/voices/${id}/export`,
  import: async (file: File): Promise<ProfileRead> => {
    const form = new FormData()
    form.append('file', file, file.name)
    let res: Response
    try {
      res = await fetch('/api/voices/import', { method: 'POST', body: form, headers: { Accept: 'application/json' } })
    } catch {
      throw new ApiError('NETWORK_ERROR', t.errors.network, 0)
    }
    const body: unknown = await res.json().catch(() => null)
    if (!res.ok) {
      if (isErrorBody(body)) throw new ApiError(body.error_code, body.message, res.status, body.details)
      throw new ApiError('UNEXPECTED_ERROR', t.errors.unexpected, res.status)
    }
    return body as ProfileRead
  },
}
