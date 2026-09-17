import { request } from '@/services/api'
import type { GenerationRequest } from '@/services/generation'
import type { GenerationAccepted } from '@/types/generation'
import type { BatchRequest, QueuedJob } from '@/types/jobs'

const json = (body: unknown): RequestInit => ({ body: JSON.stringify(body), headers: { 'Content-Type': 'application/json' } })

export const jobsApi = {
  list: () => request<QueuedJob[]>('/jobs'),
  cancel: (jobId: string) => request<unknown>(`/jobs/${jobId}/cancel`, { method: 'POST' }),
  cancelAll: (onlyQueued: boolean) => request<{ cancelled: string[] }>(`/jobs/cancel?only_queued=${onlyQueued}`, { method: 'POST' }),
  batch: (body: GenerationRequest & BatchRequest) =>
    request<{ items: GenerationAccepted[]; total: number }>('/generation/batch', { method: 'POST', ...json(body) }),
}
