import { request } from '@/services/api'
import type { CheckpointInput, CustomCheckpoint } from '@/types/checkpoints'

const json = (body: unknown): RequestInit => ({ body: JSON.stringify(body), headers: { 'Content-Type': 'application/json' } })

export const checkpointsApi = {
  list: () => request<CustomCheckpoint[]>('/models/checkpoints'),
  create: (engineId: string, body: CheckpointInput) =>
    request<CustomCheckpoint>(`/models/${engineId}/checkpoints`, { method: 'POST', ...json(body) }),
  remove: (id: string) => request<void>(`/models/checkpoints/${id}`, { method: 'DELETE' }),
}
