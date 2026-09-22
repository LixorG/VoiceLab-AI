import { request } from '@/services/api'
import type { TrainingDataset, TrainingInput, TrainingRun } from '@/types/training'

const json = (body: unknown): RequestInit => ({ body: JSON.stringify(body), headers: { 'Content-Type': 'application/json' } })

export const trainingApi = {
  dataset: (profileId: string) => request<TrainingDataset>(`/training/dataset/${profileId}`),
  list: (profileId: string) => request<TrainingRun[]>(`/training?profile_id=${encodeURIComponent(profileId)}`),
  start: (body: TrainingInput) => request<TrainingRun>('/training', { method: 'POST', ...json(body) }),
  cancel: (id: string) => request<TrainingRun>(`/training/${id}/cancel`, { method: 'POST' }),
  remove: (id: string) => request<void>(`/training/${id}`, { method: 'DELETE' }),
  sampleUrl: (id: string, index: number, kind: 'real' | 'trained' | 'normal') => `/api/training/${id}/samples/${index}_${kind}.wav`,
}
