import { request } from '@/services/api'
import type { EvaluatorInfo, ExperimentArm, ExperimentCreate, ExperimentRead, ExperimentSummary, GenerationRating } from '@/types/experiments'
import type { GenerationRead } from '@/types/generation'

const json = (body: unknown): RequestInit => ({
  body: JSON.stringify(body),
  headers: { 'Content-Type': 'application/json' },
})

export const experimentsApi = {
  list: () => request<ExperimentSummary[]>('/experiments'),
  get: (id: string) => request<ExperimentRead>(`/experiments/${id}`),
  create: (body: ExperimentCreate) => request<ExperimentRead>('/experiments', { method: 'POST', ...json(body) }),
  addArms: (id: string, arms: ExperimentArm[]) => request<ExperimentRead>(`/experiments/${id}/arms`, { method: 'POST', ...json({ arms }) }),
  update: (id: string, patch: { name?: string; notes?: string }) => request<ExperimentRead>(`/experiments/${id}`, { method: 'PATCH', ...json(patch) }),
  evaluateAll: (id: string) => request<ExperimentRead>(`/experiments/${id}/evaluate`, { method: 'POST' }),
  remove: (id: string) => request<void>(`/experiments/${id}`, { method: 'DELETE' }),
}

export const evaluationApi = {
  evaluators: () => request<EvaluatorInfo[]>('/generation/evaluators'),
  evaluate: (generationId: string) => request<GenerationRead>(`/generation/${generationId}/evaluate`, { method: 'POST' }),
  rate: (generationId: string, rating: GenerationRating) =>
    request<GenerationRead>(`/generation/${generationId}/rating`, { method: 'PUT', ...json(rating) }),
}
