import { request } from '@/services/api'
import type { ParamValue } from '@/types/engines'
import type { PostProcessCapabilities, PostProcessConfig } from '@/types/postprocess'
import type { EngineRuntimeStatus, GenerationAccepted, GenerationPlan, GenerationRead, JobEvent, SegmentRegenerate } from '@/types/generation'

export interface GenerationRequest {
  engine: string
  variant: string | null
  text: string
  params: Record<string, ParamValue>
  reference_id: string | null
  profile_id: string | null
  preview: boolean
  emotion: string | null
  intensity: number
  markup: boolean
  normalize?: boolean
  takes?: number
  postprocess: PostProcessConfig | null
}

const json = (body: unknown): RequestInit => ({
  body: JSON.stringify(body),
  headers: { 'Content-Type': 'application/json' },
})

export const generationApi = {
  create: (body: GenerationRequest) => request<GenerationAccepted>('/generation', { method: 'POST', ...json(body) }),
  plan: (body: GenerationRequest) => request<GenerationPlan>('/generation/plan', { method: 'POST', ...json(body) }),
  variations: (body: GenerationRequest, count: number) =>
    request<GenerationAccepted[]>('/generation/variations', { method: 'POST', ...json({ ...body, count }) }),
  postprocess: (id: string, config: PostProcessConfig) =>
    request<GenerationRead>(`/generation/${id}/postprocess`, { method: 'POST', ...json(config) }),
  clearPostprocess: (id: string) => request<GenerationRead>(`/generation/${id}/postprocess`, { method: 'DELETE' }),
  postprocessCapabilities: () => request<PostProcessCapabilities>('/generation/postprocess/capabilities'),
  list: (limit = 30) => request<GenerationRead[]>(`/generation?limit=${limit}`),
  get: (id: string) => request<GenerationRead>(`/generation/${id}`),
  remove: (id: string) => request<void>(`/generation/${id}`, { method: 'DELETE' }),
  regenerateSegment: (id: string, index: number, body: SegmentRegenerate) =>
    request<GenerationAccepted>(`/generation/${id}/segments/${index}/regenerate`, { method: 'POST', ...json(body) }),
  cancel: (jobId: string) => request<unknown>(`/jobs/${jobId}/cancel`, { method: 'POST' }),
  engineStatus: (engine: string, variant: string | null) =>
    request<EngineRuntimeStatus>(`/models/${engine}/status${variant ? `?variant=${encodeURIComponent(variant)}` : ''}`),
  downloadWeights: (engine: string, variant: string | null) =>
    request<EngineRuntimeStatus>(`/models/${engine}/download${variant ? `?variant=${encodeURIComponent(variant)}` : ''}`, {
      method: 'POST',
    }),
}

/** Subscribe to Server-Sent Events for a job. Returns an unsubscribe function. */
export function subscribeToJob(jobId: string, onEvent: (event: JobEvent) => void, onClose: () => void): () => void {
  if (typeof EventSource === 'undefined') {
    onClose()
    return () => undefined
  }
  const source = new EventSource(`/api/jobs/${jobId}/events`)
  const close = () => {
    source.close()
    onClose()
  }
  source.addEventListener('progress', (e) => {
    const event = JSON.parse((e as MessageEvent<string>).data) as JobEvent
    onEvent(event)
    if (['COMPLETED', 'FAILED', 'CANCELLED'].includes(event.status)) close()
  })
  source.onerror = () => {
    // The server closes the stream after a terminal event; any other error falls back to polling.
    close()
  }
  return () => source.close()
}
