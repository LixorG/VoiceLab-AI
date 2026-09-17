import { request } from '@/services/api'
import type { EngineConfig, EngineSummary, ParamValue } from '@/types/engines'

export const modelsApi = {
  list: () => request<EngineSummary[]>('/models'),
  config: (engineId: string, variant?: string | null) =>
    request<EngineConfig>(`/models/${engineId}${variant ? `?variant=${encodeURIComponent(variant)}` : ''}`),
  validate: (engineId: string, variant: string, params: Record<string, ParamValue>) =>
    request<{ variant: string; params: Record<string, ParamValue> }>(`/models/${engineId}/validate`, {
      method: 'POST',
      body: JSON.stringify({ variant, params }),
      headers: { 'Content-Type': 'application/json' },
    }),
}
