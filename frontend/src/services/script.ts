import { request } from '@/services/api'
import type { ScriptPrepared, ScriptPrepareInput } from '@/types/script'

export const scriptApi = {
  prepare: (body: ScriptPrepareInput) =>
    request<ScriptPrepared>('/script/prepare', { method: 'POST', body: JSON.stringify(body), headers: { 'Content-Type': 'application/json' } }),
}
