import { t } from '@/i18n/es'
import type {
  ApiErrorBody,
  EnvironmentReport,
  HealthResponse,
  LicenseEntry,
  SystemInfo,
} from '@/types/api'

export class ApiError extends Error {
  readonly code: string
  readonly status: number
  readonly details: unknown

  constructor(code: string, message: string, status: number, details: unknown = null) {
    super(message)
    this.name = 'ApiError'
    this.code = code
    this.status = status
    this.details = details
  }
}

export function isErrorBody(value: unknown): value is ApiErrorBody {
  return (
    typeof value === 'object' &&
    value !== null &&
    typeof (value as ApiErrorBody).error_code === 'string' &&
    typeof (value as ApiErrorBody).message === 'string'
  )
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(`/api${path}`, {
      ...init,
      headers: { Accept: 'application/json', ...init?.headers },
    })
  } catch {
    throw new ApiError('NETWORK_ERROR', t.errors.network, 0)
  }

  if (res.status === 204) return undefined as T
  const body: unknown = await res.json().catch(() => null)
  if (!res.ok) {
    if (isErrorBody(body)) throw new ApiError(body.error_code, body.message, res.status, body.details)
    throw new ApiError('UNEXPECTED_ERROR', t.errors.unexpected, res.status)
  }
  return body as T
}

export const systemApi = {
  health: () => request<HealthResponse>('/system/health'),
  info: () => request<SystemInfo>('/system/info'),
  environment: () => request<EnvironmentReport>('/system/environment'),
  licenses: () => request<LicenseEntry[]>('/system/licenses'),
}
