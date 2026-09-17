import { request } from '@/services/api'

export interface MemoryStatus {
  gpu: { name: string; total_mb: number; free_mb: number; used_mb: number; torch_allocated_mb: number; torch_reserved_mb: number } | null
  process_ram_mb: number | null
  system_ram_total_mb: number | null
  system_ram_available_mb: number | null
  tts: { engine: string; variant: string; device: string; loaded_at: string; last_used_at: string | null; in_use: boolean; idle_unload_in_s: number | null } | null
  asr: { model: string; loaded: boolean; device: string | null; last_used_at: string | null; idle_unload_in_s: number | null }
  idle_unload_minutes: { tts: number; asr: number }
  queue_active: number
  queue_waiting: number
}

export interface ReleaseResult {
  released: string[]
  skipped: Record<string, string>
  status: MemoryStatus
}

export interface StorageCategory {
  id: string
  label: string
  bytes: number
  files: number
  removable: boolean
  description: string
}

export interface StorageUsage {
  categories: StorageCategory[]
  orphans_bytes: number
  orphans_files: number
  total_bytes: number
}

export interface CleanupOptions {
  orphans: boolean
  temp: boolean
  reference_clips: boolean
  transcript_cache: boolean
}

export interface CleanupReport {
  removed_files: number
  freed_bytes: number
  details: Record<string, number>
  skipped: string[]
}

export interface JobState {
  job_id: string
  status: string
  message: string | null
}

const json = (body: unknown): RequestInit => ({ body: JSON.stringify(body), headers: { 'Content-Type': 'application/json' } })

export const resourcesApi = {
  memory: () => request<MemoryStatus>('/system/memory'),
  release: (tts: boolean, asr: boolean) => request<ReleaseResult>('/system/memory/release', { method: 'POST', ...json({ tts, asr }) }),
  storage: () => request<StorageUsage>('/system/storage'),
  cleanup: (options: CleanupOptions) => request<CleanupReport>('/system/storage/cleanup', { method: 'POST', ...json(options) }),
  preload: (engine: string, variant: string | null) =>
    request<JobState>(`/models/${engine}/load${variant ? `?variant=${encodeURIComponent(variant)}` : ''}`, { method: 'POST' }),
  job: (id: string) => request<JobState>(`/jobs/${id}`),
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  const units = ['KB', 'MB', 'GB', 'TB']
  let value = bytes / 1024
  let i = 0
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024
    i++
  }
  return `${value.toFixed(value >= 100 ? 0 : 1)} ${units[i]}`
}
