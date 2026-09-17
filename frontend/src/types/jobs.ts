import type { JobStatus } from '@/types/generation'

export interface QueuedJob {
  job_id: string
  kind: string
  status: JobStatus
  progress: number
  message: string | null
  position: number
  text: string | null
  engine: string | null
  variant: string | null
  generation_kind: string | null
}

export interface BatchRequest {
  texts: string[]
  variants: string[]
  repeat: number
}
