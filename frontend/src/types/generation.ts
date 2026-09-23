import type { TextChange } from '@/types/pronunciation'
import type { ParamValue } from '@/types/engines'
import type { GenerationEvaluation, GenerationRating } from '@/types/experiments'
import type { PostProcessConfig, PostProcessReport } from '@/types/postprocess'

export type JobStatus =
  | 'QUEUED'
  | 'LOADING_MODEL'
  | 'PROCESSING_AUDIO'
  | 'GENERATING'
  | 'POST_PROCESSING'
  | 'COMPLETED'
  | 'FAILED'
  | 'CANCELLED'

export const TERMINAL_STATUSES: JobStatus[] = ['COMPLETED', 'FAILED', 'CANCELLED']

export interface PlannedSegment {
  index: number
  text: string
  emotion: string | null
  emotion_via: 'instruction' | 'reference' | null
  instruction: string | null
  pause_before_ms: number
  pause_after_ms: number
  reference_name: string | null
  seed?: number | null
  duration_s?: number | null
}

export interface SegmentRegenerate {
  /** Corrected words for that sentence; omitted = the same ones. */
  text?: string
  /** A specific seed; omitted = a new one at random. */
  seed?: number | null
  takes?: number
}

export interface GenerationPlan {
  segments: PlannedSegment[]
  warnings: string[]
  segmented: boolean
  text_changes: TextChange[]
  normalize_language: string | null
}

export interface TakeScore {
  take: number
  seed: number | null
  duration_s: number
  speech_ratio: number
  peak: number
  wer: number | null
  similarity: number | null
  score: number
  notes: string[]
}

export interface BestTake {
  chosen: number
  segment?: number
  takes: TakeScore[]
}

export interface GenerationRead {
  id: string
  kind: 'single' | 'preview' | string
  engine: string
  variant: string | null
  text: string
  params: Record<string, ParamValue> | null
  seed: number | null
  status: JobStatus
  progress: number
  progress_available: boolean
  /** Live-preview chunks ready so far: /api/generation/{id}/stream/{n}. */
  stream_chunks?: number
  message: string | null
  error_code: string | null
  reference: { reference_id: string; name: string | null; start_s: number | null; end_s: number | null; text: string } | null
  duration_s: number | null
  audio_url: string | null
  metrics: {
    model_load_s?: number
    generation_s?: number
    total_s?: number
    rtf?: number | null
    device?: string | null
    takes?: number
    best_take?: BestTake[]
    regenerated?: { segment: number; seed: number | null; takes: number; at: string; best_take?: BestTake }[]
  } | null
  warnings: string[]
  expression: { emotion: string | null; intensity: number; markup: boolean; normalize?: boolean } | null
  takes?: number
  parent_id: string | null
  segments: PlannedSegment[]
  postprocess: { config: PostProcessConfig; report: PostProcessReport | null } | null
  raw_audio_url: string | null
  experiment_id?: string | null
  label?: string | null
  rating?: GenerationRating | null
  evaluation?: GenerationEvaluation | null
  created_at: string
  updated_at: string
}

export interface GenerationAccepted {
  generation: GenerationRead
  job_id: string
  queue_position: number
}

export interface JobEvent {
  job_id: string
  status: JobStatus
  progress: number
  message: string | null
  chunks?: number
}

export interface EngineRuntimeStatus {
  engine: string
  variant: string
  package_installed: boolean
  missing_packages: string[]
  weights_installed: boolean
  download_state: 'idle' | 'downloading' | 'failed'
  download_error: string | null
  loaded: boolean
  device: string | null
}
