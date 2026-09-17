import type { ParamValue } from '@/types/engines'
import type { GenerationRead } from '@/types/generation'
import type { PostProcessConfig } from '@/types/postprocess'

export type SegmentStatus = 'empty' | 'queued' | 'generating' | 'ready' | 'stale' | 'failed' | 'cancelled'

export interface ProjectSettings {
  engine: string | null
  variant: string | null
  params: Record<string, ParamValue>
  profile_id: string | null
  reference_id: string | null
  markup: boolean
  default_pause_ms: number
  export_postprocess: PostProcessConfig | null
}

export interface ProjectSegment {
  id: string
  position: number
  text: string
  profile_id: string | null
  reference_id: string | null
  emotion: string | null
  intensity: number | null
  pause_after_ms: number | null
  seed: number | null
  effective_pause_ms: number
  status: SegmentStatus
  generation: GenerationRead | null
}

export interface ProjectRead {
  id: string
  name: string
  description: string | null
  settings: ProjectSettings
  segments: ProjectSegment[]
  total_duration_s: number
  exported_at: string | null
  created_at: string
  updated_at: string
}

export interface ProjectSummary {
  id: string
  name: string
  description: string | null
  segments: number
  ready: number
  pending: number
  running: number
  failed: number
  updated_at: string
}

export type SegmentPatch = Partial<Pick<ProjectSegment, 'text' | 'profile_id' | 'reference_id' | 'emotion' | 'intensity' | 'pause_after_ms' | 'seed'>>

export const EMPTY_SETTINGS: ProjectSettings = {
  engine: null,
  variant: null,
  params: {},
  profile_id: null,
  reference_id: null,
  markup: true,
  default_pause_ms: 400,
  export_postprocess: null,
}
