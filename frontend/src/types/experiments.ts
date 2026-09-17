import type { ParamValue } from '@/types/engines'
import type { GenerationRead } from '@/types/generation'
import type { PostProcessConfig } from '@/types/postprocess'

export interface Metric {
  id: string
  label: string
  value: number
  display: string
  better: 'higher' | 'lower'
  description: string
}

export interface EvaluatorInfo {
  id: string
  label: string
  description: string
  available: boolean
  reason: string | null
}

export interface GenerationEvaluation {
  metrics: Metric[]
  details: Record<string, { transcript?: string; language?: string; asr_model?: string }>
  target_text: string
  unavailable: EvaluatorInfo[]
  evaluated_at: string
  stale: boolean
}

export type RatingCriterion = 'timbre' | 'naturalness' | 'pronunciation' | 'emotion'
export const RATING_CRITERIA: RatingCriterion[] = ['timbre', 'naturalness', 'pronunciation', 'emotion']

export type GenerationRating = Partial<Record<RatingCriterion, number>> & { notes?: string; rated_at?: string }

export interface ExperimentArm {
  engine: string
  variant: string | null
  params: Record<string, ParamValue>
  label?: string | null
}

export interface ExperimentCreate {
  name: string | null
  text: string
  profile_id: string | null
  reference_id: string | null
  emotion: string | null
  intensity: number
  markup: boolean
  postprocess: PostProcessConfig | null
  arms: ExperimentArm[]
}

export interface ExperimentSummary {
  id: string
  name: string
  text: string
  arms: number
  completed: number
  running: number
  failed: number
  engines: string[]
  created_at: string
  updated_at: string
}

export interface ExperimentRead {
  id: string
  name: string
  text: string
  profile_id: string | null
  reference_id: string | null
  notes: string | null
  settings: Record<string, unknown> | null
  generations: GenerationRead[]
  created_at: string
  updated_at: string
}
