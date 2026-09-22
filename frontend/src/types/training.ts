export interface TrainingReferenceSummary {
  reference_id: string
  name: string
  duration_s: number
  transcribed: boolean
  clips: number
  usable_s: number
  discarded: Record<string, number>
}

export interface TrainingDataset {
  profile_id: string
  references: TrainingReferenceSummary[]
  clips: number
  usable_minutes: number
  recorded_minutes: number
  missing_transcripts: string[]
  min_minutes: number
  recommended_minutes: [number, number]
  ready: boolean
  warnings: string[]
}

export type TrainingStatus =
  | 'queued'
  | 'preparing'
  | 'training'
  | 'saving'
  | 'evaluating'
  | 'completed'
  | 'failed'
  | 'cancelled'

export const ACTIVE_TRAINING: TrainingStatus[] = ['queued', 'preparing', 'training', 'saving', 'evaluating']

export interface TrainingSystemScore {
  similarity?: number
  wer?: number
}

export interface TrainingEvaluation {
  held_out?: number
  texts?: string[]
  systems?: Partial<Record<'trained' | 'normal' | 'real', TrainingSystemScore>>
  verdict?: string
  missing?: string[]
  skipped?: string
}

export interface TrainingEpoch {
  epoch: number
  talker_loss: number
  predictor_loss: number
}

export interface TrainingRun {
  id: string
  profile_id: string
  profile_name: string | null
  engine: string
  base_variant: string
  name: string
  status: TrainingStatus
  progress: number
  message: string | null
  params: Record<string, unknown> | null
  dataset: Record<string, unknown> | null
  history: TrainingEpoch[] | null
  evaluation: TrainingEvaluation | null
  checkpoint_id: string | null
  variant: string | null
  error_code: string | null
  error_detail: string | null
  created_at: string
  started_at: string | null
  finished_at: string | null
}

export interface TrainingInput {
  profile_id: string
  base_variant: 'base-1.7b' | 'base-0.6b'
  name?: string | null
  epochs: number
}
