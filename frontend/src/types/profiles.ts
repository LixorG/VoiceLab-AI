import type { ParamValue } from '@/types/engines'

export interface RecommendedSetting {
  variant: string
  params: Record<string, ParamValue>
  note: string | null
  updated_at: string | null
}

export interface ProfileStats {
  reference_count: number
  analyzed_count: number
  transcribed_count: number
  total_duration_s: number
  total_speech_s: number
  average_quality: number | null
  quality_label: string | null
  emotions: string[]
}

export interface ProfileRead {
  id: string
  name: string
  slug: string
  description: string | null
  language: string | null
  default_engine: string | null
  primary_reference_id: string | null
  recommended_reference_id: string | null
  recommended_settings: Record<string, RecommendedSetting>
  stats: ProfileStats
  created_at: string
  updated_at: string
}

export interface ProfileInput {
  name?: string
  description?: string | null
  language?: string | null
  default_engine?: string | null
  primary_reference_id?: string | null
}

export interface EmotionOption {
  id: string
  label: string
}
