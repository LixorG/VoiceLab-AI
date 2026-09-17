/** Mirrors backend Pydantic schemas (app/api/routes/system.py, app/core/*). */

export interface ApiErrorBody {
  error_code: string
  message: string
  details: unknown
}

export interface HealthResponse {
  status: string
  version: string
}

export interface SystemInfo {
  version: string
  app_env: string
  device: string
  default_model: string
  max_upload_mb: number
  max_audio_seconds: number
  data_dir: string
  model_dir: string
}

export interface GPUDevice {
  index: number
  name: string
  total_memory_mb: number | null
  free_memory_mb: number | null
}

export interface GPUInfo {
  backend: 'cuda' | 'rocm' | 'mps' | 'cpu'
  devices: GPUDevice[]
  torch_available: boolean
  torch_version: string | null
  cuda_version: string | null
  rocm_version: string | null
  driver_version: string | null
  bf16_supported: boolean | null
  source: 'torch' | 'nvidia-smi' | 'none'
}

export type CheckStatus = 'ok' | 'warning' | 'error' | 'missing'

export interface EnvironmentCheck {
  id: string
  label: string
  status: CheckStatus
  detail: string | null
}

export interface EnvironmentReport {
  platform: string
  python_version: string
  gpu: GPUInfo
  checks: EnvironmentCheck[]
}

export interface LicenseEntry {
  component: string
  category: string
  code_license: string | null
  weights_license: string | null
  commercial_use: 'permitido' | 'no_permitido' | 'revisar'
  notes: string | null
  url: string
}

export type ReferenceStatus = 'UPLOADED' | 'PROCESSING' | 'ANALYZED' | 'TRANSCRIBED' | 'READY' | 'FAILED'

export interface TimeRegion {
  start_s: number
  end_s: number
}

export interface SuggestedSegment extends TimeRegion {
  score: number
}

export interface QualityComponent {
  id: string
  label: string
  score: number
  weight: number
}

/** Summary stored with the reference (speech_regions omitted). */
export interface ReferenceAnalysis {
  duration_s: number
  sample_rate: number
  peak_dbfs: number
  rms_dbfs: number
  loudness_lufs: number | null
  clipping_ratio: number
  clipped_samples: number
  noise_floor_dbfs: number
  speech_level_dbfs: number
  snr_db: number | null
  speech_ratio: number
  silence_ratio: number
  effective_speech_s: number
  speech_detected: boolean
  suggested_segments: SuggestedSegment[]
  quality_score: number
  quality_label: string
  quality_components: QualityComponent[]
  warnings: string[]
  normalization_gain_db: number
  source_codec: string
  source_bit_rate: number | null
}

export interface ReferenceRead {
  id: string
  profile_id: string | null
  original_name: string
  format: string
  size_bytes: number
  sample_rate: number | null
  channels: number | null
  duration_s: number | null
  status: ReferenceStatus
  quality_score: number | null
  quality_label: string | null
  analysis: ReferenceAnalysis | { error_code: string } | null
  segment_start_s: number | null
  segment_end_s: number | null
  emotion_tag: string | null
  is_recommended: boolean
  is_primary: boolean
  created_at: string
  urls: { original: string; processed: string | null; peaks: string | null }
  transcript: TranscriptRead | null
  segment_transcript: TranscriptRead | null
  segment_text_estimate: string | null
}

export interface UploadResponse {
  reference: ReferenceRead
  duplicate: boolean
}

export interface PeaksResponse {
  buckets: number
  duration_s: number
  sample_rate: number
  peaks: number[]
}

export interface AudioFormats {
  extensions: string[]
  max_upload_mb: number
  max_audio_seconds: number
  internal_sample_rate: number
}

export function hasAnalysis(ref: ReferenceRead): ref is ReferenceRead & { analysis: ReferenceAnalysis } {
  return ref.analysis != null && 'quality_score' in ref.analysis
}

export interface TranscriptWord {
  start: number
  end: number
  word: string
  probability: number
}

export interface TranscriptRead {
  id: string
  reference_id: string
  text: string
  language: string | null
  source: 'asr' | 'manual'
  edited: boolean
  asr_model: string | null
  segment_start_s: number | null
  segment_end_s: number | null
  confidence: {
    language_probability: number | null
    avg_logprob: number | null
    max_no_speech_prob: number | null
    mean_word_probability: number | null
  }
  warnings: string[]
  words: TranscriptWord[] | null
  updated_at: string
}

export interface ASRStatus {
  engine: string
  model: string
  package_available: boolean
  installed: boolean
  loaded: boolean
  device: string | null
  compute_type: string | null
  download_state: 'idle' | 'downloading' | 'failed'
  download_error: string | null
  approx_size_mb: number | null
}

export interface LanguageOption {
  code: string
  name: string
}
