export interface PostProcessConfig {
  crossfade_ms: number | null
  denoise: { enabled: boolean; strength: 'light' | 'medium' | 'strong' }
  trim_silence: { enabled: boolean; threshold_db: number; padding_ms: number }
  time_stretch: { enabled: boolean; rate: number }
  pitch_shift: { enabled: boolean; semitones: number; preserve_formants: boolean }
  loudness: { enabled: boolean; target_lufs: number }
  peak: { enabled: boolean; target_dbfs: number }
  fades: { enabled: boolean; fade_in_ms: number; fade_out_ms: number }
}

export type ProcessorId = 'denoise' | 'trim_silence' | 'time_stretch' | 'pitch_shift' | 'loudness' | 'peak' | 'fades'

export interface LevelStats {
  duration_s: number
  peak_dbfs: number | null
  loudness_lufs: number | null
}

export interface PostProcessReport {
  steps: { id: string; label: string; detail: string }[]
  warnings: string[]
  before: LevelStats
  after: LevelStats
}

export interface PostProcessCapabilities {
  processors: Record<string, boolean>
  reasons: Record<string, string>
}

/** Mirrors the backend defaults: everything off. */
export const DEFAULT_POSTPROCESS: PostProcessConfig = {
  crossfade_ms: null,
  denoise: { enabled: false, strength: 'light' },
  trim_silence: { enabled: false, threshold_db: -45, padding_ms: 100 },
  time_stretch: { enabled: false, rate: 1 },
  pitch_shift: { enabled: false, semitones: 0, preserve_formants: true },
  loudness: { enabled: false, target_lufs: -16 },
  peak: { enabled: false, target_dbfs: -1 },
  fades: { enabled: false, fade_in_ms: 10, fade_out_ms: 30 },
}

/** Conservative mastering that does not alter timbre or timing of the voice. */
export const BASIC_MASTERING: PostProcessConfig = {
  ...DEFAULT_POSTPROCESS,
  trim_silence: { ...DEFAULT_POSTPROCESS.trim_silence, enabled: true },
  loudness: { ...DEFAULT_POSTPROCESS.loudness, enabled: true },
  peak: { ...DEFAULT_POSTPROCESS.peak, enabled: true },
  fades: { ...DEFAULT_POSTPROCESS.fades, enabled: true },
}

export const PROCESSORS: ProcessorId[] = ['denoise', 'trim_silence', 'time_stretch', 'pitch_shift', 'loudness', 'peak', 'fades']

export function isPostprocessActive(config: PostProcessConfig): boolean {
  return config.crossfade_ms != null || PROCESSORS.some((p) => config[p].enabled)
}
