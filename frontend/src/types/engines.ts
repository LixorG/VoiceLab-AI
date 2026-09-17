/** Mirrors backend app/engines/base.py and app/schemas/engines.py */

export type ControlSource = 'native' | 'instruction' | 'segmentation' | 'dsp' | 'unavailable'
export type GenericControl =
  | 'speed'
  | 'pitch'
  | 'emotion'
  | 'naturalness'
  | 'voice_color'
  | 'instruction'
  | 'seed'
  | 'target_duration'
  | 'language'
export type EngineMode = 'clone' | 'custom_voice' | 'voice_design'
export type PresetId = 'fast' | 'balanced' | 'high_fidelity'
export type ParamValue = string | number | boolean | null

export interface ControlCapability {
  source: ControlSource
  parameter: string | null
  reason: string | null
  available_from_phase: number | null
}

export interface LicenseInfo {
  code: string
  weights: string
  commercial_use: 'permitido' | 'no_permitido' | 'revisar'
  notes: string | null
  url: string
}

export interface EngineVariant {
  id: string
  label: string
  description: string
  mode: EngineMode
  repo_id: string | null
  vram_estimate_mb: number | null
  download_size_mb: number | null
}

export interface EngineCapabilities {
  variant: string
  mode: EngineMode
  sample_rate: number
  languages: string[]
  requires_reference_audio: boolean
  requires_reference_text: boolean
  reference_text_optional_reason: string | null
  reference_duration_s: [number, number] | null
  reference_strategies: string[]
  controls: Record<GenericControl, ControlCapability>
  supports_streaming: boolean
  supports_batch: boolean
  deterministic_seed: boolean
  reports_progress: boolean
  reference_text_not_required_when: Record<string, ParamValue> | null
  notes: string[]
}

export interface ParameterSpec {
  id: string
  maps_to: string | null
  type: 'slider' | 'number' | 'toggle' | 'select' | 'text' | 'seed'
  value_type: 'int' | 'float' | 'bool' | 'str'
  label: string
  tooltip: { what: string; effect: string; cost: string; typical: string }
  default: ParamValue
  min: number | null
  max: number | null
  step: number | null
  unit: string | null
  nullable: boolean
  options: { value: string | number | boolean; label: string; description: string | null }[] | null
  dynamic_options: boolean
  max_length: number | null
  group: 'basic' | 'advanced'
  control: GenericControl | null
  visible_if: Record<string, ParamValue> | null
  presets: Partial<Record<PresetId, ParamValue>>
}

export interface EngineSummary {
  id: string
  name: string
  description: string
  installed: boolean
  missing_packages: string[]
  implemented: boolean
  implementation_phase: number | null
  license: LicenseInfo
  variants: EngineVariant[]
  default_variant: string
}

export interface PresetInfo {
  id: PresetId
  label: string
  values: Record<string, ParamValue>
}

export interface EngineConfig {
  engine: EngineSummary
  variant: EngineVariant
  capabilities: EngineCapabilities
  parameters: ParameterSpec[]
  presets: PresetInfo[]
}
