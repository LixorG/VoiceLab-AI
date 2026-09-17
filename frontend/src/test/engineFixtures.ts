import type { ControlCapability, EngineConfig, EngineSummary, GenericControl, ParameterSpec } from '@/types/engines'

const tip = { what: 'Qué hace X.', effect: 'Efecto X.', cost: 'Sin coste.', typical: '1.0' }

const spec = (over: Partial<ParameterSpec> & Pick<ParameterSpec, 'id' | 'type' | 'value_type' | 'label'>): ParameterSpec => ({
  maps_to: over.id,
  tooltip: tip,
  default: null,
  min: null,
  max: null,
  step: null,
  unit: null,
  nullable: false,
  options: null,
  dynamic_options: false,
  max_length: null,
  group: 'advanced',
  control: null,
  visible_if: null,
  presets: {},
  ...over,
})

const cap = (source: ControlCapability['source'], parameter: string | null = null, reason: string | null = null, phase: number | null = null): ControlCapability => ({
  source,
  parameter,
  reason,
  available_from_phase: phase,
})

const license = { code: 'MIT', weights: 'CC-BY-NC-4.0', commercial_use: 'no_permitido' as const, notes: null, url: 'https://example.org' }

export const f5Summary: EngineSummary = {
  id: 'f5tts',
  name: 'F5-TTS',
  description: 'Clonación por flow matching.',
  installed: false,
  missing_packages: ['f5-tts'],
  implemented: false,
  implementation_phase: 5,
  license,
  variants: [{ id: 'F5TTS_v1_Base', label: 'F5-TTS v1 Base', description: 'Checkpoint v1.', mode: 'clone', repo_id: 'SWivid/F5-TTS', vram_estimate_mb: 3000, download_size_mb: 1400, source: 'builtin', base_variant: null, languages: null }],
  default_variant: 'F5TTS_v1_Base',
  supports_custom_checkpoints: true,
}

export const qwenSummary: EngineSummary = {
  ...f5Summary,
  id: 'qwen3tts',
  name: 'Qwen3-TTS',
  description: 'Modelo de lenguaje de voz.',
  implementation_phase: 6,
  license: { ...license, weights: 'Apache-2.0', code: 'Apache-2.0', commercial_use: 'permitido' },
  variants: [
    { id: 'base-1.7b', label: 'Clonación · 1.7B', description: 'Clona.', mode: 'clone', repo_id: 'Qwen/Qwen3-TTS-12Hz-1.7B-Base', vram_estimate_mb: 6000, download_size_mb: 4500, source: 'builtin', base_variant: null, languages: null },
    { id: 'custom-voice-1.7b', label: 'Voces predefinidas · 1.7B', description: 'Voces incluidas.', mode: 'custom_voice', repo_id: null, vram_estimate_mb: 6000, download_size_mb: 4500, source: 'builtin', base_variant: null, languages: null },
  ],
  default_variant: 'base-1.7b',
}

const baseControls = (over: Partial<Record<GenericControl, ControlCapability>>): Record<GenericControl, ControlCapability> => ({
  speed: cap('unavailable', null, 'No.'),
  pitch: cap('unavailable', null, 'No.'),
  emotion: cap('unavailable', null, 'No.'),
  naturalness: cap('unavailable', null, 'No existe un parámetro de naturalidad en este modelo.'),
  voice_color: cap('unavailable', null, 'No.'),
  instruction: cap('unavailable', null, 'Este modelo no acepta instrucciones de estilo en lenguaje natural.'),
  seed: cap('native', 'seed'),
  target_duration: cap('unavailable', null, 'No.'),
  language: cap('unavailable', null, 'No.'),
  ...over,
})

const capabilities = (variant: string, controls: Record<GenericControl, ControlCapability>, mode: EngineConfig['capabilities']['mode'] = 'clone') => ({
  variant,
  mode,
  sample_rate: 24000,
  languages: ['en', 'zh'],
  requires_reference_audio: mode === 'clone',
  requires_reference_text: mode === 'clone',
  reference_text_optional_reason: null,
  reference_duration_s: [3, 12] as [number, number],
  reference_strategies: ['best_reference'],
  controls,
  supports_streaming: false,
  supports_batch: false,
  deterministic_seed: true,
  reports_progress: true,
  reference_text_not_required_when: null,
  notes: ['La transcripción debe coincidir con el audio.'],
})

const presets = (values: Record<string, number>) =>
  (['fast', 'balanced', 'high_fidelity'] as const).map((id, i) => ({
    id,
    label: ['Rápido', 'Equilibrado', 'Alta fidelidad'][i],
    values: { speed: 1, seed: null, nfe_steps: Object.values(values)[i], remove_silence: false, fix_duration: null },
  }))

export const f5Config: EngineConfig = {
  engine: f5Summary,
  variant: f5Summary.variants[0],
  capabilities: capabilities(
    'F5TTS_v1_Base',
    baseControls({
      speed: cap('native', 'speed'),
      target_duration: cap('native', 'fix_duration'),
      emotion: cap('segmentation', null, 'F5-TTS no tiene control de emoción.', 9),
      pitch: cap('dsp', null, 'El modelo no controla el tono.', 10),
    }),
  ),
  parameters: [
    spec({ id: 'speed', type: 'slider', value_type: 'float', label: 'Velocidad de habla', default: 1, min: 0.3, max: 2, step: 0.05, unit: 'x', group: 'basic', control: 'speed' }),
    spec({ id: 'seed', type: 'seed', value_type: 'int', label: 'Semilla', control: 'seed' }),
    spec({ id: 'nfe_steps', type: 'slider', value_type: 'int', label: 'Pasos de inferencia', default: 32, min: 4, max: 64, step: 2, presets: { fast: 16, balanced: 32, high_fidelity: 48 } }),
    spec({ id: 'remove_silence', type: 'toggle', value_type: 'bool', label: 'Eliminar silencios largos', default: false }),
    spec({ id: 'fix_duration', type: 'number', value_type: 'float', label: 'Duración total fija', nullable: true, min: 1, max: 300, step: 0.5, unit: 's', control: 'target_duration' }),
  ],
  presets: presets({ fast: 16, balanced: 32, high_fidelity: 48 }),
}

export const qwenConfig: EngineConfig = {
  engine: qwenSummary,
  variant: qwenSummary.variants[0],
  capabilities: capabilities(
    'base-1.7b',
    baseControls({
      language: cap('native', 'language'),
      speed: cap('dsp', null, 'La clonación de Qwen3-TTS no tiene control de velocidad.', 10),
      instruction: cap('unavailable', null, 'generate_voice_clone no acepta instrucciones.'),
    }),
  ),
  parameters: [
    spec({
      id: 'language',
      type: 'select',
      value_type: 'str',
      label: 'Idioma del texto',
      default: 'Auto',
      group: 'basic',
      control: 'language',
      options: [
        { value: 'Auto', label: 'Detectar automáticamente', description: null },
        { value: 'Spanish', label: 'Español', description: null },
      ],
    }),
    spec({
      id: 'clone_mode',
      type: 'select',
      value_type: 'str',
      label: 'Modo de clonación',
      default: 'icl',
      group: 'basic',
      options: [
        { value: 'icl', label: 'Audio + transcripción (ICL)', description: 'Mayor similitud.' },
        { value: 'x_vector', label: 'Solo embedding de hablante', description: 'Menor calidad.' },
      ],
    }),
    spec({ id: 'seed', type: 'seed', value_type: 'int', label: 'Semilla', control: 'seed' }),
    spec({ id: 'temperature', type: 'slider', value_type: 'float', label: 'Temperatura', default: 0.9, min: 0.1, max: 1.5, step: 0.05 }),
  ],
  presets: (['fast', 'balanced', 'high_fidelity'] as const).map((id, i) => ({
    id,
    label: ['Rápido', 'Equilibrado', 'Alta fidelidad'][i],
    values: { language: 'Auto', clone_mode: 'icl', seed: null, temperature: 0.9 },
  })),
}
