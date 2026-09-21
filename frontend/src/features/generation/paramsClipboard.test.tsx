import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { TooltipProvider } from '@/components/ui/tooltip'
import { ComparisonTable } from '@/features/comparison/ComparisonTable'
import { armParams, newArm } from '@/features/experiments/ArmsEditor'
import { ParamsClipboardBar } from '@/features/generation/ParamsClipboard'
import { keyOf, useEngineStore } from '@/stores/engine'
import { useGenerationStore } from '@/stores/generation'
import { type CopiedSettings, settingsAsText, settingsFromGeneration, useParamsClipboard } from '@/stores/paramsClipboard'
import { f5Config, f5Summary } from '@/test/engineFixtures'
import type { GenerationRead } from '@/types/generation'
import { DEFAULT_POSTPROCESS } from '@/types/postprocess'

const gen = (over: Partial<GenerationRead> = {}): GenerationRead =>
  ({
    id: 'g1',
    kind: 'experiment',
    engine: 'f5tts',
    variant: 'F5TTS_v1_Base',
    text: 'Hola',
    params: { speed: 0.8, nfe_steps: 16 },
    seed: 1234,
    status: 'COMPLETED',
    progress: 1,
    progress_available: true,
    label: 'F5 lento',
    expression: { emotion: 'happy', intensity: 70, markup: false, normalize: false },
    postprocess: null,
    audio_url: '/api/generation/g1/audio',
    warnings: [],
    metrics: null,
    evaluation: null,
    rating: null,
    reference: null,
    segments: [],
    duration_s: 2,
    ...over,
  }) as unknown as GenerationRead

const writeText = vi.fn(() => Promise.resolve())

beforeEach(() => {
  writeText.mockClear()
  Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true })
  useParamsClipboard.setState({ copied: null })
  useEngineStore.setState({
    engines: [{ ...f5Summary, installed: true, implemented: true }],
    engineId: 'f5tts',
    variantId: 'F5TTS_v1_Base',
    config: f5Config,
    valuesByKey: { [keyOf('f5tts', 'F5TTS_v1_Base')]: { speed: 1, seed: null } },
    selectEngine: async () => undefined,
    selectVariant: async () => undefined,
  })
  useGenerationStore.setState({ emotion: null, intensity: 50, markup: true, normalize: true, postprocess: DEFAULT_POSTPROCESS })
})

describe('copying parameters', () => {
  it('keeps the seed and the expression, and leaves voice and text out', () => {
    const settings = settingsFromGeneration(gen(), 'F5 lento')
    expect(settings).toMatchObject({
      engine: 'f5tts',
      variant: 'F5TTS_v1_Base',
      params: { speed: 0.8, nfe_steps: 16, seed: 1234 },
      emotion: 'happy',
      intensity: 70,
      markup: false,
      normalize: false,
      label: 'F5 lento',
    })
    expect(settings).not.toHaveProperty('text')
    expect(settingsAsText(settings)).not.toContain('copiedAt')
  })

  it('copies from the comparison table (Experiments) into the app and the system clipboard', async () => {
    render(
      <TooltipProvider>
        <ComparisonTable generations={[gen(), gen({ id: 'g2', label: 'Qwen', engine: 'qwen3tts' })]} onUpdate={() => undefined} />
      </TooltipProvider>,
    )
    fireEvent.click(screen.getByRole('button', { name: 'Copiar parámetros: F5 lento' }))
    await waitFor(() => expect(useParamsClipboard.getState().copied?.label).toBe('F5 lento'))
    expect(writeText).toHaveBeenCalledWith(expect.stringContaining('"seed": 1234'))
    expect(await screen.findByText('Parámetros copiados')).toBeInTheDocument()
  })
})

describe('ParamsClipboardBar (Generar)', () => {
  it('pastes the copied settings, reporting parameters this engine version does not have', async () => {
    const copied: CopiedSettings = { ...settingsFromGeneration(gen({ params: { speed: 0.8, legacy_param: 3 } }), 'F5 lento') }
    useParamsClipboard.setState({ copied })
    render(<ParamsClipboardBar />)

    await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Pegar parámetros (F5 lento)' })))
    const values = useEngineStore.getState().valuesByKey[keyOf('f5tts', 'F5TTS_v1_Base')]
    expect(values.speed).toBe(0.8)
    expect(values.seed).toBe(1234)
    expect(values).not.toHaveProperty('legacy_param')
    expect(useGenerationStore.getState()).toMatchObject({ emotion: 'happy', intensity: 70, markup: false, normalize: false })
    expect(screen.getByRole('status')).toHaveTextContent('Se aplicaron 2 parámetros. No existen en esta versión del motor y se ignoraron: legacy_param.')
  })

  it('copies the current Generar settings and says so; an unknown engine is reported', async () => {
    render(<ParamsClipboardBar />)
    expect(screen.queryByRole('button', { name: /Pegar parámetros/ })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Copiar ajustes' }))
    await waitFor(() => expect(useParamsClipboard.getState().copied).toMatchObject({ engine: 'f5tts', label: 'Generar', params: { speed: 1, seed: null } }))
    expect(await screen.findByRole('button', { name: /Pegar parámetros \(Generar\)/ })).toBeInTheDocument()

    act(() => useParamsClipboard.setState({ copied: { ...useParamsClipboard.getState().copied!, engine: 'desconocido', label: 'Otro' } }))
    await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Pegar parámetros (Otro)' })))
    expect(screen.getByRole('alert')).toHaveTextContent('El motor «desconocido» no está disponible en esta instalación.')
  })
})

describe('experiment arms from copied parameters', () => {
  it('use the copied values only for the same engine and variant', () => {
    useParamsClipboard.setState({ copied: settingsFromGeneration(gen(), 'F5 lento') })
    const arm = { ...newArm('f5tts', 'F5TTS_v1_Base'), source: 'copied' as const }
    expect(armParams(arm, null, {})).toEqual({ speed: 0.8, nfe_steps: 16, seed: 1234 })
    expect(armParams({ ...arm, engine: 'qwen3tts', variant: 'base-1.7b' }, null, {})).toEqual({})
  })
})
