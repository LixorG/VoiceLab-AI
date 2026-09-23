import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, onTestFinished, vi } from 'vitest'

import { TooltipProvider } from '@/components/ui/tooltip'
import { GenerationList } from '@/features/generation/GenerationList'
import { generationBlockers } from '@/features/generation/readiness'
import { keyOf, useEngineStore } from '@/stores/engine'
import { useGenerationStore } from '@/stores/generation'
import { useReferencesStore } from '@/stores/references'
import { useUiStore } from '@/stores/ui'
import { f5Config, f5Summary, qwenConfig } from '@/test/engineFixtures'
import type { ReferenceRead, TranscriptRead } from '@/types/api'
import type { EngineRuntimeStatus, GenerationRead } from '@/types/generation'

vi.mock('@/features/audio/AudioPlayer', () => ({ AudioPlayer: ({ url }: { url: string }) => <div data-testid="player">{url}</div> }))

const implemented = { ...f5Config, engine: { ...f5Summary, installed: true, implemented: true, implementation_phase: null } }
const runtime: EngineRuntimeStatus = {
  engine: 'f5tts',
  variant: 'F5TTS_v1_Base',
  package_installed: true,
  missing_packages: [],
  weights_installed: true,
  download_state: 'idle',
  download_error: null,
  loaded: false,
  device: null,
}
const transcript = { text: 'Hola a todos.' } as TranscriptRead
const reference = (over: Partial<ReferenceRead> = {}) =>
  ({ id: 'r1', status: 'ANALYZED', duration_s: 8, segment_start_s: null, segment_end_s: null, transcript, segment_transcript: null, ...over }) as ReferenceRead

describe('variation labels and plan', () => {
  it('labels variations and lists segments', async () => {
    const seg = { index: 0, text: 'Hola.', emotion: 'happy', emotion_via: 'reference' as const, instruction: null, pause_before_ms: 0, pause_after_ms: 500, reference_name: 'feliz.wav', seed: 7, duration_s: 1 }
    useGenerationStore.setState({ items: [gen({ id: 'v1', kind: 'variation', segments: [seg] }), gen({ id: 'v2', kind: 'variation', parent_id: 'v1', created_at: '2026-09-16T00:00:01Z' })] })
    render(<TooltipProvider><GenerationList /></TooltipProvider>)
    expect(screen.getByText('Variación 1')).toBeTruthy()
    expect(screen.getByText('Variación 2')).toBeTruthy()
    fireEvent.click(screen.getByText(/1 segmento/))
    expect(screen.getByText('Hola.')).toBeTruthy()
  })
})

describe('generationBlockers', () => {
  const base = { config: implemented, runtime, values: {}, text: 'Texto', reference: reference() }

  it('is ready when engine, weights, text and reference text are available', () => {
    expect(generationBlockers(base)).toEqual([])
  })

  it('explains every missing requirement', () => {
    expect(generationBlockers({ ...base, config: f5Config })).toEqual(['La generación con F5-TTS llega en la fase 5.'])
    expect(generationBlockers({ ...base, runtime: { ...runtime, weights_installed: false } })).toContain('Descarga los pesos del modelo (panel del motor).')
    expect(generationBlockers({ ...base, text: '  ' })).toContain('Escribe el texto a generar.')
    expect(generationBlockers({ ...base, reference: undefined })).toContain('Elige una referencia de voz («Usar para generar»).')
    expect(generationBlockers({ ...base, reference: reference({ transcript: null }) })[0]).toMatch(/Transcribe la referencia/)
    expect(generationBlockers({ ...base, reference: reference({ segment_start_s: 1, segment_end_s: 4 }) })[0]).toMatch(/segmento guardado/)
    expect(generationBlockers({ ...base, reference: reference({ duration_s: 30 }) })[0]).toMatch(/admite hasta 12 s/)
  })

  it('makes the transcript optional when the engine says so for the chosen parameters', () => {
    const qwenClone = {
      ...qwenConfig,
      engine: { ...qwenConfig.engine, installed: true, implemented: true, implementation_phase: null },
      capabilities: { ...qwenConfig.capabilities, requires_reference_text: true, reference_text_not_required_when: { clone_mode: 'x_vector' } },
    }
    const noText = { ...base, config: qwenClone, runtime: null, reference: reference({ transcript: null }) }
    expect(generationBlockers({ ...noText, values: { clone_mode: 'icl' } })[0]).toMatch(/Transcribe la referencia/)
    expect(generationBlockers({ ...noText, values: { clone_mode: 'x_vector' } })).toEqual([])
  })

  it('does not require references for engines that do not clone', () => {
    const customVoice = {
      ...qwenConfig,
      engine: { ...qwenConfig.engine, installed: true, implemented: true, implementation_phase: null },
      capabilities: { ...qwenConfig.capabilities, requires_reference_audio: false, requires_reference_text: false },
    }
    expect(generationBlockers({ ...base, config: customVoice, runtime: null, reference: undefined })).toEqual([])
  })
})

const gen = (over: Partial<GenerationRead> = {}): GenerationRead => ({
  id: 'g1',
  kind: 'single',
  engine: 'f5tts',
  variant: 'F5TTS_v1_Base',
  text: 'The future belongs to those who prepare for it.',
  params: { speed: 1, seed: 1234, nfe_steps: 32 },
  seed: 1234,
  status: 'COMPLETED',
  progress: 1,
  progress_available: true,
  message: 'Completada',
  error_code: null,
  reference: { reference_id: 'r1', name: 'zira.wav', start_s: 1, end_s: 6.5, text: 'Hello.' },
  duration_s: 5.47,
  audio_url: '/api/generation/g1/audio',
  metrics: { generation_s: 3.52, rtf: 0.64, device: 'cuda' },
  warnings: [],
  expression: null,
  parent_id: null,
  segments: [],
  postprocess: null,
  raw_audio_url: null,
  created_at: '2026-09-16T00:00:00Z',
  updated_at: '2026-09-16T00:00:00Z',
  ...over,
})

const fetchMock = vi.fn()

beforeEach(() => {
  fetchMock.mockReset()
  vi.stubGlobal('fetch', fetchMock)
  useGenerationStore.setState({ items: [], text: '', referenceId: null, error: null, submitting: null })
  useEngineStore.setState({ engines: [f5Summary], config: implemented, engineId: 'f5tts', variantId: 'F5TTS_v1_Base', valuesByKey: { [keyOf('f5tts', 'F5TTS_v1_Base')]: { speed: 1.1, seed: null, nfe_steps: 24 } } })
})

describe('GenerationList empty state', () => {
  it('points at «Voces» when there is no reference yet', async () => {
    const load = useGenerationStore.getState().load
    useReferencesStore.setState({ items: [] })
    useGenerationStore.setState({ items: [], load: async () => undefined })
    onTestFinished(() => useGenerationStore.setState({ load }))
    render(
      <TooltipProvider>
        <GenerationList />
      </TooltipProvider>,
    )
    expect(await screen.findByText('Empieza por una voz')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Ir a Voces' }))
    expect(useUiStore.getState().section).toBe('voices')
  })
})

describe('GenerationList', () => {
  it('shows completed, running and failed generations', async () => {
    fetchMock.mockImplementation((url: string) => {
      const body = url.startsWith('/api/generation?')
        ? [
            gen({ id: 'g3', status: 'GENERATING', progress: 0.4, message: 'Generando voz…', audio_url: null, seed: null, metrics: null }),
            gen({ id: 'g4', engine: 'qwen3tts', status: 'GENERATING', progress: 0.1, progress_available: false, message: 'Generando voz…', audio_url: null, seed: null, metrics: null, reference: null }),
            gen({ id: 'g2', status: 'FAILED', seed: null, metrics: null, message: 'No hay suficiente memoria de GPU para ejecutar este modelo.', audio_url: null }),
            gen(),
          ]
        : gen({ id: 'g3', status: 'GENERATING', progress: 0.4, message: 'Generando voz…', audio_url: null, seed: null, metrics: null })
      return Promise.resolve(new Response(JSON.stringify(body), { status: 200 }))
    })
    render(
      <TooltipProvider>
        <GenerationList />
      </TooltipProvider>,
    )
    expect(await screen.findByTestId('player')).toHaveTextContent('/api/generation/g1/audio')
    expect(screen.getByText('Semilla 1234')).toBeInTheDocument()
    expect(screen.getByText('RTF 0.64')).toBeInTheDocument()
    expect(screen.getAllByText('Referencia: zira.wav (0:01.0–0:06.5)')).toHaveLength(3)
    expect(screen.getByRole('progressbar', { name: '' })).toHaveAttribute('aria-valuenow', '40')
    expect(screen.getByRole('progressbar', { name: 'Generando (este modelo no informa del progreso)' })).not.toHaveAttribute('aria-valuenow')
    expect(screen.getAllByText('Generando voz…')).toHaveLength(2)
    expect(screen.getByRole('alert')).toHaveTextContent('memoria de GPU')
    expect(screen.getByRole('link', { name: 'WAV (sin pérdida)' })).toHaveAttribute('href', '/api/generation/g1/audio?download=true')
    expect(screen.getByRole('link', { name: 'MP3 (para vídeo y web)' })).toHaveAttribute('href', '/api/generation/g1/audio?download=true&format=mp3')

    fireEvent.click(screen.getAllByRole('button', { name: 'Cancelar' })[0])
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/api/jobs/g3/cancel', expect.objectContaining({ method: 'POST' })))
  })
})

describe('generation store', () => {
  it('sends the selected engine parameters, text and reference', async () => {
    fetchMock.mockImplementation(() =>
      Promise.resolve(new Response(JSON.stringify({ generation: gen({ status: 'QUEUED', progress: 0, audio_url: null }), job_id: 'g1', queue_position: 1 }), { status: 202 })),
    )
    useGenerationStore.setState({ text: 'Hola mundo', referenceId: 'r1' })
    await useGenerationStore.getState().generate(true)
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('/api/generation')
    expect(JSON.parse(init.body)).toEqual({
      engine: 'f5tts',
      variant: 'F5TTS_v1_Base',
      text: 'Hola mundo',
      params: { speed: 1.1, seed: null, nfe_steps: 24 },
      reference_id: 'r1',
      takes: 1,
      profile_id: null,
      preview: true,
      emotion: null,
      intensity: 50,
      markup: true,
      normalize: true,
      postprocess: null,
    })
    expect(useGenerationStore.getState().items[0].id).toBe('g1')
  })

  it('repeat reuses text, parameters and seed of a previous generation', async () => {
    fetchMock.mockImplementation(() =>
      Promise.resolve(new Response(JSON.stringify({ generation: gen({ id: 'g9', status: 'QUEUED' }), job_id: 'g9', queue_position: 1 }), { status: 202 })),
    )
    await useGenerationStore.getState().repeat(gen())
    const post = fetchMock.mock.calls.find(([url, init]) => url === '/api/generation' && init?.method === 'POST')!
    const body = JSON.parse(post[1].body)
    expect(body.params).toEqual({ speed: 1, seed: 1234, nfe_steps: 32 })
    expect(body.text).toBe('The future belongs to those who prepare for it.')
    expect(body.reference_id).toBe('r1')
    expect(body.preview).toBe(false)
  })

  it('surfaces backend errors in Spanish', async () => {
    fetchMock.mockResolvedValue(
      new Response(JSON.stringify({ error_code: 'REFERENCE_TOO_LONG', message: 'La referencia es demasiado larga para este modelo.', details: null }), { status: 422 }),
    )
    useGenerationStore.setState({ text: 'x' })
    await useGenerationStore.getState().generate(false)
    expect(useGenerationStore.getState().error).toBe('La referencia es demasiado larga para este modelo.')
  })
})
