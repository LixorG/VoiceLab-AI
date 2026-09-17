import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { TooltipProvider } from '@/components/ui/tooltip'
import { ComparisonTable } from '@/features/comparison/ComparisonTable'
import { ExperimentForm } from '@/features/experiments/ExperimentForm'
import { GenerationList } from '@/features/generation/GenerationList'
import { useEngineStore } from '@/stores/engine'
import { useExperimentsStore } from '@/stores/experiments'
import { useGenerationStore } from '@/stores/generation'
import { useProfilesStore } from '@/stores/profiles'
import { f5Config, f5Summary, qwenConfig, qwenSummary } from '@/test/engineFixtures'
import type { GenerationRead } from '@/types/generation'
import type { ProfileRead } from '@/types/profiles'

vi.mock('@/features/audio/SyncedPlayers', () => ({
  SyncedPlayers: ({ tracks }: { tracks: { label: string }[] }) => <div data-testid="synced">{tracks.map((t) => t.label).join(' | ')}</div>,
}))
vi.mock('@/features/audio/AudioPlayer', () => ({ AudioPlayer: ({ url }: { url: string }) => <div data-testid="player">{url}</div> }))

const gen = (over: Partial<GenerationRead> = {}): GenerationRead => ({
  id: 'g1',
  kind: 'experiment',
  engine: 'f5tts',
  variant: 'F5TTS_v1_Base',
  text: 'Hola a todos',
  params: {},
  seed: 7,
  status: 'COMPLETED',
  progress: 1,
  progress_available: true,
  message: null,
  error_code: null,
  reference: null,
  duration_s: 2.5,
  audio_url: '/api/generation/g1/audio?v=1',
  metrics: { rtf: 0.6 },
  warnings: [],
  expression: null,
  parent_id: null,
  segments: [],
  postprocess: null,
  raw_audio_url: null,
  label: 'F5-TTS',
  rating: null,
  evaluation: null,
  created_at: '2026-09-16T00:00:00Z',
  updated_at: '2026-09-16T00:00:00Z',
  ...over,
})

const metric = (id: string, value: number) => ({ id, label: id === 'wer' ? 'Error de palabras (WER)' : id, value, display: `${(value * 100).toFixed(1)} %`, better: 'lower' as const, description: 'd' })
const evaluation = (wer: number, stale = false) => ({
  metrics: [metric('wer', wer)],
  details: { intelligibility: { transcript: 'hola a todos' } },
  target_text: 'Hola a todos',
  unavailable: [],
  evaluated_at: '2026-09-16T00:00:00Z',
  stale,
})

const evaluators = [
  { id: 'intelligibility', label: 'Inteligibilidad (ASR)', description: 'x', available: true, reason: null },
  { id: 'speaker_similarity', label: 'Similitud de hablante', description: 'x', available: false, reason: 'Sin modelo.' },
]

const fetchMock = vi.fn()
const ok = (body: unknown, status = 200) => Promise.resolve(new Response(JSON.stringify(body), { status }))

beforeEach(() => {
  fetchMock.mockReset()
  vi.stubGlobal('fetch', fetchMock)
  const engines = [
    { ...f5Summary, implemented: true, installed: true },
    { ...qwenSummary, implemented: true, installed: true },
  ]
  useEngineStore.setState({ engines, loadEngines: async () => undefined, valuesByKey: {} })
  useExperimentsStore.setState({ items: [], selected: 'new', detail: null, busy: null, error: null })
  useGenerationStore.setState({ items: [], compareIds: [], text: 'Hola a todos', profileId: null, referenceId: null })
})

describe('ComparisonTable', () => {
  it('shows estimates, highlights the best value, lists unavailable metrics and saves ratings', async () => {
    const onUpdate = vi.fn()
    fetchMock.mockImplementation((url: string, init?: RequestInit) => {
      if (url === '/api/generation/evaluators') return ok(evaluators)
      if (url === '/api/generation/g2/rating') return ok(gen({ id: 'g2', rating: JSON.parse(String(init!.body)) }))
      return ok({})
    })
    render(
      <TooltipProvider>
        <ComparisonTable generations={[gen({ evaluation: evaluation(0.25) }), gen({ id: 'g2', label: 'Qwen3-TTS', engine: 'qwen3tts', evaluation: evaluation(0, true) })]} onUpdate={onUpdate} />
      </TooltipProvider>,
    )
    expect(await screen.findByText('Similitud de hablante')).toBeInTheDocument()
    expect(screen.getAllByText('No disponible')).toHaveLength(2)
    const best = screen.getByText('0.0 %')
    expect(best.className).toContain('text-success')
    expect(screen.getByText('desactualizada')).toBeInTheDocument()

    const table = screen.getByTestId('comparison-table')
    fireEvent.click(within(table).getAllByRole('radio', { name: 'Timbre: 4' })[1])
    await waitFor(() => expect(onUpdate).toHaveBeenCalledWith(expect.objectContaining({ id: 'g2', rating: { timbre: 4 } })))
  })
})

describe('ExperimentForm', () => {
  it('requires a voice and sends one arm per engine, with the seed only where the engine has one', async () => {
    const profile = { id: 'p1', name: 'Narradora', recommended_settings: {} } as unknown as ProfileRead
    useProfilesStore.setState({ items: [profile], load: async () => undefined })
    fetchMock.mockImplementation((url: string, init?: RequestInit) => {
      if (url.startsWith('/api/references')) return ok([])
      if (url.startsWith('/api/models/f5tts')) return ok(f5Config)
      if (url.startsWith('/api/models/qwen3tts')) return ok({ ...qwenConfig, parameters: qwenConfig.parameters.filter((p) => p.id !== 'seed') })
      if (url === '/api/experiments' && init?.method === 'POST') return ok({ id: 'e1', name: 'x', text: 'Hola a todos', profile_id: 'p1', reference_id: null, notes: null, settings: null, generations: [], created_at: '', updated_at: '' }, 202)
      if (url === '/api/experiments') return ok([])
      return ok({})
    })
    render(
      <TooltipProvider>
        <ExperimentForm />
      </TooltipProvider>,
    )
    expect(await screen.findAllByTestId('arm-row')).toHaveLength(2)
    const submit = screen.getByRole('button', { name: 'Generar 2 versiones' })
    expect(submit).toBeDisabled()
    expect(screen.getByText(/elige un perfil o una referencia/)).toBeInTheDocument()

    fireEvent.change(screen.getByLabelText('Perfil de voz'), { target: { value: 'p1' } })
    fireEvent.change(screen.getByLabelText('Semilla común'), { target: { value: '42' } })
    fireEvent.click(submit)

    await waitFor(() => expect(fetchMock.mock.calls.some(([u, i]) => u === '/api/experiments' && i?.method === 'POST')).toBe(true))
    const body = JSON.parse(String(fetchMock.mock.calls.find(([u, i]) => u === '/api/experiments' && i?.method === 'POST')![1].body))
    expect(body.profile_id).toBe('p1')
    expect(body.arms).toEqual([
      { engine: 'f5tts', variant: 'F5TTS_v1_Base', params: { seed: 42 } },
      { engine: 'qwen3tts', variant: 'base-1.7b', params: {} },
    ])
    await waitFor(() => expect(useExperimentsStore.getState().selected).toBe('e1'))
  })
})

describe('A/B comparison in results', () => {
  it('opens the synced comparison when two results are selected', async () => {
    fetchMock.mockImplementation((url: string) => {
      if (url.startsWith('/api/generation?')) return ok([gen({ id: 'a', kind: 'single', label: null, seed: 1 }), gen({ id: 'b', kind: 'single', label: null, seed: 2 })])
      if (url === '/api/generation/evaluators') return ok(evaluators)
      return ok({})
    })
    render(
      <TooltipProvider>
        <GenerationList />
      </TooltipProvider>,
    )
    const boxes = await screen.findAllByRole('checkbox', { name: 'Comparar' })
    fireEvent.click(boxes[0])
    expect(screen.getByText('Marca otro resultado para compararlos.')).toBeInTheDocument()
    fireEvent.click(boxes[1])
    expect(await screen.findByTestId('synced')).toHaveTextContent('f5tts · 1 | f5tts · 2')
    expect(screen.getByTestId('comparison-table')).toBeInTheDocument()
  })
})
