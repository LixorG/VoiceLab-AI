import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { TrainingPanel } from '@/features/voices/TrainingPanel'
import { useEngineStore } from '@/stores/engine'
import { useUiStore } from '@/stores/ui'
import type { ProfileRead } from '@/types/profiles'
import type { TrainingDataset, TrainingRun } from '@/types/training'

const profile = { id: 'p1', name: 'Daniela' } as ProfileRead

const dataset = (over: Partial<TrainingDataset> = {}): TrainingDataset => ({
  profile_id: 'p1',
  references: [],
  clips: 35,
  usable_minutes: 5.7,
  recorded_minutes: 6.4,
  missing_transcripts: [],
  min_minutes: 5,
  recommended_minutes: [15, 30],
  ready: true,
  warnings: ['Con 5.7 minutos se puede entrenar, pero se recomiendan 15–30 para un resultado estable.'],
  ...over,
})

const run = (over: Partial<TrainingRun> = {}): TrainingRun => ({
  id: 'r1',
  profile_id: 'p1',
  profile_name: 'Daniela',
  engine: 'qwen3tts',
  base_variant: 'base-1.7b',
  name: 'Daniela (entrenada)',
  status: 'training',
  progress: 0.4,
  message: 'Época 4 de 10 · quedan ~3 min',
  params: { epochs: 10 },
  dataset: null,
  history: null,
  evaluation: null,
  checkpoint_id: null,
  variant: null,
  error_code: null,
  error_detail: null,
  created_at: '2026-09-21T00:00:00Z',
  started_at: null,
  finished_at: null,
  ...over,
})

let state: { dataset: TrainingDataset; runs: TrainingRun[] }
const fetchMock = vi.fn()

beforeEach(() => {
  state = { dataset: dataset(), runs: [] }
  fetchMock.mockReset()
  fetchMock.mockImplementation((url: string, init?: RequestInit) => {
    let body: unknown = null
    if (url.startsWith('/api/training/dataset/')) body = state.dataset
    else if (url.startsWith('/api/training?profile_id=')) body = state.runs
    else if (url === '/api/training' && init?.method === 'POST') {
      state.runs = [run({ status: 'queued', message: 'En cola…' })]
      body = state.runs[0]
    } else if (url.endsWith('/cancel')) {
      state.runs = [run({ status: 'cancelled', message: 'Cancelado.' })]
      body = state.runs[0]
    } else if (url.startsWith('/api/transcription/references/')) {
      state.dataset = dataset({ missing_transcripts: [] })
      body = { id: 't1' }
    }
    return Promise.resolve(new Response(JSON.stringify(body), { status: 200 }))
  })
  vi.stubGlobal('fetch', fetchMock)
  vi.spyOn(window, 'confirm').mockReturnValue(true)
})

describe('TrainingPanel', () => {
  it('summarises the data, transcribes what is missing and starts a training', async () => {
    state.dataset = dataset({ missing_transcripts: ['ref-a', 'ref-b'], ready: false, usable_minutes: 3 })
    render(<TrainingPanel profile={profile} referencesSignature="" />)

    expect(await screen.findByText(/^3:00 utilizables de 6:24 grabados · 35 frases/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Entrenar' })).toBeDisabled()
    fireEvent.click(screen.getByRole('button', { name: 'Transcribir las 2 grabaciones que faltan' }))
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/api/transcription/references/ref-b', expect.objectContaining({ method: 'POST' })))
    await waitFor(() => expect(screen.getByRole('button', { name: 'Entrenar' })).toBeEnabled())

    fireEvent.change(screen.getByLabelText('Épocas'), { target: { value: '12' } })
    fireEvent.change(screen.getByLabelText('Modelo base'), { target: { value: 'base-0.6b' } })
    fireEvent.click(screen.getByRole('button', { name: 'Entrenar' }))
    await waitFor(() => expect(screen.getByText('En cola')).toBeInTheDocument())
    const post = fetchMock.mock.calls.find(([url, init]) => url === '/api/training' && init?.method === 'POST')!
    expect(JSON.parse(String(post[1].body))).toEqual({ profile_id: 'p1', base_variant: 'base-0.6b', epochs: 12, name: null })
    expect(screen.getByRole('button', { name: 'Entrenar' })).toBeDisabled() // one training at a time
  })

  it('shows progress and cancels a running training', async () => {
    state.runs = [run()]
    render(<TrainingPanel profile={profile} referencesSignature="" />)
    expect(await screen.findByText('Época 4 de 10 · quedan ~3 min')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Cancelar' }))
    await waitFor(() => expect(screen.getByText('Cancelado')).toBeInTheDocument())
    expect(fetchMock).toHaveBeenCalledWith('/api/training/r1/cancel', expect.objectContaining({ method: 'POST' }))
  })

  it('shows the comparison of a finished voice and opens it in Generar', async () => {
    const loadEngines = vi.fn(() => Promise.resolve())
    const selectEngine = vi.fn(() => Promise.resolve())
    const selectVariant = vi.fn(() => Promise.resolve())
    useEngineStore.setState({ loadEngines, selectEngine, selectVariant })
    useUiStore.setState({ section: 'voices' })
    state.runs = [run({
      status: 'completed',
      progress: 1,
      variant: 'custom:daniela-entrenada',
      history: [{ epoch: 10, talker_loss: 1.09, predictor_loss: 5.1 }],
      evaluation: {
        held_out: 1,
        texts: ['Uh, everybody was like crazy.'],
        systems: { trained: { similarity: 0.94, wer: 0.076 }, normal: { similarity: 0.902, wer: 0.05 }, real: { similarity: 0.982 } },
        verdict: 'La voz entrenada se parece más a tu voz real que la clonación normal y se entiende igual (estimación automática).',
        missing: [],
      },
    })]
    render(<TrainingPanel profile={profile} referencesSignature="" />)

    expect(await screen.findByText(/se parece más a tu voz real/)).toBeInTheDocument()
    expect(screen.getByText('0.940')).toBeInTheDocument()
    expect(screen.getByText('7.6 %')).toBeInTheDocument()
    expect(screen.getByText('10 épocas · pérdida final 1.09')).toBeInTheDocument()
    fireEvent.click(screen.getByText('Escuchar'))
    expect(document.querySelector('audio[src="/api/training/r1/samples/0_trained.wav"]')).not.toBeNull()

    await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Usar en Generar' })))
    expect(selectEngine).toHaveBeenCalledWith('qwen3tts')
    expect(selectVariant).toHaveBeenCalledWith('custom:daniela-entrenada')
    expect(useUiStore.getState().section).toBe('generate')
  })

  it('explains failures and skipped comparisons', async () => {
    state.runs = [
      run({ id: 'f', status: 'failed', message: 'La GPU se quedó sin memoria al entrenar.', error_detail: 'CUDA out of memory' }),
      run({ id: 'c', status: 'completed', variant: 'custom:x', evaluation: { skipped: 'Con menos de 15 frases no se reservan grabaciones para comparar.' } }),
    ]
    render(<TrainingPanel profile={profile} referencesSignature="" />)
    expect(await screen.findByText('La GPU se quedó sin memoria al entrenar.')).toBeInTheDocument()
    expect(screen.getByText(/no se reservan grabaciones/)).toBeInTheDocument()
    expect(screen.getByText('CUDA out of memory')).toBeInTheDocument()
  })
})
