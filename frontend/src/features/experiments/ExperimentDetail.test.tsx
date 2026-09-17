import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { TooltipProvider } from '@/components/ui/tooltip'
import { ExperimentsPage } from '@/features/experiments/ExperimentsPage'
import { useEngineStore } from '@/stores/engine'
import { useExperimentsStore } from '@/stores/experiments'
import { useProfilesStore } from '@/stores/profiles'
import { f5Summary } from '@/test/engineFixtures'
import type { ExperimentRead } from '@/types/experiments'
import type { GenerationRead } from '@/types/generation'

vi.mock('@/features/audio/SyncedPlayers', () => ({
  SyncedPlayers: ({ tracks }: { tracks: { label: string }[] }) => <div data-testid="synced">{tracks.map((t) => t.label).join(' | ')}</div>,
}))

const gen = (over: Partial<GenerationRead>): GenerationRead => ({
  id: 'g1',
  kind: 'experiment',
  engine: 'f5tts',
  variant: 'F5TTS_v1_Base',
  text: 'Hola',
  params: {},
  seed: 3,
  status: 'COMPLETED',
  progress: 1,
  progress_available: true,
  message: null,
  error_code: null,
  reference: null,
  duration_s: 2,
  audio_url: '/api/generation/g1/audio',
  metrics: { rtf: 0.5 },
  warnings: [],
  expression: null,
  parent_id: null,
  segments: [],
  postprocess: null,
  raw_audio_url: null,
  label: 'F5-TTS',
  experiment_id: 'e1',
  rating: null,
  evaluation: null,
  created_at: '',
  updated_at: '',
  ...over,
})

const detail = (over: Partial<ExperimentRead> = {}): ExperimentRead => ({
  id: 'e1',
  name: 'Comparativa',
  text: 'Hola a todos',
  profile_id: null,
  reference_id: 'r1',
  notes: 'nota inicial',
  settings: null,
  generations: [
    gen({ id: 'g1', label: 'F5-TTS' }),
    gen({ id: 'g2', label: 'E2-TTS', engine: 'e2tts' }),
    gen({ id: 'g3', label: 'Qwen3-TTS', engine: 'qwen3tts', status: 'FAILED', message: 'No hay suficiente memoria de GPU.', audio_url: null }),
  ],
  created_at: '',
  updated_at: '',
  ...over,
})

const fetchMock = vi.fn()
const json = (body: unknown, status = 200) => Promise.resolve(new Response(JSON.stringify(body), { status }))
const bodyOf = (url: string, method: string) => {
  const call = fetchMock.mock.calls.find(([u, init]) => u === url && init?.method === method)
  return call ? JSON.parse(String(call[1].body)) : undefined
}

beforeEach(() => {
  fetchMock.mockReset()
  vi.stubGlobal('fetch', fetchMock)
  vi.stubGlobal('confirm', () => true)
  useEngineStore.setState({ engines: [{ ...f5Summary, implemented: true, installed: true }], loadEngines: async () => undefined, valuesByKey: {} })
  useProfilesStore.setState({ items: [], load: async () => undefined })
  useExperimentsStore.setState({ items: [], selected: null, detail: null, busy: null, error: null })
  fetchMock.mockImplementation((url: string, init?: RequestInit) => {
    const method = init?.method ?? 'GET'
    if (url === '/api/experiments' && method === 'GET') return json([{ id: 'e1', name: 'Comparativa', text: 'Hola a todos', arms: 3, completed: 2, running: 0, failed: 1, engines: ['e2tts', 'f5tts', 'qwen3tts'], created_at: '', updated_at: '' }])
    if (url === '/api/experiments/e1' && method === 'GET') return json(detail())
    if (url === '/api/experiments/e1' && method === 'PATCH') return json(detail(JSON.parse(String(init!.body))))
    if (url === '/api/experiments/e1/arms') return json(detail(), 202)
    if (url === '/api/experiments/e1/evaluate') return json(detail())
    if (url === '/api/experiments/e1' && method === 'DELETE') return Promise.resolve(new Response(null, { status: 204 }))
    if (url === '/api/generation/evaluators' || url.startsWith('/api/references')) return json([])
    return json({})
  })
})

const renderPage = () =>
  render(
    <TooltipProvider>
      <ExperimentsPage />
    </TooltipProvider>,
  )

describe('ExperimentsPage', () => {
  it('shows the intro, lists experiments and opens the creation form', async () => {
    renderPage()
    expect(await screen.findByText('Comparativa')).toBeInTheDocument()
    expect(screen.getByText('3 versiones')).toBeInTheDocument()
    expect(screen.getByText('1 con error')).toBeInTheDocument()
    fireEvent.click(screen.getAllByRole('button', { name: /Nuevo/ })[0])
    expect(await screen.findByText('Nuevo experimento')).toBeInTheDocument()
  })

  it('opens an experiment: synced players for finished versions, failures, rename, notes, evaluate, add and delete', async () => {
    renderPage()
    fireEvent.click(await screen.findByRole('button', { name: /Comparativa/ }))

    expect(await screen.findByTestId('synced')).toHaveTextContent('F5-TTS | E2-TTS') // the failed version has no audio
    expect(screen.getByText('Qwen3-TTS: No hay suficiente memoria de GPU.')).toBeInTheDocument()

    const name = screen.getByLabelText('Nombre')
    fireEvent.change(name, { target: { value: 'Comparativa final' } })
    fireEvent.blur(name)
    await waitFor(() => expect(bodyOf('/api/experiments/e1', 'PATCH')).toEqual({ name: 'Comparativa final' }))

    const notes = screen.getByPlaceholderText(/Conclusiones/)
    fireEvent.change(notes, { target: { value: 'F5 se parece más' } })
    fireEvent.blur(notes)
    await waitFor(() => expect(fetchMock.mock.calls.filter(([u, i]) => u === '/api/experiments/e1' && i?.method === 'PATCH')).toHaveLength(2))

    fireEvent.click(screen.getByRole('button', { name: /Evaluar todo/ }))
    await waitFor(() => expect(fetchMock.mock.calls.some(([u]) => u === '/api/experiments/e1/evaluate')).toBe(true))

    fireEvent.click(screen.getByRole('button', { name: 'Añadir motor' }))
    fireEvent.click(screen.getByRole('button', { name: 'Generar 1 versión' }))
    await waitFor(() => expect(bodyOf('/api/experiments/e1/arms', 'POST')).toEqual({ arms: [{ engine: 'f5tts', variant: 'F5TTS_v1_Base', params: {} }] }))

    fireEvent.click(screen.getByRole('button', { name: /Eliminar/ }))
    await waitFor(() => expect(useExperimentsStore.getState().selected).toBeNull())
  })

  it('polls while versions are still generating', async () => {
    let calls = 0
    fetchMock.mockImplementation((url: string) => {
      if (url === '/api/experiments/e1') {
        calls++
        return json(detail({ generations: [gen({ status: calls > 1 ? 'COMPLETED' : 'GENERATING', progress: 0.4 })] }))
      }
      return json([])
    })
    vi.useFakeTimers({ shouldAdvanceTime: true })
    try {
      useExperimentsStore.setState({ selected: 'e1' })
      await useExperimentsStore.getState().open('e1')
      renderPage()
      expect(await screen.findByText('Generando…')).toBeInTheDocument()
      await vi.advanceTimersByTimeAsync(1600)
      await waitFor(() => expect(calls).toBeGreaterThan(1))
      await waitFor(() => expect(screen.queryByText('Generando…')).not.toBeInTheDocument())
    } finally {
      vi.useRealTimers()
    }
  })
})
