import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useEngineStore } from '@/stores/engine'
import { useExperimentsStore } from '@/stores/experiments'
import { useGenerationStore } from '@/stores/generation'
import type { ExperimentRead } from '@/types/experiments'
import type { GenerationRead } from '@/types/generation'
import { BASIC_MASTERING, DEFAULT_POSTPROCESS } from '@/types/postprocess'

const gen = (over: Partial<GenerationRead> = {}): GenerationRead => ({
  id: 'g1',
  kind: 'single',
  engine: 'f5tts',
  variant: 'F5TTS_v1_Base',
  text: 'Hola',
  params: { seed: 1 },
  seed: 1,
  status: 'COMPLETED',
  progress: 1,
  progress_available: true,
  message: null,
  error_code: null,
  reference: null,
  duration_s: 1,
  audio_url: '/api/generation/g1/audio',
  metrics: null,
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
const json = (body: unknown, status = 200) => Promise.resolve(new Response(JSON.stringify(body), { status }))
const calls = (method: string, url: string) => fetchMock.mock.calls.filter(([u, init]) => u === url && (init?.method ?? 'GET') === method)

/** EventSource double: lets a test push SSE events for a job. */
class FakeEventSource {
  static instances: FakeEventSource[] = []
  listeners: Record<string, ((e: MessageEvent<string>) => void)[]> = {}
  onerror: (() => void) | null = null
  closed = false
  constructor(public url: string) {
    FakeEventSource.instances.push(this)
  }
  addEventListener(type: string, fn: (e: MessageEvent<string>) => void) {
    ;(this.listeners[type] ??= []).push(fn)
  }
  emit(data: unknown) {
    this.listeners.progress?.forEach((fn) => fn({ data: JSON.stringify(data) } as MessageEvent<string>))
  }
  close() {
    this.closed = true
  }
}

beforeEach(() => {
  fetchMock.mockReset()
  vi.stubGlobal('fetch', fetchMock)
  FakeEventSource.instances = []
  vi.stubGlobal('EventSource', FakeEventSource)
  useEngineStore.setState({ engineId: 'f5tts', variantId: 'F5TTS_v1_Base', valuesByKey: {} })
  useGenerationStore.setState({ items: [], text: 'Hola', referenceId: 'r1', profileId: null, error: null, submitting: null, compareIds: [], plan: null, planError: null, postprocess: DEFAULT_POSTPROCESS })
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('generation store', () => {
  it('queues variations newest-first and follows their progress through SSE', async () => {
    const accepted = [gen({ id: 'v1', kind: 'variation', status: 'QUEUED' }), gen({ id: 'v2', kind: 'variation', parent_id: 'v1', status: 'QUEUED' })]
    fetchMock.mockImplementation((url: string) => {
      if (url === '/api/generation/variations') return json(accepted.map((g) => ({ generation: g, job_id: g.id, queue_position: 1 })), 202)
      if (url === '/api/generation/v1') return json(gen({ id: 'v1', kind: 'variation', status: 'COMPLETED' }))
      return json({ engine: 'f5tts', variant: 'F5TTS_v1_Base', loaded: true })
    })
    useGenerationStore.setState({ postprocess: BASIC_MASTERING })
    await useGenerationStore.getState().generateVariations(2)

    const body = JSON.parse(String(calls('POST', '/api/generation/variations')[0][1].body))
    expect(body.count).toBe(2)
    expect(body.postprocess.loudness.enabled).toBe(true) // active post-processing travels with the request
    expect(useGenerationStore.getState().items.map((g) => g.id)).toEqual(['v1', 'v2'])

    const stream = FakeEventSource.instances.find((s) => s.url === '/api/jobs/v1/events')!
    stream.emit({ job_id: 'v1', status: 'GENERATING', progress: 0.5, message: 'Generando voz…' })
    expect(useGenerationStore.getState().items[0]).toMatchObject({ status: 'GENERATING', progress: 0.5 })
    stream.emit({ job_id: 'v1', status: 'COMPLETED', progress: 1, message: null })
    await vi.waitFor(() => expect(useGenerationStore.getState().items[0].status).toBe('COMPLETED'))
    expect(stream.closed).toBe(true)
  })

  it('reports errors from the API in Spanish without throwing', async () => {
    fetchMock.mockImplementation(() => json({ error_code: 'MARKUP_ERROR', message: 'Marca no válida.', details: {} }, 422))
    await useGenerationStore.getState().generate(false)
    expect(useGenerationStore.getState()).toMatchObject({ error: 'Marca no válida.', submitting: null })

    await useGenerationStore.getState().loadPlan()
    expect(useGenerationStore.getState()).toMatchObject({ plan: null, planError: 'Marca no válida.' })

    useGenerationStore.setState({ text: '   ' })
    await useGenerationStore.getState().loadPlan()
    expect(useGenerationStore.getState().planError).toBeNull() // empty text: nothing to plan

    expect(await useGenerationStore.getState().remaster('g1', BASIC_MASTERING)).toBe('Marca no válida.')
  })

  it('limits the A/B selection to four results and toggles entries', () => {
    const { toggleCompare } = useGenerationStore.getState()
    for (const id of ['a', 'b', 'c', 'd', 'e']) toggleCompare(id)
    expect(useGenerationStore.getState().compareIds).toEqual(['b', 'c', 'd', 'e'])
    toggleCompare('c')
    expect(useGenerationStore.getState().compareIds).toEqual(['b', 'd', 'e'])
    useGenerationStore.getState().clearCompare()
    expect(useGenerationStore.getState().compareIds).toEqual([])
  })

  it('removes, cancels and re-masters generations', async () => {
    useGenerationStore.setState({ items: [gen({ id: 'g1' }), gen({ id: 'g2', status: 'GENERATING' })] })
    fetchMock.mockImplementation((url: string, init?: RequestInit) => {
      if (init?.method === 'DELETE' && url === '/api/generation/g1') return Promise.resolve(new Response(null, { status: 204 }))
      if (url === '/api/jobs/g2/cancel') return json({ status: 'CANCELLED' })
      if (url === '/api/generation/g2') return json(gen({ id: 'g2', status: 'CANCELLED' }))
      if (url === '/api/generation/g1/postprocess' && init?.method === 'DELETE') return json(gen({ id: 'g1', postprocess: null }))
      return json({})
    })
    await useGenerationStore.getState().cancel('g2')
    expect(useGenerationStore.getState().items.find((g) => g.id === 'g2')?.status).toBe('CANCELLED')
    expect(await useGenerationStore.getState().clearMaster('g1')).toBeNull()
    await useGenerationStore.getState().remove('g1')
    expect(useGenerationStore.getState().items.map((g) => g.id)).toEqual(['g2'])
  })
})

describe('experiments store', () => {
  const experiment = (over: Partial<ExperimentRead> = {}): ExperimentRead => ({
    id: 'e1',
    name: 'Prueba',
    text: 'Hola',
    profile_id: null,
    reference_id: 'r1',
    notes: null,
    settings: null,
    generations: [gen({ id: 'a', kind: 'experiment', status: 'GENERATING' })],
    created_at: '',
    updated_at: '',
    ...over,
  })

  beforeEach(() => {
    useExperimentsStore.setState({ items: [], selected: null, detail: null, busy: null, error: null })
  })

  it('creates, refreshes while running, renames and deletes an experiment', async () => {
    let running = true
    fetchMock.mockImplementation((url: string, init?: RequestInit) => {
      const method = init?.method ?? 'GET'
      if (url === '/api/experiments' && method === 'POST') return json(experiment(), 202)
      if (url === '/api/experiments' && method === 'GET') return json([{ id: 'e1', name: 'Prueba', text: 'Hola', arms: 1, completed: running ? 0 : 1, running: running ? 1 : 0, failed: 0, engines: ['f5tts'], created_at: '', updated_at: '' }])
      if (url === '/api/experiments/e1' && method === 'GET') return json(experiment({ generations: [gen({ id: 'a', kind: 'experiment', status: running ? 'GENERATING' : 'COMPLETED' })] }))
      if (url === '/api/experiments/e1' && method === 'PATCH') return json(experiment({ name: JSON.parse(String(init!.body)).name }))
      if (url === '/api/experiments/e1' && method === 'DELETE') return Promise.resolve(new Response(null, { status: 204 }))
      return json({})
    })
    const store = useExperimentsStore.getState()
    const ok = await store.create({ name: 'Prueba', text: 'Hola', profile_id: null, reference_id: 'r1', emotion: null, intensity: 50, markup: true, postprocess: null, arms: [{ engine: 'f5tts', variant: null, params: {} }] })
    expect(ok).toBe(true)
    expect(useExperimentsStore.getState()).toMatchObject({ selected: 'e1', busy: null })

    running = false
    await useExperimentsStore.getState().refresh()
    await vi.waitFor(() => expect(useExperimentsStore.getState().items[0]?.completed).toBe(1))
    expect(useExperimentsStore.getState().detail?.generations[0].status).toBe('COMPLETED')

    useExperimentsStore.getState().replaceGeneration(gen({ id: 'a', kind: 'experiment', rating: { timbre: 4 } }))
    expect(useExperimentsStore.getState().detail?.generations[0].rating).toEqual({ timbre: 4 })

    await useExperimentsStore.getState().update({ name: 'Renombrado' })
    expect(useExperimentsStore.getState().detail?.name).toBe('Renombrado')

    await useExperimentsStore.getState().remove('e1')
    expect(useExperimentsStore.getState()).toMatchObject({ selected: null, detail: null })
  })

  it('keeps the form open with the error when creation is rejected', async () => {
    fetchMock.mockImplementation(() => json({ error_code: 'VALIDATION_ERROR', message: 'Parámetro no válido.', details: { posicion: 2 } }, 422))
    useExperimentsStore.setState({ selected: 'new' })
    const ok = await useExperimentsStore.getState().create({ name: null, text: 'Hola', profile_id: null, reference_id: 'r1', emotion: null, intensity: 50, markup: true, postprocess: null, arms: [] })
    expect(ok).toBe(false)
    expect(useExperimentsStore.getState()).toMatchObject({ selected: 'new', error: 'Parámetro no válido.' })
  })
})
