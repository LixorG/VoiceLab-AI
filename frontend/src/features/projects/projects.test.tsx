import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { TooltipProvider } from '@/components/ui/tooltip'
import { ProjectsPage } from '@/features/projects/ProjectsPage'
import { keyOf, useEngineStore } from '@/stores/engine'
import { useGenerationStore } from '@/stores/generation'
import { useProfilesStore } from '@/stores/profiles'
import { useProjectsStore } from '@/stores/projects'
import { f5Summary } from '@/test/engineFixtures'
import type { GenerationRead } from '@/types/generation'
import { EMPTY_SETTINGS, type ProjectRead, type ProjectSegment } from '@/types/projects'

vi.mock('@/features/audio/AudioPlayer', () => ({ AudioPlayer: ({ url }: { url: string }) => <div data-testid="player">{url}</div> }))

const gen = (id: string): GenerationRead =>
  ({ id, status: 'COMPLETED', progress: 1, progress_available: true, duration_s: 2.5, audio_url: `/api/generation/${id}/audio`, warnings: [], message: null }) as unknown as GenerationRead

const segment = (i: number, over: Partial<ProjectSegment> = {}): ProjectSegment => ({
  id: `s${i}`,
  position: i,
  text: `Frase ${i + 1}.`,
  profile_id: null,
  reference_id: null,
  emotion: null,
  intensity: null,
  pause_after_ms: null,
  seed: null,
  effective_pause_ms: 400,
  status: 'empty',
  generation: null,
  ...over,
})

const project = (over: Partial<ProjectRead> = {}): ProjectRead => ({
  id: 'p1',
  name: 'Vídeo',
  description: null,
  settings: { ...EMPTY_SETTINGS, engine: 'f5tts', variant: 'F5TTS_v1_Base' },
  segments: [],
  total_duration_s: 0,
  exported_at: null,
  created_at: '',
  updated_at: '',
  ...over,
})

const fetchMock = vi.fn()
const json = (body: unknown, status = 200) => Promise.resolve(new Response(JSON.stringify(body), { status }))
const lastBody = (url: string, method: string) => {
  const call = [...fetchMock.mock.calls].reverse().find(([u, i]) => u === url && i?.method === method)
  return call ? JSON.parse(String(call[1].body)) : undefined
}

const renderPage = () =>
  render(
    <TooltipProvider>
      <ProjectsPage />
    </TooltipProvider>,
  )

beforeEach(() => {
  fetchMock.mockReset()
  vi.stubGlobal('fetch', fetchMock)
  vi.stubGlobal('confirm', () => true)
  useEngineStore.setState({
    engines: [{ ...f5Summary, implemented: true, installed: true }],
    loadEngines: async () => undefined,
    engineId: 'f5tts',
    variantId: 'F5TTS_v1_Base',
    valuesByKey: { [keyOf('f5tts', 'F5TTS_v1_Base')]: { speed: 0.9, seed: null } },
  })
  useGenerationStore.setState({ profileId: 'voz1', referenceId: null, markup: true })
  useProfilesStore.setState({
    items: [{ id: 'voz1', name: 'Mateo' }] as never,
    emotions: [{ id: 'happy', label: 'Feliz' }],
    load: async () => undefined,
    loadEmotions: async () => undefined,
  })
  useProjectsStore.setState({ items: [], selected: null, detail: null, busy: null, error: null })
})

describe('ProjectsPage', () => {
  it('creates a project copying the Generate settings and imports a script', async () => {
    let current = project()
    fetchMock.mockImplementation((url: string, init?: RequestInit) => {
      const method = init?.method ?? 'GET'
      if (url === '/api/projects' && method === 'POST') {
        current = project({ name: JSON.parse(String(init!.body)).name, settings: JSON.parse(String(init!.body)).settings })
        return json(current, 201)
      }
      if (url === '/api/projects/p1/import') {
        current = { ...current, segments: [segment(0), segment(1)] }
        return json(current)
      }
      if (url === '/api/projects' || url.startsWith('/api/references')) return json([])
      return json(current)
    })
    renderPage()
    fireEvent.change(screen.getByLabelText('Nombre del nuevo proyecto'), { target: { value: 'Vídeo IA' } })
    fireEvent.click(screen.getByRole('button', { name: 'Crear' }))

    expect(await screen.findByText('Ajustes del proyecto')).toBeInTheDocument()
    expect(lastBody('/api/projects', 'POST')).toEqual({
      name: 'Vídeo IA',
      settings: { ...EMPTY_SETTINGS, engine: 'f5tts', variant: 'F5TTS_v1_Base', params: { speed: 0.9, seed: null }, profile_id: 'voz1' },
    })

    fireEvent.change(screen.getByLabelText('Importar guion'), { target: { value: 'Frase 1.\n\nFrase 2.' } })
    fireEvent.click(screen.getByLabelText('Frases'))
    fireEvent.click(screen.getByRole('button', { name: 'Añadir segmentos' }))
    expect(await screen.findAllByTestId('segment-row')).toHaveLength(2)
    expect(lastBody('/api/projects/p1/import', 'POST')).toEqual({ text: 'Frase 1.\n\nFrase 2.', split: 'sentences' })
  })

  it('edits, reorders, generates pending segments and only allows export when everything is ready', async () => {
    let current = project({
      segments: [
        segment(0, { status: 'ready', generation: gen('g0') }),
        segment(1, { status: 'stale', generation: gen('g1') }),
        segment(2),
      ],
    })
    fetchMock.mockImplementation((url: string, init?: RequestInit) => {
      const method = init?.method ?? 'GET'
      if (url === '/api/projects') return json([])
      if (url === '/api/projects/p1/generate') {
        current = { ...current, segments: current.segments.map((s) => ({ ...s, status: 'ready', generation: gen(`n${s.position}`) })) }
        return json(current, 202)
      }
      if (url === '/api/projects/p1/segments/order') {
        const ids: string[] = JSON.parse(String(init!.body)).ids
        current = { ...current, segments: ids.map((id, i) => ({ ...current.segments.find((s) => s.id === id)!, position: i })) }
        return json(current)
      }
      if (url.startsWith('/api/projects/p1/segments/') && method === 'PATCH') return json(current)
      return json(current)
    })
    useProjectsStore.setState({ selected: 'p1' })
    await useProjectsStore.getState().open('p1')
    renderPage()

    const rows = await screen.findAllByTestId('segment-row')
    expect(rows.map((r) => r.dataset.status)).toEqual(['ready', 'stale', 'empty'])
    expect(screen.getByText(/cambiaron después de generar/)).toBeInTheDocument()
    expect(screen.getAllByTestId('player')).toHaveLength(2)
    expect(screen.getByText(/1\/3 listos/)).toBeInTheDocument()
    expect(screen.getByText('Exportar audio').closest('[aria-disabled]')).toHaveAttribute('aria-disabled', 'true')
    expect(screen.getByRole('button', { name: /Exportar ZIP/ })).toBeDisabled()

    const text = within(rows[2]).getByLabelText('Texto del segmento 3')
    fireEvent.change(text, { target: { value: 'Texto nuevo.' } })
    fireEvent.blur(text)
    await waitFor(() => expect(lastBody('/api/projects/p1/segments/s2', 'PATCH')).toEqual({ text: 'Texto nuevo.' }))

    fireEvent.change(within(rows[2]).getByLabelText('Emoción'), { target: { value: 'happy' } })
    await waitFor(() => expect(lastBody('/api/projects/p1/segments/s2', 'PATCH')).toEqual({ emotion: 'happy' }))

    const pause = within(rows[0]).getByLabelText('Pausa después (ms)')
    fireEvent.change(pause, { target: { value: '1200' } })
    fireEvent.blur(pause)
    await waitFor(() => expect(lastBody('/api/projects/p1/segments/s0', 'PATCH')).toEqual({ pause_after_ms: 1200 }))

    fireEvent.click(within(rows[2]).getByRole('button', { name: 'Subir' }))
    await waitFor(() => expect(lastBody('/api/projects/p1/segments/order', 'PUT')).toEqual({ ids: ['s0', 's2', 's1'] }))

    fireEvent.click(screen.getByRole('button', { name: 'Generar 2 pendientes' }))
    await waitFor(() => expect(lastBody('/api/projects/p1/generate', 'POST')).toEqual({ segment_ids: null, only_pending: true }))
    await waitFor(() => expect(screen.getByRole('link', { name: 'WAV (sin pérdida)' })).toHaveAttribute('href', '/api/projects/p1/export?format=wav&download=true&allow_partial=false'))
    expect(screen.getByRole('link', { name: 'MP3 (para vídeo y web)' })).toHaveAttribute('href', '/api/projects/p1/export?format=mp3&download=true&allow_partial=false')
    expect(screen.getByRole('link', { name: /SRT/ })).toHaveAttribute('href', '/api/projects/p1/subtitles?format=srt&download=true')
    expect(screen.getByRole('link', { name: /WebVTT/ })).toHaveAttribute('href', '/api/projects/p1/subtitles?format=vtt&download=true')
    expect(screen.getByRole('link', { name: /Exportar ZIP/ })).toHaveAttribute('href', '/api/projects/p1/export?format=zip&download=true&allow_partial=false')

    fireEvent.click(screen.getByRole('button', { name: 'Escuchar todo' }))
    expect(screen.getAllByTestId('player')[0]).toHaveTextContent('/api/projects/p1/export?format=wav&download=false')
  })

  it('shows which segment failed validation', async () => {
    fetchMock.mockImplementation((url: string) => {
      if (url === '/api/projects/p1/generate') return json({ error_code: 'REFERENCE_TEXT_REQUIRED', message: 'Falta la transcripción.', details: { segmento: 2 } }, 422)
      if (url === '/api/projects') return json([])
      return json(project({ segments: [segment(0), segment(1)] }))
    })
    useProjectsStore.setState({ selected: 'p1' })
    await useProjectsStore.getState().open('p1')
    renderPage()
    fireEvent.click(await screen.findByRole('button', { name: 'Generar 2 pendientes' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Segmento 2: Falta la transcripción.')
  })
})
