import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { TooltipProvider } from '@/components/ui/tooltip'
import { LibraryPage } from '@/features/library/LibraryPage'
import { useLibraryStore } from '@/stores/library'
import { EMPTY_QUERY, type LibraryItem } from '@/types/library'

vi.mock('@/features/audio/AudioPlayer', () => ({ AudioPlayer: ({ url }: { url: string }) => <div data-testid="player">{url}</div> }))

const item = (over: Partial<LibraryItem> = {}): LibraryItem => ({
  id: 'g1',
  kind: 'single',
  engine: 'f5tts',
  variant: 'F5TTS_v1_Base',
  text: 'Primera prueba de voz',
  status: 'COMPLETED',
  seed: 42,
  duration_s: 2.5,
  audio_url: '/api/generation/g1/audio?v=1',
  favorite: false,
  tags: [],
  rating: null,
  profile_id: 'voz1',
  profile_name: 'Mateo',
  reference_name: 'ref.wav',
  created_at: '2026-09-17T10:00:00',
  ...over,
})

const facets = {
  tags: [{ tag: 'buena', count: 1 }],
  engines: [{ id: 'f5tts', label: 'f5tts', count: 2 }],
  profiles: [{ id: 'voz1', label: 'Mateo', count: 2 }],
  total: 2,
  favorites: 1,
}

const fetchMock = vi.fn()
const json = (body: unknown, status = 200) => Promise.resolve(new Response(JSON.stringify(body), { status }))
const urls = () => fetchMock.mock.calls.map(([url]) => String(url))
const lastSearch = () => [...urls()].reverse().find((url) => url.startsWith('/api/library?'))!
const lastBody = (method: string) => {
  const call = [...fetchMock.mock.calls].reverse().find(([, init]) => init?.method === method)
  return call ? JSON.parse(String(call[1].body)) : undefined
}

const renderPage = () =>
  render(
    <TooltipProvider>
      <LibraryPage />
    </TooltipProvider>,
  )

beforeEach(() => {
  vi.useRealTimers()
  fetchMock.mockReset()
  vi.stubGlobal('fetch', fetchMock)
  vi.stubGlobal('confirm', () => true)
  useLibraryStore.setState({ query: { ...EMPTY_QUERY }, items: [], total: 0, facets: null, selected: [], error: null, loading: false })
})

describe('LibraryPage', () => {
  it('lists results with their audio and filters by engine, favourites and tag', async () => {
    fetchMock.mockImplementation((url: string) => {
      if (url === '/api/library/facets') return json(facets)
      return json({ items: [item(), item({ id: 'g2', text: 'Segunda prueba', favorite: true })], total: 2, limit: 20, offset: 0 })
    })
    renderPage()

    expect(await screen.findAllByTestId('library-row')).toHaveLength(2)
    expect(screen.getByText('2 resultados')).toBeInTheDocument()
    expect(screen.getByText('2 en total · 1 favoritos')).toBeInTheDocument()
    expect(screen.getAllByTestId('player')[0]).toHaveTextContent('/api/generation/g1/audio?v=1')

    fireEvent.change(screen.getByLabelText('Motor'), { target: { value: 'f5tts' } })
    await waitFor(() => expect(lastSearch()).toContain('engine=f5tts'))

    fireEvent.click(screen.getByLabelText('Solo favoritos'))
    await waitFor(() => expect(lastSearch()).toContain('favorite=true'))

    fireEvent.change(screen.getByLabelText('Etiqueta'), { target: { value: 'buena' } })
    await waitFor(() => expect(lastSearch()).toContain('tag=buena'))
  })

  it('debounces the search box', async () => {
    fetchMock.mockImplementation((url: string) => (url === '/api/library/facets' ? json(facets) : json({ items: [item()], total: 1, limit: 20, offset: 0 })))
    renderPage()
    await screen.findAllByTestId('library-row')

    const box = screen.getByLabelText('Buscar en el texto')
    fireEvent.change(box, { target: { value: 'pru' } })
    fireEvent.change(box, { target: { value: 'prueba' } })
    await waitFor(() => expect(lastSearch()).toContain('q=prueba'))
    expect(urls().filter((url) => url.includes('q=pru&')).length).toBe(0)
  })

  it('marks a favourite and edits the tags', async () => {
    fetchMock.mockImplementation((url: string, init?: RequestInit) => {
      if (url === '/api/library/facets') return json(facets)
      if (url === '/api/generation/g1/favorite') return json(item({ favorite: true }))
      if (url === '/api/generation/g1/tags') return json(item({ tags: JSON.parse(String(init!.body)).tags }))
      return json({ items: [item()], total: 1, limit: 20, offset: 0 })
    })
    renderPage()
    await screen.findAllByTestId('library-row')

    fireEvent.click(screen.getByLabelText('Marcar como favorito'))
    await waitFor(() => expect(lastBody('PUT')).toEqual({ favorite: true }))
    expect(await screen.findByLabelText('Quitar de favoritos')).toBeInTheDocument()

    fireEvent.click(screen.getByLabelText(/Editar las etiquetas/))
    fireEvent.change(screen.getByLabelText(/^Etiquetas de/), { target: { value: 'buena, para el vídeo' } })
    fireEvent.blur(screen.getByLabelText(/^Etiquetas de/))
    await waitFor(() => expect(lastBody('PUT')).toEqual({ tags: ['buena', 'para el vídeo'] }))
    expect(await screen.findByText('para el vídeo')).toBeInTheDocument()
  })

  it('deletes the selected generations and reports failures', async () => {
    let remaining = [item(), item({ id: 'g2', text: 'Segunda prueba' })]
    fetchMock.mockImplementation((url: string) => {
      if (url === '/api/library/facets') return json(facets)
      if (url === '/api/library/delete') {
        remaining = []
        return json({ deleted: ['g1'], failed: ['g2'] })
      }
      return json({ items: remaining, total: remaining.length, limit: 20, offset: 0 })
    })
    renderPage()
    const rows = await screen.findAllByTestId('library-row')

    fireEvent.click(within(rows[0]).getByLabelText(/^Seleccionar/))
    fireEvent.click(within(rows[1]).getByLabelText(/^Seleccionar/))
    expect(screen.getByText('2 seleccionados')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Eliminar seleccionados' }))
    await waitFor(() => expect(lastBody('POST')).toEqual({ ids: ['g1', 'g2'] }))
    expect(await screen.findByRole('alert')).toHaveTextContent('No se pudo eliminar 1 generación.')
  })

  it('loads more and shows the filtered empty state', async () => {
    fetchMock.mockImplementation((url: string) => {
      if (url === '/api/library/facets') return json(facets)
      if (url.includes('q=nada')) return json({ items: [], total: 0, limit: 20, offset: 0 })
      if (url.includes('offset=1')) return json({ items: [item({ id: 'g2', text: 'Segunda prueba' })], total: 2, limit: 20, offset: 1 })
      return json({ items: [item()], total: 2, limit: 20, offset: 0 })
    })
    renderPage()
    await screen.findAllByTestId('library-row')

    fireEvent.click(screen.getByRole('button', { name: 'Cargar más' }))
    await waitFor(() => expect(screen.getAllByTestId('library-row')).toHaveLength(2))

    useLibraryStore.setState({ query: { ...EMPTY_QUERY, q: 'nada' } })
    await useLibraryStore.getState().load()
    expect(await screen.findByText('Ningún audio coincide con estos filtros.')).toBeInTheDocument()
  })
})
