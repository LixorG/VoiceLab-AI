import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { PronunciationCard } from '@/features/settings/PronunciationCard'
import { useProfilesStore } from '@/stores/profiles'
import type { PronunciationEntry } from '@/types/pronunciation'

const entry = (over: Partial<PronunciationEntry> = {}): PronunciationEntry => ({
  id: 'e1',
  term: 'IA',
  replacement: 'i a',
  case_sensitive: false,
  profile_id: null,
  created_at: '',
  updated_at: '',
  ...over,
})

const fetchMock = vi.fn()
const json = (body: unknown, status = 200) => Promise.resolve(new Response(JSON.stringify(body), { status }))
const lastCall = (method: string) => [...fetchMock.mock.calls].reverse().find(([, init]) => init?.method === method)

beforeEach(() => {
  fetchMock.mockReset()
  vi.stubGlobal('fetch', fetchMock)
  vi.stubGlobal('confirm', () => true)
  useProfilesStore.setState({ items: [{ id: 'voz1', name: 'Mateo' }] as never, load: async () => undefined })
})

describe('PronunciationCard', () => {
  it('lists entries, adds one for a voice and shows the IPA limitation', async () => {
    let entries: PronunciationEntry[] = [entry()]
    fetchMock.mockImplementation((url: string, init?: RequestInit) => {
      if (init?.method === 'POST' && url === '/api/pronunciation') {
        entries = [...entries, entry({ id: 'e2', term: 'NASA', replacement: 'nasa', profile_id: 'voz1' })]
        return json(entries[1], 201)
      }
      return json(entries)
    })
    render(<PronunciationCard />)

    expect(await screen.findByText('IA → i a')).toBeInTheDocument()
    expect(screen.getByText(/transcripción fonética/)).toBeInTheDocument()

    fireEvent.change(screen.getByLabelText('Diccionario'), { target: { value: 'voz1' } })
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/api/pronunciation?profile_id=voz1', expect.anything()))

    fireEvent.change(screen.getByLabelText('Escrito'), { target: { value: ' NASA ' } })
    fireEvent.change(screen.getByLabelText('Se lee como'), { target: { value: 'nasa' } })
    fireEvent.click(screen.getByRole('button', { name: 'Añadir' }))
    await waitFor(() =>
      expect(JSON.parse(String(lastCall('POST')![1].body))).toEqual({
        term: 'NASA',
        replacement: 'nasa',
        case_sensitive: false,
        profile_id: 'voz1',
      }),
    )
    expect(await screen.findByText('NASA → nasa')).toBeInTheDocument()
  })

  it('edits an entry and tests the normalisation', async () => {
    fetchMock.mockImplementation((url: string, init?: RequestInit) => {
      if (url === '/api/pronunciation/preview') {
        return json({ text: 'La i a costó quince euros.', changes: [{ original: 'IA', replacement: 'i a', kind: 'diccionario' }], language: 'es', numbers_supported: true })
      }
      if (init?.method === 'PUT') return json(entry({ replacement: 'inteligencia artificial' }))
      return json([entry()])
    })
    render(<PronunciationCard />)

    fireEvent.click(await screen.findByRole('button', { name: 'Editar: IA' }))
    fireEvent.change(screen.getByLabelText('Se lee como'), { target: { value: 'inteligencia artificial' } })
    fireEvent.click(screen.getByRole('button', { name: 'Guardar' }))
    await waitFor(() => expect(JSON.parse(String(lastCall('PUT')![1].body)).replacement).toBe('inteligencia artificial'))

    fireEvent.change(screen.getByLabelText('Texto de prueba'), { target: { value: 'La IA costó 15 €.' } })
    fireEvent.click(screen.getByRole('button', { name: 'Probar' }))
    expect(await screen.findByText('La i a costó quince euros.')).toBeInTheDocument()
  })

  it('deletes an entry', async () => {
    fetchMock.mockImplementation((_url: string, init?: RequestInit) => {
      if (init?.method === 'DELETE') return Promise.resolve(new Response(null, { status: 204 }))
      return json([entry()])
    })
    render(<PronunciationCard />)
    fireEvent.click(await screen.findByRole('button', { name: 'Eliminar: IA' }))
    await waitFor(() => expect(lastCall('DELETE')![0]).toBe('/api/pronunciation/e1'))
  })
})
