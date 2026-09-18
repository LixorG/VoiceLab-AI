import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { SpeakerModelCard } from '@/features/settings/SpeakerModelCard'

const status = (over = {}) => ({
  model: 'microsoft/wavlm-base-plus-sv',
  installed: false,
  loaded: false,
  download_state: 'idle',
  download_error: null,
  download_size_mb: 405,
  license: 'MIT',
  ...over,
})

const fetchMock = vi.fn()
const json = (body: unknown, code = 200) => Promise.resolve(new Response(JSON.stringify(body), { status: code }))

beforeEach(() => {
  fetchMock.mockReset()
  vi.stubGlobal('fetch', fetchMock)
})

describe('SpeakerModelCard', () => {
  it('explains the limits and downloads the model on demand', async () => {
    fetchMock.mockImplementation((_url: string, init?: RequestInit) =>
      json(init?.method === 'POST' ? status({ download_state: 'downloading' }) : status(), init?.method === 'POST' ? 202 : 200),
    )
    render(<SpeakerModelCard />)

    expect(await screen.findByText('Sin descargar')).toBeInTheDocument()
    expect(screen.getByText(/mide sobre todo el timbre, no el acento/)).toBeInTheDocument()
    expect(screen.getByText('microsoft/wavlm-base-plus-sv')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Descargar (~405 MB)' }))
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/api/generation/evaluators/speaker-model/download', expect.objectContaining({ method: 'POST' })))
    expect(await screen.findByRole('button', { name: 'Descargando…' })).toBeDisabled()
  })

  it('says how to use it once downloaded, and shows download errors', async () => {
    fetchMock.mockImplementation(() => json(status({ installed: true })))
    const { unmount } = render(<SpeakerModelCard />)
    expect(await screen.findByText('Descargado')).toBeInTheDocument()
    expect(screen.getByText(/aparecerá la columna «Similitud de voz»/)).toBeInTheDocument()
    expect(screen.queryByRole('button')).not.toBeInTheDocument()
    unmount()

    fetchMock.mockImplementation(() => json(status({ download_state: 'failed', download_error: 'No se pudo descargar el modelo. Revisa la conexión.' })))
    render(<SpeakerModelCard />)
    expect(await screen.findByRole('alert')).toHaveTextContent('Revisa la conexión')
  })
})
