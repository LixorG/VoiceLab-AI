import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { WeightsStatus } from '@/features/models/WeightsStatus'
import { useEngineStore } from '@/stores/engine'
import { useGenerationStore } from '@/stores/generation'
import { f5Config } from '@/test/engineFixtures'
import type { EngineRuntimeStatus } from '@/types/generation'

const runtime = (over: Partial<EngineRuntimeStatus> = {}): EngineRuntimeStatus => ({
  engine: 'f5tts',
  variant: 'F5TTS_v1_Base',
  package_installed: true,
  missing_packages: [],
  weights_installed: true,
  download_state: 'idle',
  download_error: null,
  loaded: false,
  device: null,
  ...over,
})

const fetchMock = vi.fn()
const json = (body: unknown, status = 200) => Promise.resolve(new Response(JSON.stringify(body), { status }))
const config = { ...f5Config, engine: { ...f5Config.engine, installed: true, implemented: true }, variant: { ...f5Config.variant, download_size_mb: 1340 } }

beforeEach(() => {
  fetchMock.mockReset()
  vi.stubGlobal('fetch', fetchMock)
  useEngineStore.setState({ engineId: 'f5tts', variantId: 'F5TTS_v1_Base', config })
  useGenerationStore.setState({ runtime: null, error: null })
})

describe('WeightsStatus', () => {
  it('offers the download when weights are missing and follows its progress', async () => {
    let state = runtime({ weights_installed: false })
    fetchMock.mockImplementation((url: string) => {
      if (url.includes('/download')) {
        state = runtime({ weights_installed: false, download_state: 'downloading' })
        return json(state, 202)
      }
      return json(state)
    })
    render(<WeightsStatus />)
    expect(await screen.findByText(/Pesos del modelo no descargados \(~1.3 GB\)/)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Descargar modelo' }))
    expect(await screen.findByText(/Descargando pesos/)).toBeInTheDocument()
  })

  it('shows download errors from the server', async () => {
    fetchMock.mockImplementation(() => json(runtime({ weights_installed: false, download_state: 'failed', download_error: 'Sin espacio en disco.' })))
    render(<WeightsStatus />)
    expect(await screen.findByText('Sin espacio en disco.')).toBeInTheDocument()
  })

  it('preloads the model through the queue and then shows it loaded', async () => {
    let loaded = false
    fetchMock.mockImplementation((url: string) => {
      if (url.startsWith('/api/models/f5tts/load')) return json({ job_id: 'j1', status: 'QUEUED', message: null }, 202)
      if (url === '/api/jobs/j1') {
        loaded = true
        return json({ job_id: 'j1', status: 'COMPLETED', message: null })
      }
      return json(runtime(loaded ? { loaded: true, device: 'cuda' } : {}))
    })
    render(<WeightsStatus />)
    fireEvent.click(await screen.findByRole('button', { name: 'Cargar ahora' }))
    expect(screen.getByRole('button', { name: 'Cargando modelo…' })).toBeDisabled()
    expect(await screen.findByText('Cargado en memoria (CUDA)', {}, { timeout: 3000 })).toBeInTheDocument()
  })

  it('reports a failed preload', async () => {
    fetchMock.mockImplementation((url: string) => {
      if (url.startsWith('/api/models/f5tts/load')) return json({ job_id: 'j2', status: 'QUEUED', message: null }, 202)
      if (url === '/api/jobs/j2') return json({ job_id: 'j2', status: 'FAILED', message: 'No hay suficiente memoria de GPU.' })
      return json(runtime())
    })
    render(<WeightsStatus />)
    fireEvent.click(await screen.findByRole('button', { name: 'Cargar ahora' }))
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('No hay suficiente memoria de GPU.'), { timeout: 3000 })
  })

  it('renders nothing when the engine package is not installed', () => {
    useEngineStore.setState({ config: { ...config, engine: { ...config.engine, installed: false } } })
    const { container } = render(<WeightsStatus />)
    expect(container).toBeEmptyDOMElement()
    expect(fetchMock).not.toHaveBeenCalled()
  })
})
