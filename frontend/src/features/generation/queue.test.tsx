import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { BatchPanel, QueuePanel } from '@/features/generation/QueuePanel'
import { keyOf, useEngineStore } from '@/stores/engine'
import { useGenerationStore } from '@/stores/generation'
import { f5Summary } from '@/test/engineFixtures'
import type { QueuedJob } from '@/types/jobs'

const job = (over: Partial<QueuedJob> = {}): QueuedJob => ({
  job_id: 'j1',
  kind: 'generation',
  status: 'GENERATING',
  progress: 0.42,
  message: null,
  position: 0,
  text: 'Primera frase.',
  engine: 'f5tts',
  variant: 'F5TTS_v1_Base',
  generation_kind: 'single',
  ...over,
})

const fetchMock = vi.fn()
const json = (body: unknown, status = 200) => Promise.resolve(new Response(JSON.stringify(body), { status }))
const lastCall = (method: string) => [...fetchMock.mock.calls].reverse().find(([, init]) => init?.method === method)

beforeEach(() => {
  fetchMock.mockReset()
  vi.stubGlobal('fetch', fetchMock)
  vi.stubGlobal('confirm', () => true)
  useEngineStore.setState({
    engines: [{ ...f5Summary, installed: true, implemented: true }],
    engineId: 'f5tts',
    variantId: 'F5TTS_v1_Base',
    valuesByKey: { [keyOf('f5tts', 'F5TTS_v1_Base')]: { speed: 1, seed: null } },
  })
  useGenerationStore.setState({ text: 'Texto actual.', profileId: 'voz1', referenceId: null, markup: true, normalize: true, emotion: null, intensity: 50, load: async () => undefined })
})

describe('QueuePanel', () => {
  it('shows what is running and what waits, and cancels one job', async () => {
    fetchMock.mockImplementation(() => json([job(), job({ job_id: 'j2', status: 'QUEUED', position: 1, text: 'Segunda frase.', progress: 0 })]))
    render(<QueuePanel />)

    const rows = await screen.findAllByTestId('queue-row')
    expect(rows).toHaveLength(2)
    expect(within(rows[0]).getByText('Generando')).toBeInTheDocument()
    expect(within(rows[0]).getByText('42 %')).toBeInTheDocument()
    expect(within(rows[1]).getByText('#1')).toBeInTheDocument()
    expect(screen.getByText('2 tareas')).toBeInTheDocument()

    fireEvent.click(within(rows[1]).getByLabelText('Cancelar: Segunda frase.'))
    await waitFor(() => expect(lastCall('POST')![0]).toBe('/api/jobs/j2/cancel'))
  })

  it('cancels only the waiting jobs and disappears once the queue empties', async () => {
    let cancelled = false
    fetchMock.mockImplementation((url: string) => {
      if (url.startsWith('/api/jobs/cancel')) {
        cancelled = true
        return json({ cancelled: ['j2'] })
      }
      return json(cancelled ? [] : [job(), job({ job_id: 'j2', status: 'QUEUED', position: 1 })])
    })
    render(<QueuePanel />)
    await screen.findAllByTestId('queue-row')

    fireEvent.click(screen.getByRole('button', { name: 'Cancelar las que esperan' }))
    await waitFor(() => expect(lastCall('POST')![0]).toBe('/api/jobs/cancel?only_queued=true'))
    await waitFor(() => expect(screen.queryByTestId('queue-panel')).not.toBeInTheDocument())
  })
})

describe('BatchPanel', () => {
  it('queues one generation per line with the current settings', async () => {
    fetchMock.mockImplementation(() => json({ items: [], total: 2 }, 202))
    render(<BatchPanel />)
    fireEvent.click(screen.getByRole('button', { name: 'Abrir' }))

    fireEvent.change(screen.getByLabelText('Textos (uno por línea)'), { target: { value: 'Uno.\n\nDos.  ' } })
    expect(screen.getByRole('button', { name: 'Encolar 2 generaciones' })).toBeEnabled()

    fireEvent.click(screen.getByRole('button', { name: 'Encolar 2 generaciones' }))
    await waitFor(() => expect(lastCall('POST')![0]).toBe('/api/generation/batch'))
    const body = JSON.parse(String(lastCall('POST')![1].body))
    expect(body).toMatchObject({
      engine: 'f5tts',
      variant: 'F5TTS_v1_Base',
      texts: ['Uno.', 'Dos.'],
      variants: [],
      repeat: 1,
      profile_id: 'voz1',
      params: { speed: 1, seed: null },
      normalize: true,
    })
  })

  it('multiplies by repetitions, adds the current text and reports errors', async () => {
    fetchMock.mockImplementation(() => json({ error_code: 'VALIDATION_ERROR', message: 'El lote saldría de 100 generaciones y el máximo es 60.' }, 422))
    render(<BatchPanel />)
    fireEvent.click(screen.getByRole('button', { name: 'Abrir' }))

    fireEvent.click(screen.getByRole('button', { name: 'Añadir el texto de arriba' }))
    expect(screen.getByLabelText('Textos (uno por línea)')).toHaveValue('Texto actual.')

    fireEvent.change(screen.getByLabelText('Repeticiones'), { target: { value: '3' } })
    expect(screen.getByRole('button', { name: 'Encolar 3 generaciones' })).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Encolar 3 generaciones' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('El lote saldría de 100 generaciones')
  })
})
