import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { CustomCheckpoints } from '@/features/models/CustomCheckpoints'
import { useEngineStore } from '@/stores/engine'
import { f5Summary, qwenSummary } from '@/test/engineFixtures'
import type { CustomCheckpoint } from '@/types/checkpoints'

const checkpoint = (over: Partial<CustomCheckpoint> = {}): CustomCheckpoint => ({
  id: 'c1',
  engine: 'f5tts',
  variant: 'custom:f5-espanol',
  name: 'F5 Español',
  base_variant: 'F5TTS_v1_Base',
  repo_id: 'jpgallegoar/F5-Spanish',
  ckpt_file: 'model_1200000.safetensors',
  local_path: null,
  vocab_path: null,
  languages: ['es'],
  notes: null,
  weights_installed: false,
  created_at: '',
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
    engines: [
      { ...f5Summary, installed: true, implemented: true },
      { ...qwenSummary, installed: true, implemented: true, supports_custom_checkpoints: false },
    ],
    loadEngines: async () => undefined,
  })
})

describe('CustomCheckpoints', () => {
  it('only offers engines that can load user weights and warns about what cannot be checked', async () => {
    fetchMock.mockImplementation(() => json([]))
    render(<CustomCheckpoints />)

    expect(await screen.findByText('Aún no has añadido ninguno.')).toBeInTheDocument()
    expect(screen.getByText(/no puede comprobar qué contiene un checkpoint/i)).toBeInTheDocument()
    const engines = screen.getByLabelText('Motor') as HTMLSelectElement
    expect([...engines.options].map((o) => o.textContent)).toEqual(['F5-TTS'])
    expect([...(screen.getByLabelText('Arquitectura base') as HTMLSelectElement).options].map((o) => o.value)).toEqual(['F5TTS_v1_Base'])
  })

  it('adds a Hugging Face checkpoint and refreshes the engine list', async () => {
    const onChanged = vi.fn()
    let items: CustomCheckpoint[] = []
    fetchMock.mockImplementation((_url: string, init?: RequestInit) => {
      if (init?.method === 'POST') {
        items = [checkpoint()]
        return json(items[0], 201)
      }
      return json(items)
    })
    render(<CustomCheckpoints onChanged={onChanged} />)
    await screen.findByText('Aún no has añadido ninguno.')

    fireEvent.change(screen.getByLabelText('Nombre'), { target: { value: 'F5 Español' } })
    fireEvent.change(screen.getByLabelText('Repositorio'), { target: { value: 'jpgallegoar/F5-Spanish' } })
    fireEvent.change(screen.getByLabelText('Archivo del checkpoint'), { target: { value: 'model_1200000.safetensors' } })
    fireEvent.change(screen.getByLabelText('Idiomas (separados por comas)'), { target: { value: 'es, es-CO' } })
    fireEvent.click(screen.getByRole('button', { name: 'Añadir checkpoint' }))

    await waitFor(() => expect(lastCall('POST')![0]).toBe('/api/models/f5tts/checkpoints'))
    expect(JSON.parse(String(lastCall('POST')![1].body))).toEqual({
      name: 'F5 Español',
      base_variant: 'F5TTS_v1_Base',
      repo_id: 'jpgallegoar/F5-Spanish',
      ckpt_file: 'model_1200000.safetensors',
      local_path: null,
      vocab_path: null,
      languages: ['es', 'es-CO'],
      notes: null,
    })
    expect(await screen.findByText('F5 Español')).toBeInTheDocument()
    expect(screen.getByText('Sin descargar')).toBeInTheDocument()
    await waitFor(() => expect(onChanged).toHaveBeenCalled())
  })

  it('sends a local path when that source is chosen and shows backend errors', async () => {
    fetchMock.mockImplementation((_url: string, init?: RequestInit) => {
      if (init?.method === 'POST') return json({ error_code: 'VALIDATION_ERROR', message: 'No encuentro ese archivo de checkpoint en el equipo.' }, 422)
      return json([])
    })
    render(<CustomCheckpoints />)
    await screen.findByText('Aún no has añadido ninguno.')

    fireEvent.change(screen.getByLabelText('Origen'), { target: { value: 'local' } })
    fireEvent.change(screen.getByLabelText('Nombre'), { target: { value: 'Mío' } })
    fireEvent.change(screen.getByLabelText('Ruta del archivo'), { target: { value: 'C:\\modelos\\x.safetensors' } })
    fireEvent.click(screen.getByRole('button', { name: 'Añadir checkpoint' }))

    await waitFor(() => expect(JSON.parse(String(lastCall('POST')![1].body)).local_path).toBe('C:\\modelos\\x.safetensors'))
    expect(await screen.findByRole('alert')).toHaveTextContent('No encuentro ese archivo de checkpoint en el equipo.')
  })

  it('removes a checkpoint', async () => {
    fetchMock.mockImplementation((_url: string, init?: RequestInit) => {
      if (init?.method === 'DELETE') return Promise.resolve(new Response(null, { status: 204 }))
      return json([checkpoint({ weights_installed: true })])
    })
    render(<CustomCheckpoints />)
    fireEvent.click(await screen.findByLabelText('Quitar: F5 Español'))
    await waitFor(() => expect(lastCall('DELETE')![0]).toBe('/api/models/checkpoints/c1'))
  })
})
