import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { TooltipProvider } from '@/components/ui/tooltip'
import { EngineSettingsPanel } from '@/features/models/EngineSettingsPanel'
import { f5Config, f5Summary, qwenConfig, qwenSummary } from '@/test/engineFixtures'
import { keyOf, useEngineStore } from '@/stores/engine'
import { useUiStore } from '@/stores/ui'

const fetchMock = vi.fn()

beforeEach(() => {
  useUiStore.setState({ mode: 'simple' })
  useEngineStore.setState({ engines: [], config: null, engineId: null, variantId: null, valuesByKey: {}, activePreset: null, error: null })
  fetchMock.mockReset()
  fetchMock.mockImplementation((url: string) => {
    const body = url === '/api/models' ? [f5Summary, qwenSummary] : url.startsWith('/api/models/qwen3tts') ? qwenConfig : f5Config
    return Promise.resolve(new Response(JSON.stringify(body), { status: 200 }))
  })
  vi.stubGlobal('fetch', fetchMock)
})

const renderPanel = () =>
  render(
    <TooltipProvider>
      <EngineSettingsPanel />
    </TooltipProvider>,
  )

const values = () => {
  const s = useEngineStore.getState()
  return s.valuesByKey[keyOf(s.engineId, s.variantId)]
}

describe('EngineSettingsPanel', () => {
  it('renders controls from the backend schema and honest capability badges', async () => {
    renderPanel()
    expect(await screen.findByLabelText('Velocidad de habla')).toHaveAttribute('type', 'range')
    expect(screen.getByText('Uso no comercial')).toBeInTheDocument()
    expect(screen.getByText('Generación disponible en la fase 5')).toBeInTheDocument()
    const emotion = screen.getByText('Emoción').closest('[data-control]') as HTMLElement
    expect(within(emotion).getByText('Por segmentos')).toBeInTheDocument()
    const naturalness = screen.getByText('Naturalidad').closest('[data-control]') as HTMLElement
    expect(within(naturalness).getByText('No disponible')).toBeInTheDocument()
    // simple mode hides advanced parameters
    expect(screen.queryByText('Pasos de inferencia')).not.toBeInTheDocument()
  })

  it('updates values and applies presets in advanced mode without touching the seed', async () => {
    useUiStore.setState({ mode: 'advanced' })
    renderPanel()
    fireEvent.change(await screen.findByLabelText('Velocidad de habla'), { target: { value: '1.25' } })
    expect(values().speed).toBe(1.25)

    fireEvent.click(screen.getByRole('button', { name: 'Configuración avanzada' }))
    expect(screen.getByText('Pasos de inferencia')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('checkbox', { name: 'Aleatoria' }))
    const seed = values().seed
    expect(typeof seed).toBe('number')

    fireEvent.click(screen.getByRole('button', { name: 'Rápido' }))
    expect(values().nfe_steps).toBe(16)
    expect(values().seed).toBe(seed)
    expect(screen.getByRole('button', { name: 'Rápido' }).className).toContain('bg-primary')

    fireEvent.click(screen.getByRole('button', { name: /Restablecer valores/ }))
    expect(values().nfe_steps).toBe(32)
    expect(values().seed).toBeNull()
  })

  it('switching engines swaps the whole schema and keeps values per engine', async () => {
    renderPanel()
    await screen.findByLabelText('Velocidad de habla')
    fireEvent.change(screen.getByLabelText('Velocidad de habla'), { target: { value: '0.8' } })

    await act(async () => {
      fireEvent.change(screen.getByLabelText('Motor de voz'), { target: { value: 'qwen3tts' } })
    })
    expect(await screen.findByLabelText('Modo de clonación')).toBeInTheDocument()
    expect(screen.queryByLabelText('Velocidad de habla')).not.toBeInTheDocument()
    const speed = screen.getByText('Velocidad').closest('[data-control]') as HTMLElement
    expect(within(speed).getByText('Postprocesado')).toBeInTheDocument()
    expect(screen.getByLabelText('Variante')).toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledWith('/api/models/qwen3tts', expect.anything())

    await act(async () => {
      fireEvent.change(screen.getByLabelText('Motor de voz'), { target: { value: 'f5tts' } })
    })
    await waitFor(() => expect(screen.getByLabelText('Velocidad de habla')).toHaveValue('0.8'))
  })
})
