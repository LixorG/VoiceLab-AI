import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { TooltipProvider } from '@/components/ui/tooltip'
import { RecommendedSettingsBar } from '@/features/voices/RecommendedSettingsBar'
import { VoicesPage } from '@/features/voices/VoicesPage'
import { keyOf, useEngineStore } from '@/stores/engine'
import { useGenerationStore } from '@/stores/generation'
import { useProfilesStore } from '@/stores/profiles'
import { useReferencesStore } from '@/stores/references'
import { f5Config, f5Summary } from '@/test/engineFixtures'
import type { ProfileRead } from '@/types/profiles'

vi.mock('@/features/voices/ReferenceLibrary', () => ({ ReferenceLibrary: () => <div data-testid="library" /> }))

const profile = (over: Partial<ProfileRead> = {}): ProfileRead => ({
  id: 'p1',
  name: 'Narradora',
  slug: 'narradora',
  description: 'Documentales',
  language: 'Español',
  default_engine: 'f5tts',
  primary_reference_id: null,
  recommended_reference_id: null,
  recommended_settings: { f5tts: { variant: 'F5TTS_v1_Base', params: { speed: 0.9, nfe_steps: 48, seed: null }, note: null, updated_at: null } },
  stats: {
    reference_count: 3,
    analyzed_count: 3,
    transcribed_count: 2,
    total_duration_s: 40,
    total_speech_s: 31.5,
    average_quality: 88.2,
    quality_label: 'Excelente',
    emotions: ['calm'],
  },
  created_at: '2026-09-16T00:00:00Z',
  updated_at: '2026-09-16T00:00:00Z',
  ...over,
})

const fetchMock = vi.fn()

beforeEach(() => {
  useProfilesStore.setState({ items: [], emotions: [], error: null })
  useReferencesStore.setState({ profileId: null, items: [] })
  useEngineStore.setState({ engines: [f5Summary], config: f5Config, engineId: 'f5tts', variantId: 'F5TTS_v1_Base', valuesByKey: { [keyOf('f5tts', 'F5TTS_v1_Base')]: { speed: 1, nfe_steps: 32, seed: null } } })
  useGenerationStore.setState({ profileId: null })
  fetchMock.mockReset()
  fetchMock.mockImplementation((url: string, init?: RequestInit) => {
    let body: unknown = []
    if (url === '/api/voices' && init?.method === 'POST') body = profile({ id: 'p2', name: JSON.parse(String(init.body)).name, recommended_settings: {} })
    else if (url === '/api/voices') body = [profile()]
    else if (url === '/api/voices/emotions') body = [{ id: 'calm', label: 'Calmado' }]
    else if (url.startsWith('/api/voices/p1/settings/') && init?.method === 'PUT') body = profile()
    else if (url.startsWith('/api/voices/p')) body = profile()
    return Promise.resolve(new Response(JSON.stringify(body), { status: 200 }))
  })
  vi.stubGlobal('fetch', fetchMock)
})

const renderPage = () =>
  render(
    <TooltipProvider>
      <VoicesPage />
    </TooltipProvider>,
  )

describe('VoicesPage', () => {
  it('lists profiles with stats and opens the detail scoped to its references', async () => {
    renderPage()
    expect(await screen.findByText('Narradora')).toBeInTheDocument()
    expect(screen.getByText('3 referencias')).toBeInTheDocument()
    expect(screen.getByText('Excelente')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /Narradora/ }))
    expect(await screen.findByTestId('library')).toBeInTheDocument()
    await waitFor(() => expect(useReferencesStore.getState().profileId).toBe('p1'))
    expect(screen.getByText(/Calidad media: Excelente \(88%\)/)).toBeInTheDocument()
    expect(await screen.findByText(/Emociones etiquetadas: Calmado/)).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /Exportar/ })).toHaveAttribute('href', '/api/voices/p1/export')
    expect(screen.getByText('F5TTS_v1_Base')).toBeInTheDocument()
  })

  it('creates a profile', async () => {
    renderPage()
    fireEvent.click(await screen.findByRole('button', { name: /Nueva voz/ }))
    fireEvent.change(screen.getByLabelText('Nombre'), { target: { value: 'Voz nueva' } })
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Crear perfil' }))
    })
    const post = fetchMock.mock.calls.find(([url, init]) => url === '/api/voices' && init?.method === 'POST')!
    expect(JSON.parse(post[1].body)).toMatchObject({ name: 'Voz nueva', default_engine: null })
    expect(await screen.findByRole('button', { name: /Volver a voces/ })).toBeInTheDocument()
  })
})

describe('RecommendedSettingsBar', () => {
  it('applies and saves recommended settings for the selected voice', async () => {
    useProfilesStore.setState({ items: [profile()] })
    useGenerationStore.setState({ profileId: 'p1' })
    render(<RecommendedSettingsBar />)
    fireEvent.click(screen.getByRole('button', { name: /Aplicar configuración recomendada/ }))
    await waitFor(() => expect(useEngineStore.getState().valuesByKey[keyOf('f5tts', 'F5TTS_v1_Base')]).toMatchObject({ speed: 0.9, nfe_steps: 48 }))
    fireEvent.click(screen.getByRole('button', { name: /Guardar configuración actual/ }))
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/api/voices/p1/settings/f5tts', expect.objectContaining({ method: 'PUT' })))
    const put = fetchMock.mock.calls.find(([url]) => url === '/api/voices/p1/settings/f5tts')!
    expect(JSON.parse(put[1].body)).toEqual({ variant: 'F5TTS_v1_Base', params: { speed: 0.9, nfe_steps: 48, seed: null } })
    expect(await screen.findByText('Configuración guardada en el perfil')).toBeInTheDocument()
  })

  it('renders nothing without a selected voice', () => {
    const { container } = render(<RecommendedSettingsBar />)
    expect(container).toBeEmptyDOMElement()
  })
})
