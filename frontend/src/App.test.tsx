import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import App from '@/App'
import { useUiStore } from '@/stores/ui'
import { f5Config, f5Summary } from '@/test/engineFixtures'

const envReport = {
  platform: 'Windows',
  python_version: '3.11.9',
  gpu: {
    backend: 'cuda', devices: [{ index: 0, name: 'Mock GPU', total_memory_mb: 16384, free_memory_mb: 15000 }],
    torch_available: false, torch_version: null, cuda_version: null, rocm_version: null,
    driver_version: '1.0', bf16_supported: null, source: 'nvidia-smi',
  },
  checks: [{ id: 'ffmpeg', label: 'FFmpeg disponible', status: 'ok', detail: '9.0' }],
}

const responses: Record<string, unknown> = {
  '/api/models': [f5Summary],
  '/api/models/f5tts': f5Config,
  '/api/references': [],
  '/api/audio/formats': { extensions: ['wav'], max_upload_mb: 500, max_audio_seconds: 600, internal_sample_rate: 24000 },
  '/api/system/health': { status: 'ok', version: '0.1.0' },
  '/api/system/environment': envReport,
  '/api/system/info': { version: '0.1.0', app_env: 'test', device: 'auto', default_model: 'f5tts', max_upload_mb: 500, max_audio_seconds: 600, data_dir: '/d', model_dir: '/m' },
  '/api/system/licenses': [{ component: 'F5-TTS', category: 'motor', code_license: 'MIT', weights_license: 'CC-BY-NC-4.0', commercial_use: 'no_permitido', notes: null, url: 'https://example.org' }],
}

beforeEach(() => {
  useUiStore.setState({ section: 'generate', mode: 'simple', sidebarCollapsed: false })
  vi.stubGlobal('fetch', vi.fn((url: string) => Promise.resolve(new Response(JSON.stringify(responses[url]), { status: 200 }))))
})

describe('App shell', () => {
  it('renders the Spanish generate workspace and connects to the backend', async () => {
    render(<App />)
    expect(screen.getByRole('heading', { name: 'Generar voz' })).toBeInTheDocument()
    expect(await screen.findByText('Servidor conectado')).toBeInTheDocument()
    expect(await screen.findByText(/Mock GPU/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Generar voz/ })).toBeDisabled()
  })

  it('advanced mode reveals the collapsible advanced settings', () => {
    render(<App />)
    expect(screen.queryByText('Configuración avanzada')).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('radio', { name: 'Avanzado' }))
    expect(screen.getByText('Configuración avanzada')).toBeInTheDocument()
  })

  it('settings page shows environment checks and non-commercial licenses', async () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Configuración' }))
    expect(await screen.findByText('FFmpeg disponible')).toBeInTheDocument()
    expect(await screen.findByText('No permitido')).toBeInTheDocument()
  })
})
