import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { TooltipProvider } from '@/components/ui/tooltip'
import { ReferenceLibrary } from '@/features/voices/ReferenceLibrary'
import { useReferencesStore } from '@/stores/references'
import type { ReferenceAnalysis, ReferenceRead } from '@/types/api'

vi.mock('@/features/audio/Waveform', () => ({ Waveform: () => <div data-testid="waveform" /> }))

const analysis: ReferenceAnalysis = {
  duration_s: 12,
  sample_rate: 24000,
  peak_dbfs: -3.2,
  rms_dbfs: -24,
  loudness_lufs: -23.1,
  clipping_ratio: 0,
  clipped_samples: 0,
  noise_floor_dbfs: -70,
  speech_level_dbfs: -20,
  snr_db: 38.4,
  speech_ratio: 0.7,
  silence_ratio: 0.3,
  effective_speech_s: 8.4,
  speech_detected: true,
  suggested_segments: [{ start_s: 1, end_s: 9, score: 0.9 }],
  quality_score: 91.5,
  quality_label: 'Excelente',
  quality_components: [{ id: 'snr', label: 'Relación señal/ruido', score: 1, weight: 0.3 }],
  warnings: ['Nivel de grabación muy bajo.'],
  normalization_gain_db: 3,
  source_codec: 'pcm_s16le',
  source_bit_rate: null,
}

const ref = (over: Partial<ReferenceRead> = {}): ReferenceRead => ({
  id: 'r1',
  profile_id: null,
  original_name: 'voz_01.wav',
  format: 'wav',
  size_bytes: 1_048_576,
  sample_rate: 44100,
  channels: 2,
  duration_s: 12,
  status: 'ANALYZED',
  quality_score: 91.5,
  quality_label: 'Excelente',
  analysis,
  segment_start_s: null,
  segment_end_s: null,
  emotion_tag: null,
  is_recommended: true,
  is_primary: false,
  created_at: '2026-09-16T00:00:00Z',
  urls: { original: '/o', processed: '/api/audio/references/r1/processed', peaks: '/p' },
  transcript: null,
  segment_transcript: null,
  segment_text_estimate: null,
  ...over,
})

const failed = ref({
  id: 'r2',
  original_name: 'fallida.mp3',
  status: 'FAILED',
  is_recommended: false,
  analysis: { error_code: 'AUDIO_DECODE_ERROR' },
})

const fetchMock = vi.fn()

function respond(url: string, init?: RequestInit): unknown {
  if (url === '/api/references') return [ref(), failed]
  if (url.includes('/peaks')) return { buckets: 3, duration_s: 12, sample_rate: 24000, peaks: [0.1, 0.5, 0.2] }
  if (url === '/api/audio/formats')
    return { extensions: ['wav', 'mp3'], max_upload_mb: 500, max_audio_seconds: 600, internal_sample_rate: 24000 }
  if (url === '/api/transcription/languages') return [{ code: 'es', name: 'Español' }]
  if (url === '/api/transcription/status')
    return { engine: 'faster-whisper', model: 'large-v3-turbo', package_available: true, installed: true, loaded: false,
      device: null, compute_type: null, download_state: 'idle', download_error: null, approx_size_mb: 1620 }
  if (init?.method === 'PATCH') return { ...ref(), ...JSON.parse(String(init.body)) }
  return null
}

beforeEach(() => {
  useReferencesStore.setState({ items: [], pending: [], notice: null, error: null })
  fetchMock.mockReset()
  fetchMock.mockImplementation((url: string, init?: RequestInit) =>
    Promise.resolve(new Response(JSON.stringify(respond(url, init)), { status: 200 })),
  )
  vi.stubGlobal('fetch', fetchMock)
})

const renderLibrary = () =>
  render(
    <TooltipProvider>
      <ReferenceLibrary />
    </TooltipProvider>,
  )

describe('ReferenceLibrary', () => {
  it('shows technical metadata, heuristic quality and recommendation', async () => {
    renderLibrary()
    expect(await screen.findByText('voz_01.wav')).toBeInTheDocument()
    expect(screen.getAllByText('44.1 kHz')).toHaveLength(2)
    expect(screen.getAllByText('Estéreo')[0]).toBeInTheDocument()
    expect(screen.getByText('-23.1 LUFS')).toBeInTheDocument()
    expect(screen.getByText('92%')).toBeInTheDocument()
    expect(screen.getByRole('meter', { name: 'Calidad de referencia' })).toHaveAttribute('aria-valuenow', '91.5')
    expect(screen.getByLabelText(/Indicador heurístico/)).toBeInTheDocument()
    expect(screen.getByText('Referencia recomendada')).toBeInTheDocument()
    expect(screen.getByText('Nivel de grabación muy bajo.')).toBeInTheDocument()
    expect(await screen.findByTestId('waveform')).toBeInTheDocument()
    expect(screen.getByText('No se pudo procesar el audio. Prueba a volver a analizarlo.')).toBeInTheDocument()
  })

  it('applies a suggested segment and saves it', async () => {
    renderLibrary()
    const save = await screen.findByRole('button', { name: 'Usar este segmento como referencia' })
    expect(save).toBeDisabled()
    fireEvent.click(screen.getByRole('button', { name: '0:01.0–0:09.0' }))
    expect(save).toBeEnabled()
    fireEvent.click(save)
    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        '/api/references/r1',
        expect.objectContaining({ method: 'PATCH', body: JSON.stringify({ segment_start_s: 1, segment_end_s: 9, snap_to_words: false }) }),
      ),
    )
    expect(await screen.findByText('Segmento guardado')).toBeInTheDocument()
  })
})
