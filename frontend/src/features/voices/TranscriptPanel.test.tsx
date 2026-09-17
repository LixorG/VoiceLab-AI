import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { TranscriptPanel } from '@/features/voices/TranscriptPanel'
import { useReferencesStore } from '@/stores/references'
import { useTranscriptionStore } from '@/stores/transcription'
import type { ASRStatus, ReferenceRead, TranscriptRead } from '@/types/api'

const status = (over: Partial<ASRStatus> = {}): ASRStatus => ({
  engine: 'faster-whisper',
  model: 'large-v3-turbo',
  package_available: true,
  installed: true,
  loaded: false,
  device: null,
  compute_type: null,
  download_state: 'idle',
  download_error: null,
  approx_size_mb: 1620,
  ...over,
})

const transcript = (over: Partial<TranscriptRead> = {}): TranscriptRead => ({
  id: 't1',
  reference_id: 'r1',
  text: 'Hola a todos.',
  language: 'es',
  source: 'asr',
  edited: false,
  asr_model: 'large-v3-turbo',
  segment_start_s: null,
  segment_end_s: null,
  confidence: { language_probability: 0.97, avg_logprob: -0.2, max_no_speech_prob: 0.01, mean_word_probability: 0.9 },
  warnings: [],
  words: null,
  updated_at: '2026-09-16T00:00:00Z',
  ...over,
})

const reference = (over: Partial<ReferenceRead> = {}) =>
  ({
    id: 'r1',
    segment_start_s: null,
    segment_end_s: null,
    transcript: null,
    segment_transcript: null,
    segment_text_estimate: null,
    ...over,
  }) as ReferenceRead

const fetchMock = vi.fn()

beforeEach(() => {
  useTranscriptionStore.setState({ status: status(), languages: [{ code: 'es', name: 'Español' }], busy: {}, errors: {} })
  useReferencesStore.setState({ items: [] })
  fetchMock.mockReset()
  fetchMock.mockImplementation((url: string, init?: RequestInit) => {
    const body = url.endsWith('/references')
      ? []
      : url.includes('/transcription/status')
        ? status()
        : init?.method === 'PATCH'
          ? transcript({ text: JSON.parse(String(init.body)).text, source: 'manual', edited: true })
          : transcript()
    return Promise.resolve(new Response(JSON.stringify(body), { status: 200 }))
  })
  vi.stubGlobal('fetch', fetchMock)
})

describe('TranscriptPanel', () => {
  it('offers the model download when it is not installed', () => {
    useTranscriptionStore.setState({ status: status({ installed: false }) })
    render(<TranscriptPanel reference={reference()} />)
    expect(screen.getByText(/large-v3-turbo \(~1.6 GB\) no está descargado/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Descargar modelo/ })).toBeEnabled()
    expect(screen.getByRole('button', { name: 'Transcribir' })).toBeDisabled()
  })

  it('transcribes with the selected language', async () => {
    render(<TranscriptPanel reference={reference()} />)
    fireEvent.change(screen.getByLabelText('Idioma del audio'), { target: { value: 'es' } })
    fireEvent.click(screen.getByRole('button', { name: 'Transcribir' }))
    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        '/api/transcription/references/r1',
        expect.objectContaining({ method: 'POST', body: JSON.stringify({ scope: 'full', language: 'es', force: false }) }),
      ),
    )
  })

  it('edits and saves the transcript', async () => {
    render(<TranscriptPanel reference={reference({ transcript: transcript() })} />)
    expect(screen.getByText('Automática · large-v3-turbo')).toBeInTheDocument()
    expect(screen.getByText('Español (97%)')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Guardar cambios' })).not.toBeInTheDocument()
    fireEvent.change(screen.getByRole('textbox', { name: 'Transcripción' }), { target: { value: 'Hola a todas.' } })
    fireEvent.click(screen.getByRole('button', { name: 'Guardar cambios' }))
    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        '/api/transcription/transcripts/t1',
        expect.objectContaining({ method: 'PATCH', body: JSON.stringify({ text: 'Hola a todas.' }) }),
      ),
    )
  })

  it('shows the estimated segment text and warnings', () => {
    render(
      <TranscriptPanel
        reference={reference({
          segment_start_s: 1,
          segment_end_s: 4,
          transcript: transcript({ warnings: ['Confianza baja del reconocimiento: revisa la transcripción.'] }),
          segment_text_estimate: 'a todos',
        })}
      />,
    )
    expect(screen.getByText('“a todos”')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Transcribir segmento' })).toBeEnabled()
    expect(screen.getByText(/Confianza baja/)).toBeInTheDocument()
  })
})
