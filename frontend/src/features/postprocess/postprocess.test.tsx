import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { TooltipProvider } from '@/components/ui/tooltip'
import { GenerationMastering } from '@/features/postprocess/GenerationMastering'
import { PostProcessEditor } from '@/features/postprocess/PostProcessEditor'
import { useGenerationStore } from '@/stores/generation'
import type { GenerationRead } from '@/types/generation'
import { BASIC_MASTERING, DEFAULT_POSTPROCESS, isPostprocessActive } from '@/types/postprocess'

vi.mock('@/features/audio/AudioPlayer', () => ({ AudioPlayer: ({ url }: { url: string }) => <div data-testid="player">{url}</div> }))

const gen = (over: Partial<GenerationRead> = {}): GenerationRead => ({
  id: 'g1',
  kind: 'single',
  engine: 'f5tts',
  variant: null,
  text: 'Hola',
  params: {},
  seed: 1,
  status: 'COMPLETED',
  progress: 1,
  progress_available: true,
  message: null,
  error_code: null,
  reference: null,
  duration_s: 2,
  audio_url: '/api/generation/g1/audio?v=2',
  metrics: null,
  warnings: [],
  expression: null,
  parent_id: null,
  segments: [],
  postprocess: null,
  raw_audio_url: '/api/generation/g1/audio?version=raw',
  created_at: '2026-09-16T00:00:00Z',
  updated_at: '2026-09-16T00:00:00Z',
  ...over,
})

const fetchMock = vi.fn()

beforeEach(() => {
  fetchMock.mockReset()
  vi.stubGlobal('fetch', fetchMock)
  useGenerationStore.setState({ items: [], postprocess: DEFAULT_POSTPROCESS, postprocessCaps: { processors: { pitch_shift: false }, reasons: { pitch_shift: 'Sin rubberband.' } } })
})

describe('PostProcessEditor', () => {
  it('starts disabled, shows controls when enabled and marks unavailable processors', () => {
    const onChange = vi.fn()
    render(<PostProcessEditor value={DEFAULT_POSTPROCESS} onChange={onChange} capabilities={useGenerationStore.getState().postprocessCaps} segmented={false} />)
    expect(isPostprocessActive(DEFAULT_POSTPROCESS)).toBe(false)
    expect(screen.getByText('Sin rubberband.')).toBeInTheDocument()
    const pitch = screen.getByText('Tono (DSP)').closest('label')!.querySelector('input')!
    expect(pitch).toBeDisabled()
    expect(screen.getByText('Crossfade entre segmentos').closest('label')!.querySelector('input')).toBeDisabled()

    fireEvent.click(screen.getByText('Normalizar sonoridad').closest('label')!.querySelector('input')!)
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ loudness: { enabled: true, target_lufs: -16 } }))
  })

  it('warns when the engine controls speed natively', () => {
    const value = { ...DEFAULT_POSTPROCESS, time_stretch: { enabled: true, rate: 1.2 } }
    render(<PostProcessEditor value={value} onChange={() => undefined} capabilities={null} nativeSpeedParameter="speed" />)
    expect(screen.getByText(/de forma nativa/)).toBeInTheDocument()
  })
})

describe('GenerationMastering', () => {
  it('switches between processed and original audio and shows the report', () => {
    const processed = gen({
      postprocess: {
        config: BASIC_MASTERING,
        report: {
          steps: [{ id: 'loudness', label: 'Normalización de sonoridad', detail: '-24.0 → -16.0 LUFS (+8.0 dB).' }],
          warnings: ['Aviso de prueba'],
          before: { duration_s: 2, peak_dbfs: -9, loudness_lufs: -24 },
          after: { duration_s: 1.8, peak_dbfs: -1, loudness_lufs: -16 },
        },
      },
    })
    render(
      <TooltipProvider>
        <GenerationMastering gen={processed} />
      </TooltipProvider>,
    )
    expect(screen.getByTestId('player')).toHaveTextContent('audio?v=2')
    fireEvent.click(screen.getByRole('radio', { name: 'Original' }))
    expect(screen.getByTestId('player')).toHaveTextContent('version=raw')
    expect(screen.getByText('Aviso de prueba')).toBeInTheDocument()
    expect(screen.getByText(/1 paso/)).toBeInTheDocument()
  })

  it('applies post-processing to a finished generation', async () => {
    fetchMock.mockResolvedValue(new Response(JSON.stringify(gen({ postprocess: { config: BASIC_MASTERING, report: null } })), { status: 200 }))
    useGenerationStore.setState({ items: [gen()] })
    render(
      <TooltipProvider>
        <GenerationMastering gen={gen()} />
      </TooltipProvider>,
    )
    fireEvent.click(screen.getByRole('button', { name: /Posprocesar/ }))
    fireEvent.click(screen.getByText('Normalizar pico').closest('label')!.querySelector('input')!)
    fireEvent.click(screen.getByRole('button', { name: 'Aplicar' }))
    await waitFor(() => expect(fetchMock).toHaveBeenCalled())
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('/api/generation/g1/postprocess')
    expect(JSON.parse(init.body).peak).toEqual({ enabled: true, target_dbfs: -1 })
    await waitFor(() => expect(useGenerationStore.getState().items[0].postprocess?.config).toEqual(BASIC_MASTERING))
  })
})
