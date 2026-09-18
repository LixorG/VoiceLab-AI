import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { Recorder } from '@/features/voices/Recorder'
import { analyse, encodeWav, issuesOf, mixToMono, recordingName } from '@/lib/recording'
import { useReferencesStore } from '@/stores/references'

vi.mock('@/features/audio/AudioPlayer', () => ({ AudioPlayer: ({ url }: { url: string }) => <div data-testid="player">{url}</div> }))

const SR = 48_000
const sine = (seconds: number, amplitude: number) => Float32Array.from({ length: Math.round(seconds * SR) }, (_, i) => amplitude * Math.sin((2 * Math.PI * 220 * i) / SR))

describe('recording helpers', () => {
  it('encodes a 16-bit mono WAV with a valid header', async () => {
    const wav = encodeWav(Float32Array.from([0, 1, -1, 0.5]), 24_000)
    const view = new DataView(await wav.arrayBuffer())
    const text = (o: number) => String.fromCharCode(...[0, 1, 2, 3].map((i) => view.getUint8(o + i)))
    expect(wav.type).toBe('audio/wav')
    expect([text(0), text(8), text(12), text(36)]).toEqual(['RIFF', 'WAVE', 'fmt ', 'data'])
    expect(view.getUint16(22, true)).toBe(1) // mono
    expect(view.getUint32(24, true)).toBe(24_000)
    expect(view.getUint16(34, true)).toBe(16)
    expect(view.getUint32(40, true)).toBe(8)
    expect([view.getInt16(46, true), view.getInt16(48, true)]).toEqual([32767, -32768])
  })

  it('mixes channels and flags short, quiet, clipped and silent takes', () => {
    expect(Array.from(mixToMono([Float32Array.from([1, 0]), Float32Array.from([0, 1])]))).toEqual([0.5, 0.5])

    const good = analyse(sine(12, 0.3), SR)
    expect(good.durationS).toBeCloseTo(12)
    expect(issuesOf(good)).toEqual([])

    expect(issuesOf(analyse(sine(2, 0.3), SR))).toEqual(['tooShort'])
    expect(issuesOf(analyse(sine(40, 0.3), SR))).toEqual(['tooLong'])
    expect(issuesOf(analyse(sine(10, 0.005), SR))).toEqual(['tooQuiet'])
    expect(issuesOf(analyse(sine(10, 1.2).map((s) => Math.max(-1, Math.min(1, s))), SR))).toContain('clipping')
    expect(issuesOf(analyse(new Float32Array(SR * 10), SR))).toEqual(['silent'])
  })

  it('names recordings by date', () => {
    expect(recordingName(new Date(2026, 8, 18, 9, 5, 7))).toBe('grabacion-20260918-090507.wav')
  })
})

// ---------------------------------------------------------------- browser audio stand-ins
let decoded = sine(12, 0.3)
const trackStop = vi.fn()

class FakeMediaRecorder {
  static instances: FakeMediaRecorder[] = []
  mimeType = 'audio/webm'
  ondataavailable: ((e: { data: Blob }) => void) | null = null
  onstop: (() => void) | null = null
  constructor() {
    FakeMediaRecorder.instances.push(this)
  }
  start() {}
  stop() {
    this.ondataavailable?.({ data: new Blob(['x'], { type: 'audio/webm' }) })
    this.onstop?.()
  }
}

class FakeAudioContext {
  createAnalyser() {
    return { fftSize: 0, getFloatTimeDomainData: (b: Float32Array) => b.fill(0.25) }
  }
  createMediaStreamSource() {
    return { connect: () => undefined }
  }
  decodeAudioData() {
    return Promise.resolve({ numberOfChannels: 1, sampleRate: SR, getChannelData: () => decoded })
  }
  close() {
    return Promise.resolve()
  }
}

const getUserMedia = vi.fn()

beforeEach(() => {
  decoded = sine(12, 0.3)
  FakeMediaRecorder.instances = []
  getUserMedia.mockReset().mockResolvedValue({ getTracks: () => [{ stop: trackStop }] })
  vi.stubGlobal('MediaRecorder', FakeMediaRecorder)
  vi.stubGlobal('AudioContext', FakeAudioContext)
  Object.defineProperty(navigator, 'mediaDevices', { value: { getUserMedia }, configurable: true })
  vi.stubGlobal('URL', Object.assign(URL, { createObjectURL: () => 'blob:toma', revokeObjectURL: vi.fn() }))
})

afterEach(() => vi.unstubAllGlobals())

describe('Recorder', () => {
  it('records without browser voice processing, reviews the take and uploads it as WAV', async () => {
    const uploadFiles = vi.fn(async () => undefined)
    const onClose = vi.fn()
    useReferencesStore.setState({ uploadFiles })
    render(<Recorder onClose={onClose} />)

    const firstScript = screen.getByTestId('recorder-script').textContent
    fireEvent.click(screen.getByRole('button', { name: 'Otro texto' }))
    expect(screen.getByTestId('recorder-script').textContent).not.toBe(firstScript)

    fireEvent.click(screen.getByRole('button', { name: 'Empezar a grabar' }))
    expect(await screen.findByText('Grabando…')).toBeInTheDocument()
    expect(getUserMedia).toHaveBeenCalledWith({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
    })
    expect(screen.getByRole('meter', { name: 'Nivel de entrada' })).toBeInTheDocument()

    await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Detener' })))
    expect(await screen.findByText('Buena toma: duración y nivel adecuados.')).toBeInTheDocument()
    expect(screen.getByTestId('player')).toHaveTextContent('blob:toma')
    expect(screen.getByText(/12\.0 s · pico/)).toBeInTheDocument()
    expect(trackStop).toHaveBeenCalled() // microphone released

    fireEvent.click(screen.getByRole('button', { name: 'Guardar como referencia' }))
    await waitFor(() => expect(uploadFiles).toHaveBeenCalledTimes(1))
    const [[files]] = uploadFiles.mock.calls as unknown as [[File[]]]
    expect(files[0].name).toMatch(/^grabacion-\d{8}-\d{6}\.wav$/)
    expect(files[0].type).toBe('audio/wav')
    await waitFor(() => expect(onClose).toHaveBeenCalled())
  })

  it('warns about a poor take and lets you record again; silence cannot be saved', async () => {
    decoded = new Float32Array(SR * 3)
    render(<Recorder onClose={() => undefined} />)
    fireEvent.click(screen.getByRole('button', { name: 'Empezar a grabar' }))
    await screen.findByText('Grabando…')
    await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Detener' })))

    expect(await screen.findByText(/No se captó sonido/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Guardar como referencia' })).toBeDisabled()
    fireEvent.click(screen.getByRole('button', { name: 'Grabar de nuevo' }))
    expect(screen.getByRole('button', { name: 'Empezar a grabar' })).toBeInTheDocument()
  })

  it('explains a denied permission and an unsupported browser', async () => {
    getUserMedia.mockRejectedValueOnce(new DOMException('no', 'NotAllowedError'))
    const { unmount } = render(<Recorder onClose={() => undefined} />)
    fireEvent.click(screen.getByRole('button', { name: 'Empezar a grabar' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('No hay permiso para usar el micrófono')
    unmount()

    vi.stubGlobal('MediaRecorder', undefined)
    render(<Recorder onClose={() => undefined} />)
    fireEvent.click(screen.getByRole('button', { name: 'Empezar a grabar' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Este navegador no permite grabar audio')
  })
})
