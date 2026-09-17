import { act, fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { AudioPlayer } from '@/features/audio/AudioPlayer'
import { SyncedPlayers } from '@/features/audio/SyncedPlayers'
import { Waveform } from '@/features/audio/Waveform'
import { FakeRegions, FakeWaveSurfer, resetWaveSurferMock } from '@/test/wavesurferMock'

vi.mock('wavesurfer.js', async () => (await import('@/test/wavesurferMock')).wavesurferModule)
vi.mock('wavesurfer.js/dist/plugins/regions.esm.js', async () => (await import('@/test/wavesurferMock')).regionsModule)

beforeEach(() => {
  resetWaveSurferMock()
})

describe('AudioPlayer', () => {
  it('creates one waveform per url, toggles playback and destroys it on unmount', () => {
    const { rerender, unmount } = render(<AudioPlayer url="/a.wav" duration={4} />)
    const ws = FakeWaveSurfer.instances[0]
    expect(ws.options.url).toBe('/a.wav')
    act(() => ws.ready(4))
    expect(screen.getByText('0:00.0 / 0:04.0')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Reproducir' }))
    expect(ws.playing).toBe(true)
    act(() => ws.setTime(1.5))
    expect(screen.getByText('0:01.5 / 0:04.0')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Pausar' }))
    expect(ws.playing).toBe(false)

    rerender(<AudioPlayer url="/b.wav" duration={4} />)
    expect(ws.destroyed).toBe(true)
    expect(FakeWaveSurfer.instances).toHaveLength(2)
    unmount()
    expect(FakeWaveSurfer.instances[1].destroyed).toBe(true)
  })
})

const PEAKS = [0.1, 0.5] // stable reference, like the store provides

describe('Waveform', () => {
  it('applies the saved selection only after ready, reports user selections and keeps a single region', () => {
    const onChange = vi.fn()
    const { rerender } = render(<Waveform url="/r.wav" peaks={PEAKS} duration={10} selection={{ start_s: 1, end_s: 3 }} onSelectionChange={onChange} />)
    const ws = FakeWaveSurfer.instances[0]
    const regions = FakeRegions.instances[0]
    expect(regions.regions).toHaveLength(0) // not before the real duration is known
    act(() => ws.ready(10))
    expect(regions.regions.map((r) => [r.start, r.end])).toEqual([[1, 3]])
    expect(onChange).not.toHaveBeenCalled() // programmatic region does not echo back

    act(() => regions.userSelect(2.345, 5.678))
    expect(onChange).toHaveBeenCalledWith({ start_s: 2.35, end_s: 5.68 })
    expect(regions.regions).toHaveLength(1)

    rerender(<Waveform url="/r.wav" peaks={PEAKS} duration={10} selection={{ start_s: 4, end_s: 6 }} onSelectionChange={onChange} />)
    expect(regions.regions.map((r) => [r.start, r.end])).toEqual([[4, 6]])
    fireEvent.click(screen.getByRole('button', { name: 'Reproducir selección' }))
    expect(regions.regions[0].played).toBe(1)

    rerender(<Waveform url="/r.wav" peaks={PEAKS} duration={10} selection={null} onSelectionChange={onChange} />)
    expect(regions.regions).toHaveLength(0)
    expect(screen.getByRole('button', { name: 'Reproducir selección' })).toBeDisabled()
  })
})

describe('SyncedPlayers', () => {
  const tracks = [
    { id: 'a', label: 'F5-TTS', url: '/a.wav' },
    { id: 'b', label: 'Qwen3-TTS', url: '/b.wav' },
  ]

  it('waits for every track, plays all together, keeps only the selected one audible and seeks all', () => {
    vi.spyOn(window, 'requestAnimationFrame').mockImplementation(() => 1)
    vi.spyOn(window, 'cancelAnimationFrame').mockImplementation(() => undefined)
    render(<SyncedPlayers tracks={tracks} />)
    const [a, b] = FakeWaveSurfer.instances
    const play = screen.getByRole('button', { name: 'Reproducir' })
    expect(play).toBeDisabled()

    act(() => {
      a.ready(6)
      b.ready(4)
    })
    expect(play).toBeEnabled()
    expect([a.muted, b.muted]).toEqual([false, true])
    expect(screen.getByText('0:00.0 / 0:06.0')).toBeInTheDocument()

    fireEvent.click(play)
    expect([a.playing, b.playing]).toEqual([true, true])

    fireEvent.click(screen.getByRole('radio', { name: 'B' }))
    expect([a.muted, b.muted]).toEqual([true, false])
    fireEvent.keyDown(screen.getByRole('radio', { name: 'B' }), { key: '1' })
    expect([a.muted, b.muted]).toEqual([false, true])

    act(() => a.emit('interaction', 5))
    expect(a.time).toBe(5)
    expect(b.time).toBe(4) // clamped to its own duration
    expect(b.playing).toBe(false) // already past its end

    fireEvent.click(screen.getByRole('button', { name: 'Pausar' }))
    expect(a.playing).toBe(false)

    fireEvent.change(screen.getByLabelText('Zoom de la forma de onda'), { target: { value: '120' } })
    expect([a.zoomLevel, b.zoomLevel]).toEqual([120, 120])

    fireEvent.click(screen.getByRole('button', { name: 'Volver al inicio' }))
    expect([a.time, b.time]).toEqual([0, 0])
  })
})
