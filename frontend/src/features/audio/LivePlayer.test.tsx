import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { LivePlayer } from '@/features/audio/LivePlayer'

const started: { at: number; duration: number }[] = []
let closed = 0

class FakeContext {
  currentTime = 10
  destination = {}
  decodeAudioData(buffer: ArrayBuffer) {
    return Promise.resolve({ duration: new Uint8Array(buffer)[0] / 10 }) // first byte = duration in tenths
  }
  createBufferSource() {
    const node = {
      buffer: null as { duration: number } | null,
      connect: () => undefined,
      start: (at: number) => started.push({ at, duration: node.buffer!.duration }),
      stop: () => undefined,
    }
    return node
  }
  close() {
    closed += 1
    return Promise.resolve()
  }
}

const fetchMock = vi.fn((url: string) => {
  const index = Number(url.split('/').pop())
  return Promise.resolve(new Response(new Uint8Array([5 + index]), { status: 200 })) // 0.5 s, 0.6 s, …
})

beforeEach(() => {
  started.length = 0
  closed = 0
  fetchMock.mockClear()
  vi.stubGlobal('AudioContext', FakeContext)
  vi.stubGlobal('fetch', fetchMock)
})
afterEach(() => vi.unstubAllGlobals())

describe('LivePlayer', () => {
  it('stays hidden until there is audio, and never offers to start once the generation ended', () => {
    const { rerender } = render(<LivePlayer generationId="g1" chunks={0} finished={false} />)
    expect(screen.queryByTestId('live-player')).not.toBeInTheDocument()
    rerender(<LivePlayer generationId="g1" chunks={3} finished />)
    expect(screen.queryByTestId('live-player')).not.toBeInTheDocument()
  })

  it('waits for two chunks, then plays them back to back and keeps up with new ones', async () => {
    const { rerender } = render(<LivePlayer generationId="g1" chunks={1} finished={false} />)
    fireEvent.click(screen.getByRole('button', { name: 'Escuchar en directo' }))

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/api/generation/g1/stream/0'))
    expect(started).toEqual([])  // one chunk is not enough to start without running dry

    rerender(<LivePlayer generationId="g1" chunks={2} finished={false} />)
    await waitFor(() => expect(started).toHaveLength(2))
    expect(started[0].at).toBeCloseTo(10.05)
    expect(started[1].at).toBeCloseTo(10.05 + 0.5)  // gapless: right after the first one

    rerender(<LivePlayer generationId="g1" chunks={3} finished={false} />)
    await waitFor(() => expect(started).toHaveLength(3))
    expect(started[2].at).toBeCloseTo(10.05 + 0.5 + 0.6)
    expect(screen.getByText('Reproduciendo en directo…')).toBeInTheDocument()
  })

  it('starts with a single chunk if the generation ends first, and stops on demand', async () => {
    const { rerender } = render(<LivePlayer generationId="g2" chunks={1} finished={false} />)
    fireEvent.click(screen.getByRole('button', { name: 'Escuchar en directo' }))
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    rerender(<LivePlayer generationId="g2" chunks={1} finished />)
    await waitFor(() => expect(started).toHaveLength(1))

    await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Detener' })))
    expect(closed).toBe(1)
    expect(screen.queryByTestId('live-player')).not.toBeInTheDocument()  // finished + stopped: nothing left to offer
  })

  it('reports a chunk that cannot be decoded', async () => {
    vi.stubGlobal('AudioContext', class extends FakeContext {
      decodeAudioData() {
        return Promise.reject(new Error('bad'))
      }
    })
    render(<LivePlayer generationId="g3" chunks={2} finished={false} />)
    fireEvent.click(screen.getByRole('button', { name: 'Escuchar en directo' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('No se pudo reproducir la vista previa en directo.')
  })
})
