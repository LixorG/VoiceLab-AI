import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { SentenceList } from '@/features/generation/SentenceEditor'
import { useGenerationStore } from '@/stores/generation'
import type { GenerationRead, PlannedSegment } from '@/types/generation'

const segment = (index: number, text: string): PlannedSegment => ({
  index,
  text,
  emotion: null,
  emotion_via: null,
  instruction: null,
  pause_before_ms: 0,
  pause_after_ms: 500,
  reference_name: null,
  seed: 100 + index,
})

const generation = (over: Partial<GenerationRead> = {}): GenerationRead =>
  ({
    id: 'g1',
    status: 'COMPLETED',
    updated_at: '2026-09-23T10:00:00Z',
    segments: [segment(0, 'Primera frase.'), segment(1, 'Segunda frase.')],
    ...over,
  }) as unknown as GenerationRead

beforeEach(() => {
  useGenerationStore.setState({ items: [] })
  vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(new Response(JSON.stringify({
    generation: { ...generation(), status: 'QUEUED' }, job_id: 'g1', queue_position: 0,
  }), { status: 202, headers: { 'Content-Type': 'application/json' } }))))
})

describe('SentenceList', () => {
  it('repeats only the chosen sentence, with its takes and seed', async () => {
    render(<SentenceList gen={generation()} />)
    expect(screen.getByText('Si una frase salió mal, repítela sola: el resto del audio no cambia.')).toBeInTheDocument()

    fireEvent.change(screen.getByLabelText('Semilla 2'), { target: { value: '555' } })
    fireEvent.change(screen.getByLabelText('Tomas 2'), { target: { value: '3' } })
    fireEvent.click(screen.getAllByRole('button', { name: 'Repetir frase' })[1])

    await waitFor(() => expect(fetch).toHaveBeenCalled())
    const [url, init] = (fetch as unknown as { mock: { calls: [string, RequestInit][] } }).mock.calls[0]
    expect(url).toBe('/api/generation/g1/segments/1/regenerate')
    expect(JSON.parse(init.body as string)).toEqual({ seed: 555, takes: 3 })
  })

  it('sends the corrected words when the sentence is edited', async () => {
    render(<SentenceList gen={generation()} />)
    fireEvent.click(screen.getByLabelText('Corregir el texto 1'))
    fireEvent.change(screen.getByLabelText('Texto de la frase'), { target: { value: 'Primera frase corregida.' } })
    fireEvent.click(screen.getAllByRole('button', { name: 'Repetir frase' })[0])

    await waitFor(() => expect(fetch).toHaveBeenCalled())
    const [, init] = (fetch as unknown as { mock: { calls: [string, RequestInit][] } }).mock.calls[0]
    expect(JSON.parse(init.body as string)).toEqual({ text: 'Primera frase corregida.', takes: 1 })
  })

  it('explains that a single-sentence generation is repeated whole', () => {
    render(<SentenceList gen={generation({ segments: [segment(0, 'Una sola frase.')] })} />)
    expect(screen.getByText(/usa «Repetir»/)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Repetir frase' })).not.toBeInTheDocument()
  })

  it('does not offer to repeat a sentence while the generation is still running', () => {
    render(<SentenceList gen={generation({ status: 'GENERATING' })} />)
    expect(screen.queryByRole('button', { name: 'Repetir frase' })).not.toBeInTheDocument()
    expect(screen.getByText('Segunda frase.')).toBeInTheDocument()
  })
})
