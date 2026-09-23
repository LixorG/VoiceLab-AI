import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { BestTakeControl, BestTakeReport } from '@/features/generation/BestTakePanel'
import { useGenerationStore } from '@/stores/generation'
import type { GenerationRead, TakeScore } from '@/types/generation'

const take = (over: Partial<TakeScore> = {}): TakeScore => ({
  take: 0,
  seed: 100,
  duration_s: 2.4,
  speech_ratio: 0.9,
  peak: 0.5,
  wer: 0.0,
  similarity: 0.94,
  score: 0.96,
  notes: [],
  ...over,
})

const finished = (): GenerationRead =>
  ({
    id: 'g1',
    status: 'COMPLETED',
    takes: 3,
    metrics: {
      takes: 3,
      best_take: [{
        chosen: 1,
        segment: 0,
        takes: [
          take({ take: 0, seed: 100, wer: 0.25, duration_s: 5.1, notes: ['mucho más larga que las demás tomas'] }),
          take({ take: 1, seed: 101 }),
          take({ take: 2, seed: 102, wer: 0.08, similarity: 0.9 }),
        ],
      }],
    },
  }) as unknown as GenerationRead

beforeEach(() => {
  useGenerationStore.setState({ takes: 1 })
  vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(new Response('[]', { status: 200 }))))
})

describe('BestTakeControl', () => {
  it('explains the cost and stores the number of takes', () => {
    render(<BestTakeControl />)
    expect(screen.getByText('Una toma por frase. Si alguna sale mal, se repite a mano.')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('radio', { name: '3' }))
    expect(useGenerationStore.getState().takes).toBe(3)
    expect(screen.getByText(/3 tomas de cada frase .* Tarda unas 3 veces más\./)).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: '3' })).toBeChecked()
  })
})

describe('BestTakeReport', () => {
  it('shows which take was kept and what the others scored', () => {
    render(<BestTakeReport generation={finished()} />)
    expect(screen.getByText('Mejor de 3')).toBeInTheDocument()
    fireEvent.click(screen.getByText('Ver las tomas y por qué se eligió esta'))

    expect(screen.getByText('Frase 1')).toBeInTheDocument()
    expect(screen.getAllByText(/Toma \d/)).toHaveLength(3)
    expect(screen.getByText('mucho más larga que las demás tomas')).toBeInTheDocument()
    expect(screen.getByText('palabras mal 25 %')).toBeInTheDocument()
    expect(screen.getAllByText('voz 0.94')).toHaveLength(2)  // the long take and the winner share it
  })

  it('shows nothing when a single take was generated', () => {
    const { container } = render(<BestTakeReport generation={{ metrics: null } as GenerationRead} />)
    expect(container).toBeEmptyDOMElement()
  })
})
