import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it } from 'vitest'

import { DialoguePanel } from '@/features/generation/DialoguePanel'
import { useGenerationStore } from '@/stores/generation'
import { useProfilesStore } from '@/stores/profiles'
import type { GenerationPlan } from '@/types/generation'
import type { ProfileRead } from '@/types/profiles'

const plan = (speakers: string[]): GenerationPlan => ({
  segments: [],
  warnings: [],
  segmented: true,
  text_changes: [],
  normalize_language: 'es',
  detected_speakers: speakers,
})

const profile = (id: string, name: string) => ({ id, name, slug: id }) as unknown as ProfileRead

beforeEach(() => {
  useProfilesStore.setState({ items: [profile('p1', 'Hanna'), profile('p2', 'Carlos')] })
  useGenerationStore.setState({ plan: null, speakers: {}, turnPauseMs: 450 })
})

describe('DialoguePanel', () => {
  it('stays out of the way when the text has no characters', () => {
    const { container } = render(<DialoguePanel />)
    expect(container).toBeEmptyDOMElement()
  })

  it('asks for one voice per character and remembers the choice', () => {
    useGenerationStore.setState({ plan: plan(['Ana', 'Luis']) })
    render(<DialoguePanel />)

    expect(screen.getByText('Diálogo a varias voces')).toBeInTheDocument()
    expect(screen.getAllByText('Falta asignar voz')).toHaveLength(2)

    fireEvent.change(screen.getByLabelText('Voz de Ana'), { target: { value: 'p1' } })
    fireEvent.change(screen.getByLabelText('Voz de Luis'), { target: { value: 'p2' } })

    expect(useGenerationStore.getState().speakers).toEqual({ Ana: 'p1', Luis: 'p2' })
    expect(screen.queryByText('Falta asignar voz')).not.toBeInTheDocument()
  })

  it('lets a character go back to unassigned', () => {
    useGenerationStore.setState({ plan: plan(['Ana']), speakers: { Ana: 'p1' } })
    render(<DialoguePanel />)
    fireEvent.change(screen.getByLabelText('Voz de Ana'), { target: { value: '' } })
    expect(useGenerationStore.getState().speakers).toEqual({})
  })

  it('keeps the pause between turns within what the server accepts', () => {
    useGenerationStore.setState({ plan: plan(['Ana']) })
    render(<DialoguePanel />)
    const pause = screen.getByLabelText('Pausa entre turnos')

    fireEvent.change(pause, { target: { value: '700' } })
    expect(useGenerationStore.getState().turnPauseMs).toBe(700)

    fireEvent.change(pause, { target: { value: '99999' } })
    expect(useGenerationStore.getState().turnPauseMs).toBe(5000)
  })
})
