import { fireEvent, render, screen } from '@testing-library/react'
import { useRef } from 'react'
import { beforeEach, describe, expect, it } from 'vitest'

import { TooltipProvider } from '@/components/ui/tooltip'
import { MarkupToolbar, PlanPreview } from '@/features/generation/MarkupTools'
import { useGenerationStore } from '@/stores/generation'
import { useProfilesStore } from '@/stores/profiles'

function Editor() {
  const ref = useRef<HTMLTextAreaElement>(null)
  const { text, setText } = useGenerationStore()
  return (
    <TooltipProvider>
      <MarkupToolbar textareaRef={ref} />
      <textarea ref={ref} aria-label="texto" value={text} onChange={(e) => setText(e.target.value)} />
    </TooltipProvider>
  )
}

beforeEach(() => {
  useGenerationStore.setState({ text: 'Hola mundo', markup: true, plan: null, planError: null })
  useProfilesStore.setState({ emotions: [{ id: 'happy', label: 'Feliz' }] } as never)
})

describe('MarkupToolbar', () => {
  it('wraps the selected text and inserts standalone tags at the cursor', () => {
    render(<Editor />)
    const area = screen.getByLabelText('texto') as HTMLTextAreaElement
    area.setSelectionRange(5, 10) // "mundo"
    fireEvent.click(screen.getByRole('button', { name: 'Susurro' }))
    expect(useGenerationStore.getState().text).toBe('Hola [susurro]mundo[/susurro]')

    area.setSelectionRange(4, 4)
    fireEvent.click(screen.getByRole('button', { name: 'Pausa' }))
    expect(useGenerationStore.getState().text).toBe('Hola[pausa:500ms] [susurro]mundo[/susurro]')
  })

  it('disables the tag buttons when markup interpretation is off', () => {
    render(<Editor />)
    fireEvent.click(screen.getByLabelText('Interpretar marcas'))
    expect(useGenerationStore.getState().markup).toBe(false)
    expect(screen.getByRole('button', { name: 'Énfasis' })).toBeDisabled()
  })
})

describe('PlanPreview', () => {
  it('shows segments with how each emotion is applied, pauses and warnings', () => {
    useGenerationStore.setState({
      plan: {
        segmented: true,
        warnings: ['Ningún motor actual genera risas.'],
        text_changes: [{ original: '15 €', replacement: 'quince euros', kind: 'número' }],
        normalize_language: 'es',
        segments: [
          { index: 0, text: 'Hola.', emotion: 'happy', emotion_via: 'reference', instruction: null, pause_before_ms: 0, pause_after_ms: 1500, reference_name: 'feliz.wav' },
          { index: 1, text: 'Adiós.', emotion: null, emotion_via: null, instruction: 'Whisper softly.', pause_before_ms: 0, pause_after_ms: 0, reference_name: null },
        ],
      },
    })
    render(
      <TooltipProvider>
        <PlanPreview />
      </TooltipProvider>,
    )
    expect(screen.getByText('Feliz')).toBeInTheDocument()
    expect(screen.getByText('referencia «feliz.wav»')).toBeInTheDocument()
    expect(screen.getByText('pausa 1.5 s')).toBeInTheDocument()
    expect(screen.getByText('por instrucción')).toBeInTheDocument()
    expect(screen.getByText('Ningún motor actual genera risas.')).toBeInTheDocument()
  })

  it('shows markup errors instead of the plan and nothing for empty text', () => {
    useGenerationStore.setState({ planError: 'Duración de pausa no válida.' })
    const { unmount } = render(<PlanPreview />)
    expect(screen.getByRole('alert')).toHaveTextContent('Duración de pausa no válida.')
    unmount()
    useGenerationStore.setState({ text: '  ' })
    const { container } = render(<PlanPreview />)
    expect(container).toBeEmptyDOMElement()
  })
})
