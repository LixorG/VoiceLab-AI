import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { Shortcuts } from '@/features/shortcuts/Shortcuts'
import { useGenerationStore } from '@/stores/generation'
import { useUiStore } from '@/stores/ui'

const generate = vi.fn()

beforeEach(() => {
  generate.mockReset()
  useUiStore.setState({ section: 'generate' })
  useGenerationStore.setState({ generate, submitting: null })
})

describe('Shortcuts', () => {
  it('generates with Ctrl+Enter and previews with Ctrl+Shift+Enter, only from Generar', () => {
    render(<Shortcuts />)

    fireEvent.keyDown(window, { key: 'Enter', ctrlKey: true })
    expect(generate).toHaveBeenCalledWith(false)

    fireEvent.keyDown(window, { key: 'Enter', ctrlKey: true, shiftKey: true })
    expect(generate).toHaveBeenCalledWith(true)

    useUiStore.setState({ section: 'voices' })
    fireEvent.keyDown(window, { key: 'Enter', ctrlKey: true })
    expect(generate).toHaveBeenCalledTimes(2)  // ignored outside Generar
  })

  it('does not generate while another one is being submitted', () => {
    useGenerationStore.setState({ submitting: 'full' })
    render(<Shortcuts />)
    fireEvent.keyDown(window, { key: 'Enter', ctrlKey: true })
    expect(generate).not.toHaveBeenCalled()
  })

  it('jumps between sections with Alt+number', () => {
    render(<Shortcuts />)
    fireEvent.keyDown(window, { key: '2', altKey: true })
    expect(useUiStore.getState().section).toBe('library')
    fireEvent.keyDown(window, { key: '7', altKey: true })
    expect(useUiStore.getState().section).toBe('settings')
  })

  it('opens the help with «?» and closes it with Esc or the button', async () => {
    render(<Shortcuts />)
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()

    fireEvent.keyDown(window, { key: '?' })
    const dialog = await screen.findByRole('dialog', { name: 'Atajos de teclado' })
    expect(dialog).toHaveTextContent('Ctrl + Enter')
    expect(dialog).toHaveTextContent('Alt + 1…7')

    fireEvent.keyDown(window, { key: 'Escape' })
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument())

    fireEvent.keyDown(window, { key: '?' })
    fireEvent.click(await screen.findByRole('button', { name: 'Cerrar' }))
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument())
  })

  it('ignores «?» while typing in a field', () => {
    render(
      <>
        <Shortcuts />
        <textarea aria-label="texto" />
      </>,
    )
    fireEvent.keyDown(screen.getByLabelText('texto'), { key: '?' })
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
  })
})
