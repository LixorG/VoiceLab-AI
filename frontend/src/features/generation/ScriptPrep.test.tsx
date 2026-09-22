import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { ScriptPrep } from '@/features/generation/ScriptPrep'
import { keyOf, useEngineStore } from '@/stores/engine'
import { useGenerationStore } from '@/stores/generation'
import type { ScriptPrepared } from '@/types/script'

const prepared = (over: Partial<ScriptPrepared> = {}): ScriptPrepared => ({
  text: '[emoción:entusiasmado] That is growth.[/emoción]\nFirst idea, and the second idea goes on and on and on and on and on and on and on.',
  changed: true,
  language: 'en',
  changes: [
    { kind: 'etiqueta', before: '[excited]', after: '[emoción:entusiasmado]', reason: 'Emoción hasta el final de la frase.' },
    { kind: 'emoji', before: '💪', after: '', reason: 'Los emojis no se leen.' },
  ],
  alerts: [{ kind: 'idioma', message: 'Posibles palabras en otro idioma.', excerpt: 'cuatro' }],
  suggestions: [{
    before: 'First idea, and the second idea goes on and on and on and on and on and on and on.',
    after: 'First idea. And the second idea goes on and on and on and on and on and on and on.',
    reason: '22 palabras en una sola frase.',
  }],
  stats: { words: 26, sentences: 2, paragraphs: 1, average_words: 13, longest_words: 22, seconds: 9.5 },
  ...over,
})

const fetchMock = vi.fn()

beforeEach(() => {
  useGenerationStore.setState({ text: '[excited] That is growth 💪', profileId: 'p1', markup: false })
  useEngineStore.setState({ engineId: 'qwen3tts', variantId: 'base-1.7b', valuesByKey: { [keyOf('qwen3tts', 'base-1.7b')]: { language: 'English' } } })
  fetchMock.mockReset()
  fetchMock.mockImplementation(() => Promise.resolve(new Response(JSON.stringify(prepared()), { status: 200 })))
  vi.stubGlobal('fetch', fetchMock)
})

describe('ScriptPrep', () => {
  it('prepares the script with the engine language and the voice, and uses it', async () => {
    render(<ScriptPrep />)
    fireEvent.click(screen.getByRole('button', { name: 'Preparar guion' }))
    expect(await screen.findByText('Guion listo para voz')).toBeInTheDocument()
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('/api/script/prepare')
    expect(JSON.parse(String(init.body))).toEqual({ text: '[excited] That is growth 💪', profile_id: 'p1', params: { language: 'English' } })

    expect(screen.getByText(/26 palabras · 2 frases · 13 palabras por frase/)).toBeInTheDocument()
    expect(screen.getByText('cuatro')).toBeInTheDocument()
    expect(screen.getByText('2 cambios')).toBeInTheDocument()
    expect(screen.getByText('(se quita)')).toBeInTheDocument()
    expect(screen.getByText('Las marcas se activan porque el texto preparado las usa.')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Partir' }))
    expect(screen.getByTestId('prepared-text')).toHaveTextContent('First idea. And the second idea')
    expect(screen.getByRole('button', { name: 'Aplicado' })).toBeDisabled()

    fireEvent.click(screen.getByRole('button', { name: 'Usar este texto' }))
    const state = useGenerationStore.getState()
    expect(state.text).toContain('First idea. And the second idea')
    expect(state.markup).toBe(true)
    expect(screen.queryByText('Guion listo para voz')).not.toBeInTheDocument()
  })

  it('says when there is nothing to change and reports errors', async () => {
    fetchMock.mockImplementationOnce(() => Promise.resolve(new Response(JSON.stringify(prepared({ changed: false, changes: [], suggestions: [], alerts: [] })), { status: 200 })))
    render(<ScriptPrep />)
    fireEvent.click(screen.getByRole('button', { name: 'Preparar guion' }))
    expect(await screen.findByText('El texto ya estaba listo: no hay nada que cambiar.')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Usar este texto' })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Cerrar' }))

    fetchMock.mockImplementationOnce(() =>
      Promise.resolve(new Response(JSON.stringify({ error_code: 'VALIDATION_ERROR', message: 'Texto demasiado largo.' }), { status: 422 })),
    )
    fireEvent.click(screen.getByRole('button', { name: 'Preparar guion' }))
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('Texto demasiado largo.'))
  })

  it('is disabled without text', () => {
    useGenerationStore.setState({ text: '  ' })
    render(<ScriptPrep />)
    expect(screen.getByRole('button', { name: 'Preparar guion' })).toBeDisabled()
  })
})
