import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { AppearanceCard } from '@/features/settings/AppearanceCard'
import { resolveTheme, useUiStore } from '@/stores/ui'

const systemPrefersDark = (dark: boolean) =>
  vi.stubGlobal('matchMedia', (query: string) => ({
    matches: dark && query.includes('dark'),
    media: query,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  }))

beforeEach(() => {
  useUiStore.setState({ theme: 'dark' })
  systemPrefersDark(false)
})

describe('AppearanceCard', () => {
  it('switches between day, night and the system setting, and says which one is active', () => {
    render(<AppearanceCard />)
    expect(screen.getByRole('radio', { name: 'Oscuro' })).toBeChecked()
    expect(screen.getByText('Se guarda en este equipo y se aplica a toda la aplicación.')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('radio', { name: 'Claro' }))
    expect(useUiStore.getState().theme).toBe('light')
    expect(screen.getByRole('radio', { name: 'Claro' })).toBeChecked()

    fireEvent.click(screen.getByRole('radio', { name: 'Automático' }))
    expect(useUiStore.getState().theme).toBe('system')
    expect(screen.getByText('Sigue al sistema: ahora está en claro.')).toBeInTheDocument()
  })

  it('follows the system preference when «Automático» is chosen', () => {
    systemPrefersDark(true)
    useUiStore.setState({ theme: 'system' })
    render(<AppearanceCard />)
    expect(resolveTheme('system')).toBe('dark')
    expect(screen.getByText('Sigue al sistema: ahora está en oscuro.')).toBeInTheDocument()
  })
})

describe('theme state', () => {
  it('the sidebar switch flips the palette that is actually painted', () => {
    systemPrefersDark(true)
    useUiStore.setState({ theme: 'system' })
    useUiStore.getState().toggleTheme() // showing dark (from the system) → day mode
    expect(useUiStore.getState().theme).toBe('light')
    useUiStore.getState().toggleTheme()
    expect(useUiStore.getState().theme).toBe('dark')
    expect(resolveTheme('light')).toBe('light')
  })
})
