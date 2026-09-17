import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { beforeEach, expect, it, vi } from 'vitest'

import { TooltipProvider } from '@/components/ui/tooltip'
import { ModelsPage } from '@/features/models/ModelsPage'
import { f5Config, f5Summary, qwenConfig, qwenSummary } from '@/test/engineFixtures'

const fetchMock = vi.fn()

beforeEach(() => {
  fetchMock.mockReset()
  vi.stubGlobal('fetch', fetchMock)
})

it('lists engines with licence warnings and switches the capability view per variant (mouse and keyboard)', async () => {
  fetchMock.mockImplementation((url: string) => {
    const body = url === '/api/models' ? [f5Summary, qwenSummary] : url.startsWith('/api/models/qwen3tts') ? qwenConfig : f5Config
    return Promise.resolve(new Response(JSON.stringify(body), { status: 200 }))
  })
  render(
    <TooltipProvider>
      <ModelsPage />
    </TooltipProvider>,
  )
  expect(await screen.findByRole('heading', { name: 'F5-TTS' })).toBeInTheDocument()
  expect(screen.getAllByText('Uso no comercial').length).toBeGreaterThan(0)

  const qwenCard = screen.getByRole('heading', { name: 'Qwen3-TTS' }).closest<HTMLElement>('[data-slot="card"]')!
  const rows = within(qwenCard).getAllByRole('row').filter((r) => r.getAttribute('aria-selected') != null)
  expect(rows[0]).toHaveAttribute('aria-selected', 'true')

  fireEvent.click(rows[1])
  expect(rows[1]).toHaveAttribute('aria-selected', 'true')
  fireEvent.keyDown(rows[0], { key: 'Enter' })
  expect(rows[0]).toHaveAttribute('aria-selected', 'true')
  await waitFor(() => expect(fetchMock.mock.calls.some(([u]) => String(u).includes('variant=custom-voice-1.7b'))).toBe(true))
})
