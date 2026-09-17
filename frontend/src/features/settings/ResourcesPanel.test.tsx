import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { MemoryCard, StorageCard } from '@/features/settings/ResourcesPanel'
import { formatBytes, type MemoryStatus } from '@/services/resources'

const memory = (over: Partial<MemoryStatus> = {}): MemoryStatus => ({
  gpu: { name: 'RTX 3080 Laptop', total_mb: 16384, free_mb: 12000, used_mb: 4384, torch_allocated_mb: 2100, torch_reserved_mb: 2500 },
  process_ram_mb: 1800,
  system_ram_total_mb: 32000,
  system_ram_available_mb: 20000,
  tts: { engine: 'f5tts', variant: 'F5TTS_v1_Base', device: 'cuda', loaded_at: '', last_used_at: '', in_use: false, idle_unload_in_s: 530 },
  asr: { model: 'large-v3-turbo', loaded: false, device: null, last_used_at: null, idle_unload_in_s: null },
  idle_unload_minutes: { tts: 15, asr: 10 },
  queue_active: 0,
  queue_waiting: 0,
  ...over,
})

const fetchMock = vi.fn()
const ok = (body: unknown) => Promise.resolve(new Response(JSON.stringify(body), { status: 200 }))

beforeEach(() => {
  fetchMock.mockReset()
  vi.stubGlobal('fetch', fetchMock)
})

describe('MemoryCard', () => {
  it('shows GPU usage, the loaded model with its auto-unload countdown and releases it', async () => {
    fetchMock.mockImplementation((url: string) =>
      url === '/api/system/memory/release' ? ok({ released: ['tts'], skipped: {}, status: memory({ tts: null }) }) : ok(memory()),
    )
    render(<MemoryCard />)
    expect(await screen.findByText('GPU · RTX 3080 Laptop')).toBeInTheDocument()
    expect(screen.getByText('se libera en 9 min sin uso')).toBeInTheDocument()
    const [releaseTts, releaseAsr] = screen.getAllByRole('button', { name: 'Liberar' })
    expect(releaseAsr).toBeDisabled()
    fireEvent.click(releaseTts)
    expect(await screen.findByText('Memoria liberada.')).toBeInTheDocument()
    const [, init] = fetchMock.mock.calls.find(([u]) => u === '/api/system/memory/release')!
    expect(JSON.parse(init.body)).toEqual({ tts: true, asr: false })
  })

  it('does not allow releasing a model in use', async () => {
    fetchMock.mockImplementation(() => ok(memory({ tts: { ...memory().tts!, in_use: true, idle_unload_in_s: null }, queue_active: 1 })))
    render(<MemoryCard />)
    expect(await screen.findByText('En uso')).toBeInTheDocument()
    expect(screen.getAllByRole('button', { name: 'Liberar' })[0]).toBeDisabled()
    expect(screen.getByText(/1 en curso/)).toBeInTheDocument()
  })
})

describe('StorageCard', () => {
  it('lists usage and cleans the selected categories after confirming', async () => {
    const usage = {
      categories: [
        { id: 'generated', label: 'Audios generados', bytes: 5_000_000, files: 12, removable: false, description: '' },
        { id: 'transcript_cache', label: 'Caché de transcripciones', bytes: 2048, files: 3, removable: true, description: '' },
      ],
      orphans_bytes: 1_048_576,
      orphans_files: 2,
      total_bytes: 5_002_048,
    }
    fetchMock.mockImplementation((url: string) =>
      url === '/api/system/storage/cleanup' ? ok({ removed_files: 2, freed_bytes: 1_048_576, details: { orphans: 2 }, skipped: [] }) : ok(usage),
    )
    vi.stubGlobal('confirm', () => true)
    render(<StorageCard />)
    expect(await screen.findByText(/2 archivos huérfanos \(1\.0 MB\)/)).toBeInTheDocument()
    fireEvent.click(screen.getByLabelText(/Caché de transcripciones/, { selector: 'input' }))
    fireEvent.click(screen.getByRole('button', { name: 'Limpiar seleccionados' }))
    await waitFor(() => expect(screen.getByText('2 archivos eliminados, 1.0 MB liberados.')).toBeInTheDocument())
    const [, init] = fetchMock.mock.calls.find(([u]) => u === '/api/system/storage/cleanup')!
    expect(JSON.parse(init.body)).toEqual({ orphans: true, temp: true, reference_clips: false, transcript_cache: true })
  })
})

it('formats byte sizes', () => {
  expect([formatBytes(512), formatBytes(2048), formatBytes(5_000_000), formatBytes(3 * 1024 ** 3)]).toEqual(['512 B', '2.0 KB', '4.8 MB', '3.0 GB'])
})
