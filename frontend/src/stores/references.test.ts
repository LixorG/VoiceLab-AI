import { beforeEach, describe, expect, it, vi } from 'vitest'

import { ApiError } from '@/services/api'
import { referencesApi, uploadReference } from '@/services/references'
import { useReferencesStore } from '@/stores/references'

vi.mock('@/services/references', () => ({
  uploadReference: vi.fn(),
  referencesApi: {
    list: vi.fn(),
    formats: vi.fn(),
    update: vi.fn(),
    reanalyze: vi.fn(),
    remove: vi.fn(),
    peaks: vi.fn(),
  },
}))

const file = (name: string) => new File(['x'], name, { type: 'audio/wav' })

beforeEach(() => {
  vi.mocked(referencesApi.list).mockResolvedValue([])
  useReferencesStore.setState({ items: [], pending: [], notice: null, error: null })
})

describe('references store', () => {
  it('uploads sequentially, removes finished jobs and reports duplicates', async () => {
    const order: string[] = []
    vi.mocked(uploadReference).mockImplementation(async (f, onProgress) => {
      order.push(f.name)
      onProgress(1)
      return { reference: {} as never, duplicate: f.name === 'b.wav' }
    })
    await useReferencesStore.getState().uploadFiles([file('a.wav'), file('b.wav')])
    const state = useReferencesStore.getState()
    expect(order).toEqual(['a.wav', 'b.wav'])
    expect(state.pending).toEqual([])
    expect(state.notice).toMatch(/ya estaba/)
    expect(referencesApi.list).toHaveBeenCalled()
  })

  it('keeps failed uploads with the backend message', async () => {
    vi.mocked(uploadReference).mockRejectedValue(
      new ApiError('UNSUPPORTED_FORMAT', 'Formato de audio no compatible.', 415),
    )
    await useReferencesStore.getState().uploadFiles([file('x.wav')])
    const [job] = useReferencesStore.getState().pending
    expect(job).toMatchObject({ phase: 'error', error: 'Formato de audio no compatible.' })
    useReferencesStore.getState().dismissPending(job.key)
    expect(useReferencesStore.getState().pending).toEqual([])
  })
})
