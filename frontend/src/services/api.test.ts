import { afterEach, describe, expect, it, vi } from 'vitest'

import { ApiError, request } from '@/services/api'

afterEach(() => vi.unstubAllGlobals())

describe('request', () => {
  it('returns JSON on success', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ status: 'ok' }), { status: 200 })))
    await expect(request('/system/health')).resolves.toEqual({ status: 'ok' })
    expect(fetch).toHaveBeenCalledWith('/api/system/health', expect.anything())
  })

  it('maps backend error format to ApiError', async () => {
    const body = { error_code: 'GPU_MEMORY_ERROR', message: 'No hay suficiente memoria de GPU', details: null }
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify(body), { status: 507 })))
    const err = await request('/x').catch((e: unknown) => e)
    expect(err).toBeInstanceOf(ApiError)
    expect(err).toMatchObject({ code: 'GPU_MEMORY_ERROR', status: 507, message: body.message })
  })

  it('gives a Spanish message when the server is unreachable', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')))
    await expect(request('/x')).rejects.toMatchObject({ code: 'NETWORK_ERROR', message: expect.stringContaining('servidor local') })
  })

  it('handles non-JSON error responses', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('<html>', { status: 502 })))
    await expect(request('/x')).rejects.toMatchObject({ code: 'UNEXPECTED_ERROR', status: 502 })
  })
})
