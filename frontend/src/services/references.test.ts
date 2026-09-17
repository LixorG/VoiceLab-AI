import { afterEach, describe, expect, it, vi } from 'vitest'

import { ApiError } from '@/services/api'
import { uploadReference } from '@/services/references'

/** XMLHttpRequest double that lets each test decide how the upload ends. */
class FakeXHR {
  static last: FakeXHR
  upload: { onprogress: ((e: { lengthComputable: boolean; loaded: number; total: number }) => void) | null; onload: (() => void) | null } = { onprogress: null, onload: null }
  onload: (() => void) | null = null
  onerror: (() => void) | null = null
  status = 0
  responseText = ''
  method = ''
  url = ''
  headers: Record<string, string> = {}
  body: FormData | null = null

  constructor() {
    FakeXHR.last = this
  }
  open(method: string, url: string) {
    this.method = method
    this.url = url
  }
  setRequestHeader(name: string, value: string) {
    this.headers[name] = value
  }
  send(body: FormData) {
    this.body = body
  }
  respond(status: number, text: string) {
    this.status = status
    this.responseText = text
    this.onload?.()
  }
}

afterEach(() => vi.unstubAllGlobals())

const file = () => new File(['RIFF'], 'voz.wav', { type: 'audio/wav' })

describe('uploadReference', () => {
  it('reports progress and resolves with the created reference, sending the profile', async () => {
    vi.stubGlobal('XMLHttpRequest', FakeXHR)
    const progress: number[] = []
    const promise = uploadReference(file(), (f) => progress.push(f), 'p1')
    const xhr = FakeXHR.last
    expect([xhr.method, xhr.url, xhr.headers.Accept]).toEqual(['POST', '/api/references', 'application/json'])
    expect(xhr.body?.get('profile_id')).toBe('p1')
    expect((xhr.body!.get('file') as File).name).toBe('voz.wav')

    xhr.upload.onprogress?.({ lengthComputable: true, loaded: 50, total: 200 })
    xhr.upload.onprogress?.({ lengthComputable: false, loaded: 0, total: 0 })
    xhr.upload.onload?.()
    xhr.respond(201, JSON.stringify({ reference: { id: 'r1' }, duplicate: false }))
    await expect(promise).resolves.toEqual({ reference: { id: 'r1' }, duplicate: false })
    expect(progress).toEqual([0.25, 1])
  })

  it('turns API errors, unexpected responses and network failures into Spanish ApiErrors', async () => {
    vi.stubGlobal('XMLHttpRequest', FakeXHR)

    const rejected = uploadReference(file(), () => undefined)
    expect(FakeXHR.last.body?.get('profile_id')).toBeNull()
    FakeXHR.last.respond(422, JSON.stringify({ error_code: 'UNSUPPORTED_FORMAT', message: 'Formato no admitido.', details: { formato: 'exe' } }))
    await expect(rejected).rejects.toMatchObject({ code: 'UNSUPPORTED_FORMAT', message: 'Formato no admitido.', status: 422 })

    const html = uploadReference(file(), () => undefined)
    FakeXHR.last.respond(502, '<html>Bad gateway</html>')
    const error = await html.catch((e: unknown) => e)
    expect(error).toBeInstanceOf(ApiError)
    expect(error).toMatchObject({ code: 'UNEXPECTED_ERROR', status: 502 })

    const offline = uploadReference(file(), () => undefined)
    FakeXHR.last.onerror?.()
    await expect(offline).rejects.toMatchObject({ code: 'NETWORK_ERROR', status: 0 })
  })
})
