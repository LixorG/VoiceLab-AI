import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useProfilesStore } from '@/stores/profiles'
import type { ProfileRead } from '@/types/profiles'

const profile = (id: string, name: string) => ({ id, name, recommended_settings: {} }) as unknown as ProfileRead
const fetchMock = vi.fn()
const json = (body: unknown, status = 200) => Promise.resolve(new Response(JSON.stringify(body), { status }))

beforeEach(() => {
  fetchMock.mockReset()
  vi.stubGlobal('fetch', fetchMock)
  useProfilesStore.setState({ items: [], emotions: [], loading: false, error: null })
})

afterEach(() => vi.unstubAllGlobals())

describe('profiles store', () => {
  it('keeps profiles sorted in Spanish order and removes them after a 204', async () => {
    fetchMock.mockImplementation((url: string, init?: RequestInit) => {
      if (url === '/api/voices' && init?.method === 'POST') return json(profile('p3', 'Álvaro'), 201)
      if (url === '/api/voices') return json([profile('p1', 'Zoe'), profile('p2', 'Beatriz')])
      if (url === '/api/voices/p1?delete_references=true') return Promise.resolve(new Response(null, { status: 204 }))
      if (url === '/api/voices/p2/settings/f5tts' && init?.method === 'PUT') return json({ error_code: 'VALIDATION_ERROR', message: 'Parámetro no válido.' }, 422)
      return json({})
    })
    const store = useProfilesStore.getState()
    await store.load()
    expect(useProfilesStore.getState().items.map((p) => p.name)).toEqual(['Beatriz', 'Zoe'])
    await store.create({ name: 'Álvaro' })
    expect(useProfilesStore.getState().items.map((p) => p.name)).toEqual(['Álvaro', 'Beatriz', 'Zoe'])

    expect(await store.remove('p1', true)).toBe(true)
    expect(useProfilesStore.getState().items.map((p) => p.id)).not.toContain('p1')

    expect(await store.saveRecommended('p2', 'f5tts', null, { speed: 9 })).toBe(false)
    expect(useProfilesStore.getState().error).toBe('Parámetro no válido.')
    store.clearError()
    expect(useProfilesStore.getState().error).toBeNull()
  })

  it('imports a .voiceprofile and reports server, unexpected and network errors in Spanish', async () => {
    const file = new File(['PK'], 'narradora.voiceprofile')
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify(profile('p9', 'Narradora (importado)')), { status: 201 }))
    expect((await useProfilesStore.getState().importFile(file))?.id).toBe('p9')
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('/api/voices/import')
    expect((init.body as FormData).get('file')).toBeInstanceOf(File)

    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify({ error_code: 'PROFILE_IMPORT_ERROR', message: 'El archivo no es un perfil válido.' }), { status: 422 }))
    expect(await useProfilesStore.getState().importFile(file)).toBeNull()
    expect(useProfilesStore.getState().error).toBe('El archivo no es un perfil válido.')

    fetchMock.mockResolvedValueOnce(new Response('<html>', { status: 500 }))
    await useProfilesStore.getState().importFile(file)
    expect(useProfilesStore.getState().error).toMatch(/inesperado/i)

    fetchMock.mockRejectedValueOnce(new TypeError('Failed to fetch'))
    await useProfilesStore.getState().importFile(file)
    expect(useProfilesStore.getState().error).toMatch(/conectar|servidor/i)
  })

  it('loads the emotion catalogue only once', async () => {
    fetchMock.mockImplementation(() => json([{ id: 'happy', label: 'Feliz' }]))
    await useProfilesStore.getState().loadEmotions()
    await useProfilesStore.getState().loadEmotions()
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(useProfilesStore.getState().emotions).toEqual([{ id: 'happy', label: 'Feliz' }])
  })
})

describe('ui store', () => {
  it('falls back to memory when browser storage is blocked', async () => {
    vi.resetModules()
    const setItem = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new DOMException('blocked', 'SecurityError')
    })
    try {
      const { useUiStore } = await import('@/stores/ui')
      useUiStore.getState().toggleTheme()
      useUiStore.getState().toggleSidebar()
      useUiStore.getState().setSection('experiments')
      useUiStore.getState().setMode('advanced')
      expect(useUiStore.getState()).toMatchObject({ theme: 'light', sidebarCollapsed: true, section: 'experiments', mode: 'advanced' })
    } finally {
      setItem.mockRestore()
    }
  })
})
