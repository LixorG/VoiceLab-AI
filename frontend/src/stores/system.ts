import { create } from 'zustand'

import { ApiError, systemApi } from '@/services/api'
import type { EnvironmentReport, LicenseEntry, SystemInfo } from '@/types/api'

type Connection = 'checking' | 'connected' | 'disconnected'

interface SystemState {
  connection: Connection
  version: string | null
  info: SystemInfo | null
  environment: EnvironmentReport | null
  licenses: LicenseEntry[] | null
  error: string | null
  checkHealth: () => Promise<void>
  loadEnvironment: () => Promise<void>
  loadInfo: () => Promise<void>
  loadLicenses: () => Promise<void>
}

const messageOf = (err: unknown) => (err instanceof ApiError ? err.message : String(err))

export const useSystemStore = create<SystemState>()((set) => ({
  connection: 'checking',
  version: null,
  info: null,
  environment: null,
  licenses: null,
  error: null,

  checkHealth: async () => {
    try {
      const health = await systemApi.health()
      set({ connection: 'connected', version: health.version })
    } catch (err) {
      set({ connection: 'disconnected', error: messageOf(err) })
    }
  },
  loadEnvironment: async () => {
    try {
      set({ environment: await systemApi.environment(), error: null })
    } catch (err) {
      set({ error: messageOf(err) })
    }
  },
  loadInfo: async () => {
    try {
      set({ info: await systemApi.info() })
    } catch (err) {
      set({ error: messageOf(err) })
    }
  },
  loadLicenses: async () => {
    try {
      set({ licenses: await systemApi.licenses() })
    } catch (err) {
      set({ error: messageOf(err) })
    }
  },
}))
