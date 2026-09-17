import { create } from 'zustand'

import { ApiError } from '@/services/api'
import { transcriptionApi } from '@/services/transcription'
import { useReferencesStore } from '@/stores/references'
import type { ASRStatus, LanguageOption } from '@/types/api'

type Scope = 'full' | 'segment'

interface TranscriptionState {
  status: ASRStatus | null
  languages: LanguageOption[]
  /** `${referenceId}:${scope}` currently running */
  busy: Record<string, boolean>
  errors: Record<string, string | undefined>
  loadStatus: () => Promise<void>
  loadLanguages: () => Promise<void>
  download: () => Promise<void>
  unload: () => Promise<void>
  transcribe: (referenceId: string, scope: Scope, language: string | null, force?: boolean) => Promise<void>
  edit: (referenceId: string, transcriptId: string, text: string) => Promise<void>
  revert: (referenceId: string, transcriptId: string) => Promise<void>
}

const messageOf = (err: unknown) => (err instanceof ApiError ? err.message : String(err))
const POLL_MS = 3000
let pollTimer: number | undefined

export const useTranscriptionStore = create<TranscriptionState>()((set, get) => {
  const run = async (key: string, fn: () => Promise<unknown>) => {
    set((s) => ({ busy: { ...s.busy, [key]: true }, errors: { ...s.errors, [key]: undefined } }))
    try {
      await fn()
      await useReferencesStore.getState().load()
    } catch (err) {
      set((s) => ({ errors: { ...s.errors, [key]: messageOf(err) } }))
      if (err instanceof ApiError && err.code === 'ASR_MODEL_NOT_INSTALLED') void get().loadStatus()
    } finally {
      set((s) => ({ busy: { ...s.busy, [key]: false } }))
    }
  }

  return {
    status: null,
    languages: [],
    busy: {},
    errors: {},

    loadStatus: async () => {
      try {
        const status = await transcriptionApi.status()
        set({ status })
        window.clearTimeout(pollTimer)
        if (status.download_state === 'downloading') pollTimer = window.setTimeout(() => void get().loadStatus(), POLL_MS)
      } catch {
        /* backend offline: status bar already reports it */
      }
    },
    loadLanguages: async () => {
      if (get().languages.length) return
      try {
        const languages = await transcriptionApi.languages()
        set({ languages: Array.isArray(languages) ? languages : [] })
      } catch {
        /* keep auto-detect only */
      }
    },
    download: async () => {
      set({ status: await transcriptionApi.download() })
      void get().loadStatus()
    },
    unload: async () => {
      set({ status: await transcriptionApi.unload() })
    },
    transcribe: (referenceId, scope, language, force = false) =>
      run(`${referenceId}:${scope}`, async () => {
        await transcriptionApi.transcribe(referenceId, { scope, language, force })
        void get().loadStatus()
      }),
    edit: (referenceId, transcriptId, text) => run(`${referenceId}:edit`, () => transcriptionApi.edit(transcriptId, text)),
    revert: (referenceId, transcriptId) => run(`${referenceId}:edit`, () => transcriptionApi.revert(transcriptId)),
  }
})
