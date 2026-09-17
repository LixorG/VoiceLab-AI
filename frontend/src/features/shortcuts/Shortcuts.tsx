import { useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { t } from '@/i18n/es'
import { useGenerationStore } from '@/stores/generation'
import { type Section, useUiStore } from '@/stores/ui'

const ts = t.shortcuts

/** Sections reachable with Alt+1…7, in the order of the sidebar. */
export const SECTION_KEYS: Section[] = ['generate', 'library', 'voices', 'experiments', 'projects', 'models', 'settings']

const isTyping = (target: EventTarget | null) => {
  const el = target as HTMLElement | null
  return !!el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable)
}

/** Global keyboard shortcuts plus the help panel that lists them (no hidden magic: everything here is shown). */
export function Shortcuts() {
  const [open, setOpen] = useState(false)
  const setSection = useUiStore((s) => s.setSection)
  const closeRef = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const { generate, submitting } = useGenerationStore.getState()
      if (event.key === 'Escape' && open) {
        setOpen(false)
        return
      }
      if (event.key === '?' && !isTyping(event.target)) {
        event.preventDefault()
        setOpen((v) => !v)
        return
      }
      if (event.altKey && /^[1-7]$/.test(event.key)) {
        event.preventDefault()
        setSection(SECTION_KEYS[Number(event.key) - 1])
        return
      }
      if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        event.preventDefault()
        if (useUiStore.getState().section !== 'generate' || submitting) return
        void generate(event.shiftKey)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, setSection])

  useEffect(() => {
    if (open) closeRef.current?.focus()
  }, [open])

  if (!open) return null

  const rows: [string, string][] = [
    ['Ctrl + Enter', ts.generate],
    ['Ctrl + Shift + Enter', ts.preview],
    ['Alt + 1…7', ts.sections],
    ['?', ts.help],
    ['Esc', ts.close],
  ]

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div role="dialog" aria-modal="true" aria-label={ts.title} className="w-full max-w-md rounded-lg border bg-panel p-5 shadow-lg">
        <h2 className="text-base font-semibold">{ts.title}</h2>
        <dl className="mt-3 divide-y">
          {rows.map(([keys, label]) => (
            <div key={keys} className="flex items-center justify-between gap-4 py-1.5 text-sm">
              <dt className="text-muted-foreground">{label}</dt>
              <dd className="font-mono text-xs">{keys}</dd>
            </div>
          ))}
        </dl>
        <div className="mt-4 text-right">
          <Button ref={closeRef} size="sm" variant="outline" onClick={() => setOpen(false)}>
            {t.common.close}
          </Button>
        </div>
      </div>
    </div>
  )
}

/** Screen-reader announcements for what the app is doing right now (generation status). */
export function LiveStatus() {
  const items = useGenerationStore((s) => s.items)
  const current = items.find((g) => g.status !== 'COMPLETED' && g.status !== 'FAILED' && g.status !== 'CANCELLED')
  const last = items[0]
  const message = current
    ? `${t.generation.status[current.status]} ${Math.round((current.progress ?? 0) * 100)} %`
    : last
      ? `${t.generation.status[last.status]}`
      : ''

  return (
    <p role="status" aria-live="polite" className="sr-only">
      {message}
    </p>
  )
}
