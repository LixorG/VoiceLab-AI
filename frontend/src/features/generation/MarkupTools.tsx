import { AlertTriangle, CircleHelp, ListTree } from 'lucide-react'
import type { RefObject } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { t } from '@/i18n/es'
import { useEngineStore } from '@/stores/engine'
import { useGenerationStore } from '@/stores/generation'
import { useProfilesStore } from '@/stores/profiles'

const tx = t.expression

/** Only tags that do something: [tono] is gone (no engine changes pitch from the text; that is Posprocesado). */
const SNIPPETS: { id: keyof typeof tx.tags; open: string; close?: string; needs?: 'emotion' | 'instruction' }[] = [
  { id: 'pause', open: '[pausa:500ms]' },
  { id: 'longPause', open: '[pausa:1s]' },
  { id: 'breath', open: '[respira]' },
  { id: 'emotion', open: '[emoción:feliz]', close: '[/emoción]', needs: 'emotion' },
  { id: 'emphasis', open: '[énfasis]', close: '[/énfasis]', needs: 'instruction' },
  { id: 'whisper', open: '[susurro]', close: '[/susurro]', needs: 'instruction' },
]

/** Buttons that wrap the current selection (or insert at the cursor) with SpeechMarkup tags. */
export function MarkupToolbar({ textareaRef }: { textareaRef: RefObject<HTMLTextAreaElement | null> }) {
  const { text, setText, markup, setMarkup, normalize, setNormalize } = useGenerationStore()
  const controls = useEngineStore((s) => s.config?.capabilities.controls)

  /** A tag this engine cannot apply is shown disabled with the engine's own reason (never silently ignored). */
  const unusable = (needs?: 'emotion' | 'instruction'): string | null => {
    if (!needs || !controls) return null
    // emphasis and whisper only reach a model that takes style instructions
    const capability = controls[needs === 'emotion' ? 'emotion' : 'instruction']
    const ok = needs === 'emotion' ? capability.source !== 'unavailable' : capability.source === 'native'
    return ok ? null : tx.unavailableTag(capability.reason ?? '')
  }

  const insert = (open: string, close?: string) => {
    const el = textareaRef.current
    const start = el?.selectionStart ?? text.length
    const end = el?.selectionEnd ?? text.length
    const selected = text.slice(start, end)
    const insertion = close ? `${open}${selected}${close}` : open
    setText(text.slice(0, start) + insertion + text.slice(end))
    requestAnimationFrame(() => {
      if (!el) return
      el.focus()
      const cursor = close && !selected ? start + open.length : start + insertion.length
      el.setSelectionRange(cursor, cursor)
    })
  }

  return (
    <div className="flex flex-wrap items-center gap-1" role="toolbar" aria-label={tx.insert}>
      {SNIPPETS.map((s) => {
        const reason = unusable(s.needs)
        return (
          <Button
            key={s.id}
            type="button"
            size="sm"
            variant="outline"
            className="h-7 px-2 text-[11px]"
            disabled={!markup || reason !== null}
            title={reason ?? tx.tagHints[s.id]}
            onClick={() => insert(s.open, s.close)}
          >
            {tx.tags[s.id]}
          </Button>
        )
      })}
      <Tooltip>
        <TooltipTrigger asChild>
          <button type="button" aria-label={tx.help} className="text-muted-foreground hover:text-foreground">
            <CircleHelp className="size-4" />
          </button>
        </TooltipTrigger>
        <TooltipContent className="max-w-96">{tx.help}</TooltipContent>
      </Tooltip>
      <label className="ml-auto flex items-center gap-1.5 text-xs text-muted-foreground" title={tx.markupHint}>
        <input type="checkbox" checked={markup} onChange={(e) => setMarkup(e.target.checked)} className="accent-[var(--primary)]" />
        {tx.markup}
      </label>
      <label className="flex items-center gap-1.5 text-xs text-muted-foreground" title={tx.normalizeHint}>
        <input type="checkbox" checked={normalize} onChange={(e) => setNormalize(e.target.checked)} className="accent-[var(--primary)]" />
        {tx.normalize}
      </label>
    </div>
  )
}

/** Shows how the backend will split and apply the text for the selected engine (no generation involved). */
export function PlanPreview() {
  const { plan, planError, text } = useGenerationStore()
  const emotions = useProfilesStore((s) => s.emotions)
  const label = (id: string | null) => emotions.find((e) => e.id === id)?.label ?? id

  if (!text.trim()) return null
  if (planError) {
    return (
      <p role="alert" className="flex items-start gap-1.5 text-xs text-destructive">
        <AlertTriangle className="mt-px size-3.5 shrink-0" />
        {planError}
      </p>
    )
  }
  if (!plan) return null

  return (
    <div className="space-y-2 rounded-md border border-dashed p-3" data-testid="plan-preview">
      <div className="flex items-center gap-1.5 text-xs font-medium">
        <ListTree className="size-3.5" />
        {tx.plan}
        <span className="font-normal text-muted-foreground">
          · {plan.segmented ? tx.planSegments(plan.segments.length) : tx.planSimple}
        </span>
      </div>
      {plan.segmented && (
        <ol className="space-y-1">
          {plan.segments.map((s) => (
            <li key={s.index} className="flex flex-wrap items-center gap-1.5 text-xs">
              <span className="font-mono text-muted-foreground">{s.index + 1}.</span>
              <span className="min-w-0 max-w-full truncate">{s.text}</span>
              {s.emotion && <Badge variant="accent">{label(s.emotion)}</Badge>}
              {s.emotion_via === 'reference' && s.reference_name && <Badge variant="outline">{tx.viaReferenceShort(s.reference_name)}</Badge>}
              {s.instruction && (
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Badge variant="outline" tabIndex={0} className="cursor-help">
                      {tx.viaInstructionShort}
                    </Badge>
                  </TooltipTrigger>
                  <TooltipContent>{s.instruction}</TooltipContent>
                </Tooltip>
              )}
              {s.pause_after_ms > 0 && <Badge>{tx.pauseAfter(s.pause_after_ms)}</Badge>}
            </li>
          ))}
        </ol>
      )}
      {plan.text_changes.length > 0 && (
        <div className="space-y-1 border-t pt-2">
          <p className="text-xs font-medium">{tx.textChanges}</p>
          <ul className="flex flex-wrap gap-1.5">
            {plan.text_changes.map((c) => (
              <li key={`${c.kind}-${c.original}`} className="rounded bg-muted px-1.5 py-0.5 font-mono text-[11px]" title={c.kind}>
                {c.original} → {c.replacement}
              </li>
            ))}
          </ul>
        </div>
      )}
      {plan.warnings.map((w) => (
        <p key={w} className="flex items-start gap-1.5 text-xs text-warning">
          <AlertTriangle className="mt-px size-3.5 shrink-0" />
          {w}
        </p>
      ))}
    </div>
  )
}
