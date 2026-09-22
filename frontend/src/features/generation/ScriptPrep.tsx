import { AlertTriangle, Check, Loader2, Scissors, Wand2, X } from 'lucide-react'
import { useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { t } from '@/i18n/es'
import { formatDuration } from '@/lib/utils'
import { ApiError } from '@/services/api'
import { scriptApi } from '@/services/script'
import { keyOf, useEngineStore } from '@/stores/engine'
import { useGenerationStore } from '@/stores/generation'
import type { ScriptPrepared } from '@/types/script'

const ts = t.script
const messageOf = (err: unknown) => (err instanceof ApiError ? err.message : String(err))
const hasTags = (text: string) => /(?<!\\)\[[^[\]\n]{1,40}\]/.test(text)

/** «Preparar guion»: the same words, ready for speech (punctuation, paragraphs, formatting, tags). */
export function ScriptPrep() {
  const { text, setText, profileId, markup, setMarkup } = useGenerationStore()
  const [result, setResult] = useState<ScriptPrepared | null>(null)
  const [draft, setDraft] = useState('')
  const [applied, setApplied] = useState<Set<number>>(new Set())
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const run = async () => {
    const { engineId, variantId, valuesByKey } = useEngineStore.getState()
    setBusy(true)
    setError(null)
    try {
      const prepared = await scriptApi.prepare({ text, profile_id: profileId, params: valuesByKey[keyOf(engineId, variantId)] ?? {} })
      setResult(prepared)
      setDraft(prepared.text)
      setApplied(new Set())
    } catch (err) {
      setError(messageOf(err))
    } finally {
      setBusy(false)
    }
  }

  const use = () => {
    setText(draft)
    if (!markup && hasTags(draft)) setMarkup(true)
    setResult(null)
  }

  const split = (index: number) => {
    const s = result?.suggestions[index]
    if (!s || !draft.includes(s.before)) return
    setDraft(draft.replace(s.before, s.after))
    setApplied(new Set(applied).add(index))
  }

  return (
    <div className="space-y-2">
      <Button type="button" size="sm" variant="outline" disabled={busy || !text.trim()} onClick={() => void run()} title={ts.prepareHint}>
        {busy ? <Loader2 className="animate-spin" /> : <Wand2 />}
        {busy ? ts.preparing : ts.prepare}
      </Button>
      {error && <p role="alert" className="text-xs text-destructive">{error}</p>}

      {result && (
        <section aria-label={ts.title} className="space-y-3 rounded-lg border bg-muted/30 p-3">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="text-sm font-medium">{ts.title}</h3>
            <span className="text-xs text-muted-foreground" title={ts.statsHint}>
              {ts.stats(result.stats.words, result.stats.sentences, result.stats.average_words, formatDuration(result.stats.seconds))}
            </span>
            <Button type="button" size="icon" variant="ghost" className="ml-auto size-7" aria-label={ts.close} onClick={() => setResult(null)}>
              <X />
            </Button>
          </div>
          <p className="text-xs text-muted-foreground">{result.changed ? ts.sameWords : ts.nothing}</p>

          {result.alerts.length > 0 && (
            <div className="space-y-1">
              <p className="text-xs font-medium">{ts.alerts}</p>
              <ul className="space-y-1">
                {result.alerts.map((a) => (
                  <li key={`${a.kind}-${a.excerpt}`} className="flex items-start gap-1.5 text-xs text-warning">
                    <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
                    <span>
                      {a.excerpt && <span className="mr-1 rounded bg-warning/10 px-1 font-mono">{a.excerpt}</span>}
                      {a.message}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {result.suggestions.length > 0 && (
            <div className="space-y-1">
              <p className="text-xs font-medium">{ts.suggestions}</p>
              <p className="text-xs text-muted-foreground">{ts.suggestionsHint} {ts.rhythm}</p>
              <ul className="space-y-2">
                {result.suggestions.map((s, i) => (
                  <li key={s.before} className="space-y-1 rounded-md border bg-background p-2 text-xs">
                    <p className="text-muted-foreground">{s.reason}</p>
                    <p>{s.after}</p>
                    <Button type="button" size="sm" variant="outline" className="h-7" disabled={applied.has(i) || !draft.includes(s.before)} onClick={() => split(i)}>
                      {applied.has(i) ? <Check /> : <Scissors />}
                      {applied.has(i) ? ts.applied : ts.applySuggestion}
                    </Button>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {result.changed && (
            <>
              <div className="space-y-1">
                <p className="text-xs font-medium">{ts.result}</p>
                <p className="max-h-60 overflow-y-auto whitespace-pre-wrap rounded-md border bg-background p-2 text-sm" data-testid="prepared-text">
                  {draft}
                </p>
              </div>
              <details>
                <summary className="cursor-pointer text-xs text-primary">{ts.changes(result.changes.length)}</summary>
                <ul className="mt-1 space-y-1">
                  {result.changes.map((c, i) => (
                    // eslint-disable-next-line react/no-array-index-key -- the same change can repeat; order is stable
                    <li key={i} className="flex flex-wrap items-center gap-1.5 text-xs">
                      <Badge variant="outline">{ts.kinds[c.kind] ?? c.kind}</Badge>
                      <code className="rounded bg-muted px-1">{c.before.replaceAll('\n', '⏎')}</code>
                      <span aria-hidden>→</span>
                      <code className="rounded bg-muted px-1">{c.after ? c.after.replaceAll('\n', '⏎') : ts.removed}</code>
                      <span className="text-muted-foreground">{c.reason}</span>
                    </li>
                  ))}
                </ul>
              </details>
              {!markup && hasTags(draft) && <p className="text-xs text-muted-foreground">{ts.markupOn}</p>}
              <div className="flex gap-2">
                <Button type="button" size="sm" onClick={use}>
                  <Check />
                  {ts.use}
                </Button>
                <Button type="button" size="sm" variant="ghost" onClick={() => setResult(null)}>
                  {ts.close}
                </Button>
              </div>
            </>
          )}
        </section>
      )}
    </div>
  )
}
