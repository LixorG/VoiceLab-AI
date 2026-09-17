import { AlertTriangle, Gauge, Loader2, Star } from 'lucide-react'
import { useEffect, useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { t } from '@/i18n/es'
import { cn, formatDuration } from '@/lib/utils'
import { evaluationApi } from '@/services/experiments'
import { ApiError } from '@/services/api'
import { useEngineStore } from '@/stores/engine'
import { type EvaluatorInfo, type GenerationRating, RATING_CRITERIA, type RatingCriterion } from '@/types/experiments'
import type { GenerationRead } from '@/types/generation'

const tc = t.compare

function Stars({ value, onChange, label }: { value: number | undefined; onChange: (v: number | undefined) => void; label: string }) {
  return (
    <div className="flex" role="radiogroup" aria-label={label}>
      {[1, 2, 3, 4, 5].map((n) => (
        <button
          key={n}
          type="button"
          role="radio"
          aria-checked={value === n}
          aria-label={`${label}: ${n}`}
          onClick={() => onChange(value === n ? undefined : n)}
          className="p-0.5 text-muted-foreground hover:text-primary"
        >
          <Star className={cn('size-3.5', value != null && n <= value && 'fill-primary text-primary')} />
        </button>
      ))}
    </div>
  )
}

export function RatingControl({ gen, onSaved }: { gen: GenerationRead; onSaved: (gen: GenerationRead) => void }) {
  const saved = (gen.rating ?? {}) as GenerationRating
  const [notes, setNotes] = useState(saved.notes ?? '')

  const save = async (patch: GenerationRating) => {
    const { rated_at: _ignored, ...current } = (gen.rating ?? {}) as GenerationRating
    void _ignored
    try {
      onSaved(await evaluationApi.rate(gen.id, { ...current, ...patch }))
    } catch {
      // keep the previous value; the table stays usable
    }
  }

  return (
    <div className="space-y-1">
      {RATING_CRITERIA.map((c: RatingCriterion) => (
        <div key={c} className="flex items-center justify-between gap-2 text-[11px]">
          <span className="text-muted-foreground">{tc.criteria[c]}</span>
          <Stars value={saved[c]} label={tc.criteria[c]} onChange={(v) => void save({ [c]: v })} />
        </div>
      ))}
      <input
        value={notes}
        onChange={(e) => setNotes(e.target.value)}
        onBlur={() => notes !== (saved.notes ?? '') && void save({ notes: notes || undefined })}
        placeholder={tc.notes}
        aria-label={tc.notes}
        maxLength={1000}
        className="h-7 w-full rounded-md border bg-background px-2 text-[11px]"
      />
    </div>
  )
}

function Row({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <tr className="border-t align-top">
      <th scope="row" className="py-2 pr-3 text-left text-[11px] font-medium whitespace-nowrap text-muted-foreground">
        {hint ? (
          <Tooltip>
            <TooltipTrigger asChild>
              <span tabIndex={0} className="cursor-help underline decoration-dotted underline-offset-2">
                {label}
              </span>
            </TooltipTrigger>
            <TooltipContent className="max-w-72">{hint}</TooltipContent>
          </Tooltip>
        ) : (
          label
        )}
      </th>
      {children}
    </tr>
  )
}

/** Side-by-side facts, automatic estimates and manual ratings for several generations. */
export function ComparisonTable({ generations, onUpdate }: { generations: GenerationRead[]; onUpdate: (gen: GenerationRead) => void }) {
  const engines = useEngineStore((s) => s.engines)
  const [evaluators, setEvaluators] = useState<EvaluatorInfo[]>([])
  const [busy, setBusy] = useState<string | null>(null)
  const [errors, setErrors] = useState<Record<string, string>>({})

  useEffect(() => {
    evaluationApi.evaluators().then(setEvaluators, () => setEvaluators([]))
  }, [])

  const canEvaluate = evaluators.some((e) => e.available)
  const unavailable = evaluators.filter((e) => !e.available)
  const metricIds = Array.from(new Set(generations.flatMap((g) => g.evaluation?.metrics.map((m) => m.id) ?? [])))
  const metricInfo = (id: string) => generations.flatMap((g) => g.evaluation?.metrics ?? []).find((m) => m.id === id)!
  const best = (id: string) => {
    const values = generations.map((g) => g.evaluation?.metrics.find((m) => m.id === id)).filter(Boolean)
    if (values.length < 2) return null
    const better = values[0]!.better
    return Math.min(...values.map((m) => (better === 'lower' ? m!.value : -m!.value))) * (better === 'lower' ? 1 : -1)
  }

  const evaluate = async (gen: GenerationRead) => {
    setBusy(gen.id)
    setErrors((e) => ({ ...e, [gen.id]: '' }))
    try {
      onUpdate(await evaluationApi.evaluate(gen.id))
    } catch (err) {
      setErrors((e) => ({ ...e, [gen.id]: err instanceof ApiError ? err.message : String(err) }))
    } finally {
      setBusy(null)
    }
  }

  const cell = (children: React.ReactNode, key: string) => (
    <td key={key} className="min-w-40 px-2 py-2 text-xs">
      {children}
    </td>
  )

  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse" data-testid="comparison-table">
        <thead>
          <tr>
            <th className="w-32">
              <span className="sr-only">{tc.metric}</span>
            </th>
            {generations.map((g, i) => (
              <th key={g.id} scope="col" className="px-2 pb-2 text-left text-xs font-semibold">
                <span className="mr-1 font-mono text-primary">{'ABCDEFGHIJ'[i]}</span>
                {g.label ?? engines.find((e) => e.id === g.engine)?.name ?? g.engine}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          <Row label={tc.engine}>{generations.map((g) => cell(<span className="font-mono text-[11px]">{g.variant ?? g.engine}</span>, g.id))}</Row>
          <Row label={tc.status}>
            {generations.map((g) =>
              cell(
                <Badge variant={g.status === 'COMPLETED' ? 'success' : g.status === 'FAILED' ? 'destructive' : 'warning'}>
                  {!['COMPLETED', 'FAILED', 'CANCELLED'].includes(g.status) && <Loader2 className="animate-spin" />}
                  {t.generation.status[g.status]}
                  {!['COMPLETED', 'FAILED', 'CANCELLED'].includes(g.status) && g.progress_available && ` ${Math.round(g.progress * 100)}%`}
                </Badge>,
                g.id,
              ),
            )}
          </Row>
          <Row label={tc.duration}>{generations.map((g) => cell(<span className="font-mono">{g.duration_s != null ? formatDuration(g.duration_s) : '—'}</span>, g.id))}</Row>
          <Row label={tc.seed}>{generations.map((g) => cell(<span className="font-mono">{g.seed ?? '—'}</span>, g.id))}</Row>
          <Row label={tc.rtf} hint={t.generation.rtfHint}>
            {generations.map((g) => cell(<span className="font-mono">{g.metrics?.rtf != null ? g.metrics.rtf.toFixed(2) : '—'}</span>, g.id))}
          </Row>
          <Row label={tc.postprocess}>{generations.map((g) => cell(g.postprocess ? tc.postprocessSteps(g.postprocess.report?.steps.length ?? 0) : tc.none, g.id))}</Row>

          {metricIds.map((id) => {
            const info = metricInfo(id)
            const top = best(id)
            return (
              <Row key={id} label={info.label} hint={`${info.description} ${tc.estimate}`}>
                {generations.map((g) => {
                  const m = g.evaluation?.metrics.find((x) => x.id === id)
                  return cell(
                    m ? (
                      <span className={cn('font-mono', top != null && m.value === top && 'font-semibold text-success')}>
                        {m.display}
                        {g.evaluation?.stale && (
                          <Badge variant="outline" className="ml-1" title={tc.staleHint}>
                            {tc.stale}
                          </Badge>
                        )}
                      </span>
                    ) : (
                      '—'
                    ),
                    g.id,
                  )
                })}
              </Row>
            )
          })}

          {generations.some((g) => g.evaluation?.details.intelligibility?.transcript) && (
            <Row label={tc.heard} hint={tc.heardHint}>
              {generations.map((g) => cell(<span className="text-[11px] italic">{g.evaluation?.details.intelligibility?.transcript ?? '—'}</span>, g.id))}
            </Row>
          )}

          {unavailable.map((e) => (
            <Row key={e.id} label={e.label} hint={e.description}>
              {generations.map((g) =>
                cell(
                  <span className="text-[11px] text-muted-foreground" title={e.reason ?? undefined}>
                    {tc.unavailable}
                  </span>,
                  g.id,
                ),
              )}
            </Row>
          ))}

          <Row label={tc.evaluate} hint={tc.evaluateHint}>
            {generations.map((g) =>
              cell(
                <div className="space-y-1">
                  <Button size="sm" variant="outline" className="h-7 text-[11px]" disabled={!canEvaluate || g.status !== 'COMPLETED' || busy != null} onClick={() => void evaluate(g)} title={canEvaluate ? tc.evaluateHint : evaluators.find((e) => e.id === 'intelligibility')?.reason ?? undefined}>
                    {busy === g.id ? <Loader2 className="animate-spin" /> : <Gauge />}
                    {g.evaluation ? tc.reevaluate : tc.evaluate}
                  </Button>
                  {errors[g.id] && (
                    <p role="alert" className="flex gap-1 text-[11px] text-destructive">
                      <AlertTriangle className="size-3 shrink-0" />
                      {errors[g.id]}
                    </p>
                  )}
                </div>,
                g.id,
              ),
            )}
          </Row>
          <Row label={tc.rating} hint={tc.ratingHint}>
            {generations.map((g) => cell(g.status === 'COMPLETED' ? <RatingControl gen={g} onSaved={onUpdate} /> : '—', g.id))}
          </Row>
        </tbody>
      </table>
    </div>
  )
}
