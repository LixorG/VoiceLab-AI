import { Check, Dices } from 'lucide-react'

import { Badge } from '@/components/ui/badge'
import { t } from '@/i18n/es'
import { cn } from '@/lib/utils'
import { useGenerationStore } from '@/stores/generation'
import type { BestTake, GenerationRead } from '@/types/generation'

const tb = t.bestTake
const CHOICES = [1, 2, 3, 5]

/** How many takes of each sentence to generate; the one that reads the text best is the one that is kept. */
export function BestTakeControl() {
  const { takes, setTakes } = useGenerationStore()
  return (
    <div className="space-y-1">
      <div className="flex items-center gap-2">
        <Dices className="size-3.5 text-muted-foreground" />
        <span className="text-xs font-medium">{tb.title}</span>
        <div className="ml-auto flex gap-1" role="radiogroup" aria-label={tb.title}>
          {CHOICES.map((n) => (
            <button
              key={n}
              type="button"
              role="radio"
              aria-checked={takes === n}
              onClick={() => setTakes(n)}
              className={cn(
                'h-7 w-8 rounded-md border text-xs transition-colors',
                takes === n ? 'border-primary bg-primary/10 text-foreground' : 'text-muted-foreground hover:bg-muted',
              )}
            >
              {n}
            </button>
          ))}
        </div>
      </div>
      <p className="text-xs text-muted-foreground">{takes === 1 ? tb.offHint : tb.onHint(takes)}</p>
    </div>
  )
}

const percent = (value: number | null) => (value == null ? '—' : `${(value * 100).toFixed(0)} %`)

/** What the app kept and why, on a finished generation. */
export function BestTakeReport({ generation }: { generation: GenerationRead }) {
  const report = (generation.metrics?.best_take ?? null) as BestTake[] | null
  if (!report?.length) return null
  const total = report[0]?.takes.length ?? 0

  return (
    <details className="rounded-md border bg-muted/30 p-2">
      <summary className="cursor-pointer text-xs">
        <Badge variant="accent">{tb.badge(total)}</Badge>
        <span className="ml-2 text-muted-foreground">{tb.reportHint}</span>
      </summary>
      <ul className="mt-2 space-y-2">
        {report.map((entry) => (
          <li key={entry.segment ?? 0} className="space-y-1">
            {entry.segment != null && <p className="text-[11px] text-muted-foreground">{tb.segment(entry.segment + 1)}</p>}
            <ul className="space-y-0.5">
              {entry.takes.map((take) => (
                <li
                  key={take.take}
                  className={cn('flex flex-wrap items-center gap-2 rounded px-1.5 py-1 text-[11px]',
                    take.take === entry.chosen ? 'bg-primary/10' : 'text-muted-foreground')}
                >
                  {take.take === entry.chosen ? <Check className="size-3 text-primary" /> : <span className="w-3" />}
                  <span className="font-mono">{tb.take(take.take + 1)}</span>
                  <span className="font-mono" title={tb.wer}>{tb.werShort(percent(take.wer))}</span>
                  <span className="font-mono" title={tb.similarity}>{tb.similarityShort(take.similarity == null ? '—' : take.similarity.toFixed(2))}</span>
                  <span className="font-mono">{take.duration_s.toFixed(1)} s</span>
                  {take.notes?.length > 0 && <span className="text-warning">{take.notes.join(' · ')}</span>}
                </li>
              ))}
            </ul>
          </li>
        ))}
      </ul>
    </details>
  )
}
