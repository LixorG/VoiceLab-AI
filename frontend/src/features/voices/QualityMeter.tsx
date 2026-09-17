import { Info } from 'lucide-react'

import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { t } from '@/i18n/es'
import { cn } from '@/lib/utils'
import type { QualityComponent } from '@/types/api'

const SEGMENTS = 20

export function qualityTone(score: number) {
  return score >= 85 ? 'text-success' : score >= 70 ? 'text-primary' : score >= 50 ? 'text-warning' : 'text-destructive'
}

/** Segmented meter (VU-style). Always labelled as a heuristic indicator. */
export function QualityMeter({
  score,
  label,
  components,
}: {
  score: number
  label: string | null
  components: QualityComponent[]
}) {
  const lit = Math.round((score / 100) * SEGMENTS)
  const tone = qualityTone(score)
  return (
    <div className="space-y-1.5">
      <div className="flex items-center gap-1.5 text-[11px] font-semibold tracking-[0.14em] text-muted-foreground uppercase">
        {t.references.quality}
        <Tooltip>
          <TooltipTrigger asChild>
            <button type="button" aria-label={t.references.qualityHint} className="text-muted-foreground hover:text-foreground">
              <Info className="size-3.5" />
            </button>
          </TooltipTrigger>
          <TooltipContent className="space-y-2">
            <p>{t.references.qualityHint}</p>
            <ul className="space-y-0.5 font-mono text-[11px]">
              {components.map((c) => (
                <li key={c.id} className="flex justify-between gap-4">
                  <span>{c.label}</span>
                  <span>{Math.round(c.score * 100)}%</span>
                </li>
              ))}
            </ul>
          </TooltipContent>
        </Tooltip>
      </div>
      <div className="flex items-center gap-3">
        <div className={cn('flex flex-1 gap-[3px]', tone)} role="meter" aria-valuenow={score} aria-valuemin={0} aria-valuemax={100} aria-label={t.references.quality}>
          {Array.from({ length: SEGMENTS }, (_, i) => (
            <span key={i} className={cn('h-2.5 flex-1 rounded-[2px]', i < lit ? 'bg-current' : 'bg-muted')} />
          ))}
        </div>
        <span className={cn('font-mono text-sm font-semibold tabular-nums', tone)}>{Math.round(score)}%</span>
        {label && <span className="text-xs text-muted-foreground">{label}</span>}
      </div>
    </div>
  )
}
