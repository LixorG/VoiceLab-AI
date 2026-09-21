import { AlertTriangle, Columns2, Loader2, Repeat, Trash2, X } from 'lucide-react'
import { useEffect, useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { AudioDownloadMenu } from '@/components/ui/download-menu'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { LivePlayer } from '@/features/audio/LivePlayer'
import { CopyParamsButton } from '@/features/generation/ParamsClipboard'
import { SyncedPlayers } from '@/features/audio/SyncedPlayers'
import { ComparisonTable } from '@/features/comparison/ComparisonTable'
import { GenerationMastering } from '@/features/postprocess/GenerationMastering'
import { t } from '@/i18n/es'
import { cn, formatDuration } from '@/lib/utils'
import { useEngineStore } from '@/stores/engine'
import { useGenerationStore } from '@/stores/generation'
import { useReferencesStore } from '@/stores/references'
import { useUiStore } from '@/stores/ui'
import { type GenerationRead, TERMINAL_STATUSES } from '@/types/generation'

const tg = t.generation

function GenerationItem({ gen, variationIndex }: { gen: GenerationRead; variationIndex: number | null }) {
  const { cancel, remove, repeat, compareIds, toggleCompare } = useGenerationStore()
  const engineName = useEngineStore((s) => s.engines.find((e) => e.id === gen.engine)?.name ?? gen.engine)
  const running = !TERMINAL_STATUSES.includes(gen.status)
  const [showSegments, setShowSegments] = useState(false)
  const ref = gen.reference
  const range = ref && ref.start_s != null && ref.end_s != null ? ` (${formatDuration(ref.start_s)}–${formatDuration(ref.end_s)})` : ''

  return (
    <article className="space-y-2 rounded-lg border bg-background/40 p-3" data-status={gen.status}>
      <header className="flex flex-wrap items-center gap-1.5">
        <Badge variant={gen.kind === 'preview' ? 'outline' : gen.kind === 'variation' ? 'accent' : 'default'}>
          {gen.kind === 'variation' && variationIndex != null ? t.variations.label(variationIndex) : (tg.kind[gen.kind as 'preview' | 'single'] ?? gen.kind)}
        </Badge>
        <span className="text-xs font-medium">{engineName}</span>
        {gen.variant && <span className="font-mono text-[11px] text-muted-foreground">{gen.variant}</span>}
        {gen.status === 'COMPLETED' && (
          <label className="ml-auto flex items-center gap-1 text-[11px] text-muted-foreground" title={t.compare.selectHint}>
            <input type="checkbox" className="accent-[var(--primary)]" checked={compareIds.includes(gen.id)} onChange={() => toggleCompare(gen.id)} />
            {t.compare.select}
          </label>
        )}
        <Badge variant={gen.status === 'COMPLETED' ? 'success' : gen.status === 'FAILED' ? 'destructive' : gen.status === 'CANCELLED' ? 'outline' : 'warning'} className={gen.status === 'COMPLETED' ? '' : 'ml-auto'}>
          {running && <Loader2 className="animate-spin" />}
          {tg.status[gen.status]}
        </Badge>
      </header>

      <p className="line-clamp-2 text-sm" title={gen.text}>
        {gen.text}
      </p>

      <LivePlayer generationId={gen.id} chunks={gen.stream_chunks ?? 0} finished={!running} />
      {running && (
        <div className="space-y-1" aria-live="polite">
          {gen.progress_available || gen.status !== 'GENERATING' ? (
            <div className="h-1.5 overflow-hidden rounded-full bg-muted" role="progressbar" aria-valuenow={Math.round(gen.progress * 100)} aria-valuemin={0} aria-valuemax={100}>
              <div className="h-full bg-primary transition-[width] duration-300" style={{ width: `${Math.max(3, Math.round(gen.progress * 100))}%` }} />
            </div>
          ) : (
            <div className="relative h-1.5 overflow-hidden rounded-full bg-muted" role="progressbar" aria-busy aria-label={tg.indeterminate}>
              <div className="absolute inset-y-0 w-1/3 animate-[voicelab-indeterminate_1.4s_ease-in-out_infinite] rounded-full bg-primary" />
            </div>
          )}
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>{gen.message ?? tg.status[gen.status]}</span>
            <Button size="sm" variant="ghost" className="h-7" onClick={() => void cancel(gen.id)}>
              <X />
              {tg.cancel}
            </Button>
          </div>
        </div>
      )}

      {gen.status === 'FAILED' && (
        <p role="alert" className="flex items-start gap-1.5 text-xs text-destructive">
          <AlertTriangle className="mt-px size-3.5 shrink-0" />
          {gen.message ?? tg.failed}
        </p>
      )}

      {gen.status === 'COMPLETED' && gen.audio_url && <GenerationMastering gen={gen} />}

      {gen.segments.length > 0 && (
        <div className="text-xs">
          <button type="button" className="text-muted-foreground underline-offset-2 hover:underline" onClick={() => setShowSegments((v) => !v)} aria-expanded={showSegments}>
            {tg.segments(gen.segments.length)} · {tg.showSegments}
          </button>
          {showSegments && (
            <ol className="mt-1 space-y-0.5">
              {gen.segments.map((s) => (
                <li key={s.index} className="flex flex-wrap gap-1.5 text-muted-foreground">
                  <span className="font-mono">{s.index + 1}.</span>
                  <span className="text-foreground">{s.text}</span>
                  {s.emotion && <Badge variant="accent">{s.emotion}</Badge>}
                  {s.reference_name && <span>· {s.reference_name}</span>}
                  {s.seed != null && <span className="font-mono">· {tg.seed} {s.seed}</span>}
                  {s.pause_after_ms > 0 && <span>· {t.expression.pauseAfter(s.pause_after_ms)}</span>}
                </li>
              ))}
            </ol>
          )}
        </div>
      )}

      {gen.warnings.map((w) => (
        <p key={w} className="flex items-start gap-1.5 text-xs text-warning">
          <AlertTriangle className="mt-px size-3.5 shrink-0" />
          {w}
        </p>
      ))}

      <footer className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-muted-foreground">
        {gen.seed != null && (
          <span className="font-mono">
            {tg.seed} {gen.seed}
          </span>
        )}
        {gen.duration_s != null && <span className="font-mono">{formatDuration(gen.duration_s)}</span>}
        {gen.metrics?.generation_s != null && (
          <span className="font-mono">
            {tg.time} {gen.metrics.generation_s.toFixed(1)} s
          </span>
        )}
        {gen.metrics?.rtf != null && (
          <Tooltip>
            <TooltipTrigger asChild>
              <span className="cursor-help font-mono" tabIndex={0}>
                {tg.rtf(gen.metrics.rtf)}
              </span>
            </TooltipTrigger>
            <TooltipContent>{tg.rtfHint}</TooltipContent>
          </Tooltip>
        )}
        {ref && <span className="truncate">{tg.reference(ref.name ?? '', range)}</span>}
        <span className="ml-auto flex gap-0.5">
          {gen.audio_url && <AudioDownloadMenu url={gen.audio_url} />}
          <CopyParamsButton compact gen={gen} label={gen.label ?? `${gen.variant ?? gen.engine} · «${gen.text.length > 24 ? `${gen.text.slice(0, 24)}…` : gen.text}»`} />
          {!running && (
            <Tooltip>
              <TooltipTrigger asChild>
                <Button size="icon" variant="ghost" className="size-7" onClick={() => void repeat(gen)} aria-label={tg.repeat}>
                  <Repeat />
                </Button>
              </TooltipTrigger>
              <TooltipContent>{tg.repeatHint}</TooltipContent>
            </Tooltip>
          )}
          <Button size="icon" variant="ghost" className={cn('size-7 hover:text-destructive')} onClick={() => void remove(gen.id)} aria-label={tg.remove} title={tg.remove}>
            <Trash2 />
          </Button>
        </span>
      </footer>
    </article>
  )
}

/** 1-based position of a variation within its group (the group root is the first created). */
function variationIndexOf(items: GenerationRead[], gen: GenerationRead): number | null {
  if (gen.kind !== 'variation') return null
  const root = gen.parent_id ?? gen.id
  const group = items
    .filter((g) => g.kind === 'variation' && (g.id === root || g.parent_id === root))
    .sort((a, b) => a.created_at.localeCompare(b.created_at))
  const index = group.findIndex((g) => g.id === gen.id)
  return index >= 0 ? index + 1 : null
}

function ComparePanel() {
  const { items, compareIds, clearCompare, replaceItem } = useGenerationStore()
  const selected = compareIds.map((id) => items.find((g) => g.id === id)).filter((g): g is GenerationRead => !!g && g.status === 'COMPLETED' && !!g.audio_url)
  if (selected.length < 2) return null
  return (
    <section className="space-y-3 rounded-lg border border-primary/40 bg-primary/5 p-3" aria-label={t.compare.title}>
      <header className="flex items-center gap-2">
        <Columns2 className="size-4" />
        <h3 className="text-sm font-semibold">{t.compare.title}</h3>
        <Button size="sm" variant="ghost" className="ml-auto h-7" onClick={clearCompare}>
          <X />
          {t.compare.close}
        </Button>
      </header>
      <SyncedPlayers tracks={selected.map((g) => ({ id: g.id, label: g.label ?? `${g.engine} · ${g.seed ?? '—'}`, url: g.audio_url! }))} />
      <p className="text-[11px] text-muted-foreground">{t.compare.disclaimer}</p>
      <ComparisonTable generations={selected} onUpdate={replaceItem} />
    </section>
  )
}

export function GenerationList() {
  const { items, load, compareIds } = useGenerationStore()
  const references = useReferencesStore((s) => s.items)
  const setSection = useUiStore((s) => s.setSection)

  useEffect(() => {
    void load()
  }, [load])

  return (
    <Card>
      <CardHeader>
        <CardTitle>{tg.results}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {compareIds.length === 1 && <p className="text-xs text-muted-foreground">{t.compare.pickAnother}</p>}
        <ComparePanel />
        {items.length ? (
          items.map((gen) => <GenerationItem key={gen.id} gen={gen} variationIndex={variationIndexOf(items, gen)} />)
        ) : references.length === 0 ? (
          <div className="space-y-2 rounded-md border border-dashed p-4 text-sm">
            <p className="font-medium">{t.emptyStart.title}</p>
            <p className="text-muted-foreground">{t.emptyStart.body}</p>
            <Button size="sm" variant="outline" onClick={() => setSection('voices')}>
              {t.emptyStart.action}
            </Button>
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">{tg.empty}</p>
        )}
      </CardContent>
    </Card>
  )
}
