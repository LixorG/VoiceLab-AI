import { AlertTriangle, CheckCircle2, Crown, FileAudio, RotateCcw, Sparkles, Star, Trash2 } from 'lucide-react'
import { useEffect, useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { Waveform } from '@/features/audio/Waveform'
import { QualityMeter } from '@/features/voices/QualityMeter'
import { TranscriptPanel } from '@/features/voices/TranscriptPanel'
import { t } from '@/i18n/es'
import { cn, formatBytes, formatDb, formatDuration } from '@/lib/utils'
import { referencesApi } from '@/services/references'
import { useGenerationStore } from '@/stores/generation'
import { useProfilesStore } from '@/stores/profiles'
import { useReferencesStore } from '@/stores/references'
import { hasAnalysis, type ReferenceRead, type TimeRegion } from '@/types/api'

const PEAK_BUCKETS = 800

function Meta({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="min-w-0">
      <dt className="text-[11px] text-muted-foreground">{label}</dt>
      <dd className="truncate font-mono text-xs tabular-nums">{value}</dd>
    </div>
  )
}

const sameRegion = (a: TimeRegion | null, b: TimeRegion | null) =>
  a === b || (!!a && !!b && a.start_s === b.start_s && a.end_s === b.end_s)

export function ReferenceCard({ reference: ref }: { reference: ReferenceRead }) {
  const { update, reanalyze, remove } = useReferencesStore()
  const selectedForGeneration = useGenerationStore((s) => s.referenceId === ref.id)
  const setReference = useGenerationStore((s) => s.setReference)
  const { emotions, loadEmotions, update: updateProfile } = useProfilesStore()
  const loadReferences = useReferencesStore((s) => s.load)

  useEffect(() => {
    void loadEmotions()
  }, [loadEmotions])

  const makePrimary = async () => {
    if (!ref.profile_id) return
    await updateProfile(ref.profile_id, { primary_reference_id: ref.id })
    await loadReferences()
  }
  const [peaks, setPeaks] = useState<number[] | null>(null)
  const saved: TimeRegion | null =
    ref.segment_start_s != null && ref.segment_end_s != null
      ? { start_s: ref.segment_start_s, end_s: ref.segment_end_s }
      : null
  const [draft, setDraft] = useState<TimeRegion | null>(saved)
  const analysis = hasAnalysis(ref) ? ref.analysis : null
  const ready = ref.status === 'ANALYZED' && analysis
  const isReady = Boolean(ready)

  useEffect(() => {
    setDraft(
      ref.segment_start_s != null && ref.segment_end_s != null
        ? { start_s: ref.segment_start_s, end_s: ref.segment_end_s }
        : null,
    )
  }, [ref.segment_start_s, ref.segment_end_s])

  useEffect(() => {
    if (!isReady) return
    let cancelled = false
    referencesApi
      .peaks(ref.id, PEAK_BUCKETS)
      .then((res) => !cancelled && setPeaks(res.peaks))
      .catch(() => !cancelled && setPeaks([]))
    return () => {
      cancelled = true
    }
  }, [ref.id, isReady])

  const draftLength = draft ? draft.end_s - draft.start_s : 0
  const dirty = !sameRegion(draft, saved)
  const errorCode = ref.analysis && 'error_code' in ref.analysis ? ref.analysis.error_code : null

  const onRemove = () => {
    if (window.confirm(t.references.confirmRemove)) void remove(ref.id)
  }

  return (
    <article
      className={cn(
        'rounded-lg border bg-background/40 p-4 transition-colors',
        ref.is_recommended && 'border-primary/60',
        selectedForGeneration && 'ring-2 ring-primary/40',
      )}
    >
      <header className="flex flex-wrap items-center gap-2">
        <FileAudio className="size-4 shrink-0 text-muted-foreground" />
        <h3 className="min-w-0 flex-1 truncate text-sm font-medium" title={ref.original_name}>
          {ref.original_name}
        </h3>
        {ref.is_recommended && (
          <Tooltip>
            <TooltipTrigger asChild>
              <Badge variant="accent" tabIndex={0}>
                <Star />
                {t.references.recommended}
              </Badge>
            </TooltipTrigger>
            <TooltipContent>{t.references.recommendedHint}</TooltipContent>
          </Tooltip>
        )}
        {ref.is_primary && (
          <Tooltip>
            <TooltipTrigger asChild>
              <Badge variant="accent" tabIndex={0}>
                <Crown />
                {t.generation.primary}
              </Badge>
            </TooltipTrigger>
            <TooltipContent>{t.generation.primaryHint}</TooltipContent>
          </Tooltip>
        )}
        {ref.status === 'ANALYZED' &&
          (selectedForGeneration ? (
            <Badge variant="success">
              <CheckCircle2 />
              {t.generation.inUse}
            </Badge>
          ) : (
            <Button size="sm" variant="outline" className="h-7" onClick={() => setReference(ref.id)}>
              {t.generation.selectReference}
            </Button>
          ))}
        <Badge
          variant={ref.status === 'FAILED' ? 'destructive' : ref.status === 'ANALYZED' ? 'success' : 'warning'}
        >
          {t.references.status[ref.status]}
        </Badge>
        <div className="flex">
          <Button size="icon" variant="ghost" className="size-8" onClick={() => void reanalyze(ref.id)} aria-label={t.references.reanalyze} title={t.references.reanalyze}>
            <RotateCcw />
          </Button>
          <Button size="icon" variant="ghost" className="size-8 hover:text-destructive" onClick={onRemove} aria-label={t.references.remove} title={t.references.remove}>
            <Trash2 />
          </Button>
        </div>
      </header>

      <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
        <label className="text-muted-foreground" htmlFor={`emotion-${ref.id}`}>
          {t.generation.emotion}
        </label>
        <select
          id={`emotion-${ref.id}`}
          value={ref.emotion_tag ?? ''}
          onChange={(e) => void update(ref.id, { emotion_tag: e.target.value || null })}
          className="h-7 rounded-md border bg-background px-2 text-xs"
        >
          <option value="">{t.generation.noEmotion}</option>
          {emotions.map((em) => (
            <option key={em.id} value={em.id}>
              {em.label}
            </option>
          ))}
        </select>
        {ref.profile_id && !ref.is_primary && ref.status === 'ANALYZED' && (
          <Button size="sm" variant="ghost" className="h-7" onClick={() => void makePrimary()} title={t.generation.primaryHint}>
            <Crown />
            {t.generation.makePrimary}
          </Button>
        )}
      </div>

      <dl className="mt-3 grid grid-cols-3 gap-x-4 gap-y-2 sm:grid-cols-5">
        <Meta label={t.references.meta.duration} value={formatDuration(ref.duration_s)} />
        <Meta label={t.references.meta.sampleRate} value={ref.sample_rate ? `${(ref.sample_rate / 1000).toFixed(1)} kHz` : '—'} />
        <Meta label={t.references.meta.channels} value={t.references.channelsValue(ref.channels)} />
        <Meta label={t.references.meta.format} value={`${ref.format.toUpperCase()}${analysis ? ` · ${analysis.source_codec}` : ''}`} />
        <Meta label={t.references.meta.size} value={formatBytes(ref.size_bytes)} />
        {analysis && (
          <>
            <Meta label={t.references.meta.level} value={formatDb(analysis.loudness_lufs, 'LUFS')} />
            <Meta label={t.references.meta.peak} value={formatDb(analysis.peak_dbfs, 'dBFS')} />
            <Meta label={t.references.meta.snr} value={analysis.snr_db == null ? t.references.notEstimable : formatDb(analysis.snr_db)} />
            <Meta label={t.references.meta.speech} value={formatDuration(analysis.effective_speech_s)} />
            <Meta label={t.references.meta.clipping} value={`${(analysis.clipping_ratio * 100).toFixed(2)} %`} />
          </>
        )}
      </dl>

      {ref.status === 'PROCESSING' && (
        <div className="mt-4 space-y-2" aria-busy>
          <Skeleton className="h-3 w-1/3" />
          <Skeleton className="h-16" />
        </div>
      )}

      {ref.status === 'FAILED' && (
        <p role="alert" className="mt-4 flex items-center gap-2 text-sm text-destructive">
          <AlertTriangle className="size-4" />
          {(errorCode && t.references.errors[errorCode]) ?? t.errors.unexpected}
        </p>
      )}

      {ready && (
        <div className="mt-4 space-y-4">
          <QualityMeter score={analysis.quality_score} label={analysis.quality_label} components={analysis.quality_components} />

          {analysis.warnings.length > 0 && (
            <ul className="space-y-1">
              {analysis.warnings.map((w) => (
                <li key={w} className="flex items-start gap-2 text-xs text-warning">
                  <AlertTriangle className="mt-px size-3.5 shrink-0" />
                  {w}
                </li>
              ))}
            </ul>
          )}

          {peaks && ref.urls.processed ? (
            <Waveform url={ref.urls.processed} peaks={peaks} duration={analysis.duration_s} selection={draft} onSelectionChange={setDraft} />
          ) : (
            <Skeleton className="h-[104px]" />
          )}

          <section className="space-y-2 rounded-md border border-dashed p-3">
            <div className="flex flex-wrap items-center gap-2">
              <h4 className="text-xs font-medium">{t.references.segment.title}</h4>
              <span className="font-mono text-xs text-muted-foreground">
                {draft
                  ? t.references.segment.range(formatDuration(draft.start_s), formatDuration(draft.end_s), formatDuration(draftLength))
                  : t.references.segment.none}
              </span>
            </div>
            <p className="text-xs text-muted-foreground">{t.references.segment.hint}</p>
            {ref.transcript && <p className="text-xs text-muted-foreground">{t.references.segment.snapHint}</p>}

            {analysis.suggested_segments.length > 0 && (
              <div className="flex flex-wrap items-center gap-1.5">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <span className="flex items-center gap-1 text-xs text-muted-foreground" tabIndex={0}>
                      <Sparkles className="size-3.5" />
                      {t.references.segment.suggested}
                    </span>
                  </TooltipTrigger>
                  <TooltipContent>{t.references.segment.suggestedHint}</TooltipContent>
                </Tooltip>
                {analysis.suggested_segments.map((s) => (
                  <button
                    key={`${s.start_s}-${s.end_s}`}
                    type="button"
                    onClick={() => setDraft({ start_s: s.start_s, end_s: s.end_s })}
                    className={cn(
                      'rounded-full border px-2 py-0.5 font-mono text-[11px] transition-colors hover:border-primary hover:text-primary',
                      sameRegion(draft, s) && 'border-primary text-primary',
                    )}
                  >
                    {formatDuration(s.start_s)}–{formatDuration(s.end_s)}
                  </button>
                ))}
              </div>
            )}

            {draft && draftLength < 3 && <p className="text-xs text-warning">{t.references.segment.tooShort}</p>}
            {draft && draftLength > 12 && <p className="text-xs text-warning">{t.references.segment.tooLong}</p>}

            <div className="flex flex-wrap gap-2 pt-1">
              <Button
                size="sm"
                disabled={!draft || !dirty || draftLength < 0.5}
                onClick={() =>
                  draft &&
                  void update(ref.id, { segment_start_s: draft.start_s, segment_end_s: draft.end_s, snap_to_words: !!ref.transcript })
                }
              >
                {t.references.segment.use}
              </Button>
              <Button
                size="sm"
                variant="outline"
                disabled={!saved && !draft}
                onClick={() => {
                  setDraft(null)
                  if (saved) void update(ref.id, { segment_start_s: null, segment_end_s: null })
                }}
              >
                {t.references.segment.clear}
              </Button>
              {saved && !dirty && <span className="self-center text-xs text-success">{t.references.segment.saved}</span>}
            </div>
          </section>

          <TranscriptPanel reference={ref} />
        </div>
      )}
    </article>
  )
}
