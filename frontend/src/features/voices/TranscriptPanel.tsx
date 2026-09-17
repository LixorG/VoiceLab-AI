import { AlertTriangle, Download, Languages, Loader2, RotateCcw, Scissors, Wand2 } from 'lucide-react'
import { useEffect, useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { Textarea } from '@/components/ui/textarea'
import { t } from '@/i18n/es'
import { useTranscriptionStore } from '@/stores/transcription'
import type { ReferenceRead, TranscriptRead } from '@/types/api'

const tx = t.transcription

function TranscriptEditor({ referenceId, transcript, rows = 3 }: { referenceId: string; transcript: TranscriptRead; rows?: number }) {
  const { edit, revert, busy, languages } = useTranscriptionStore()
  const [draft, setDraft] = useState(transcript.text)
  useEffect(() => setDraft(transcript.text), [transcript.text, transcript.id])

  const saving = busy[`${referenceId}:edit`]
  const dirty = draft.trim() !== transcript.text
  const langName = languages.find((l) => l.code === transcript.language)?.name ?? transcript.language ?? '—'
  const langPct = transcript.confidence.language_probability
  const aria = transcript.segment_start_s == null ? tx.title : tx.segmentTitle

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-1.5">
        {transcript.source === 'manual' ? (
          <Badge variant="accent">{tx.manual}</Badge>
        ) : (
          <Badge>{tx.automatic(transcript.asr_model ?? 'ASR')}</Badge>
        )}
        <Badge variant="outline">
          <Languages />
          {tx.detected(langName, langPct == null ? null : Math.round(langPct * 100))}
        </Badge>
      </div>
      <Textarea
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        rows={rows}
        className="min-h-0 text-[13px] leading-relaxed"
        aria-label={aria}
        disabled={saving}
      />
      {transcript.warnings.map((w) => (
        <p key={w} className="flex items-start gap-1.5 text-xs text-warning">
          <AlertTriangle className="mt-px size-3.5 shrink-0" />
          {w}
        </p>
      ))}
      <div className="flex flex-wrap items-center gap-2">
        {dirty && (
          <>
            <Button size="sm" disabled={!draft.trim() || saving} onClick={() => void edit(referenceId, transcript.id, draft)}>
              {saving && <Loader2 className="animate-spin" />}
              {tx.save}
            </Button>
            <Button size="sm" variant="ghost" onClick={() => setDraft(transcript.text)}>
              {tx.discard}
            </Button>
          </>
        )}
        {transcript.edited && !dirty && (
          <Button size="sm" variant="ghost" onClick={() => void revert(referenceId, transcript.id)} disabled={saving}>
            <RotateCcw />
            {tx.revert}
          </Button>
        )}
        <span className="text-[11px] text-muted-foreground">{tx.editHint}</span>
      </div>
    </div>
  )
}

function ModelNotice() {
  const { status, download } = useTranscriptionStore()
  if (!status || status.installed) return null
  if (!status.package_available) return <p className="text-xs text-muted-foreground">{tx.model.notAvailable}</p>
  return (
    <div className="flex flex-wrap items-center gap-3 rounded-md border border-dashed px-3 py-2 text-xs">
      {status.download_state === 'downloading' ? (
        <span className="flex items-center gap-2 text-muted-foreground">
          <Loader2 className="size-3.5 animate-spin text-primary" />
          {tx.model.downloading}
        </span>
      ) : (
        <>
          <span className="text-muted-foreground">{status.download_error ?? tx.model.notInstalled(status.model, status.approx_size_mb)}</span>
          <Button size="sm" variant="secondary" onClick={() => void download()}>
            <Download />
            {tx.model.download}
          </Button>
        </>
      )}
    </div>
  )
}

export function TranscriptPanel({ reference: ref }: { reference: ReferenceRead }) {
  const { status, languages, busy, errors, loadStatus, loadLanguages, transcribe } = useTranscriptionStore()
  const [language, setLanguage] = useState<string>(ref.transcript?.language ?? '')

  useEffect(() => {
    if (!status) void loadStatus()
    void loadLanguages()
  }, [status, loadStatus, loadLanguages])

  const ready = !!status?.installed
  const fullKey = `${ref.id}:full`
  const segKey = `${ref.id}:segment`
  const hasSegment = ref.segment_start_s != null && ref.segment_end_s != null
  const error = errors[fullKey] ?? errors[segKey] ?? errors[`${ref.id}:edit`]

  return (
    <section className="space-y-3" aria-label={tx.title}>
      <div className="flex flex-wrap items-center gap-2">
        <h4 className="mr-auto text-[11px] font-semibold tracking-[0.14em] text-muted-foreground uppercase">{tx.title}</h4>
        <label className="sr-only" htmlFor={`lang-${ref.id}`}>
          {tx.language}
        </label>
        <select
          id={`lang-${ref.id}`}
          value={language}
          onChange={(e) => setLanguage(e.target.value)}
          className="h-8 rounded-md border bg-background px-2 text-xs"
          title={tx.language}
        >
          <option value="">{tx.auto}</option>
          {languages.map((l) => (
            <option key={l.code} value={l.code}>
              {l.name}
            </option>
          ))}
        </select>
        <Button
          size="sm"
          variant={ref.transcript ? 'outline' : 'default'}
          disabled={!ready || busy[fullKey]}
          onClick={() => void transcribe(ref.id, 'full', language || null, !!ref.transcript)}
        >
          {busy[fullKey] ? <Loader2 className="animate-spin" /> : <Wand2 />}
          {busy[fullKey] ? tx.transcribing : ref.transcript ? tx.retranscribe : tx.transcribe}
        </Button>
      </div>

      <ModelNotice />

      {error && (
        <p role="alert" className="flex items-start gap-1.5 text-xs text-destructive">
          <AlertTriangle className="mt-px size-3.5 shrink-0" />
          {error}
        </p>
      )}

      {busy[fullKey] ? (
        <div className="space-y-2" aria-busy>
          <Skeleton className="h-16" />
          {status && !status.loaded && <p className="text-[11px] text-muted-foreground">{tx.firstLoadHint}</p>}
        </div>
      ) : ref.transcript ? (
        <TranscriptEditor referenceId={ref.id} transcript={ref.transcript} />
      ) : (
        <p className="text-xs text-muted-foreground">{tx.empty}</p>
      )}

      {hasSegment && (
        <div className="space-y-2 border-l-2 border-primary/40 pl-3">
          <div className="flex flex-wrap items-center gap-2">
            <h5 className="mr-auto flex items-center gap-1.5 text-xs font-medium">
              <Scissors className="size-3.5" />
              {tx.segmentTitle}
            </h5>
            <Button
              size="sm"
              variant="outline"
              disabled={!ready || busy[segKey]}
              onClick={() => void transcribe(ref.id, 'segment', language || null, !!ref.segment_transcript)}
            >
              {busy[segKey] ? <Loader2 className="animate-spin" /> : <Wand2 />}
              {busy[segKey] ? tx.transcribing : tx.transcribeSegment}
            </Button>
          </div>
          {busy[segKey] ? (
            <Skeleton className="h-10" />
          ) : ref.segment_transcript ? (
            <TranscriptEditor referenceId={ref.id} transcript={ref.segment_transcript} rows={2} />
          ) : ref.segment_text_estimate ? (
            <p className="text-xs text-muted-foreground">
              {tx.segmentEstimate} <span className="text-foreground italic">“{ref.segment_text_estimate}”</span>
            </p>
          ) : (
            <p className="text-xs text-muted-foreground">{tx.segmentNoEstimate}</p>
          )}
        </div>
      )}
    </section>
  )
}
