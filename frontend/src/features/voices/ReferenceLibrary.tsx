import { AlertTriangle, Loader2, Upload, X } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { ReferenceCard } from '@/features/voices/ReferenceCard'
import { t } from '@/i18n/es'
import { cn, formatBytes, formatDuration } from '@/lib/utils'
import { useGenerationStore } from '@/stores/generation'
import { useReferencesStore } from '@/stores/references'
import { hasAnalysis } from '@/types/api'

const DEFAULT_EXTENSIONS = ['wav', 'mp3', 'flac', 'm4a', 'ogg', 'opus', 'aiff', 'aif']

export function ReferenceLibrary() {
  const { items, pending, formats, loading, error, notice, load, loadFormats, uploadFiles, dismissPending, clearNotice } =
    useReferencesStore()
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const dragDepth = useRef(0)

  useEffect(() => {
    void load()
    void loadFormats()
  }, [load, loadFormats])

  const { referenceId, setReference } = useGenerationStore()

  // Keep a valid reference selected: the recommended one, otherwise the first analysed.
  useEffect(() => {
    if (referenceId && items.some((r) => r.id === referenceId)) return
    const analysed = items.filter((r) => r.status === 'ANALYZED')
    const candidate = analysed.find((r) => r.is_primary) ?? analysed.find((r) => r.is_recommended) ?? analysed[0]
    if (candidate?.id !== referenceId) setReference(candidate?.id ?? null)
  }, [items, referenceId, setReference])

  const extensions = formats?.extensions ?? DEFAULT_EXTENSIONS
  const accept = extensions.map((e) => `.${e}`).join(',')
  const totalSpeech = items.reduce((sum, r) => sum + (hasAnalysis(r) ? r.analysis.effective_speech_s : 0), 0)

  const submit = (files: FileList | null) => {
    if (files?.length) void uploadFiles(Array.from(files))
  }

  return (
    <Card>
      <CardHeader className="flex-wrap">
        <CardTitle>{t.references.title}</CardTitle>
        {items.length > 0 && (
          <span className="font-mono text-xs text-muted-foreground">
            {t.references.count(items.length)} · {t.references.totalSpeech(formatDuration(totalSpeech))}
          </span>
        )}
      </CardHeader>
      <CardContent className="space-y-3">
        <div
          role="button"
          tabIndex={0}
          aria-label={t.generate.addAudio}
          onClick={() => inputRef.current?.click()}
          onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && inputRef.current?.click()}
          onDragEnter={(e) => {
            e.preventDefault()
            dragDepth.current++
            setDragging(true)
          }}
          onDragOver={(e) => e.preventDefault()}
          onDragLeave={() => {
            dragDepth.current = Math.max(0, dragDepth.current - 1)
            if (!dragDepth.current) setDragging(false)
          }}
          onDrop={(e) => {
            e.preventDefault()
            dragDepth.current = 0
            setDragging(false)
            submit(e.dataTransfer.files)
          }}
          className={cn(
            'flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border border-dashed px-6 text-center transition-colors outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50',
            items.length ? 'py-5' : 'py-10',
            dragging ? 'border-primary bg-primary/5' : 'bg-muted/30 hover:border-muted-foreground/50',
          )}
        >
          <div className={cn('grid size-10 place-items-center rounded-full bg-muted', dragging && 'bg-primary/15 text-primary')}>
            <Upload className="size-5" />
          </div>
          <p className="text-sm">
            {dragging ? (
              t.references.dropActive
            ) : (
              <>
                {t.references.dropHint} <span className="text-primary underline-offset-2 hover:underline">{t.references.browse}</span>
              </>
            )}
          </p>
          <p className="text-xs text-muted-foreground">
            {t.references.limits(
              extensions.filter((e) => e !== 'aif').map((e) => e.toUpperCase()).join(', '),
              formats?.max_upload_mb ?? 500,
              Math.round((formats?.max_audio_seconds ?? 600) / 60),
            )}
          </p>
          <input
            ref={inputRef}
            type="file"
            multiple
            accept={accept}
            className="hidden"
            data-testid="reference-input"
            onChange={(e) => {
              submit(e.target.files)
              e.target.value = ''
            }}
          />
        </div>

        {(notice || error) && (
          <div
            role={error ? 'alert' : 'status'}
            className={cn(
              'flex items-start gap-2 rounded-md border px-3 py-2 text-sm',
              error ? 'border-destructive/30 bg-destructive/10' : 'border-border bg-muted/40',
            )}
          >
            {error && <AlertTriangle className="mt-0.5 size-4 shrink-0 text-destructive" />}
            <span className="flex-1">{error ?? notice}</span>
            <button type="button" onClick={clearNotice} aria-label={t.references.dismiss}>
              <X className="size-4" />
            </button>
          </div>
        )}

        {pending.map((p) => (
          <div key={p.key} className="rounded-lg border px-4 py-3">
            <div className="flex items-center gap-2 text-sm">
              {p.phase === 'error' ? (
                <AlertTriangle className="size-4 text-destructive" />
              ) : (
                <Loader2 className="size-4 animate-spin text-primary" />
              )}
              <span className="min-w-0 flex-1 truncate">{p.name}</span>
              <span className="font-mono text-xs text-muted-foreground">{formatBytes(p.size)}</span>
              {p.phase === 'error' && (
                <button type="button" onClick={() => dismissPending(p.key)} aria-label={t.references.dismiss}>
                  <X className="size-4" />
                </button>
              )}
            </div>
            {p.phase === 'error' ? (
              <p className="mt-1 text-xs text-destructive">
                {t.references.uploadFailed}: {p.error}
              </p>
            ) : (
              <>
                <div className="mt-2 h-1 overflow-hidden rounded-full bg-muted">
                  <div
                    className={cn('h-full bg-primary transition-[width]', p.phase === 'processing' && 'animate-pulse')}
                    style={{ width: `${Math.round(p.progress * 100)}%` }}
                  />
                </div>
                <p className="mt-1 text-xs text-muted-foreground">
                  {p.phase === 'processing' ? t.references.processing : `${t.references.uploading} ${Math.round(p.progress * 100)}%`}
                </p>
              </>
            )}
          </div>
        ))}

        {loading && !items.length ? (
          <Skeleton className="h-40" />
        ) : (
          items.map((ref) => <ReferenceCard key={ref.id} reference={ref} />)
        )}

        {!loading && !items.length && !pending.length && (
          <p className="text-center text-xs text-muted-foreground">{t.references.empty}</p>
        )}
        <p className="text-xs text-muted-foreground">{t.generate.consent}</p>
      </CardContent>
    </Card>
  )
}
