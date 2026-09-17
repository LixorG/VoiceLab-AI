import { Layers, X } from 'lucide-react'
import { useEffect, useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { t } from '@/i18n/es'
import { ApiError } from '@/services/api'
import { jobsApi } from '@/services/jobs'
import { keyOf, useEngineStore } from '@/stores/engine'
import { useGenerationStore } from '@/stores/generation'
import type { QueuedJob } from '@/types/jobs'

const tq = t.queue
const POLL_MS = 2000

/** What the GPU is generating now and what is waiting, with per-job and bulk cancel. */
export function QueuePanel() {
  const [jobs, setJobs] = useState<QueuedJob[]>([])
  const [error, setError] = useState<string | null>(null)
  const load = useGenerationStore((s) => s.load)

  useEffect(() => {
    let active = true
    const tick = async () => {
      try {
        const list = await jobsApi.list()
        if (active) setJobs(Array.isArray(list) ? list : [])
      } catch (err) {
        if (active) setError(err instanceof ApiError ? err.message : String(err))
      }
    }
    void tick()
    const timer = window.setInterval(() => void tick(), POLL_MS)
    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [])

  const cancelAll = async (onlyQueued: boolean) => {
    if (!confirm(onlyQueued ? tq.confirmCancelQueued : tq.confirmCancelAll)) return
    try {
      await jobsApi.cancelAll(onlyQueued)
      setJobs(await jobsApi.list())
      await load()
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err))
    }
  }

  const cancelOne = async (job: QueuedJob) => {
    await jobsApi.cancel(job.job_id)
    setJobs(await jobsApi.list())
    await load()
  }

  if (jobs.length === 0 && !error) return null

  return (
    <Card data-testid="queue-panel">
      <CardHeader className="flex flex-row items-center justify-between gap-2">
        <CardTitle className="flex items-center gap-1.5">
          <Layers className="size-4" />
          {tq.title}
        </CardTitle>
        <span className="text-xs text-muted-foreground">{tq.summary(jobs.length)}</span>
      </CardHeader>
      <CardContent className="space-y-2">
        {error && (
          <p role="alert" className="text-xs text-destructive">
            {error}
          </p>
        )}
        <ul className="divide-y rounded-md border">
          {jobs.map((job) => (
            <li key={job.job_id} className="flex items-center gap-2 px-3 py-1.5 text-xs" data-testid="queue-row">
              <Badge variant={job.position === 0 ? 'accent' : 'outline'}>{job.position === 0 ? tq.running : tq.position(job.position)}</Badge>
              <span className="min-w-0 flex-1 truncate">{job.text ?? tq.otherJob}</span>
              {job.position === 0 && job.progress > 0 && <span className="font-mono">{Math.round(job.progress * 100)} %</span>}
              <Button size="icon" variant="ghost" aria-label={`${tq.cancel}: ${job.text ?? job.job_id}`} onClick={() => void cancelOne(job)}>
                <X className="size-4" />
              </Button>
            </li>
          ))}
        </ul>
        {jobs.length > 1 && (
          <div className="flex flex-wrap gap-2">
            <Button size="sm" variant="outline" onClick={() => void cancelAll(true)}>
              {tq.cancelQueued}
            </Button>
            <Button size="sm" variant="destructive" onClick={() => void cancelAll(false)}>
              {tq.cancelAll}
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

/** Queue many generations at once: one text per line, optionally across variants and with repetitions. */
export function BatchPanel() {
  const { text, referenceId, profileId, emotion, intensity, markup, normalize, load } = useGenerationStore()
  const { engineId, variantId, valuesByKey, engines } = useEngineStore()
  const [open, setOpen] = useState(false)
  const [lines, setLines] = useState('')
  const [variants, setVariants] = useState<string[]>([])
  const [repeat, setRepeat] = useState(1)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const texts = lines.split('\n').map((l) => l.trim()).filter(Boolean)
  const engineVariants = engines.find((e) => e.id === engineId)?.variants ?? []
  const total = Math.max(texts.length, 0) * Math.max(variants.length || 1, 1) * repeat

  const submit = async () => {
    setError(null)
    setBusy(true)
    try {
      await jobsApi.batch({
        engine: engineId ?? '',
        variant: variantId,
        text: '',
        params: valuesByKey[keyOf(engineId ?? '', variantId)] ?? {},
        reference_id: referenceId,
        profile_id: profileId,
        preview: false,
        emotion,
        intensity,
        markup,
        normalize,
        postprocess: null,
        texts,
        variants,
        repeat,
      })
      setLines('')
      await load()
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Card data-testid="batch-panel">
      <CardHeader className="flex flex-row items-center justify-between gap-2">
        <CardTitle>{t.batch.title}</CardTitle>
        <Button size="sm" variant="ghost" onClick={() => setOpen(!open)} aria-expanded={open}>
          {open ? t.batch.hide : t.batch.show}
        </Button>
      </CardHeader>
      {open && (
        <CardContent className="space-y-3">
          <p className="text-xs text-muted-foreground">{t.batch.intro}</p>
          <label className="flex flex-col gap-1 text-xs text-muted-foreground">
            {t.batch.texts}
            <textarea
              aria-label={t.batch.texts}
              className="min-h-28 rounded-md border bg-background p-2 text-sm"
              placeholder={t.batch.textsPlaceholder}
              value={lines}
              onChange={(e) => setLines(e.target.value)}
            />
          </label>
          <div className="flex flex-wrap items-end gap-3">
            <Button size="sm" variant="outline" onClick={() => setLines(text.trim() ? `${lines}${lines ? '\n' : ''}${text.trim()}` : lines)}>
              {t.batch.addCurrent}
            </Button>
            <label className="flex flex-col gap-1 text-xs text-muted-foreground">
              {t.batch.repeat}
              <input
                type="number"
                min={1}
                max={8}
                aria-label={t.batch.repeat}
                className="h-8 w-20 rounded-md border bg-background px-2 text-sm"
                value={repeat}
                onChange={(e) => setRepeat(Math.min(8, Math.max(1, Number(e.target.value) || 1)))}
              />
            </label>
          </div>
          {engineVariants.length > 1 && (
            <fieldset className="space-y-1">
              <legend className="text-xs text-muted-foreground">{t.batch.variants}</legend>
              <div className="flex flex-wrap gap-2">
                {engineVariants.map((v) => (
                  <label key={v.id} className="flex items-center gap-1.5 text-xs">
                    <input
                      type="checkbox"
                      className="accent-[var(--primary)]"
                      checked={variants.includes(v.id)}
                      onChange={(e) => setVariants(e.target.checked ? [...variants, v.id] : variants.filter((x) => x !== v.id))}
                    />
                    {v.label}
                  </label>
                ))}
              </div>
            </fieldset>
          )}
          {error && (
            <p role="alert" className="text-xs text-destructive">
              {error}
            </p>
          )}
          <div className="flex items-center gap-2">
            <Button size="sm" disabled={busy || texts.length === 0} onClick={() => void submit()}>
              {t.batch.enqueue(total)}
            </Button>
            <span className="text-xs text-muted-foreground">{t.batch.hint}</span>
          </div>
        </CardContent>
      )}
    </Card>
  )
}
