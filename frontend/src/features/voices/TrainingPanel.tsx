import { AlertTriangle, GraduationCap, Loader2, Play, Trash2, X } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { t } from '@/i18n/es'
import { formatDuration } from '@/lib/utils'
import { ApiError } from '@/services/api'
import { trainingApi } from '@/services/training'
import { transcriptionApi } from '@/services/transcription'
import { useEngineStore } from '@/stores/engine'
import { useUiStore } from '@/stores/ui'
import type { ProfileRead } from '@/types/profiles'
import { ACTIVE_TRAINING, TRAINABLE, type TrainableEngine, type TrainingDataset, type TrainingRun } from '@/types/training'

const tt = t.training
const BASE_LABELS: Record<string, string> = {
  'base-1.7b': tt.base17,
  'base-0.6b': tt.base06,
  F5TTS_v1_Base: tt.baseF5v1,
  F5TTS_Base: tt.baseF5,
}

const POLL_MS = 2000
const messageOf = (err: unknown) => (err instanceof ApiError ? err.message : String(err))

const statusVariant = (run: TrainingRun) =>
  run.status === 'completed' ? 'success' : run.status === 'failed' ? 'destructive' : run.status === 'cancelled' ? 'outline' : 'accent'

function DatasetSummary({ data, onTranscribe, transcribing }: {
  data: TrainingDataset
  onTranscribe: () => void
  transcribing: string | null
}) {
  const [from, to] = data.recommended_minutes
  const fill = Math.min(100, (data.usable_minutes / to) * 100)
  return (
    <div className="space-y-2">
      <p className="text-sm font-medium">
        {tt.dataset(formatDuration(data.usable_minutes * 60), formatDuration(data.recorded_minutes * 60), data.clips)}
      </p>
      <div className="relative h-2 overflow-hidden rounded-full bg-muted" aria-hidden>
        <div className={`h-full ${data.ready ? 'bg-success' : 'bg-warning'}`} style={{ width: `${fill}%` }} />
        <div className="absolute inset-y-0 w-px bg-foreground/40" style={{ left: `${(data.min_minutes / to) * 100}%` }} />
      </div>
      <p className="text-xs text-muted-foreground">{tt.needs(data.min_minutes, from, to)}</p>
      {data.warnings.map((w) => (
        <p key={w} className="flex items-start gap-1.5 text-xs text-warning">
          <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
          {w}
        </p>
      ))}
      {data.missing_transcripts.length > 0 && (
        <Button size="sm" variant="outline" disabled={transcribing !== null} onClick={onTranscribe}>
          {transcribing !== null && <Loader2 className="animate-spin" />}
          {transcribing ?? tt.transcribeMissing(data.missing_transcripts.length)}
        </Button>
      )}
    </div>
  )
}

function Comparison({ run }: { run: TrainingRun }) {
  const ev = run.evaluation
  if (!ev) return null
  if (ev.skipped) return <p className="text-xs text-muted-foreground">{ev.skipped}</p>
  const systems = (['trained', 'normal', 'real'] as const).filter((k) => ev.systems?.[k])
  return (
    <div className="space-y-2 rounded-lg border p-3">
      <p className="text-sm font-medium">{tt.compare}</p>
      <p className="text-xs text-muted-foreground">{tt.compareHint}</p>
      {ev.verdict && <p className="text-sm">{ev.verdict}</p>}
      <table className="w-full text-xs">
        <thead>
          <tr className="text-left text-muted-foreground">
            <th className="py-1 font-normal"><span className="sr-only">{tt.systemColumn}</span></th>
            <th className="py-1 font-normal" title={tt.similarityHint}>{tt.similarity}</th>
            <th className="py-1 font-normal">{tt.wer}</th>
          </tr>
        </thead>
        <tbody>
          {systems.map((k) => {
            const s = ev.systems?.[k] ?? {}
            return (
              <tr key={k} className="border-t">
                <td className="py-1">{tt.system[k]}</td>
                <td className="py-1 font-mono">
                  {s.similarity != null ? s.similarity.toFixed(3) : '—'}
                  {k === 'real' && <span className="ml-1 font-sans text-muted-foreground">({tt.ceiling})</span>}
                </td>
                <td className="py-1 font-mono">{s.wer != null ? `${(s.wer * 100).toFixed(1)} %` : '—'}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
      {(ev.missing ?? []).map((m) => (
        <p key={m} className="text-xs text-muted-foreground">{tt.missing[m as 'similarity' | 'wer']}</p>
      ))}
      {(ev.texts ?? []).length > 0 && (
        <details>
          <summary className="cursor-pointer text-xs text-primary">{tt.samples}</summary>
          <ol className="mt-2 space-y-3">
            {(ev.texts ?? []).map((text, i) => (
              <li key={text} className="space-y-1">
                <p className="text-xs">
                  <span className="text-muted-foreground">{tt.text(i + 1)}: </span>
                  {text}
                </p>
                <div className="grid gap-1 sm:grid-cols-3">
                  {(['real', 'trained', 'normal'] as const).map((k) => (
                    <label key={k} className="space-y-0.5 text-[11px] text-muted-foreground">
                      {tt.system[k]}
                      {/* oxlint-disable-next-line jsx-a11y/media-has-caption -- speech samples, the text is shown above */}
                      <audio controls preload="none" src={trainingApi.sampleUrl(run.id, i, k)} className="h-8 w-full" />
                    </label>
                  ))}
                </div>
              </li>
            ))}
          </ol>
        </details>
      )}
    </div>
  )
}

function RunCard({ run, onChanged }: { run: TrainingRun; onChanged: () => void }) {
  const active = ACTIVE_TRAINING.includes(run.status)
  const last = run.history?.at(-1)
  const [error, setError] = useState<string | null>(null)

  const act = async (fn: () => Promise<unknown>, confirmText: string) => {
    if (!window.confirm(confirmText)) return
    try {
      setError(null)
      await fn()
      onChanged()
    } catch (err) {
      setError(messageOf(err))
    }
  }

  const use = async () => {
    const engine = useEngineStore.getState()
    await engine.loadEngines()
    await engine.selectEngine(run.engine)
    if (run.variant) await useEngineStore.getState().selectVariant(run.variant)
    useUiStore.getState().setSection('generate')
  }

  return (
    <li className="space-y-2 py-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-medium">{run.name}</span>
        <Badge variant={statusVariant(run)}>{tt.status[run.status]}</Badge>
        <span className="text-xs text-muted-foreground">{run.base_variant}</span>
        <div className="ml-auto flex gap-1">
          {run.status === 'completed' && run.variant && (
            <Button size="sm" onClick={() => void use()}>
              <Play />
              {tt.use}
            </Button>
          )}
          {active ? (
            <Button size="sm" variant="outline" onClick={() => void act(() => trainingApi.cancel(run.id), tt.confirmCancel)}>
              <X />
              {tt.cancel}
            </Button>
          ) : (
            <Button size="sm" variant="ghost" className="hover:text-destructive" onClick={() => void act(() => trainingApi.remove(run.id), tt.confirmRemove)}>
              <Trash2 />
              {tt.remove}
            </Button>
          )}
        </div>
      </div>
      {active && (
        <div className="space-y-1">
          <div className="h-1.5 overflow-hidden rounded-full bg-muted">
            <div className="h-full bg-primary transition-[width]" style={{ width: `${Math.round(run.progress * 100)}%` }} />
          </div>
          <p className="text-xs text-muted-foreground" aria-live="polite">{run.message}</p>
        </div>
      )}
      {!active && last && <p className="text-xs text-muted-foreground">{tt.epochsDone(last.epoch, last.talker_loss)}</p>}
      {run.status === 'failed' && (
        <div className="text-xs text-destructive">
          <p>{run.message}</p>
          {run.error_detail && (
            <details className="text-muted-foreground">
              <summary className="cursor-pointer">{tt.detail}</summary>
              <pre className="whitespace-pre-wrap">{run.error_detail}</pre>
            </details>
          )}
        </div>
      )}
      {run.status === 'completed' && <Comparison run={run} />}
      {error && <p className="text-xs text-destructive" role="alert">{error}</p>}
    </li>
  )
}

export function TrainingPanel({ profile, referencesSignature }: { profile: ProfileRead; referencesSignature: string }) {
  const [data, setData] = useState<TrainingDataset | null>(null)
  const [runs, setRuns] = useState<TrainingRun[]>([])
  const [engine, setEngine] = useState<TrainableEngine>('qwen3tts')
  const [base, setBase] = useState<string>(TRAINABLE.qwen3tts[0])
  const [epochs, setEpochs] = useState(10)
  const [name, setName] = useState('')
  const [busy, setBusy] = useState(false)
  const [transcribing, setTranscribing] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const reload = useCallback(async () => {
    try {
      const [dataset, list] = await Promise.all([trainingApi.dataset(profile.id), trainingApi.list(profile.id)])
      setData(dataset)
      setRuns((prev) => {
        // a run that just finished adds an engine variant: refresh the engine list once
        if (prev.some((p) => ACTIVE_TRAINING.includes(p.status) && list.find((r) => r.id === p.id)?.status === 'completed')) {
          void useEngineStore.getState().loadEngines()
        }
        return list
      })
    } catch (err) {
      setError(messageOf(err))
    }
  }, [profile.id])

  useEffect(() => {
    void reload()
  }, [reload, referencesSignature])

  const anyActive = runs.some((r) => ACTIVE_TRAINING.includes(r.status))
  useEffect(() => {
    if (!anyActive) return
    const id = window.setInterval(() => void reload(), POLL_MS)
    return () => window.clearInterval(id)
  }, [anyActive, reload])

  const transcribeMissing = async () => {
    if (!data) return
    const ids = data.missing_transcripts
    setError(null)
    try {
      for (const [i, id] of ids.entries()) {
        setTranscribing(tt.transcribing(i + 1, ids.length))
        await transcriptionApi.transcribe(id, { scope: 'full', language: null })
      }
    } catch {
      setError(tt.transcribeFailed)
    } finally {
      setTranscribing(null)
      void reload()
    }
  }

  const start = async () => {
    setBusy(true)
    setError(null)
    try {
      await trainingApi.start({ profile_id: profile.id, engine, base_variant: base, epochs, name: name.trim() || null })
      await reload()
    } catch (err) {
      setError(messageOf(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <GraduationCap className="size-5 text-primary" />
          {tt.title}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-sm text-muted-foreground">{tt.intro}</p>
        {data && <DatasetSummary data={data} onTranscribe={() => void transcribeMissing()} transcribing={transcribing} />}

        <div className="grid gap-3 sm:grid-cols-[0.9fr_1.2fr_0.6fr_1.2fr_auto] sm:items-end">
          <label className="space-y-1 text-xs text-muted-foreground">
            {tt.engine}
            <select
              value={engine}
              aria-label={tt.engine}
              onChange={(e) => {
                const next = e.target.value as TrainableEngine
                setEngine(next)
                setBase(TRAINABLE[next][0])
              }}
              className="h-9 w-full rounded-md border bg-background px-2 text-sm text-foreground"
            >
              <option value="qwen3tts">{tt.engineQwen}</option>
              <option value="f5tts">{tt.engineF5}</option>
            </select>
          </label>
          <label className="space-y-1 text-xs text-muted-foreground">
            {tt.base}
            <select value={base} aria-label={tt.base} onChange={(e) => setBase(e.target.value)} className="h-9 w-full rounded-md border bg-background px-2 text-sm text-foreground">
              {TRAINABLE[engine].map((id) => (
                <option key={id} value={id}>
                  {BASE_LABELS[id] ?? id}
                </option>
              ))}
            </select>
          </label>
          <label className="space-y-1 text-xs text-muted-foreground" title={tt.epochsHint}>
            {tt.epochs}
            <input type="number" min={1} max={50} value={epochs} onChange={(e) => setEpochs(Math.max(1, Math.min(50, Number(e.target.value) || 1)))} className="h-9 w-full rounded-md border bg-background px-2 text-sm text-foreground" />
          </label>
          <label className="space-y-1 text-xs text-muted-foreground">
            {tt.name}
            <input value={name} onChange={(e) => setName(e.target.value)} maxLength={60} placeholder={`${profile.name} (entrenada)`} className="h-9 w-full rounded-md border bg-background px-2 text-sm text-foreground" />
          </label>
          <Button disabled={busy || anyActive || !data?.ready} onClick={() => void start()}>
            {busy ? <Loader2 className="animate-spin" /> : <GraduationCap />}
            {busy ? tt.starting : tt.start}
          </Button>
        </div>
        <p className="text-xs text-muted-foreground">{tt.engineHint[engine]}</p>
        <p className="text-xs text-muted-foreground">{tt.gpuNote}</p>
        {error && <p className="text-sm text-destructive" role="alert">{error}</p>}

        <div>
          <p className="text-sm font-medium">{tt.runs}</p>
          {runs.length === 0 ? (
            <p className="text-xs text-muted-foreground">{tt.noRuns}</p>
          ) : (
            <ul className="divide-y">
              {runs.map((run) => (
                <RunCard key={run.id} run={run} onChanged={() => void reload()} />
              ))}
            </ul>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
