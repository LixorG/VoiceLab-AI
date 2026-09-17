import { AlertTriangle, Loader2, SlidersHorizontal, Undo2 } from 'lucide-react'
import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { AudioPlayer } from '@/features/audio/AudioPlayer'
import { PostProcessEditor } from '@/features/postprocess/PostProcessEditor'
import { t } from '@/i18n/es'
import { cn } from '@/lib/utils'
import { useGenerationStore } from '@/stores/generation'
import type { GenerationRead } from '@/types/generation'
import { DEFAULT_POSTPROCESS, type LevelStats, type PostProcessConfig } from '@/types/postprocess'

const tp = t.postprocess

const levels = (s: LevelStats) =>
  `${s.duration_s.toFixed(2)} s · ${tp.peak} ${s.peak_dbfs != null ? `${s.peak_dbfs.toFixed(1)} dBFS` : '—'} · ${s.loudness_lufs != null ? `${s.loudness_lufs.toFixed(1)} LUFS` : '—'}`

/** Player with processed/original (A/B) switch, the processing report and an inline editor to re-master. */
export function GenerationMastering({ gen }: { gen: GenerationRead }) {
  const { remaster, clearMaster, postprocessCaps, loadPostprocessCaps } = useGenerationStore()
  const [version, setVersion] = useState<'final' | 'raw'>('final')
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState<PostProcessConfig>(gen.postprocess?.config ?? DEFAULT_POSTPROCESS)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const pp = gen.postprocess
  const url = version === 'raw' && pp && gen.raw_audio_url ? gen.raw_audio_url : gen.audio_url

  const run = async (action: () => Promise<string | null>) => {
    setBusy(true)
    setError(null)
    const err = await action()
    setBusy(false)
    if (err) setError(err)
    else {
      setEditing(false)
      setVersion('final')
    }
  }

  if (!url) return null
  return (
    <div className="space-y-2">
      <AudioPlayer url={url} duration={version === 'raw' ? null : gen.duration_s} />
      <div className="flex flex-wrap items-center gap-2 text-[11px]">
        {pp && (
          <div className="inline-flex rounded-md border p-0.5" role="radiogroup" aria-label={tp.compare}>
            {(['final', 'raw'] as const).map((v) => (
              <button
                key={v}
                type="button"
                role="radio"
                aria-checked={version === v}
                onClick={() => setVersion(v)}
                className={cn('rounded px-2 py-0.5', version === v ? 'bg-primary text-primary-foreground' : 'text-muted-foreground')}
              >
                {v === 'final' ? tp.processed : tp.original}
              </button>
            ))}
          </div>
        )}
        <Button
          size="sm"
          variant="ghost"
          className="h-7 text-[11px]"
          aria-expanded={editing}
          onClick={() => {
            if (!editing) {
              setDraft(pp?.config ?? DEFAULT_POSTPROCESS)
              void loadPostprocessCaps()
            }
            setEditing((v) => !v)
          }}
        >
          <SlidersHorizontal />
          {pp ? tp.edit : tp.apply}
        </Button>
        {pp && (
          <Button size="sm" variant="ghost" className="h-7 text-[11px]" disabled={busy} onClick={() => void run(() => clearMaster(gen.id))}>
            <Undo2 />
            {tp.remove}
          </Button>
        )}
      </div>

      {pp?.report && (
        <details className="text-[11px] text-muted-foreground">
          <summary className="cursor-pointer">{tp.reportTitle(pp.report.steps.length)}</summary>
          <ul className="mt-1 space-y-0.5">
            {pp.report.steps.map((s) => (
              <li key={s.id}>
                <span className="text-foreground">{s.label}:</span> {s.detail}
              </li>
            ))}
            <li className="font-mono">
              {tp.before}: {levels(pp.report.before)}
            </li>
            <li className="font-mono">
              {tp.after}: {levels(pp.report.after)}
            </li>
          </ul>
        </details>
      )}
      {pp?.report?.warnings.map((w) => (
        <p key={w} className="flex items-start gap-1.5 text-xs text-warning">
          <AlertTriangle className="mt-px size-3.5 shrink-0" />
          {w}
        </p>
      ))}

      {editing && (
        <div className="space-y-3 rounded-md border border-dashed p-3">
          <PostProcessEditor value={draft} onChange={setDraft} capabilities={postprocessCaps} segmented={gen.segments.length > 1} idPrefix={`pp-${gen.id}`} />
          {error && (
            <p role="alert" className="text-xs text-destructive">
              {error}
            </p>
          )}
          <div className="flex justify-end gap-2">
            <Button size="sm" variant="ghost" onClick={() => setEditing(false)}>
              {tp.cancel}
            </Button>
            <Button size="sm" disabled={busy} onClick={() => void run(() => remaster(gen.id, draft))}>
              {busy && <Loader2 className="animate-spin" />}
              {tp.applyNow}
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}
