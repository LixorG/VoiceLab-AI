import { Loader2, Pencil, Play, RefreshCw } from 'lucide-react'
import { useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { t } from '@/i18n/es'
import { useGenerationStore } from '@/stores/generation'
import type { GenerationRead, PlannedSegment } from '@/types/generation'

const ts = t.sentence
const tg = t.generation
const TAKES = [1, 2, 3]

/** One sentence of a finished generation: listen to it alone, fix its words, or ask for another reading. */
function Sentence({ gen, segment, editable }: { gen: GenerationRead; segment: PlannedSegment; editable: boolean }) {
  const regenerateSegment = useGenerationStore((s) => s.regenerateSegment)
  const [editing, setEditing] = useState(false)
  const [text, setText] = useState(segment.text)
  const [takes, setTakes] = useState(1)
  const [seed, setSeed] = useState('')
  const [busy, setBusy] = useState(false)
  const [playing, setPlaying] = useState(false)

  const repeat = async () => {
    setBusy(true)
    try {
      await regenerateSegment(gen.id, segment.index, {
        ...(editing && text.trim() && text.trim() !== segment.text ? { text: text.trim() } : {}),
        ...(seed.trim() ? { seed: Number(seed) } : {}),
        takes,
      })
      setEditing(false)
    } finally {
      setBusy(false)
    }
  }

  return (
    <li className="space-y-1 rounded-md px-1.5 py-1 hover:bg-muted/40">
      <div className="flex flex-wrap items-center gap-1.5 text-muted-foreground">
        <span className="font-mono">{segment.index + 1}.</span>
        {!editing && <span className="text-foreground">{segment.text}</span>}
        {segment.emotion && <Badge variant="accent">{segment.emotion}</Badge>}
        {segment.reference_name && <span>· {segment.reference_name}</span>}
        {segment.seed != null && <span className="font-mono">· {tg.seed} {segment.seed}</span>}
        {segment.pause_after_ms > 0 && <span>· {t.expression.pauseAfter(segment.pause_after_ms)}</span>}
        <button
          type="button"
          className="ml-auto inline-flex items-center gap-1 hover:text-foreground"
          aria-label={`${ts.listen} ${segment.index + 1}`}
          onClick={() => setPlaying((v) => !v)}
        >
          <Play className="size-3" />
        </button>
        {editable && (
          <button
            type="button"
            className="inline-flex items-center gap-1 hover:text-foreground"
            aria-label={`${editing ? ts.cancelEdit : ts.edit} ${segment.index + 1}`}
            onClick={() => {
              setText(segment.text)
              setEditing((v) => !v)
            }}
          >
            <Pencil className="size-3" />
          </button>
        )}
      </div>

      {playing && (
        // eslint-disable-next-line jsx-a11y/media-has-caption
        <audio className="h-8 w-full" controls autoPlay src={`/api/generation/${gen.id}/segments/${segment.index}/audio?v=${gen.updated_at}`} />
      )}

      {editable && (
        <div className="space-y-1.5">
          {editing && (
            <Textarea value={text} onChange={(e) => setText(e.target.value)} aria-label={ts.textLabel} rows={2} maxLength={5000} />
          )}
          <div className="flex flex-wrap items-center gap-2">
            <label className="flex items-center gap-1" title={ts.takesHint}>
              {ts.takes}
              <select
                aria-label={`${ts.takes} ${segment.index + 1}`}
                value={takes}
                onChange={(e) => setTakes(Number(e.target.value))}
                className="h-7 rounded-md border bg-background px-1 text-xs"
              >
                {TAKES.map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex items-center gap-1" title={ts.seedHint}>
              {ts.seed}
              <input
                aria-label={`${ts.seed} ${segment.index + 1}`}
                value={seed}
                onChange={(e) => setSeed(e.target.value.replace(/\D/g, '').slice(0, 10))}
                inputMode="numeric"
                placeholder="—"
                className="h-7 w-24 rounded-md border bg-background px-2 font-mono text-xs"
              />
            </label>
            <Button size="sm" variant="secondary" className="h-7" disabled={busy} onClick={() => void repeat()}>
              {busy ? <Loader2 className="animate-spin" /> : <RefreshCw />}
              {busy ? ts.regenerating : ts.regenerate}
            </Button>
          </div>
        </div>
      )}
    </li>
  )
}

/** The sentences of a generation, each one repeatable on its own once the generation has finished. */
export function SentenceList({ gen }: { gen: GenerationRead }) {
  const editable = gen.status === 'COMPLETED' && gen.segments.length > 1
  return (
    <>
      <ol className="mt-1 space-y-0.5">
        {gen.segments.map((s) => (
          <Sentence key={s.index} gen={gen} segment={s} editable={editable} />
        ))}
      </ol>
      <p className="mt-1 text-[11px] text-muted-foreground">{editable ? ts.hint : gen.segments.length > 1 ? '' : ts.unavailable}</p>
    </>
  )
}
