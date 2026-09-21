import { Radio, Square } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { t } from '@/i18n/es'

const tl = t.live
/** Start once this many chunks are buffered (or the generation ended): the first seconds must not run dry. */
const START_AFTER = 2
const LEAD_S = 0.05

interface Props {
  generationId: string
  chunks: number
  finished: boolean
}

/**
 * Plays the live-preview chunks of a running generation back to back (Web Audio, gapless scheduling).
 * The browser only lets audio start from a click, hence the button. It is a preview: the final file (with any
 * post-processing) replaces it when the generation ends.
 */
export function LivePlayer({ generationId, chunks, finished }: Props) {
  const [playing, setPlaying] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const state = useRef<{ context: AudioContext; next: number; endAt: number; started: boolean; buffers: AudioBuffer[]; sources: AudioBufferSourceNode[]; busy: boolean } | null>(null)

  const stop = () => {
    const s = state.current
    state.current = null
    setPlaying(false)
    if (!s) return
    s.sources.forEach((source) => {
      try {
        source.stop()
      } catch {
        /* already finished */
      }
    })
    void s.context.close()
  }

  useEffect(() => stop, []) // eslint-disable-line react-hooks/exhaustive-deps

  const schedule = (s: NonNullable<typeof state.current>, buffer: AudioBuffer) => {
    const source = s.context.createBufferSource()
    source.buffer = buffer
    source.connect(s.context.destination)
    const at = Math.max(s.context.currentTime + LEAD_S, s.endAt)
    source.start(at)
    s.endAt = at + buffer.duration
    s.sources.push(source)
  }

  // Fetch every chunk that became available and schedule it right after the previous one.
  useEffect(() => {
    const s = state.current
    if (!playing || !s || s.busy) return
    const flushOnly = finished && !s.started && s.buffers.length > 0  // ended before the start threshold
    if (s.next >= chunks && !flushOnly) return
    s.busy = true
    const load = async () => {
      try {
        while (state.current === s && s.next < chunks) {
          const res = await fetch(`/api/generation/${generationId}/stream/${s.next}`)
          if (!res.ok) break
          s.buffers.push(await s.context.decodeAudioData(await res.arrayBuffer()))
          s.next += 1
        }
        if (state.current !== s) return
        if (!s.started && (s.buffers.length >= START_AFTER || finished)) s.started = true
        if (s.started) s.buffers.splice(0).forEach((buffer) => schedule(s, buffer))
      } catch {
        setError(tl.error)
        stop()
      } finally {
        s.busy = false
      }
    }
    void load()
  }, [playing, chunks, finished, generationId]) // eslint-disable-line react-hooks/exhaustive-deps

  // Once everything was scheduled after the generation ended, release the audio device when playback ends.
  useEffect(() => {
    const s = state.current
    if (!playing || !s || !finished || s.next < chunks || s.buffers.length) return
    const remaining = Math.max(0, s.endAt - s.context.currentTime)
    const timer = window.setTimeout(stop, remaining * 1000 + 200)
    return () => window.clearTimeout(timer)
  }, [playing, finished, chunks]) // eslint-disable-line react-hooks/exhaustive-deps

  const start = () => {
    setError(null)
    const context = new AudioContext()
    state.current = { context, next: 0, endAt: 0, started: false, buffers: [], sources: [], busy: false }
    setPlaying(true)
  }

  if (!playing && (chunks === 0 || finished)) return null

  return (
    <div className="flex items-center gap-2 text-xs" data-testid="live-player">
      {playing ? (
        <Button size="sm" variant="outline" onClick={stop}>
          <Square />
          {tl.stop}
        </Button>
      ) : (
        !finished && (
          <Button size="sm" variant="outline" onClick={start}>
            <Radio />
            {tl.listen}
          </Button>
        )
      )}
      <span className="text-muted-foreground">{playing ? tl.playing : tl.hint}</span>
      {error && (
        <span role="alert" className="text-destructive">
          {error}
        </span>
      )}
    </div>
  )
}
