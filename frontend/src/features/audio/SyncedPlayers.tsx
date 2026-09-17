import { Pause, Play, SkipBack, ZoomIn } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import WaveSurfer from 'wavesurfer.js'

import { Button } from '@/components/ui/button'
import { t } from '@/i18n/es'
import { cn, formatDuration } from '@/lib/utils'
import { useUiStore } from '@/stores/ui'

const tc = t.compare
const cssVar = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim()
const DRIFT_S = 0.08
const LETTERS = 'ABCDEFGHIJ'

export interface SyncedTrack {
  id: string
  label: string
  url: string
}

/**
 * Several waveforms sharing one transport: play/pause/seek move every track to the same time, and only the selected
 * track is audible, so switching A/B is instant and keeps the position. Zoom applies to all tracks.
 */
export function SyncedPlayers({ tracks }: { tracks: SyncedTrack[] }) {
  const containers = useRef<(HTMLDivElement | null)[]>([])
  const players = useRef<WaveSurfer[]>([])
  const durations = useRef<number[]>([])
  const clock = useRef<{ startedAt: number; offset: number } | null>(null)
  const frame = useRef<number | null>(null)
  const [playing, setPlaying] = useState(false)
  const [time, setTime] = useState(0)
  const [total, setTotal] = useState(0)
  const [ready, setReady] = useState(0)
  const [solo, setSolo] = useState(0)
  const [zoom, setZoom] = useState(0)
  const theme = useUiStore((s) => s.theme)
  const key = tracks.map((tr) => tr.url).join('|')

  const current = () => (clock.current ? clock.current.offset + (performance.now() - clock.current.startedAt) / 1000 : time)

  const stopClock = useCallback(() => {
    if (frame.current != null) cancelAnimationFrame(frame.current)
    frame.current = null
    clock.current = null
  }, [])

  const seekAll = useCallback(
    (seconds: number, keepPlaying: boolean) => {
      const at = Math.max(0, seconds)
      players.current.forEach((ws, i) => {
        const d = durations.current[i] ?? 0
        ws.setTime(Math.min(at, d))
        if (keepPlaying && at < d) void ws.play()
        else ws.pause()
      })
      if (keepPlaying) clock.current = { startedAt: performance.now(), offset: at }
      setTime(at)
    },
    [],
  )

  const tick = useCallback(() => {
    const now = current()
    const longest = Math.max(0, ...durations.current)
    if (now >= longest) {
      players.current.forEach((ws) => ws.pause())
      stopClock()
      setPlaying(false)
      setTime(longest)
      return
    }
    players.current.forEach((ws, i) => {
      const d = durations.current[i] ?? 0
      if (now < d && Math.abs(ws.getCurrentTime() - now) > DRIFT_S) ws.setTime(now)
    })
    setTime(now)
    frame.current = requestAnimationFrame(tick)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stopClock])

  useEffect(() => {
    const created = tracks.map((track, i) => {
      const ws = WaveSurfer.create({
        container: containers.current[i]!,
        url: track.url,
        height: 56,
        normalize: false, // same scale for every track, so level differences stay visible
        barWidth: 2,
        barGap: 1,
        barRadius: 2,
        waveColor: cssVar('--muted-foreground'),
        progressColor: cssVar('--primary'),
        cursorColor: cssVar('--foreground'),
        cursorWidth: 1,
        autoScroll: true,
      })
      ws.on('ready', (d) => {
        durations.current[i] = d
        setTotal(Math.max(0, ...durations.current))
        setReady((n) => n + 1)
      })
      ws.on('interaction', (seconds) => {
        const wasPlaying = clock.current != null
        seekAll(seconds, wasPlaying)
      })
      return ws
    })
    players.current = created
    durations.current = tracks.map(() => 0)
    setReady(0)
    setTime(0)
    setPlaying(false)
    return () => {
      stopClock()
      created.forEach((ws) => ws.destroy())
      players.current = []
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, theme])

  useEffect(() => {
    players.current.forEach((ws, i) => ws.setMuted(i !== solo))
  }, [solo, ready])

  useEffect(() => {
    if (ready < tracks.length) return
    players.current.forEach((ws) => {
      try {
        ws.zoom(zoom)
      } catch {
        // not decoded yet
      }
    })
  }, [zoom, ready, tracks.length])

  const toggle = () => {
    if (playing) {
      const now = current()
      players.current.forEach((ws) => ws.pause())
      stopClock()
      setTime(now)
      setPlaying(false)
    } else {
      const from = time >= total ? 0 : time
      seekAll(from, true)
      setPlaying(true)
      frame.current = requestAnimationFrame(tick)
    }
  }

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === ' ') {
      e.preventDefault()
      toggle()
    }
    const n = parseInt(e.key, 10)
    if (n >= 1 && n <= tracks.length) setSolo(n - 1)
  }

  const allReady = ready >= tracks.length

  return (
    <div className="space-y-3" role="group" aria-label={tc.title} data-testid="synced-players">
      <div className="flex flex-wrap items-center gap-2">
        <Button size="icon" className="size-9" onClick={toggle} onKeyDown={onKey} disabled={!allReady} aria-label={playing ? t.references.pause : t.references.play}>
          {playing ? <Pause /> : <Play />}
        </Button>
        <Button size="icon" variant="ghost" className="size-9" onClick={() => seekAll(0, playing)} disabled={!allReady} aria-label={tc.restart}>
          <SkipBack />
        </Button>
        <span className="font-mono text-xs tabular-nums text-muted-foreground">
          {formatDuration(time)} / {formatDuration(total)}
        </span>
        <div className="ml-auto flex items-center gap-2 text-xs text-muted-foreground">
          <ZoomIn className="size-4" />
          <label htmlFor="compare-zoom" className="sr-only">
            {tc.zoom}
          </label>
          <input id="compare-zoom" type="range" min={0} max={400} step={10} value={zoom} disabled={!allReady} onChange={(e) => setZoom(parseInt(e.target.value, 10))} className="h-1.5 w-28 accent-[var(--primary)]" />
        </div>
      </div>
      <p className="text-[11px] text-muted-foreground">{tc.hint}</p>
      <div className="space-y-2" role="radiogroup" aria-label={tc.listenTo}>
        {tracks.map((track, i) => (
          <div key={track.id} className={cn('flex items-center gap-2 rounded-md border p-2 transition-colors', i === solo ? 'border-primary bg-primary/5' : 'opacity-80')}>
            <button
              type="button"
              role="radio"
              aria-checked={i === solo}
              onClick={() => setSolo(i)}
              onKeyDown={onKey}
              className={cn('grid size-8 shrink-0 place-items-center rounded-md border font-mono text-sm font-semibold', i === solo ? 'border-primary bg-primary text-primary-foreground' : 'text-muted-foreground')}
              title={tc.listen(track.label)}
            >
              {LETTERS[i] ?? i + 1}
            </button>
            <div className="w-28 shrink-0 truncate text-xs" title={track.label}>
              {track.label}
            </div>
            <div ref={(el) => void (containers.current[i] = el)} className="min-w-0 flex-1" />
          </div>
        ))}
      </div>
    </div>
  )
}
