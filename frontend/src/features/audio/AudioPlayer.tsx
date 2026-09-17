import { Pause, Play } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import WaveSurfer from 'wavesurfer.js'

import { Button } from '@/components/ui/button'
import { t } from '@/i18n/es'
import { formatDuration } from '@/lib/utils'
import { useUiStore } from '@/stores/ui'

const cssVar = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim()

/** Compact player for generated audio (waveform decoded from the file). */
export function AudioPlayer({ url, duration }: { url: string; duration?: number | null }) {
  const containerRef = useRef<HTMLDivElement>(null)
  const wsRef = useRef<WaveSurfer | null>(null)
  const [playing, setPlaying] = useState(false)
  const [time, setTime] = useState(0)
  const [total, setTotal] = useState(duration ?? 0)
  const theme = useUiStore((s) => s.theme)

  useEffect(() => {
    if (!containerRef.current) return
    const ws = WaveSurfer.create({
      container: containerRef.current,
      url,
      height: 48,
      normalize: true,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      waveColor: cssVar('--muted-foreground'),
      progressColor: cssVar('--primary'),
      cursorColor: cssVar('--foreground'),
      cursorWidth: 1,
    })
    ws.on('ready', (d) => setTotal(d))
    ws.on('play', () => setPlaying(true))
    ws.on('pause', () => setPlaying(false))
    ws.on('finish', () => setPlaying(false))
    ws.on('timeupdate', (s) => setTime(s))
    wsRef.current = ws
    return () => {
      ws.destroy()
      wsRef.current = null
    }
  }, [url, theme])

  return (
    <div className="flex items-center gap-2">
      <Button
        size="icon"
        variant="secondary"
        className="size-9 shrink-0"
        onClick={() => wsRef.current?.playPause()}
        aria-label={playing ? t.references.pause : t.references.play}
      >
        {playing ? <Pause /> : <Play />}
      </Button>
      <div ref={containerRef} className="min-w-0 flex-1" data-testid="audio-player" />
      <span className="shrink-0 font-mono text-xs text-muted-foreground tabular-nums">
        {formatDuration(time)} / {formatDuration(total)}
      </span>
    </div>
  )
}
