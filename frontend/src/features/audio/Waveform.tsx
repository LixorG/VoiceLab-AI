import { Pause, Play, PlaySquare } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import WaveSurfer from 'wavesurfer.js'
import RegionsPlugin, { type Region } from 'wavesurfer.js/dist/plugins/regions.esm.js'

import { Button } from '@/components/ui/button'
import { t } from '@/i18n/es'
import { formatDuration } from '@/lib/utils'
import { useUiStore } from '@/stores/ui'
import type { TimeRegion } from '@/types/api'

interface WaveformProps {
  url: string
  peaks: number[]
  duration: number
  selection: TimeRegion | null
  onSelectionChange: (selection: TimeRegion | null) => void
}

const cssVar = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim()

/** Audio waveform with transport and single-range selection (WaveSurfer + Regions). */
export function Waveform({ url, peaks, duration, selection, onSelectionChange }: WaveformProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const wsRef = useRef<WaveSurfer | null>(null)
  const regionsRef = useRef<RegionsPlugin | null>(null)
  const onChangeRef = useRef(onSelectionChange)
  const selectionRef = useRef(selection)
  const readyRef = useRef(false)
  const programmaticRef = useRef(false)
  const [playing, setPlaying] = useState(false)
  const [time, setTime] = useState(0)
  const theme = useUiStore((s) => s.theme)

  onChangeRef.current = onSelectionChange
  selectionRef.current = selection

  useEffect(() => {
    if (!containerRef.current) return
    const regions = RegionsPlugin.create()
    const ws = WaveSurfer.create({
      container: containerRef.current,
      url,
      peaks: [peaks],
      duration,
      height: 72,
      normalize: true,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      waveColor: cssVar('--muted-foreground'),
      progressColor: cssVar('--primary'),
      cursorColor: cssVar('--foreground'),
      cursorWidth: 1,
      plugins: [regions],
    })
    const regionColor = `color-mix(in oklch, ${cssVar('--primary')} 22%, transparent)`
    // Regions need the real duration: apply the current selection once the waveform is ready.
    readyRef.current = false
    ws.on('ready', () => {
      readyRef.current = true
      const sel = selectionRef.current
      if (sel) addRegionSilently(regions, sel, programmaticRef)
    })

    const keepSingle = (region: Region) => {
      regions.getRegions().forEach((r) => r !== region && r.remove())
      if (programmaticRef.current) return
      onChangeRef.current({ start_s: round(region.start), end_s: round(region.end) })
    }
    const unsubscribeDrag = regions.enableDragSelection({ color: regionColor })
    regions.on('region-created', keepSingle)
    regions.on('region-updated', keepSingle)
    ws.on('play', () => setPlaying(true))
    ws.on('pause', () => setPlaying(false))
    ws.on('finish', () => setPlaying(false))
    ws.on('timeupdate', (s) => setTime(s))

    wsRef.current = ws
    regionsRef.current = regions
    return () => {
      unsubscribeDrag()
      ws.destroy()
      wsRef.current = null
      regionsRef.current = null
    }
  }, [url, peaks, duration, theme])

  // Reflect externally-set selection (saved segment, suggestion click) without re-emitting it.
  useEffect(() => {
    const regions = regionsRef.current
    if (!regions || !readyRef.current) return
    const current = regions.getRegions()[0]
    if (!selection) {
      if (current) regions.clearRegions()
      return
    }
    if (current && round(current.start) === selection.start_s && round(current.end) === selection.end_s) return
    regions.clearRegions()
    addRegionSilently(regions, selection, programmaticRef)
  }, [selection, url, theme])

  const playSelection = () => regionsRef.current?.getRegions()[0]?.play()

  return (
    <div className="space-y-2">
      <div ref={containerRef} className="rounded-md bg-muted/30 px-2 py-1" data-testid="waveform" />
      <div className="flex items-center gap-2">
        <Button
          size="icon"
          variant="secondary"
          className="size-8"
          onClick={() => wsRef.current?.playPause()}
          aria-label={playing ? t.references.pause : t.references.play}
        >
          {playing ? <Pause /> : <Play />}
        </Button>
        <Button size="sm" variant="ghost" disabled={!selection} onClick={playSelection}>
          <PlaySquare />
          {t.references.segment.playSelection}
        </Button>
        <span className="ml-auto font-mono text-xs text-muted-foreground tabular-nums">
          {formatDuration(time)} / {formatDuration(duration)}
        </span>
      </div>
    </div>
  )
}

function addRegionSilently(regions: RegionsPlugin, sel: TimeRegion, flag: { current: boolean }) {
  flag.current = true
  try {
    regions.addRegion({
      start: sel.start_s,
      end: sel.end_s,
      color: `color-mix(in oklch, ${cssVar('--primary')} 22%, transparent)`,
      drag: true,
      resize: true,
    })
  } finally {
    flag.current = false
  }
}

function round(value: number) {
  return Math.round(value * 100) / 100
}
