import { Mic, RotateCcw, Save, Square } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { AudioPlayer } from '@/features/audio/AudioPlayer'
import { t } from '@/i18n/es'
import { cn, formatDuration } from '@/lib/utils'
import { type RecordingIssue, type RecordingStats, analyse, encodeWav, issuesOf, mixToMono, recordingName, toDb } from '@/lib/recording'
import { useReferencesStore } from '@/stores/references'

const tr = t.recorder

/** Seconds since the call (kept outside the component: reading the clock is not a render-time concern). */
const stopwatch = () => {
  const started = performance.now()
  return () => (performance.now() - started) / 1000
}

type Phase = 'idle' | 'recording' | 'review' | 'saving'

interface Take {
  url: string
  file: File
  stats: RecordingStats
  issues: RecordingIssue[]
}

/**
 * Record a reference in the browser. The browser's own voice processing (echo cancellation, noise suppression,
 * automatic gain) is switched off: it reshapes the voice, which is exactly what a cloning reference must avoid.
 * The take is converted to 16-bit WAV locally and uploaded through the normal reference pipeline.
 */
export function Recorder({ onClose }: { onClose: () => void }) {
  const uploadFiles = useReferencesStore((s) => s.uploadFiles)
  const [phase, setPhase] = useState<Phase>('idle')
  const [scriptIndex, setScriptIndex] = useState(0)
  const [elapsed, setElapsed] = useState(0)
  const [level, setLevel] = useState(-Infinity)
  const [take, setTake] = useState<Take | null>(null)
  const [error, setError] = useState<string | null>(null)
  const media = useRef<{ stream: MediaStream; recorder: MediaRecorder; context: AudioContext; timer: number; frame: number } | null>(null)
  const takeUrl = useRef<string | null>(null)

  const stopDevices = () => {
    const m = media.current
    if (!m) return
    window.clearInterval(m.timer)
    window.cancelAnimationFrame(m.frame)
    m.stream.getTracks().forEach((track) => track.stop())
    void m.context.close()
    media.current = null
  }

  useEffect(() => () => {
    stopDevices()
    if (takeUrl.current) URL.revokeObjectURL(takeUrl.current)
  }, [])

  const start = async () => {
    setError(null)
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      setError(tr.unsupported)
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
      })
      const context = new AudioContext()
      const analyser = context.createAnalyser()
      analyser.fftSize = 2048
      context.createMediaStreamSource(stream).connect(analyser)
      const buffer = new Float32Array(analyser.fftSize)
      const recorder = new MediaRecorder(stream)
      const chunks: Blob[] = []
      recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data)
      recorder.onstop = () => void finish(new Blob(chunks, { type: recorder.mimeType }))

      const elapsedS = stopwatch()
      const meter = () => {
        analyser.getFloatTimeDomainData(buffer)
        let peak = 0
        for (const s of buffer) peak = Math.max(peak, Math.abs(s))
        setLevel(toDb(peak))
        if (media.current) media.current.frame = window.requestAnimationFrame(meter)
      }
      media.current = {
        stream, recorder, context,
        timer: window.setInterval(() => setElapsed(elapsedS()), 200),
        frame: window.requestAnimationFrame(meter),
      }
      setElapsed(0)
      recorder.start()
      setPhase('recording')
    } catch (err) {
      const denied = err instanceof DOMException && (err.name === 'NotAllowedError' || err.name === 'SecurityError')
      setError(denied ? tr.denied : tr.noMicrophone)
      stopDevices()
    }
  }

  const stop = () => media.current?.recorder.stop()

  const finish = async (blob: Blob) => {
    const context = media.current?.context
    try {
      const decoder = context ?? new AudioContext()
      const audio = await decoder.decodeAudioData(await blob.arrayBuffer())
      const samples = mixToMono(Array.from({ length: audio.numberOfChannels }, (_, i) => audio.getChannelData(i)))
      const stats = analyse(samples, audio.sampleRate)
      const wav = encodeWav(samples, audio.sampleRate)
      const file = new File([wav], recordingName(), { type: 'audio/wav' })
      takeUrl.current = URL.createObjectURL(wav)
      setTake({ url: takeUrl.current, file, stats, issues: issuesOf(stats) })
      setPhase('review')
    } catch {
      setError(tr.decodeFailed)
      setPhase('idle')
    } finally {
      stopDevices()
    }
  }

  const discard = () => {
    if (takeUrl.current) URL.revokeObjectURL(takeUrl.current)
    takeUrl.current = null
    setTake(null)
    setPhase('idle')
  }

  const save = async () => {
    if (!take) return
    setPhase('saving')
    await uploadFiles([take.file])
    discard()
    onClose()
  }

  const script = tr.scripts[scriptIndex % tr.scripts.length]
  const meterPercent = Number.isFinite(level) ? Math.min(100, Math.max(0, ((level + 60) / 60) * 100)) : 0

  return (
    <div className="space-y-3 rounded-lg border p-4" data-testid="recorder">
      <div className="flex items-start justify-between gap-2">
        <div>
          <p className="text-sm font-medium">{tr.title}</p>
          <p className="text-xs text-muted-foreground">{tr.intro}</p>
        </div>
        <Button size="sm" variant="ghost" onClick={onClose} disabled={phase === 'recording' || phase === 'saving'}>
          {t.common.close}
        </Button>
      </div>

      <div className="space-y-1 rounded-md bg-muted/50 p-3">
        <div className="flex items-center justify-between gap-2">
          <p className="text-[11px] font-semibold tracking-[0.14em] text-muted-foreground uppercase">{tr.readThis}</p>
          <Button size="sm" variant="ghost" className="h-6 px-2 text-xs" disabled={phase === 'recording'} onClick={() => setScriptIndex((i) => i + 1)}>
            {tr.otherText}
          </Button>
        </div>
        <p className="text-sm leading-relaxed" data-testid="recorder-script">
          {script}
        </p>
        <p className="text-[11px] text-muted-foreground">{tr.tips}</p>
      </div>

      {phase === 'recording' && (
        <div className="space-y-1" aria-live="polite">
          <div className="flex items-center justify-between text-xs">
            <span className="flex items-center gap-1.5 text-destructive">
              <span className="size-2 animate-pulse rounded-full bg-destructive" />
              {tr.recording}
            </span>
            <span className="font-mono">{formatDuration(elapsed)}</span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-muted" role="meter" aria-label={tr.level} aria-valuemin={-60} aria-valuemax={0} aria-valuenow={Number.isFinite(level) ? Math.round(level) : -60}>
            <div className={cn('h-full transition-[width]', level > -1 ? 'bg-destructive' : level > -30 ? 'bg-success' : 'bg-warning')} style={{ width: `${meterPercent}%` }} />
          </div>
        </div>
      )}

      {take && phase !== 'idle' && (
        <div className="space-y-2">
          <AudioPlayer url={take.url} duration={take.stats.durationS} />
          <p className="font-mono text-xs text-muted-foreground">{tr.stats(take.stats.durationS, take.stats.peakDb, take.stats.rmsDb)}</p>
          {take.issues.length === 0 ? (
            <p className="text-xs text-success">{tr.looksGood}</p>
          ) : (
            <ul className="space-y-0.5 text-xs text-warning">
              {take.issues.map((issue) => (
                <li key={issue}>{tr.issues[issue]}</li>
              ))}
            </ul>
          )}
        </div>
      )}

      {error && (
        <p role="alert" className="text-xs text-destructive">
          {error}
        </p>
      )}

      <div className="flex flex-wrap gap-2">
        {phase === 'idle' && (
          <Button size="sm" onClick={() => void start()}>
            <Mic />
            {tr.start}
          </Button>
        )}
        {phase === 'recording' && (
          <Button size="sm" variant="destructive" onClick={stop}>
            <Square />
            {tr.stop}
          </Button>
        )}
        {(phase === 'review' || phase === 'saving') && (
          <>
            <Button size="sm" disabled={phase === 'saving' || take?.issues.includes('silent')} onClick={() => void save()}>
              <Save />
              {tr.save}
            </Button>
            <Button size="sm" variant="outline" disabled={phase === 'saving'} onClick={discard}>
              <RotateCcw />
              {tr.again}
            </Button>
          </>
        )}
      </div>
    </div>
  )
}
