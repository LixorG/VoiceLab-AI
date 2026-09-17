import { Cpu, HardDrive, Loader2, MemoryStick, RefreshCw, Trash2 } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { t } from '@/i18n/es'
import { formatMegabytes } from '@/lib/utils'
import { ApiError } from '@/services/api'
import { type CleanupOptions, type CleanupReport, formatBytes, type MemoryStatus, resourcesApi, type StorageUsage } from '@/services/resources'

const tr = t.resources
const POLL_MS = 5000
const messageOf = (err: unknown) => (err instanceof ApiError ? err.message : String(err))

function Bar({ used, total, label }: { used: number; total: number; label: string }) {
  const pct = total > 0 ? Math.min(100, Math.round((used / total) * 100)) : 0
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono tabular-nums">
          {formatMegabytes(used)} / {formatMegabytes(total)}
        </span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-muted" role="progressbar" aria-label={label} aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
        <div className={`h-full ${pct > 90 ? 'bg-destructive' : pct > 70 ? 'bg-warning' : 'bg-primary'}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

const minutes = (s: number | null) => (s == null ? null : Math.max(1, Math.ceil(s / 60)))

export function MemoryCard() {
  const [status, setStatus] = useState<MemoryStatus | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [message, setMessage] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      setStatus(await resourcesApi.memory())
    } catch {
      // backend offline: keep the last snapshot
    }
  }, [])

  useEffect(() => {
    void load()
    const id = window.setInterval(() => void load(), POLL_MS)
    return () => window.clearInterval(id)
  }, [load])

  const release = async (tts: boolean, asr: boolean) => {
    setBusy(tts && asr ? 'all' : tts ? 'tts' : 'asr')
    setMessage(null)
    try {
      const res = await resourcesApi.release(tts, asr)
      setStatus(res.status)
      const skipped = Object.values(res.skipped)
      setMessage(skipped.length ? skipped.join(' ') : res.released.length ? tr.released : tr.nothingToRelease)
    } catch (err) {
      setMessage(messageOf(err))
    } finally {
      setBusy(null)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <MemoryStick className="size-4" />
          {tr.memoryTitle}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4" data-testid="memory-card">
        {!status ? (
          <Loader2 className="size-4 animate-spin text-muted-foreground" />
        ) : (
          <>
            {status.gpu ? (
              <div className="space-y-1">
                <Bar used={status.gpu.used_mb} total={status.gpu.total_mb} label={tr.gpu(status.gpu.name)} />
                <p className="text-[11px] text-muted-foreground">{tr.torch(formatMegabytes(status.gpu.torch_allocated_mb), formatMegabytes(status.gpu.torch_reserved_mb))}</p>
              </div>
            ) : (
              <p className="text-xs text-muted-foreground">{tr.noGpu}</p>
            )}
            {status.system_ram_total_mb != null && status.system_ram_available_mb != null && (
              <div className="space-y-1">
                <Bar used={status.system_ram_total_mb - status.system_ram_available_mb} total={status.system_ram_total_mb} label={tr.ram} />
                {status.process_ram_mb != null && <p className="text-[11px] text-muted-foreground">{tr.process(formatMegabytes(status.process_ram_mb))}</p>}
              </div>
            )}

            <div className="space-y-2 text-sm">
              <div className="flex flex-wrap items-center gap-2">
                <Cpu className="size-4 text-muted-foreground" />
                <span className="font-medium">{tr.tts}</span>
                {status.tts ? (
                  <>
                    <span className="font-mono text-xs">
                      {status.tts.engine} · {status.tts.variant} ({status.tts.device.toUpperCase()})
                    </span>
                    {status.tts.in_use ? <Badge variant="warning">{tr.inUse}</Badge> : minutes(status.tts.idle_unload_in_s) != null && <Badge variant="outline">{tr.autoUnload(minutes(status.tts.idle_unload_in_s)!)}</Badge>}
                  </>
                ) : (
                  <span className="text-xs text-muted-foreground">{tr.notLoaded}</span>
                )}
                <Button size="sm" variant="ghost" className="ml-auto h-7" disabled={!status.tts || status.tts.in_use || busy != null} onClick={() => void release(true, false)}>
                  {busy === 'tts' && <Loader2 className="animate-spin" />}
                  {tr.release}
                </Button>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <Cpu className="size-4 text-muted-foreground" />
                <span className="font-medium">{tr.asr}</span>
                {status.asr.loaded ? (
                  <>
                    <span className="font-mono text-xs">
                      {status.asr.model} ({(status.asr.device ?? '').toUpperCase()})
                    </span>
                    {minutes(status.asr.idle_unload_in_s) != null && <Badge variant="outline">{tr.autoUnload(minutes(status.asr.idle_unload_in_s)!)}</Badge>}
                  </>
                ) : (
                  <span className="text-xs text-muted-foreground">{tr.notLoaded}</span>
                )}
                <Button size="sm" variant="ghost" className="ml-auto h-7" disabled={!status.asr.loaded || busy != null} onClick={() => void release(false, true)}>
                  {busy === 'asr' && <Loader2 className="animate-spin" />}
                  {tr.release}
                </Button>
              </div>
            </div>

            <p className="text-[11px] text-muted-foreground">
              {tr.policy(status.idle_unload_minutes.tts, status.idle_unload_minutes.asr)}
              {(status.queue_active > 0 || status.queue_waiting > 0) && ` ${tr.queue(status.queue_active, status.queue_waiting)}`}
            </p>
            {message && (
              <p role="status" className="text-xs">
                {message}
              </p>
            )}
          </>
        )}
      </CardContent>
    </Card>
  )
}

const OPTION_KEYS: (keyof CleanupOptions)[] = ['orphans', 'temp', 'reference_clips', 'transcript_cache']

export function StorageCard() {
  const [usage, setUsage] = useState<StorageUsage | null>(null)
  const [options, setOptions] = useState<CleanupOptions>({ orphans: true, temp: true, reference_clips: false, transcript_cache: false })
  const [loading, setLoading] = useState(false)
  const [cleaning, setCleaning] = useState(false)
  const [report, setReport] = useState<CleanupReport | null>(null)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      setUsage(await resourcesApi.storage())
      setError(null)
    } catch (err) {
      setError(messageOf(err))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  const cleanup = async () => {
    if (!window.confirm(tr.confirmCleanup)) return
    setCleaning(true)
    setError(null)
    try {
      setReport(await resourcesApi.cleanup(options))
      await load()
    } catch (err) {
      setError(messageOf(err))
    } finally {
      setCleaning(false)
    }
  }

  const max = Math.max(1, ...(usage?.categories.map((c) => c.bytes) ?? [1]))

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <HardDrive className="size-4" />
          {tr.storageTitle}
        </CardTitle>
        <Button size="sm" variant="ghost" className="h-7" onClick={() => void load()} disabled={loading} aria-label={t.common.retry}>
          <RefreshCw className={loading ? 'animate-spin' : ''} />
        </Button>
      </CardHeader>
      <CardContent className="space-y-4" data-testid="storage-card">
        {error && (
          <p role="alert" className="text-xs text-destructive">
            {error}
          </p>
        )}
        {usage && (
          <>
            <ul className="space-y-2">
              {usage.categories.map((c) => (
                <li key={c.id} className="space-y-0.5" title={c.description}>
                  <div className="flex justify-between gap-2 text-xs">
                    <span>
                      {c.label}
                      {c.removable && <span className="text-muted-foreground"> · {tr.regenerable}</span>}
                    </span>
                    <span className="font-mono tabular-nums text-muted-foreground">{formatBytes(c.bytes)}</span>
                  </div>
                  <div className="h-1 overflow-hidden rounded-full bg-muted">
                    <div className="h-full bg-primary/60" style={{ width: `${Math.max(c.bytes ? 1 : 0, (c.bytes / max) * 100)}%` }} />
                  </div>
                </li>
              ))}
            </ul>
            <p className="text-xs text-muted-foreground">
              {tr.total(formatBytes(usage.total_bytes))} · {usage.orphans_files > 0 ? tr.orphans(usage.orphans_files, formatBytes(usage.orphans_bytes)) : tr.noOrphans}
            </p>

            <fieldset className="space-y-1.5">
              <legend className="mb-1 text-xs font-medium">{tr.cleanupTitle}</legend>
              {OPTION_KEYS.map((key) => (
                <label key={key} className="flex items-start gap-2 text-xs">
                  <input type="checkbox" className="mt-0.5 accent-[var(--primary)]" checked={options[key]} onChange={(e) => setOptions({ ...options, [key]: e.target.checked })} />
                  <span>
                    {tr.options[key]}
                    <span className="block text-[11px] text-muted-foreground">{tr.optionHints[key]}</span>
                  </span>
                </label>
              ))}
            </fieldset>
            <div className="flex flex-wrap items-center gap-3">
              <Button size="sm" variant="outline" disabled={cleaning || !OPTION_KEYS.some((k) => options[k])} onClick={() => void cleanup()}>
                {cleaning ? <Loader2 className="animate-spin" /> : <Trash2 />}
                {tr.cleanup}
              </Button>
              {report && (
                <span role="status" className="text-xs text-muted-foreground">
                  {tr.cleaned(report.removed_files, formatBytes(report.freed_bytes))} {report.skipped.join(' ')}
                </span>
              )}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  )
}
