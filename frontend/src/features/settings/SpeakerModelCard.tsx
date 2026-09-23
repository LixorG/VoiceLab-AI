import { Download, Loader2 } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { t } from '@/i18n/es'
import { ApiError, request } from '@/services/api'

const tv = t.speakerModel
const POLL_MS = 3000

interface SpeakerModelStatus {
  model: string
  installed: boolean
  loaded: boolean
  download_state: 'idle' | 'downloading' | 'failed'
  download_error: string | null
  download_size_mb: number
  license: string
}

const api = {
  status: () => request<SpeakerModelStatus>('/generation/evaluators/speaker-model'),
  download: () => request<SpeakerModelStatus>('/generation/evaluators/speaker-model/download', { method: 'POST' }),
}

/** Optional model behind the «Similitud de voz» estimate: explains what it measures and downloads it on demand. */
export function SpeakerModelCard() {
  const [status, setStatus] = useState<SpeakerModelStatus | null>(null)
  const [error, setError] = useState<string | null>(null)

  const alive = useRef(true)
  const read = useCallback(
    () =>
      api
        .status()
        .then((s) => alive.current && setStatus(s))
        .catch((err) => alive.current && setError(err instanceof ApiError ? err.message : String(err))),
    [],
  )

  useEffect(() => {
    alive.current = true
    void read()
    return () => {
      alive.current = false
    }
  }, [read])

  // Only while it downloads: asking again on every state change would undo what the download button just showed.
  useEffect(() => {
    if (status?.download_state !== 'downloading') return
    const timer = window.setInterval(() => void read(), POLL_MS)
    return () => window.clearInterval(timer)
  }, [status?.download_state, read])

  const download = async () => {
    setError(null)
    try {
      setStatus(await api.download())
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err))
    }
  }

  const downloading = status?.download_state === 'downloading'

  return (
    <Card data-testid="speaker-model-card">
      <CardHeader className="flex flex-row items-center justify-between gap-2">
        <CardTitle>{tv.title}</CardTitle>
        {status && (
          <Badge variant={status.installed ? 'success' : 'outline'}>
            {status.installed ? tv.ready : downloading ? tv.downloading : tv.notDownloaded}
          </Badge>
        )}
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        <p className="text-muted-foreground">{tv.intro}</p>
        <p className="rounded-md border border-dashed px-3 py-2 text-xs text-muted-foreground">{tv.limits}</p>
        {status && (
          <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-xs">
            <dt className="text-muted-foreground">{tv.model}</dt>
            <dd className="font-mono">{status.model}</dd>
            <dt className="text-muted-foreground">{tv.size}</dt>
            <dd>~{status.download_size_mb} MB</dd>
            <dt className="text-muted-foreground">{tv.license}</dt>
            <dd>{status.license}</dd>
          </dl>
        )}
        {(error || status?.download_error) && (
          <p role="alert" className="text-xs text-destructive">
            {error ?? status?.download_error}
          </p>
        )}
        {status && !status.installed && (
          <Button size="sm" disabled={downloading} onClick={() => void download()}>
            {downloading ? <Loader2 className="animate-spin" /> : <Download />}
            {downloading ? tv.downloading : tv.download(status.download_size_mb)}
          </Button>
        )}
        {status?.installed && <p className="text-xs text-muted-foreground">{tv.howToUse}</p>}
      </CardContent>
    </Card>
  )
}
