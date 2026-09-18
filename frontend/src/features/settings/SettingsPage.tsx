import { AlertTriangle, CheckCircle2, CircleDashed, Download, ExternalLink, Loader2, RefreshCw, XCircle } from 'lucide-react'
import { useEffect } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { PronunciationCard } from '@/features/settings/PronunciationCard'
import { SpeakerModelCard } from '@/features/settings/SpeakerModelCard'
import { MemoryCard, StorageCard } from '@/features/settings/ResourcesPanel'
import { t } from '@/i18n/es'
import { formatMegabytes } from '@/lib/utils'
import { useSystemStore } from '@/stores/system'
import { useTranscriptionStore } from '@/stores/transcription'
import type { CheckStatus, LicenseEntry } from '@/types/api'

const STATUS_ICON: Record<CheckStatus, React.ReactNode> = {
  ok: <CheckCircle2 className="size-4 text-success" />,
  warning: <AlertTriangle className="size-4 text-warning" />,
  error: <XCircle className="size-4 text-destructive" />,
  missing: <CircleDashed className="size-4 text-muted-foreground" />,
}

const COMMERCIAL_VARIANT: Record<LicenseEntry['commercial_use'], 'success' | 'destructive' | 'warning'> = {
  permitido: 'success',
  no_permitido: 'destructive',
  revisar: 'warning',
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex justify-between gap-4 py-1.5 text-sm">
      <span className="shrink-0 text-muted-foreground">{label}</span>
      <span className="min-w-0 truncate text-right font-mono text-xs">{value ?? '—'}</span>
    </div>
  )
}

export function SettingsPage() {
  const { environment, info, licenses, error, loadEnvironment, loadInfo, loadLicenses } = useSystemStore()

  useEffect(() => {
    void loadEnvironment()
    void loadInfo()
    void loadLicenses()
  }, [loadEnvironment, loadInfo, loadLicenses])

  const gpu = environment?.gpu
  const asr = useTranscriptionStore()
  const asrStatus = asr.status
  const tm = t.transcription.model

  useEffect(() => {
    void asr.loadStatus()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-5xl space-y-5 p-4 sm:p-6">
        <header className="flex items-end justify-between">
          <h1 className="text-2xl font-semibold tracking-tight">{t.settings.title}</h1>
          <Button variant="outline" size="sm" onClick={() => void loadEnvironment()}>
            <RefreshCw />
            {t.common.retry}
          </Button>
        </header>

        {error && !environment && (
          <div role="alert" className="rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm">
            {error}
          </div>
        )}

        <div className="grid gap-5 lg:grid-cols-2">
          <MemoryCard />
          <StorageCard />
        </div>

        <div className="grid gap-5 lg:grid-cols-2 [&>*]:min-w-0">
          <PronunciationCard />
          <SpeakerModelCard />
        </div>

        <div className="grid gap-5 lg:grid-cols-[1.4fr_1fr] [&>*]:min-w-0">
          <Card>
            <CardHeader>
              <CardTitle>{t.settings.environment}</CardTitle>
            </CardHeader>
            <CardContent>
              {environment ? (
                <ul className="divide-y">
                  {environment.checks.map((check) => (
                    <li key={check.id} className="flex items-center gap-3 py-2">
                      <span title={t.settings.checkStatus[check.status]}>{STATUS_ICON[check.status]}</span>
                      <span className="shrink-0 text-sm">{check.label}</span>
                      <span className="ml-auto min-w-0 truncate font-mono text-xs text-muted-foreground" title={check.detail ?? undefined}>
                        {check.detail}
                      </span>
                    </li>
                  ))}
                </ul>
              ) : (
                <div className="space-y-3">
                  {Array.from({ length: 6 }, (_, i) => (
                    <Skeleton key={i} className="h-6" />
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          <div className="space-y-5">
            <Card>
              <CardHeader>
                <CardTitle>{t.settings.gpu}</CardTitle>
              </CardHeader>
              <CardContent>
                {gpu ? (
                  <>
                    {gpu.devices.map((d) => (
                      <div key={d.index} className="mb-2">
                        <div className="text-sm font-medium">{d.name}</div>
                        <Row
                          label={t.settings.vram}
                          value={`${formatMegabytes(d.free_memory_mb)} ${t.status.free} / ${formatMegabytes(d.total_memory_mb)}`}
                        />
                      </div>
                    ))}
                    <Row label={t.settings.backend} value={gpu.backend.toUpperCase()} />
                    <Row label={t.settings.torch} value={gpu.torch_version ?? t.common.notInstalled} />
                    <Row label="CUDA" value={gpu.cuda_version} />
                    <Row label={t.settings.driver} value={gpu.driver_version} />
                  </>
                ) : (
                  <Skeleton className="h-24" />
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>{tm.title}</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {asrStatus ? (
                  <>
                    <Row label={tm.engine} value={`${asrStatus.engine} · ${asrStatus.model}`} />
                    <Row
                      label={tm.state}
                      value={
                        !asrStatus.package_available
                          ? t.common.notInstalled
                          : asrStatus.download_state === 'downloading'
                            ? tm.downloading
                            : !asrStatus.installed
                              ? tm.notInstalled(asrStatus.model, asrStatus.approx_size_mb)
                              : asrStatus.loaded
                                ? tm.loaded(asrStatus.device, asrStatus.compute_type)
                                : tm.idle
                      }
                    />
                    <p className="text-xs text-muted-foreground">{tm.settingsHint}</p>
                    <div className="flex gap-2">
                      {asrStatus.package_available && !asrStatus.installed && (
                        <Button size="sm" variant="secondary" disabled={asrStatus.download_state === 'downloading'} onClick={() => void asr.download()}>
                          {asrStatus.download_state === 'downloading' ? <Loader2 className="animate-spin" /> : <Download />}
                          {tm.download}
                        </Button>
                      )}
                      {asrStatus.loaded && (
                        <Button size="sm" variant="outline" onClick={() => void asr.unload()}>
                          {tm.unload}
                        </Button>
                      )}
                    </div>
                  </>
                ) : (
                  <Skeleton className="h-16" />
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>{t.settings.info}</CardTitle>
              </CardHeader>
              <CardContent>
                {info ? (
                  <>
                    <Row label={t.settings.version} value={info.version} />
                    <Row label={t.settings.dataDir} value={info.data_dir} />
                    <Row label={t.settings.modelDir} value={info.model_dir} />
                    <Row label={t.settings.maxUpload} value={`${info.max_upload_mb} MB`} />
                  </>
                ) : (
                  <Skeleton className="h-20" />
                )}
              </CardContent>
            </Card>
          </div>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>{t.settings.licenses}</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="mb-3 text-sm text-muted-foreground">{t.settings.licensesHint}</p>
            {licenses ? (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="text-left text-xs text-muted-foreground">
                    <tr className="border-b">
                      <th className="py-2 pr-4 font-medium">{t.settings.component}</th>
                      <th className="py-2 pr-4 font-medium">{t.settings.code}</th>
                      <th className="py-2 pr-4 font-medium">{t.settings.weights}</th>
                      <th className="py-2 font-medium">{t.settings.commercial}</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y">
                    {licenses.map((l) => (
                      <tr key={l.component} className="align-top">
                        <td className="py-2 pr-4">
                          <a
                            href={l.url}
                            target="_blank"
                            rel="noreferrer"
                            className="inline-flex items-center gap-1 hover:text-primary"
                          >
                            {l.component}
                            <ExternalLink className="size-3" />
                          </a>
                          {l.notes && <div className="text-xs text-muted-foreground">{l.notes}</div>}
                        </td>
                        <td className="py-2 pr-4 font-mono text-xs">{l.code_license ?? '—'}</td>
                        <td className="py-2 pr-4 font-mono text-xs">{l.weights_license ?? '—'}</td>
                        <td className="py-2">
                          <Badge variant={COMMERCIAL_VARIANT[l.commercial_use]}>
                            {t.settings.commercialUse[l.commercial_use]}
                          </Badge>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <Skeleton className="h-40" />
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
