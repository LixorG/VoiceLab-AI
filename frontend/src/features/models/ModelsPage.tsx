import { ExternalLink } from 'lucide-react'
import { useEffect, useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { SourceBadge } from '@/features/models/GenericControlRow'
import { t } from '@/i18n/es'
import { formatMegabytes } from '@/lib/utils'
import { modelsApi } from '@/services/models'
import type { EngineConfig, EngineSummary, GenericControl } from '@/types/engines'

const te = t.engines
const CONTROL_ORDER: GenericControl[] = ['speed', 'pitch', 'emotion', 'naturalness', 'voice_color', 'instruction', 'language', 'seed', 'target_duration']

function EngineCard({ engine }: { engine: EngineSummary }) {
  const [variantId, setVariantId] = useState(engine.default_variant)
  const [config, setConfig] = useState<EngineConfig | null>(null)

  useEffect(() => {
    let cancelled = false
    modelsApi
      .config(engine.id, variantId)
      .then((c) => !cancelled && setConfig(c))
      .catch(() => !cancelled && setConfig(null))
    return () => {
      cancelled = true
    }
  }, [engine.id, variantId])

  return (
    <Card>
      <CardHeader className="flex-wrap items-start">
        <div className="min-w-0 space-y-1">
          <h2 className="text-base font-semibold">{engine.name}</h2>
          <p className="text-xs text-muted-foreground">{engine.description}</p>
        </div>
        <div className="flex flex-wrap gap-1.5">
          <Badge variant={engine.installed ? 'success' : 'outline'} title={engine.installed ? undefined : te.missing(engine.missing_packages.join(', '))}>
            {engine.installed ? te.installed : te.notInstalled}
          </Badge>
          {engine.license.commercial_use === 'no_permitido' && <Badge variant="destructive">{te.nonCommercial}</Badge>}
          {!engine.implemented && engine.implementation_phase != null && <Badge variant="outline">{te.generationPhase(engine.implementation_phase)}</Badge>}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead className="text-left text-muted-foreground">
              <tr className="border-b">
                <th className="py-1.5 pr-3 font-medium">{te.variant}</th>
                <th className="py-1.5 pr-3 font-medium">{te.models.modeLabel}</th>
                <th className="py-1.5 pr-3 font-medium">{te.models.vram}</th>
                <th className="py-1.5 font-medium">{te.models.download}</th>
              </tr>
            </thead>
            <tbody className="divide-y">
              {engine.variants.map((v) => (
                <tr
                  key={v.id}
                  onClick={() => setVariantId(v.id)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault()
                      setVariantId(v.id)
                    }
                  }}
                  tabIndex={0}
                  className={`cursor-pointer hover:bg-muted/40 focus-visible:outline-2 focus-visible:outline-primary ${v.id === variantId ? 'bg-muted/60' : ''}`}
                  aria-selected={v.id === variantId}
                >
                  <td className="py-1.5 pr-3">
                    <div>{v.label}</div>
                    <div className="font-mono text-[10px] text-muted-foreground">{v.repo_id}</div>
                  </td>
                  <td className="py-1.5 pr-3">{te.models.mode[v.mode]}</td>
                  <td className="py-1.5 pr-3 font-mono">~{formatMegabytes(v.vram_estimate_mb)}</td>
                  <td className="py-1.5 font-mono">~{formatMegabytes(v.download_size_mb)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-1 text-[11px] text-muted-foreground">{te.models.estimate}</p>
        </div>

        <div className="space-y-2">
          <h3 className="text-[11px] font-semibold tracking-[0.14em] text-muted-foreground uppercase">
            {te.models.capabilities} · {engine.variants.find((v) => v.id === variantId)?.label}
          </h3>
          {config ? (
            <>
              <dl className="grid grid-cols-2 gap-x-4 gap-y-1.5 sm:grid-cols-3">
                {CONTROL_ORDER.map((c) => (
                  <div key={c} className="flex items-center justify-between gap-2 text-xs">
                    <dt className="text-muted-foreground">{te.controls[c]}</dt>
                    <dd>
                      <SourceBadge capability={config.capabilities.controls[c]} />
                    </dd>
                  </div>
                ))}
              </dl>
              <p className="text-xs text-muted-foreground">
                {config.capabilities.requires_reference_audio ? te.requiresReference : te.noReference}
                {config.capabilities.requires_reference_text && ` · ${te.requiresText}`}
              </p>
            </>
          ) : (
            <Skeleton className="h-16" />
          )}
        </div>

        <a href={engine.license.url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-primary">
          {te.licenseHint(engine.license.weights, engine.license.code)}
          <ExternalLink className="size-3" />
        </a>
      </CardContent>
    </Card>
  )
}

export function ModelsPage() {
  const [engines, setEngines] = useState<EngineSummary[] | null>(null)

  useEffect(() => {
    modelsApi.list().then(setEngines).catch(() => setEngines([]))
  }, [])

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-5xl space-y-5 p-4 sm:p-6">
        <header>
          <h1 className="text-2xl font-semibold tracking-tight">{te.models.title}</h1>
          <p className="text-sm text-muted-foreground">{te.models.subtitle}</p>
        </header>
        {engines == null ? <Skeleton className="h-64" /> : engines.map((e) => <EngineCard key={e.id} engine={e} />)}
      </div>
    </div>
  )
}
