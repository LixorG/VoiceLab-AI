import { Download, Loader2, Zap } from 'lucide-react'
import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { t } from '@/i18n/es'
import { formatMegabytes } from '@/lib/utils'
import { resourcesApi } from '@/services/resources'
import { useEngineStore } from '@/stores/engine'
import { useGenerationStore } from '@/stores/generation'

const tw = t.generation.weights

/** Download state of the selected engine variant's weights (only relevant once the package is installed). */
export function WeightsStatus() {
  const { engineId, variantId, config } = useEngineStore()
  const { runtime, loadRuntime, downloadWeights } = useGenerationStore()
  const [preloading, setPreloading] = useState(false)
  const [preloadError, setPreloadError] = useState<string | null>(null)

  const preload = async () => {
    setPreloading(true)
    setPreloadError(null)
    try {
      const job = await resourcesApi.preload(engineId!, variantId)
      for (;;) {
        await new Promise((r) => setTimeout(r, 1000))
        const state = await resourcesApi.job(job.job_id)
        if (state.status === 'COMPLETED') break
        if (['FAILED', 'CANCELLED'].includes(state.status)) {
          setPreloadError(state.message)
          break
        }
      }
    } catch (err) {
      setPreloadError(err instanceof Error ? err.message : String(err))
    } finally {
      setPreloading(false)
      void loadRuntime()
    }
  }

  useEffect(() => {
    if (engineId && config?.engine.installed) void loadRuntime()
  }, [engineId, variantId, config?.engine.installed, loadRuntime])

  if (!config?.engine.installed || !config.engine.implemented || !runtime || runtime.engine !== engineId || runtime.variant !== variantId) {
    return null
  }
  if (runtime.download_state === 'downloading') {
    return (
      <p className="flex items-center gap-2 text-xs text-muted-foreground">
        <Loader2 className="size-3.5 animate-spin text-primary" />
        {tw.downloading}
      </p>
    )
  }
  if (!runtime.weights_installed) {
    return (
      <div className="flex flex-wrap items-center gap-2 rounded-md border border-dashed px-3 py-2 text-xs">
        <span className="text-muted-foreground">
          {runtime.download_error ?? tw.missing(config.variant.download_size_mb ? formatMegabytes(config.variant.download_size_mb) : '')}
        </span>
        <Button size="sm" variant="secondary" onClick={() => void downloadWeights()}>
          <Download />
          {tw.download}
        </Button>
      </div>
    )
  }
  if (runtime.loaded && runtime.device) return <p className="text-xs text-success">{tw.loaded(runtime.device)}</p>
  return (
    <div className="flex flex-wrap items-center gap-2 text-xs">
      <span className="text-success">{tw.ready}</span>
      <Button size="sm" variant="ghost" className="h-7 text-[11px]" disabled={preloading} onClick={() => void preload()} title={tw.preloadHint}>
        {preloading ? <Loader2 className="animate-spin" /> : <Zap />}
        {preloading ? tw.preloading : tw.preload}
      </Button>
      {preloadError && (
        <span role="alert" className="text-destructive">
          {preloadError}
        </span>
      )}
    </div>
  )
}
