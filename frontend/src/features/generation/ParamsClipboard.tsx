import { Check, ClipboardPaste, Copy } from 'lucide-react'
import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { t } from '@/i18n/es'
import { useGenerationStore } from '@/stores/generation'
import { settingsFromGeneration, useParamsClipboard } from '@/stores/paramsClipboard'
import type { GenerationRead } from '@/types/generation'

const tc = t.paramsClipboard

/** Brief «copied» feedback on the button that was pressed. */
function useFlash(): [boolean, () => void] {
  const [on, setOn] = useState(false)
  useEffect(() => {
    if (!on) return
    const timer = window.setTimeout(() => setOn(false), 1500)
    return () => window.clearTimeout(timer)
  }, [on])
  return [on, () => setOn(true)]
}

/** Copy the settings a generation was made with (used in Experiments, the comparison table and the history). */
export function CopyParamsButton({ gen, label, compact = false }: { gen: GenerationRead; label: string; compact?: boolean }) {
  const copy = useParamsClipboard((s) => s.copy)
  const [flash, trigger] = useFlash()
  const onClick = () => {
    void copy(settingsFromGeneration(gen, label))
    trigger()
  }
  const icon = flash ? <Check className="text-success" /> : <Copy />
  const button = compact ? (
    <Button size="icon" variant="ghost" className="size-7" aria-label={`${tc.copy}: ${label}`} onClick={onClick}>
      {icon}
    </Button>
  ) : (
    <Button size="sm" variant="outline" className="h-7 px-2 text-[11px]" aria-label={`${tc.copy}: ${label}`} onClick={onClick}>
      {icon}
      {flash ? tc.copied : tc.copy}
    </Button>
  )
  return (
    <Tooltip>
      <TooltipTrigger asChild>{button}</TooltipTrigger>
      <TooltipContent className="max-w-72">{flash ? tc.copied : tc.copyHint}</TooltipContent>
    </Tooltip>
  )
}

/** Generar: copy the current settings, or paste the ones copied from a result. */
export function ParamsClipboardBar() {
  const { copied, copy } = useParamsClipboard()
  const { applySettings, currentSettings } = useGenerationStore()
  const [flash, trigger] = useFlash()
  const [report, setReport] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const onCopy = () => {
    const settings = currentSettings()
    if (!settings) return
    void copy(settings)
    trigger()
  }

  const onPaste = async () => {
    if (!copied) return
    setBusy(true)
    setError(null)
    setReport(null)
    try {
      const result = await applySettings(copied)
      setReport([tc.applied(result.applied), result.skipped.length ? tc.skipped(result.skipped.join(', ')) : ''].filter(Boolean).join(' '))
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="space-y-1.5" data-testid="params-clipboard">
      <div className="flex flex-wrap gap-2">
        <Button size="sm" variant="outline" onClick={onCopy} title={tc.copyHint}>
          {flash ? <Check className="text-success" /> : <Copy />}
          {flash ? tc.copied : tc.copyCurrent}
        </Button>
        {copied && (
          <Button size="sm" variant="outline" disabled={busy} onClick={() => void onPaste()} title={tc.pasteHint}>
            <ClipboardPaste />
            {tc.paste(copied.label)}
          </Button>
        )}
      </div>
      {report && (
        <p role="status" className="text-[11px] text-muted-foreground">
          {report}
        </p>
      )}
      {error && (
        <p role="alert" className="text-[11px] text-destructive">
          {error}
        </p>
      )}
    </div>
  )
}
