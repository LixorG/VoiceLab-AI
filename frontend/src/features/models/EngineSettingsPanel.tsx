import { AlertTriangle, ChevronDown, Info, RotateCcw } from 'lucide-react'
import { useEffect, useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { EmotionControl } from '@/features/generation/EmotionControl'
import { GenericControlRow } from '@/features/models/GenericControlRow'
import { ParameterControl } from '@/features/models/ParameterControl'
import { WeightsStatus } from '@/features/models/WeightsStatus'
import { RecommendedSettingsBar } from '@/features/voices/RecommendedSettingsBar'
import { t } from '@/i18n/es'
import { cn } from '@/lib/utils'
import { keyOf, useEngineStore } from '@/stores/engine'
import { useSystemStore } from '@/stores/system'
import { useUiStore } from '@/stores/ui'
import type { GenericControl, ParamValue, ParameterSpec } from '@/types/engines'

const te = t.engines

const SECTIONS: { title: string; controls: GenericControl[]; advancedOnly?: GenericControl[] }[] = [
  { title: te.sections.expression, controls: ['emotion', 'naturalness', 'instruction', 'voice_color'], advancedOnly: ['voice_color'] },
  { title: te.sections.generation, controls: ['language', 'speed', 'pitch', 'target_duration'], advancedOnly: ['pitch', 'target_duration'] },
]

function SectionTitle({ children }: { children: React.ReactNode }) {
  return <h2 className="text-[11px] font-semibold tracking-[0.14em] text-muted-foreground uppercase">{children}</h2>
}

function isVisible(spec: ParameterSpec, values: Record<string, ParamValue>) {
  return !spec.visible_if || Object.entries(spec.visible_if).every(([k, v]) => values[k] === v)
}

export function EngineSettingsPanel() {
  const { engines, config, engineId, variantId, valuesByKey, activePreset, loading, error, loadEngines, selectEngine, selectVariant, setValue, applyPreset, resetDefaults } =
    useEngineStore()
  const mode = useUiStore((s) => s.mode)
  const connection = useSystemStore((s) => s.connection)
  const [advancedOpen, setAdvancedOpen] = useState(false)

  // (Re)load when the backend becomes reachable, e.g. the page opened while the server was starting.
  useEffect(() => {
    if (!engines.length && connection !== 'disconnected') void loadEngines()
  }, [engines.length, connection, loadEngines])

  const values = valuesByKey[keyOf(engineId, variantId)] ?? {}
  const summary = engines.find((e) => e.id === engineId)
  const specs = config?.parameters ?? []
  const byId = Object.fromEntries(specs.map((s) => [s.id, s]))
  const caps = config?.capabilities
  const boundToControls = new Set(Object.values(caps?.controls ?? {}).map((c) => c.parameter).filter(Boolean) as string[])
  const renderedInSections = new Set(
    SECTIONS.flatMap((s) => s.controls)
      .map((c) => caps?.controls[c]?.parameter)
      .filter(Boolean) as string[],
  )
  const voiceParams = specs.filter((s) => s.group === 'basic' && !boundToControls.has(s.id) && isVisible(s, values))
  const advancedParams = specs.filter((s) => !renderedInSections.has(s.id) && s.group === 'advanced' && isVisible(s, values))

  return (
    <div className="space-y-6">
      <section className="space-y-2">
        <SectionTitle>{te.engine}</SectionTitle>
        <select
          aria-label={te.engine}
          value={engineId ?? ''}
          onChange={(e) => void selectEngine(e.target.value)}
          className="h-10 w-full rounded-md border bg-card px-3 text-sm"
          disabled={!engines.length}
        >
          {engines.map((e) => (
            <option key={e.id} value={e.id}>
              {e.name}
              {e.installed ? '' : ` · ${te.notInstalled}`}
            </option>
          ))}
        </select>
        {summary && summary.variants.length > 1 && (
          <select
            aria-label={te.variant}
            value={variantId ?? ''}
            onChange={(e) => void selectVariant(e.target.value)}
            className="h-8 w-full rounded-md border bg-card px-2 text-xs"
          >
            {summary.variants.map((v) => (
              <option key={v.id} value={v.id}>
                {v.label}
              </option>
            ))}
          </select>
        )}
        {summary && (
          <div className="space-y-1.5">
            <p className="text-xs leading-relaxed text-muted-foreground">{config?.variant.description ?? summary.description}</p>
            <div className="flex flex-wrap gap-1.5">
              <Badge variant={summary.installed ? 'success' : 'outline'} title={summary.installed ? undefined : te.missing(summary.missing_packages.join(', '))}>
                {summary.installed ? te.installed : te.notInstalled}
              </Badge>
              {summary.license.commercial_use === 'no_permitido' && (
                <Badge variant="destructive" title={te.licenseHint(summary.license.weights, summary.license.code)}>
                  {te.nonCommercial}
                </Badge>
              )}
              {!summary.implemented && summary.implementation_phase != null && (
                <Badge variant="outline">{te.generationPhase(summary.implementation_phase)}</Badge>
              )}
            </div>
          </div>
        )}
        <WeightsStatus />
        <RecommendedSettingsBar />
        {error && (
          <p role="alert" className="flex items-center gap-1.5 text-xs text-destructive">
            <AlertTriangle className="size-3.5" />
            {error}
          </p>
        )}
      </section>

      {loading && !config ? (
        <div className="space-y-3">
          <Skeleton className="h-6" />
          <Skeleton className="h-6" />
          <Skeleton className="h-6" />
        </div>
      ) : config && caps ? (
        <div className={cn('space-y-6', loading && 'pointer-events-none opacity-60')}>
          {voiceParams.length > 0 && (
            <section className="space-y-4">
              <SectionTitle>{te.sections.voice}</SectionTitle>
              {voiceParams.map((spec) => (
                <ParameterControl key={spec.id} spec={spec} value={values[spec.id] ?? null} onChange={(v) => setValue(spec.id, v)} />
              ))}
            </section>
          )}

          {SECTIONS.map((section) => (
            <section key={section.title} className="space-y-3">
              <SectionTitle>{section.title}</SectionTitle>
              {section.controls
                .filter((c) => mode === 'advanced' || !section.advancedOnly?.includes(c))
                .map((control) => {
                  const cap = caps.controls[control]
                  const spec = cap?.parameter ? byId[cap.parameter] : undefined
                  if (control === 'emotion') return <EmotionControl key={control} capability={cap} />
                  return (
                    <GenericControlRow
                      key={control}
                      control={control}
                      capability={cap}
                      spec={spec}
                      value={spec ? (values[spec.id] ?? null) : null}
                      onChange={(v) => spec && setValue(spec.id, v)}
                    />
                  )
                })}
            </section>
          ))}

          {mode === 'advanced' && (
            <section className="rounded-lg border bg-card">
              <button
                type="button"
                onClick={() => setAdvancedOpen((o) => !o)}
                aria-expanded={advancedOpen}
                className="flex w-full items-center justify-between px-3 py-2.5 text-sm"
              >
                {te.advanced}
                <ChevronDown className={cn('size-4 transition-transform', advancedOpen && 'rotate-180')} />
              </button>
              {advancedOpen && (
                <div className="space-y-4 border-t px-3 py-3">
                  {config.presets.length > 0 && (
                  <div className="space-y-1.5">
                    <div className="flex items-center gap-1.5 text-xs">
                      {te.presets}
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <button type="button" aria-label={te.presetHint} className="text-muted-foreground">
                            <Info className="size-3.5" />
                          </button>
                        </TooltipTrigger>
                        <TooltipContent>{te.presetHint}</TooltipContent>
                      </Tooltip>
                    </div>
                    <div className="grid grid-cols-3 gap-1" role="group" aria-label={te.presets}>
                      {config.presets.map((p) => (
                        <Button key={p.id} size="sm" variant={activePreset === p.id ? 'default' : 'outline'} className="h-7 px-1 text-[11px]" onClick={() => applyPreset(p.id)}>
                          {p.label}
                        </Button>
                      ))}
                    </div>
                  </div>
                  )}
                  {advancedParams.map((spec) => (
                    <ParameterControl key={spec.id} spec={spec} value={values[spec.id] ?? null} onChange={(v) => setValue(spec.id, v)} />
                  ))}
                  <Button size="sm" variant="ghost" onClick={resetDefaults}>
                    <RotateCcw />
                    {te.reset}
                  </Button>
                  {caps.notes.length > 0 && (
                    <div className="space-y-1 border-t pt-3">
                      <p className="text-[11px] font-medium">{te.notes}</p>
                      <ul className="list-disc space-y-0.5 pl-4 text-[11px] text-muted-foreground">
                        {caps.notes.map((n) => (
                          <li key={n}>{n}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </section>
          )}
        </div>
      ) : null}
    </div>
  )
}
