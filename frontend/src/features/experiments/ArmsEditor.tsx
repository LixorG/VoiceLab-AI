import { Plus, Trash2 } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { t } from '@/i18n/es'
import { keyOf, useEngineStore } from '@/stores/engine'
import type { ParamValue } from '@/types/engines'
import type { ProfileRead } from '@/types/profiles'

const tx = t.experiments
export const MAX_ARMS = 6

export type ParamSource = 'default' | 'profile' | 'current'

export interface ArmDraft {
  key: number
  engine: string
  variant: string
  source: ParamSource
}

let nextKey = 1
export const newArm = (engine: string, variant: string): ArmDraft => ({ key: nextKey++, engine, variant, source: 'default' })

/** Parameters an arm will send, from the chosen source (backend fills and validates the rest). */
export function armParams(arm: ArmDraft, profile: ProfileRead | null, valuesByKey: Record<string, Record<string, ParamValue>>): Record<string, ParamValue> {
  if (arm.source === 'profile') {
    const rec = profile?.recommended_settings[arm.engine]
    return rec && rec.variant === arm.variant ? rec.params : {}
  }
  if (arm.source === 'current') return valuesByKey[keyOf(arm.engine, arm.variant)] ?? {}
  return {}
}

export function ArmsEditor({ arms, onChange, profile }: { arms: ArmDraft[]; onChange: (arms: ArmDraft[]) => void; profile: ProfileRead | null }) {
  const { engines, valuesByKey } = useEngineStore()
  const usable = engines.filter((e) => e.implemented)

  const patch = (key: number, update: Partial<ArmDraft>) => onChange(arms.map((a) => (a.key === key ? { ...a, ...update } : a)))

  return (
    <div className="space-y-2">
      {arms.map((arm, i) => {
        const engine = engines.find((e) => e.id === arm.engine)
        const rec = profile?.recommended_settings[arm.engine]
        const profileOk = !!rec && rec.variant === arm.variant
        const currentOk = !!valuesByKey[keyOf(arm.engine, arm.variant)]
        return (
          <div key={arm.key} className="flex flex-wrap items-center gap-2 rounded-md border p-2" data-testid="arm-row">
            <span className="w-5 font-mono text-sm font-semibold text-primary">{'ABCDEF'[i]}</span>
            <select
              aria-label={tx.engine}
              value={arm.engine}
              onChange={(e) => {
                const next = engines.find((x) => x.id === e.target.value)
                patch(arm.key, { engine: e.target.value, variant: next?.default_variant ?? '', source: 'default' })
              }}
              className="h-8 min-w-0 flex-1 rounded-md border bg-background px-2 text-xs"
            >
              {usable.map((e) => (
                <option key={e.id} value={e.id}>
                  {e.name}
                  {e.installed ? '' : ` (${tx.notInstalled})`}
                </option>
              ))}
            </select>
            {engine && engine.variants.length > 1 && (
              <select aria-label={tx.variant} value={arm.variant} onChange={(e) => patch(arm.key, { variant: e.target.value, source: 'default' })} className="h-8 min-w-0 flex-1 rounded-md border bg-background px-2 text-xs">
                {engine.variants.map((v) => (
                  <option key={v.id} value={v.id}>
                    {v.label}
                  </option>
                ))}
              </select>
            )}
            <select aria-label={tx.params} value={arm.source} onChange={(e) => patch(arm.key, { source: e.target.value as ParamSource })} className="h-8 rounded-md border bg-background px-2 text-xs">
              <option value="default">{tx.sources.default}</option>
              <option value="profile" disabled={!profileOk}>
                {tx.sources.profile}
              </option>
              <option value="current" disabled={!currentOk}>
                {tx.sources.current}
              </option>
            </select>
            <Button size="icon" variant="ghost" className="size-8" disabled={arms.length <= 1} onClick={() => onChange(arms.filter((a) => a.key !== arm.key))} aria-label={tx.removeArm}>
              <Trash2 />
            </Button>
          </div>
        )
      })}
      <Button
        size="sm"
        variant="outline"
        disabled={arms.length >= MAX_ARMS || usable.length === 0}
        onClick={() => {
          const used = new Set(arms.map((a) => a.engine))
          const next = usable.find((e) => !used.has(e.id)) ?? usable[0]
          onChange([...arms, newArm(next.id, next.default_variant)])
        }}
      >
        <Plus />
        {tx.addArm}
      </Button>
    </div>
  )
}
