import { useEffect } from 'react'

import { SourceBadge } from '@/features/models/GenericControlRow'
import { t } from '@/i18n/es'
import { useGenerationStore } from '@/stores/generation'
import { useProfilesStore } from '@/stores/profiles'
import type { ControlCapability } from '@/types/engines'

const tx = t.expression

/** Global emotion + intensity. How (or whether) it is applied depends on the engine's declared capability. */
export function EmotionControl({ capability }: { capability: ControlCapability | undefined }) {
  const { emotion, intensity, setEmotion, setIntensity } = useGenerationStore()
  const { emotions, loadEmotions } = useProfilesStore()

  useEffect(() => {
    void loadEmotions()
  }, [loadEmotions])

  if (!capability) return null
  const usable = capability.source === 'instruction' || capability.source === 'segmentation'
  const intensityUsable = capability.source === 'instruction'

  return (
    <div className="space-y-2" data-control="emotion">
      <div className="flex items-center gap-2 text-xs">
        <label htmlFor="global-emotion" className={usable ? '' : 'text-muted-foreground'}>
          {tx.emotion}
        </label>
        <span className="ml-auto">
          <SourceBadge capability={capability} />
        </span>
      </div>
      {usable && (
        <>
          <select
            id="global-emotion"
            value={emotion ?? ''}
            onChange={(e) => setEmotion(e.target.value || null)}
            className="h-8 w-full rounded-md border bg-background px-2 text-xs"
          >
            <option value="">{tx.none}</option>
            {emotions.map((em) => (
              <option key={em.id} value={em.id}>
                {em.label}
              </option>
            ))}
          </select>
          <p className="text-[11px] text-muted-foreground">
            {capability.source === 'segmentation' ? tx.viaReference : tx.viaInstruction}
          </p>
          <div className="space-y-1">
            <div className="flex items-center text-xs">
              <label htmlFor="global-intensity" className={intensityUsable ? '' : 'text-muted-foreground'}>
                {tx.intensity}
              </label>
              <span className="ml-auto font-mono tabular-nums text-muted-foreground">{intensity}%</span>
            </div>
            <input
              id="global-intensity"
              type="range"
              min={0}
              max={100}
              step={5}
              value={intensity}
              disabled={!intensityUsable || !emotion}
              onChange={(e) => setIntensity(parseInt(e.target.value, 10))}
              title={intensityUsable ? tx.intensityHint : tx.intensityUnavailable}
              className="h-1.5 w-full cursor-pointer accent-[var(--primary)] disabled:cursor-not-allowed disabled:opacity-50"
            />
            {!intensityUsable && <p className="text-[11px] text-muted-foreground">{tx.intensityUnavailable}</p>}
          </div>
        </>
      )}
    </div>
  )
}
