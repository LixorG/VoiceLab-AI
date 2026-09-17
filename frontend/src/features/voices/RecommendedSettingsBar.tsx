import { BookmarkCheck, BookmarkPlus } from 'lucide-react'
import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { t } from '@/i18n/es'
import { keyOf, useEngineStore } from '@/stores/engine'
import { useGenerationStore } from '@/stores/generation'
import { useProfilesStore } from '@/stores/profiles'

const tv = t.voices

/** Apply / save the selected profile's recommended settings for the current engine. */
export function RecommendedSettingsBar() {
  const profileId = useGenerationStore((s) => s.profileId)
  const profile = useProfilesStore((s) => s.items.find((p) => p.id === profileId))
  const saveRecommended = useProfilesStore((s) => s.saveRecommended)
  const { engineId, variantId, valuesByKey, selectVariant } = useEngineStore()
  const [notice, setNotice] = useState<string | null>(null)

  if (!profile || !engineId) return null
  const setting = profile.recommended_settings[engineId]

  const apply = async () => {
    if (!setting) return
    if (setting.variant !== variantId) await selectVariant(setting.variant)
    for (const [id, value] of Object.entries(setting.params)) useEngineStore.getState().setValue(id, value)
    setNotice(tv.applied)
  }

  const save = async () => {
    const ok = await saveRecommended(profile.id, engineId, variantId, valuesByKey[keyOf(engineId, variantId)] ?? {})
    if (ok) setNotice(tv.saved)
  }

  return (
    <div className="space-y-1.5 rounded-md border border-dashed p-2.5" data-testid="recommended-settings">
      <p className="text-[11px] text-muted-foreground">{tv.forVoice(profile.name)}</p>
      <div className="flex flex-wrap gap-1.5">
        {setting && (
          <Button size="sm" variant="secondary" className="h-7" onClick={() => void apply()}>
            <BookmarkCheck />
            {tv.apply}
          </Button>
        )}
        <Button size="sm" variant="ghost" className="h-7" onClick={() => void save()} title={tv.recommendedHint}>
          <BookmarkPlus />
          {tv.saveCurrent}
        </Button>
      </div>
      {notice && <p className="text-[11px] text-success" role="status">{notice}</p>}
    </div>
  )
}
