import { useEffect } from 'react'

import { t } from '@/i18n/es'
import { useGenerationStore } from '@/stores/generation'
import { useProfilesStore } from '@/stores/profiles'
import { useReferencesStore } from '@/stores/references'

const tv = t.voices

/** Chooses the voice profile used in Generar: scopes the reference library and generation requests. */
export function VoiceSelector() {
  const { items, load } = useProfilesStore()
  const { profileId, setProfile } = useGenerationStore()
  const setReferencesProfile = useReferencesStore((s) => s.setProfile)

  useEffect(() => {
    void load()
  }, [load])

  // Keep the reference library scoped to the selected voice (also when returning from the Voces page).
  useEffect(() => {
    void setReferencesProfile(profileId)
  }, [profileId, setReferencesProfile])

  // A deleted profile must not stay selected.
  useEffect(() => {
    if (profileId && items.length && !items.some((p) => p.id === profileId)) setProfile(null)
  }, [items, profileId, setProfile])

  return (
    <div className="flex flex-wrap items-center gap-2">
      <label htmlFor="voice-selector" className="text-[11px] font-semibold tracking-[0.14em] text-muted-foreground uppercase">
        {tv.selector}
      </label>
      <select
        id="voice-selector"
        value={profileId ?? ''}
        onChange={(e) => setProfile(e.target.value || null)}
        className="h-8 min-w-48 flex-1 rounded-md border bg-background px-2 text-sm sm:flex-none"
      >
        <option value="">{tv.allReferences}</option>
        {items.map((p) => (
          <option key={p.id} value={p.id}>
            {p.name} · {tv.references(p.stats.reference_count)}
          </option>
        ))}
      </select>
    </div>
  )
}
