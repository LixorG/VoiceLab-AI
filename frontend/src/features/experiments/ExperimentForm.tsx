import { FlaskConical, Loader2 } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'
import { type ArmDraft, ArmsEditor, armParams, newArm } from '@/features/experiments/ArmsEditor'
import { t } from '@/i18n/es'
import { modelsApi } from '@/services/models'
import { referencesApi } from '@/services/references'
import { useEngineStore } from '@/stores/engine'
import { useExperimentsStore } from '@/stores/experiments'
import { useGenerationStore } from '@/stores/generation'
import { useProfilesStore } from '@/stores/profiles'
import type { ReferenceRead } from '@/types/api'
import type { ParamValue } from '@/types/engines'
import { isPostprocessActive } from '@/types/postprocess'

const tx = t.experiments

export function ExperimentForm() {
  const { engines, valuesByKey, loadEngines } = useEngineStore()
  const { items: profiles, load: loadProfiles } = useProfilesStore()
  const generation = useGenerationStore()
  const { create, busy, error } = useExperimentsStore()

  const [name, setName] = useState('')
  const [text, setText] = useState(generation.text)
  const [profileId, setProfileId] = useState<string | null>(generation.profileId)
  const [referenceId, setReferenceId] = useState<string | null>(generation.referenceId)
  const [references, setReferences] = useState<ReferenceRead[]>([])
  const [seed, setSeed] = useState('')
  const [applyPostprocess, setApplyPostprocess] = useState(false)
  const [arms, setArms] = useState<ArmDraft[]>([])

  useEffect(() => {
    void loadEngines()
    void loadProfiles()
  }, [loadEngines, loadProfiles])

  useEffect(() => {
    if (arms.length || !engines.length) return
    const initial = engines.filter((e) => e.implemented && e.installed).slice(0, 3)
    setArms((initial.length ? initial : engines.filter((e) => e.implemented).slice(0, 1)).map((e) => newArm(e.id, e.default_variant)))
  }, [engines, arms.length])

  useEffect(() => {
    referencesApi.list(profileId ?? undefined).then(
      (refs) => setReferences(refs.filter((r) => r.status === 'ANALYZED')),
      () => setReferences([]),
    )
  }, [profileId])

  const profile = profiles.find((p) => p.id === profileId) ?? null
  const ppActive = isPostprocessActive(generation.postprocess)
  const valid = text.trim().length > 0 && arms.length > 0 && (profileId != null || referenceId != null)
  const seedValue = seed.trim() === '' ? null : parseInt(seed, 10)
  const seedInvalid = seed.trim() !== '' && (Number.isNaN(seedValue) || (seedValue ?? 0) < 0)

  const referenceOptions = useMemo(() => references, [references])

  const submit = async () => {
    const built = await Promise.all(
      arms.map(async (arm) => {
        const params: Record<string, ParamValue> = { ...armParams(arm, profile, valuesByKey) }
        if (seedValue != null) {
          try {
            const config = await modelsApi.config(arm.engine, arm.variant)
            if (config.parameters.some((p) => p.id === 'seed')) params.seed = seedValue
          } catch {
            // without the schema the seed is simply not sent
          }
        }
        return { engine: arm.engine, variant: arm.variant, params }
      }),
    )
    await create({
      name: name.trim() || null,
      text,
      profile_id: profileId,
      reference_id: referenceId,
      emotion: null,
      intensity: 50,
      markup: true,
      postprocess: applyPostprocess && ppActive ? generation.postprocess : null,
      arms: built,
    })
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>{tx.newTitle}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-xs text-muted-foreground">{tx.newHint}</p>
        <div className="grid gap-3 sm:grid-cols-2">
          <label className="space-y-1 text-xs">
            <span className="text-muted-foreground">{tx.name}</span>
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder={tx.namePlaceholder} maxLength={120} className="h-8 w-full rounded-md border bg-background px-2" />
          </label>
          <label className="space-y-1 text-xs">
            <span className="text-muted-foreground">{tx.seed}</span>
            <input value={seed} onChange={(e) => setSeed(e.target.value.replace(/[^\d]/g, ''))} placeholder={tx.seedPlaceholder} inputMode="numeric" aria-invalid={seedInvalid} className="h-8 w-full rounded-md border bg-background px-2 font-mono" />
          </label>
          <label className="space-y-1 text-xs">
            <span className="text-muted-foreground">{tx.voice}</span>
            <select
              value={profileId ?? ''}
              onChange={(e) => {
                setProfileId(e.target.value || null)
                setReferenceId(null)
              }}
              className="h-8 w-full rounded-md border bg-background px-2"
            >
              <option value="">{tx.noProfile}</option>
              {profiles.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
          </label>
          <label className="space-y-1 text-xs">
            <span className="text-muted-foreground">{tx.reference}</span>
            <select value={referenceId ?? ''} onChange={(e) => setReferenceId(e.target.value || null)} className="h-8 w-full rounded-md border bg-background px-2">
              <option value="">{profileId ? tx.profileReference : tx.chooseReference}</option>
              {referenceOptions.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.original_name}
                  {r.is_primary ? ` · ${tx.primary}` : ''}
                </option>
              ))}
            </select>
          </label>
        </div>

        <label className="block space-y-1 text-xs">
          <span className="text-muted-foreground">{tx.text}</span>
          <Textarea value={text} onChange={(e) => setText(e.target.value)} maxLength={5000} placeholder={t.generate.textPlaceholder} className="min-h-24" />
        </label>

        <div className="space-y-2">
          <p className="text-xs font-medium">{tx.arms}</p>
          <ArmsEditor arms={arms} onChange={setArms} profile={profile} />
        </div>

        <label className={`flex items-center gap-2 text-xs ${ppActive ? '' : 'text-muted-foreground'}`}>
          <input type="checkbox" className="accent-[var(--primary)]" checked={applyPostprocess && ppActive} disabled={!ppActive} onChange={(e) => setApplyPostprocess(e.target.checked)} />
          {ppActive ? tx.applyPostprocess : tx.noPostprocess}
        </label>

        {error && (
          <p role="alert" className="text-xs text-destructive">
            {error}
          </p>
        )}
        <div className="flex items-center justify-end gap-3">
          {!valid && <span className="text-xs text-muted-foreground">{tx.needs}</span>}
          <Button disabled={!valid || seedInvalid || busy === 'create'} onClick={() => void submit()}>
            {busy === 'create' ? <Loader2 className="animate-spin" /> : <FlaskConical />}
            {tx.run(arms.length)}
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
