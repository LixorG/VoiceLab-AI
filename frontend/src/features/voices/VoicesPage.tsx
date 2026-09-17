import { AlertTriangle, ArrowLeft, Download, Loader2, Mic2, Plus, Trash2, Upload, X } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { Textarea } from '@/components/ui/textarea'
import { ReferenceLibrary } from '@/features/voices/ReferenceLibrary'
import { t } from '@/i18n/es'
import { formatDuration } from '@/lib/utils'
import { profilesApi } from '@/services/profiles'
import { useEngineStore } from '@/stores/engine'
import { useProfilesStore } from '@/stores/profiles'
import { useReferencesStore } from '@/stores/references'
import type { ProfileInput, ProfileRead } from '@/types/profiles'

const tv = t.voices

function Field({ label, children, htmlFor }: { label: string; children: React.ReactNode; htmlFor: string }) {
  return (
    <div className="space-y-1">
      <label htmlFor={htmlFor} className="text-xs text-muted-foreground">
        {label}
      </label>
      {children}
    </div>
  )
}

function ProfileForm({ initial, onSubmit, onCancel, submitLabel }: {
  initial?: ProfileRead
  onSubmit: (input: ProfileInput & { name: string }) => Promise<void>
  onCancel?: () => void
  submitLabel: string
}) {
  const engines = useEngineStore((s) => s.engines)
  const [name, setName] = useState(initial?.name ?? '')
  const [description, setDescription] = useState(initial?.description ?? '')
  const [language, setLanguage] = useState(initial?.language ?? '')
  const [engine, setEngine] = useState(initial?.default_engine ?? '')
  const [busy, setBusy] = useState(false)

  return (
    <form
      className="grid gap-3 sm:grid-cols-2"
      onSubmit={async (e) => {
        e.preventDefault()
        if (!name.trim()) return
        setBusy(true)
        await onSubmit({ name: name.trim(), description: description.trim() || null, language: language.trim() || null, default_engine: engine || null })
        setBusy(false)
      }}
    >
      <Field label={tv.name} htmlFor="profile-name">
        <input id="profile-name" value={name} onChange={(e) => setName(e.target.value)} placeholder={tv.namePlaceholder} maxLength={80} required className="h-9 w-full rounded-md border bg-background px-3 text-sm" />
      </Field>
      <Field label={tv.language} htmlFor="profile-language">
        <input id="profile-language" value={language} onChange={(e) => setLanguage(e.target.value)} placeholder={tv.languagePlaceholder} maxLength={40} className="h-9 w-full rounded-md border bg-background px-3 text-sm" />
      </Field>
      <Field label={tv.defaultEngine} htmlFor="profile-engine">
        <select id="profile-engine" value={engine} onChange={(e) => setEngine(e.target.value)} className="h-9 w-full rounded-md border bg-background px-2 text-sm">
          <option value="">{tv.none}</option>
          {engines.map((en) => (
            <option key={en.id} value={en.id}>
              {en.name}
            </option>
          ))}
        </select>
      </Field>
      <div className="sm:col-span-2">
        <Field label={tv.description} htmlFor="profile-description">
          <Textarea id="profile-description" value={description} onChange={(e) => setDescription(e.target.value)} maxLength={1000} className="min-h-16" />
        </Field>
      </div>
      <div className="flex gap-2 sm:col-span-2">
        <Button type="submit" disabled={!name.trim() || busy}>
          {busy && <Loader2 className="animate-spin" />}
          {submitLabel}
        </Button>
        {onCancel && (
          <Button type="button" variant="ghost" onClick={onCancel}>
            {tv.cancel}
          </Button>
        )}
      </div>
    </form>
  )
}

function ProfileCard({ profile, onOpen }: { profile: ProfileRead; onOpen: () => void }) {
  const s = profile.stats
  return (
    <button type="button" onClick={onOpen} className="group rounded-xl border bg-card p-4 text-left transition-colors hover:border-primary/60">
      <div className="flex items-start gap-3">
        <div className="grid size-10 shrink-0 place-items-center rounded-full bg-muted text-primary">
          <Mic2 className="size-5" />
        </div>
        <div className="min-w-0 flex-1">
          <h2 className="truncate font-medium">{profile.name}</h2>
          <p className="truncate text-xs text-muted-foreground">{profile.description || profile.language || '—'}</p>
        </div>
      </div>
      <div className="mt-3 flex flex-wrap gap-1.5 text-[11px]">
        <Badge>{tv.references(s.reference_count)}</Badge>
        {s.total_speech_s > 0 && <Badge variant="outline">{tv.speech(formatDuration(s.total_speech_s))}</Badge>}
        {s.quality_label && <Badge variant="success">{s.quality_label}</Badge>}
        {Object.keys(profile.recommended_settings).length > 0 && <Badge variant="accent">{tv.recommended}</Badge>}
      </div>
    </button>
  )
}

function ProfileDetail({ profile, onBack }: { profile: ProfileRead; onBack: () => void }) {
  const { update, remove, removeRecommended, emotions, loadEmotions, refresh } = useProfilesStore()
  const engines = useEngineStore((s) => s.engines)
  const { setProfile, items: references } = useReferencesStore()
  const [deleteRefs, setDeleteRefs] = useState(false)

  useEffect(() => {
    void setProfile(profile.id)
    void loadEmotions()
  }, [profile.id, setProfile, loadEmotions])

  // Stats depend on the references: refresh the profile when they change.
  const signature = references.map((r) => `${r.id}:${r.status}:${r.emotion_tag}:${r.is_primary}`).join('|')
  useEffect(() => {
    void refresh(profile.id)
  }, [signature, profile.id, refresh])

  const s = profile.stats
  const emotionLabel = (id: string) => emotions.find((e) => e.id === id)?.label ?? id
  const engineName = (id: string) => engines.find((e) => e.id === id)?.name ?? id

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-2">
        <Button variant="ghost" size="sm" onClick={onBack}>
          <ArrowLeft />
          {tv.back}
        </Button>
        <h1 className="mr-auto text-2xl font-semibold tracking-tight">{profile.name}</h1>
        <Button asChild variant="outline" size="sm" title={tv.exportHint}>
          <a href={profilesApi.exportUrl(profile.id)} download>
            <Download />
            {tv.export}
          </a>
        </Button>
      </div>

      <div className="grid gap-5 lg:grid-cols-[1.3fr_1fr] [&>*]:min-w-0">
        <Card>
          <CardHeader>
            <CardTitle>{tv.details}</CardTitle>
          </CardHeader>
          <CardContent>
            <ProfileForm key={profile.updated_at} initial={profile} submitLabel={tv.save} onSubmit={(input) => update(profile.id, input)} />
          </CardContent>
        </Card>

        <div className="space-y-5">
          <Card>
            <CardContent className="space-y-2 pt-4 text-sm">
              <div className="flex flex-wrap gap-1.5">
                <Badge>{tv.references(s.reference_count)}</Badge>
                <Badge variant="outline">{tv.speech(formatDuration(s.total_speech_s))}</Badge>
              </div>
              {s.average_quality != null && s.quality_label && (
                <p className="text-xs" title={tv.qualityHint}>
                  {tv.quality(s.quality_label, s.average_quality)}
                </p>
              )}
              <p className="text-xs text-muted-foreground">
                {tv.emotions}: {s.emotions.length ? s.emotions.map(emotionLabel).join(', ') : tv.noEmotions}
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>{tv.recommended}</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <p className="text-xs text-muted-foreground">{tv.recommendedHint}</p>
              {Object.entries(profile.recommended_settings).length === 0 ? (
                <p className="text-xs text-muted-foreground">{tv.noRecommended}</p>
              ) : (
                <ul className="divide-y">
                  {Object.entries(profile.recommended_settings).map(([engineId, setting]) => (
                    <li key={engineId} className="flex items-center gap-2 py-2 text-xs">
                      <span className="font-medium">{engineName(engineId)}</span>
                      <span className="font-mono text-muted-foreground">{setting.variant}</span>
                      <Button size="sm" variant="ghost" className="ml-auto h-7" onClick={() => void removeRecommended(profile.id, engineId)}>
                        {tv.removeSetting}
                      </Button>
                    </li>
                  ))}
                </ul>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardContent className="space-y-2 pt-4">
              <label className="flex items-center gap-2 text-xs">
                <input type="checkbox" checked={deleteRefs} onChange={(e) => setDeleteRefs(e.target.checked)} className="accent-[var(--destructive)]" />
                {tv.alsoReferences}
              </label>
              <Button
                variant="outline"
                size="sm"
                className="hover:text-destructive"
                onClick={async () => {
                  if (window.confirm(tv.confirmRemove) && (await remove(profile.id, deleteRefs))) onBack()
                }}
              >
                <Trash2 />
                {tv.remove}
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>

      <ReferenceLibrary />
    </div>
  )
}

export function VoicesPage() {
  const { items, loading, error, load, create, importFile, clearError } = useProfilesStore()
  const loadEngines = useEngineStore((s) => s.loadEngines)
  const enginesLoaded = useEngineStore((s) => s.engines.length > 0)
  const [creating, setCreating] = useState(false)
  const [openId, setOpenId] = useState<string | null>(null)
  const [importing, setImporting] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    void load()
    if (!enginesLoaded) void loadEngines()
  }, [load, loadEngines, enginesLoaded])

  const open = items.find((p) => p.id === openId)

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-5xl space-y-5 p-4 sm:p-6">
        {error && (
          <div role="alert" className="flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm">
            <AlertTriangle className="mt-0.5 size-4 shrink-0 text-destructive" />
            <span className="flex-1">{error}</span>
            <button type="button" onClick={clearError} aria-label={t.references.dismiss}>
              <X className="size-4" />
            </button>
          </div>
        )}

        {open ? (
          <ProfileDetail profile={open} onBack={() => setOpenId(null)} />
        ) : (
          <>
            <header className="flex flex-wrap items-end gap-3">
              <div className="mr-auto">
                <h1 className="text-2xl font-semibold tracking-tight">{tv.title}</h1>
                <p className="text-sm text-muted-foreground">{tv.subtitle}</p>
              </div>
              <Button variant="outline" onClick={() => fileRef.current?.click()} disabled={importing} title={tv.importHint}>
                {importing ? <Loader2 className="animate-spin" /> : <Upload />}
                {importing ? tv.importing : tv.import}
              </Button>
              <input
                ref={fileRef}
                type="file"
                accept=".voiceprofile,.zip"
                className="hidden"
                data-testid="profile-import"
                onChange={async (e) => {
                  const file = e.target.files?.[0]
                  e.target.value = ''
                  if (!file) return
                  setImporting(true)
                  const profile = await importFile(file)
                  setImporting(false)
                  if (profile) setOpenId(profile.id)
                }}
              />
              <Button onClick={() => setCreating(true)} disabled={creating}>
                <Plus />
                {tv.new}
              </Button>
            </header>

            {creating && (
              <Card>
                <CardHeader>
                  <CardTitle>{tv.new}</CardTitle>
                </CardHeader>
                <CardContent>
                  <ProfileForm
                    submitLabel={tv.create}
                    onCancel={() => setCreating(false)}
                    onSubmit={async (input) => {
                      const profile = await create(input)
                      if (profile) {
                        setCreating(false)
                        setOpenId(profile.id)
                      }
                    }}
                  />
                </CardContent>
              </Card>
            )}

            {loading && !items.length ? (
              <Skeleton className="h-32" />
            ) : items.length ? (
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {items.map((p) => (
                  <ProfileCard key={p.id} profile={p} onOpen={() => setOpenId(p.id)} />
                ))}
              </div>
            ) : (
              !creating && <p className="rounded-xl border border-dashed p-8 text-center text-sm text-muted-foreground">{tv.empty}</p>
            )}
          </>
        )}
      </div>
    </div>
  )
}
