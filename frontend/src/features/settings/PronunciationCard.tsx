import { Pencil, Plus, Trash2 } from 'lucide-react'
import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { t } from '@/i18n/es'
import { ApiError } from '@/services/api'
import { pronunciationApi } from '@/services/pronunciation'
import { useProfilesStore } from '@/stores/profiles'
import type { NormalizePreview, PronunciationEntry } from '@/types/pronunciation'

const tp = t.pronunciation
const EMPTY = { term: '', replacement: '', case_sensitive: false }

/** User pronunciation dictionary (global or per voice) plus a box to try the normalisation. */
export function PronunciationCard() {
  const profiles = useProfilesStore((s) => s.items)
  const loadProfiles = useProfilesStore((s) => s.load)
  const [profileId, setProfileId] = useState<string | null>(null)
  const [entries, setEntries] = useState<PronunciationEntry[]>([])
  const [draft, setDraft] = useState(EMPTY)
  const [editing, setEditing] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [sample, setSample] = useState('')
  const [preview, setPreview] = useState<NormalizePreview | null>(null)

  const reload = async (id: string | null) => {
    try {
      const list = await pronunciationApi.list(id)
      setEntries(Array.isArray(list) ? list : [])
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err))
    }
  }

  useEffect(() => {
    void loadProfiles()
  }, [loadProfiles])

  useEffect(() => {
    void reload(profileId)
  }, [profileId])

  const submit = async () => {
    setError(null)
    const body = { ...draft, term: draft.term.trim(), replacement: draft.replacement.trim(), profile_id: profileId }
    if (!body.term || !body.replacement) return
    try {
      if (editing) await pronunciationApi.update(editing, body)
      else await pronunciationApi.create(body)
      setDraft(EMPTY)
      setEditing(null)
      await reload(profileId)
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err))
    }
  }

  const remove = async (entry: PronunciationEntry) => {
    if (!confirm(tp.confirmRemove)) return
    await pronunciationApi.remove(entry.id)
    await reload(profileId)
  }

  const test = async () => {
    if (!sample.trim()) return
    setPreview(await pronunciationApi.preview({ text: sample, language: 'es', profile_id: profileId }))
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>{tp.title}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-sm text-muted-foreground">{tp.intro}</p>
        <p className="rounded-md border border-dashed px-3 py-2 text-xs text-muted-foreground">{tp.ipa}</p>

        <label className="flex flex-wrap items-center gap-2 text-sm">
          <span className="text-muted-foreground">{tp.scope}</span>
          <select
            aria-label={tp.scope}
            className="h-8 rounded-md border bg-background px-2 text-sm"
            value={profileId ?? ''}
            onChange={(e) => setProfileId(e.target.value || null)}
          >
            <option value="">{tp.global}</option>
            {profiles.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>
        </label>

        {error && (
          <p role="alert" className="text-xs text-destructive">
            {error}
          </p>
        )}

        {entries.length === 0 ? (
          <p className="text-sm text-muted-foreground">{tp.empty}</p>
        ) : (
          <ul className="divide-y rounded-md border" data-testid="pronunciation-list">
            {entries.map((entry) => (
              <li key={entry.id} className="flex items-center gap-2 px-3 py-1.5 text-sm">
                <span className="min-w-0 flex-1 truncate font-mono text-xs">
                  {entry.term} → {entry.replacement}
                </span>
                {profileId && entry.profile_id === null && <span className="text-[11px] text-muted-foreground">{tp.globalBadge}</span>}
                <Button
                  size="icon"
                  variant="ghost"
                  aria-label={`${tp.edit}: ${entry.term}`}
                  disabled={profileId !== null && entry.profile_id === null}
                  onClick={() => {
                    setEditing(entry.id)
                    setDraft({ term: entry.term, replacement: entry.replacement, case_sensitive: entry.case_sensitive })
                  }}
                >
                  <Pencil className="size-4" />
                </Button>
                <Button
                  size="icon"
                  variant="ghost"
                  aria-label={`${tp.remove}: ${entry.term}`}
                  disabled={profileId !== null && entry.profile_id === null}
                  onClick={() => void remove(entry)}
                >
                  <Trash2 className="size-4" />
                </Button>
              </li>
            ))}
          </ul>
        )}

        <div className="flex flex-wrap items-end gap-2">
          <label className="flex flex-col gap-1 text-xs text-muted-foreground">
            {tp.term}
            <input
              aria-label={tp.term}
              className="h-8 w-40 rounded-md border bg-background px-2 text-sm"
              placeholder={tp.termPlaceholder}
              value={draft.term}
              onChange={(e) => setDraft({ ...draft, term: e.target.value })}
            />
          </label>
          <label className="flex flex-col gap-1 text-xs text-muted-foreground">
            {tp.replacement}
            <input
              aria-label={tp.replacement}
              className="h-8 w-52 rounded-md border bg-background px-2 text-sm"
              placeholder={tp.replacementPlaceholder}
              value={draft.replacement}
              onChange={(e) => setDraft({ ...draft, replacement: e.target.value })}
            />
          </label>
          <label className="flex items-center gap-1.5 pb-1.5 text-xs text-muted-foreground">
            <input
              type="checkbox"
              className="accent-[var(--primary)]"
              checked={draft.case_sensitive}
              onChange={(e) => setDraft({ ...draft, case_sensitive: e.target.checked })}
            />
            {tp.caseSensitive}
          </label>
          <Button size="sm" onClick={() => void submit()}>
            <Plus />
            {editing ? tp.save : tp.add}
          </Button>
          {editing && (
            <Button
              size="sm"
              variant="ghost"
              onClick={() => {
                setEditing(null)
                setDraft(EMPTY)
              }}
            >
              {tp.cancel}
            </Button>
          )}
        </div>

        <div className="space-y-2 border-t pt-3">
          <p className="text-xs font-medium">{tp.testTitle}</p>
          <div className="flex flex-wrap items-end gap-2">
            <label className="flex min-w-0 flex-1 flex-col gap-1 text-xs text-muted-foreground">
              {tp.testLabel}
              <input
                aria-label={tp.testLabel}
                className="h-8 rounded-md border bg-background px-2 text-sm"
                placeholder={tp.testPlaceholder}
                value={sample}
                onChange={(e) => setSample(e.target.value)}
              />
            </label>
            <Button size="sm" variant="outline" onClick={() => void test()}>
              {tp.testButton}
            </Button>
          </div>
          {preview && (
            <div className="space-y-1 rounded-md border border-dashed p-2">
              <p className="text-xs text-muted-foreground">{tp.result}</p>
              <p className="text-sm">{preview.text}</p>
              {preview.changes.length === 0 && <p className="text-xs text-muted-foreground">{tp.noChanges}</p>}
              {!preview.numbers_supported && <p className="text-xs text-warning">{tp.unsupported}</p>}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
