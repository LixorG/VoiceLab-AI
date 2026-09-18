import { RotateCcw, Star, Trash2 } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { AudioDownloadMenu } from '@/components/ui/download-menu'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { AudioPlayer } from '@/features/audio/AudioPlayer'
import { t } from '@/i18n/es'
import { generationApi } from '@/services/generation'
import { useGenerationStore } from '@/stores/generation'
import { useLibraryStore } from '@/stores/library'
import { useUiStore } from '@/stores/ui'
import type { LibraryItem } from '@/types/library'

const tl = t.library
const field = 'h-8 rounded-md border bg-background px-2 text-sm'
const STATUSES = ['COMPLETED', 'FAILED', 'CANCELLED', 'GENERATING', 'QUEUED'] as const

function Filters() {
  const { query, facets, setQuery, resetQuery } = useLibraryStore()
  const [draft, setDraft] = useState(query.q)
  const timer = useRef<number | undefined>(undefined)

  // Debounced search: typing should not fire a request per keystroke.
  useEffect(() => {
    if (draft === query.q) return
    window.clearTimeout(timer.current)
    timer.current = window.setTimeout(() => setQuery({ q: draft }), 300)
    return () => window.clearTimeout(timer.current)
  }, [draft]) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="flex flex-wrap items-end gap-2">
      <label className="flex min-w-48 flex-1 flex-col gap-1 text-xs text-muted-foreground">
        {tl.search}
        <input type="search" aria-label={tl.search} className={field} placeholder={tl.searchPlaceholder} value={draft} onChange={(e) => setDraft(e.target.value)} />
      </label>
      <label className="flex flex-col gap-1 text-xs text-muted-foreground">
        {tl.engine}
        <select aria-label={tl.engine} className={field} value={query.engine} onChange={(e) => setQuery({ engine: e.target.value })}>
          <option value="">{tl.any}</option>
          {facets?.engines.map((e) => (
            <option key={e.id} value={e.id}>
              {e.label} ({e.count})
            </option>
          ))}
        </select>
      </label>
      <label className="flex flex-col gap-1 text-xs text-muted-foreground">
        {tl.voice}
        <select aria-label={tl.voice} className={field} value={query.profile_id} onChange={(e) => setQuery({ profile_id: e.target.value })}>
          <option value="">{tl.any}</option>
          {facets?.profiles.map((p) => (
            <option key={p.id} value={p.id}>
              {p.label} ({p.count})
            </option>
          ))}
        </select>
      </label>
      <label className="flex flex-col gap-1 text-xs text-muted-foreground">
        {tl.status}
        <select aria-label={tl.status} className={field} value={query.status} onChange={(e) => setQuery({ status: e.target.value })}>
          <option value="">{tl.any}</option>
          {STATUSES.map((s) => (
            <option key={s} value={s}>
              {t.generation.status[s]}
            </option>
          ))}
        </select>
      </label>
      <label className="flex flex-col gap-1 text-xs text-muted-foreground">
        {tl.tag}
        <select aria-label={tl.tag} className={field} value={query.tag} onChange={(e) => setQuery({ tag: e.target.value })}>
          <option value="">{tl.any}</option>
          {facets?.tags.map((tag) => (
            <option key={tag.tag} value={tag.tag}>
              {tag.tag} ({tag.count})
            </option>
          ))}
        </select>
      </label>
      <label className="flex flex-col gap-1 text-xs text-muted-foreground">
        {tl.since}
        <input type="date" aria-label={tl.since} className={field} value={query.since} onChange={(e) => setQuery({ since: e.target.value })} />
      </label>
      <label className="flex flex-col gap-1 text-xs text-muted-foreground">
        {tl.until}
        <input type="date" aria-label={tl.until} className={field} value={query.until} onChange={(e) => setQuery({ until: e.target.value })} />
      </label>
      <label className="flex items-center gap-1.5 pb-1.5 text-xs text-muted-foreground">
        <input type="checkbox" className="accent-[var(--primary)]" checked={query.favorite} onChange={(e) => setQuery({ favorite: e.target.checked })} />
        {tl.onlyFavorites}
      </label>
      <Button
        size="sm"
        variant="ghost"
        onClick={() => {
          setDraft('')
          resetQuery()
        }}
      >
        {tl.clearFilters}
      </Button>
    </div>
  )
}

function TagEditor({ item }: { item: LibraryItem }) {
  const setTags = useLibraryStore((s) => s.setTags)
  const [value, setValue] = useState(item.tags.join(', '))
  const [open, setOpen] = useState(false)

  if (!open) {
    return (
      <button type="button" className="flex flex-wrap items-center gap-1" onClick={() => setOpen(true)} aria-label={tl.editTags(item.text.slice(0, 20))}>
        {item.tags.map((tag) => (
          <Badge key={tag} variant="outline">
            {tag}
          </Badge>
        ))}
        <span className="text-xs text-muted-foreground underline">{item.tags.length ? tl.editTagsShort : tl.addTags}</span>
      </button>
    )
  }
  const save = () => {
    setOpen(false)
    const tags = value.split(',').map((tag) => tag.trim()).filter(Boolean)
    if (tags.join(',') !== item.tags.join(',')) void setTags(item.id, tags)
  }
  return (
    <input
      ref={(el) => el?.focus()}
      aria-label={tl.tagsOf(item.text.slice(0, 20))}
      className={`${field} w-56`}
      placeholder={tl.tagsPlaceholder}
      value={value}
      onChange={(e) => setValue(e.target.value)}
      onBlur={save}
      onKeyDown={(e) => {
        if (e.key === 'Enter') save()
        if (e.key === 'Escape') setOpen(false)
      }}
    />
  )
}

function Row({ item }: { item: LibraryItem }) {
  const { selected, toggleSelected, toggleFavorite } = useLibraryStore()
  const repeat = useGenerationStore((s) => s.repeat)
  const setSection = useUiStore((s) => s.setSection)
  const checked = selected.includes(item.id)

  const reuse = async () => {
    const full = await generationApi.get(item.id)
    setSection('generate')
    await repeat(full)
  }

  return (
    <li className="space-y-2 px-3 py-2.5" data-testid="library-row" data-id={item.id}>
      <div className="flex items-start gap-2">
        <input
          type="checkbox"
          className="mt-1 accent-[var(--primary)]"
          aria-label={tl.select(item.text.slice(0, 20))}
          checked={checked}
          onChange={() => toggleSelected(item.id)}
        />
        <div className="min-w-0 flex-1 space-y-1">
          <p className="text-sm">{item.text}</p>
          <div className="flex flex-wrap items-center gap-1.5 text-xs text-muted-foreground">
            <Badge variant={item.status === 'COMPLETED' ? 'success' : item.status === 'FAILED' ? 'destructive' : 'outline'}>
              {t.generation.status[item.status]}
            </Badge>
            <span>{item.engine}</span>
            {item.profile_name && <span>· {item.profile_name}</span>}
            {item.duration_s != null && <span>· {item.duration_s.toFixed(1)} s</span>}
            <span>· {new Date(item.created_at).toLocaleString('es')}</span>
          </div>
          <TagEditor item={item} />
        </div>
        <button
          type="button"
          aria-label={item.favorite ? tl.unfavorite : tl.favorite}
          aria-pressed={item.favorite}
          className="p-1 text-muted-foreground hover:text-foreground"
          onClick={() => void toggleFavorite(item.id)}
        >
          <Star className={item.favorite ? 'size-4 fill-[var(--primary)] text-[var(--primary)]' : 'size-4'} />
        </button>
        {item.audio_url && <AudioDownloadMenu url={item.audio_url} />}
        <button type="button" aria-label={tl.reuse} className="p-1 text-muted-foreground hover:text-foreground" onClick={() => void reuse()}>
          <RotateCcw className="size-4" />
        </button>
      </div>
      {item.audio_url && <AudioPlayer url={item.audio_url} duration={item.duration_s} />}
    </li>
  )
}

export function LibraryPage() {
  const { items, total, selected, loading, error, facets, load, loadMore, loadFacets, removeSelected, clearSelection, query } = useLibraryStore()

  useEffect(() => {
    void load()
    void loadFacets()
  }, [load, loadFacets])

  const filtered = Object.values(query).some((v) => v !== '' && v !== false)

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-5xl space-y-5 p-4 sm:p-6">
        <header>
          <h1 className="text-2xl font-semibold tracking-tight">{t.nav.library}</h1>
          <p className="mt-1 text-sm text-muted-foreground">{tl.intro}</p>
        </header>

        <Card>
          <CardHeader>
            <CardTitle>{tl.filters}</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <Filters />
            {facets && (
              <p className="text-xs text-muted-foreground">{tl.summary(facets.total, facets.favorites)}</p>
            )}
          </CardContent>
        </Card>

        {error && (
          <div role="alert" className="rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm">
            {error}
          </div>
        )}

        <Card>
          <CardHeader className="flex flex-row items-center justify-between gap-3">
            <CardTitle>{tl.results(total)}</CardTitle>
            {selected.length > 0 && (
              <div className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground">{tl.selected(selected.length)}</span>
                <Button size="sm" variant="ghost" onClick={clearSelection}>
                  {tl.clearSelection}
                </Button>
                <Button
                  size="sm"
                  variant="destructive"
                  onClick={() => {
                    if (confirm(tl.confirmDelete(selected.length))) void removeSelected()
                  }}
                >
                  <Trash2 />
                  {tl.deleteSelected}
                </Button>
              </div>
            )}
          </CardHeader>
          <CardContent>
            {items.length === 0 ? (
              <p className="text-sm text-muted-foreground">{loading ? t.common.loading : filtered ? tl.noResults : tl.empty}</p>
            ) : (
              <ul className="divide-y rounded-md border">
                {items.map((item) => (
                  <Row key={item.id} item={item} />
                ))}
              </ul>
            )}
            {items.length < total && (
              <div className="pt-3 text-center">
                <Button size="sm" variant="outline" disabled={loading} onClick={() => void loadMore()}>
                  {tl.loadMore}
                </Button>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
