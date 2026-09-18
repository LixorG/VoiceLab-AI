import { AlertTriangle, ArrowDown, ArrowUp, Captions, Clapperboard, FileArchive, Headphones, Loader2, Plus, RefreshCw, Sparkles, Trash2, Wand2, X } from 'lucide-react'
import { useEffect, useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { DownloadMenu, useExportFormats } from '@/components/ui/download-menu'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'
import { AudioPlayer } from '@/features/audio/AudioPlayer'
import { t } from '@/i18n/es'
import { cn, formatDuration } from '@/lib/utils'
import { projectsApi } from '@/services/projects'
import { referencesApi } from '@/services/references'
import { keyOf, useEngineStore } from '@/stores/engine'
import { useGenerationStore } from '@/stores/generation'
import { isProjectRunning, useProjectsStore } from '@/stores/projects'
import { useProfilesStore } from '@/stores/profiles'
import type { ReferenceRead } from '@/types/api'
import { BASIC_MASTERING } from '@/types/postprocess'
import { EMPTY_SETTINGS, type ProjectRead, type ProjectSegment, type ProjectSettings, type SegmentStatus } from '@/types/projects'

const tp = t.projects
const POLL_MS = 1500
const selectClass = 'h-8 w-full rounded-md border bg-background px-2 text-xs'

const STATUS_VARIANT: Record<SegmentStatus, 'outline' | 'warning' | 'success' | 'destructive'> = {
  empty: 'outline',
  queued: 'warning',
  generating: 'warning',
  ready: 'success',
  stale: 'warning',
  failed: 'destructive',
  cancelled: 'outline',
}

function SettingsCard({ project }: { project: ProjectRead }) {
  const { saveSettings, busy } = useProjectsStore()
  const engines = useEngineStore((s) => s.engines).filter((e) => e.implemented)
  const profiles = useProfilesStore((s) => s.items)
  const [references, setReferences] = useState<ReferenceRead[]>([])
  const cfg = project.settings
  const engine = engines.find((e) => e.id === cfg.engine)

  useEffect(() => {
    if (!cfg.profile_id) {
      setReferences([])
      return
    }
    referencesApi.list(cfg.profile_id).then(
      (refs) => setReferences(refs.filter((r) => r.status === 'ANALYZED')),
      () => setReferences([]),
    )
  }, [cfg.profile_id])

  const save = (patch: Partial<ProjectSettings>) => void saveSettings({ ...cfg, ...patch })

  const copyFromGenerate = () => {
    const { engineId, variantId, valuesByKey } = useEngineStore.getState()
    const generation = useGenerationStore.getState()
    save({
      engine: engineId,
      variant: variantId,
      params: engineId ? (valuesByKey[keyOf(engineId, variantId)] ?? {}) : {},
      profile_id: generation.profileId,
      reference_id: generation.referenceId,
      markup: generation.markup,
    })
  }

  const paramCount = Object.keys(cfg.params).length

  return (
    <Card>
      <CardHeader>
        <CardTitle>{tp.settings}</CardTitle>
        <Button size="sm" variant="outline" className="h-7" onClick={copyFromGenerate} disabled={busy === 'settings'} title={tp.copyHint}>
          {busy === 'settings' ? <Loader2 className="animate-spin" /> : <Wand2 />}
          {tp.copyFromGenerate}
        </Button>
      </CardHeader>
      <CardContent className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        <label className="space-y-1 text-xs">
          <span className="text-muted-foreground">{tp.engine}</span>
          <select
            className={selectClass}
            value={cfg.engine ?? ''}
            onChange={(e) => {
              const next = engines.find((x) => x.id === e.target.value)
              save({ engine: next?.id ?? null, variant: next?.default_variant ?? null, params: {} })
            }}
          >
            <option value="">{tp.chooseEngine}</option>
            {engines.map((e) => (
              <option key={e.id} value={e.id}>
                {e.name}
              </option>
            ))}
          </select>
        </label>
        <label className="space-y-1 text-xs">
          <span className="text-muted-foreground">{tp.variant}</span>
          <select className={selectClass} value={cfg.variant ?? ''} disabled={!engine} onChange={(e) => save({ variant: e.target.value, params: {} })}>
            {engine?.variants.map((v) => (
              <option key={v.id} value={v.id}>
                {v.label}
              </option>
            ))}
          </select>
        </label>
        <div className="space-y-1 text-xs">
          <span className="text-muted-foreground">{tp.params}</span>
          <p className="flex h-8 items-center text-muted-foreground">{paramCount ? tp.paramsCustom(paramCount) : tp.paramsDefault}</p>
        </div>
        <label className="space-y-1 text-xs">
          <span className="text-muted-foreground">{tp.voice}</span>
          <select className={selectClass} value={cfg.profile_id ?? ''} onChange={(e) => save({ profile_id: e.target.value || null, reference_id: null })}>
            <option value="">{tp.noProfile}</option>
            {profiles.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>
        </label>
        <label className="space-y-1 text-xs">
          <span className="text-muted-foreground">{tp.reference}</span>
          <select className={selectClass} value={cfg.reference_id ?? ''} disabled={!cfg.profile_id} onChange={(e) => save({ reference_id: e.target.value || null })}>
            <option value="">{tp.profileReference}</option>
            {references.map((r) => (
              <option key={r.id} value={r.id}>
                {r.original_name}
                {r.is_primary ? ` · ${tp.primary}` : ''}
              </option>
            ))}
          </select>
        </label>
        <label className="space-y-1 text-xs">
          <span className="text-muted-foreground">{tp.defaultPause}</span>
          <input
            type="number"
            min={0}
            max={10000}
            step={50}
            className={selectClass}
            defaultValue={cfg.default_pause_ms}
            key={`pause-${project.id}-${cfg.default_pause_ms}`}
            onBlur={(e) => {
              const value = Math.max(0, Math.min(10000, parseInt(e.target.value, 10) || 0))
              if (value !== cfg.default_pause_ms) save({ default_pause_ms: value })
            }}
          />
        </label>
        <label className="flex items-center gap-2 text-xs">
          <input type="checkbox" className="accent-[var(--primary)]" checked={cfg.markup} onChange={(e) => save({ markup: e.target.checked })} />
          {tp.markup}
        </label>
        <label className="flex items-center gap-2 text-xs sm:col-span-2" title={tp.masteringHint}>
          <input
            type="checkbox"
            className="accent-[var(--primary)]"
            checked={!!cfg.export_postprocess}
            onChange={(e) => save({ export_postprocess: e.target.checked ? BASIC_MASTERING : null })}
          />
          {tp.mastering}
        </label>
      </CardContent>
    </Card>
  )
}

function ImportCard({ hasSegments }: { hasSegments: boolean }) {
  const { importScript, busy } = useProjectsStore()
  const [text, setText] = useState('')
  const [split, setSplit] = useState<'paragraphs' | 'sentences'>('paragraphs')
  const [open, setOpen] = useState(!hasSegments)

  if (!open) {
    return (
      <Button size="sm" variant="outline" onClick={() => setOpen(true)}>
        <Plus />
        {tp.importMore}
      </Button>
    )
  }
  return (
    <Card>
      <CardHeader>
        <CardTitle>{tp.importTitle}</CardTitle>
        {hasSegments && (
          <Button size="icon" variant="ghost" className="size-7" onClick={() => setOpen(false)} aria-label={tp.close}>
            <X />
          </Button>
        )}
      </CardHeader>
      <CardContent className="space-y-3">
        <Textarea value={text} onChange={(e) => setText(e.target.value)} placeholder={tp.importPlaceholder} aria-label={tp.importTitle} className="min-h-32" />
        <div className="flex flex-wrap items-center gap-4 text-xs">
          <span className="text-muted-foreground">{tp.splitBy}</span>
          {(['paragraphs', 'sentences'] as const).map((mode) => (
            <label key={mode} className="flex items-center gap-1.5">
              <input type="radio" name="split" className="accent-[var(--primary)]" checked={split === mode} onChange={() => setSplit(mode)} />
              {tp.split[mode]}
            </label>
          ))}
          <Button
            size="sm"
            className="ml-auto"
            disabled={!text.trim() || busy === 'import'}
            onClick={async () => {
              if (await importScript(text, split)) {
                setText('')
                setOpen(false)
              }
            }}
          >
            {busy === 'import' ? <Loader2 className="animate-spin" /> : <Plus />}
            {tp.import}
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

function SegmentRow({ project, segment, count }: { project: ProjectRead; segment: ProjectSegment; count: number }) {
  const { updateSegment, removeSegment, move, generate, busy } = useProjectsStore()
  const profiles = useProfilesStore((s) => s.items)
  const emotions = useProfilesStore((s) => s.emotions)
  const [text, setText] = useState(segment.text)
  const gen = segment.generation
  const running = segment.status === 'queued' || segment.status === 'generating'
  const rowBusy = busy === `segment:${segment.id}`

  useEffect(() => setText(segment.text), [segment.text])

  return (
    <article className="space-y-2 rounded-lg border bg-background/40 p-3" data-testid="segment-row" data-status={segment.status}>
      <header className="flex flex-wrap items-center gap-2">
        <span className="font-mono text-sm font-semibold text-primary">{segment.position + 1}</span>
        <Badge variant={STATUS_VARIANT[segment.status]}>
          {running && <Loader2 className="animate-spin" />}
          {tp.status[segment.status]}
          {running && gen?.progress_available && gen.progress > 0 ? ` ${Math.round(gen.progress * 100)}%` : ''}
        </Badge>
        {gen?.duration_s != null && segment.status !== 'failed' && <span className="font-mono text-[11px] text-muted-foreground">{formatDuration(gen.duration_s)}</span>}
        <span className="ml-auto flex gap-0.5">
          <Button size="icon" variant="ghost" className="size-7" disabled={segment.position === 0 || busy != null} onClick={() => void move(segment.id, -1)} aria-label={tp.moveUp}>
            <ArrowUp />
          </Button>
          <Button size="icon" variant="ghost" className="size-7" disabled={segment.position === count - 1 || busy != null} onClick={() => void move(segment.id, 1)} aria-label={tp.moveDown}>
            <ArrowDown />
          </Button>
          <Button size="sm" variant="ghost" className="h-7 text-[11px]" disabled={running || rowBusy || !project.settings.engine} onClick={() => void generate([segment.id], false)}>
            {rowBusy ? <Loader2 className="animate-spin" /> : <RefreshCw />}
            {segment.status === 'empty' ? tp.generateOne : tp.regenerate}
          </Button>
          <Button
            size="icon"
            variant="ghost"
            className="size-7 hover:text-destructive"
            disabled={rowBusy}
            onClick={() => window.confirm(tp.confirmDeleteSegment) && void removeSegment(segment.id)}
            aria-label={tp.deleteSegment}
          >
            <Trash2 />
          </Button>
        </span>
      </header>

      <Textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        onBlur={() => text.trim() && text !== segment.text && void updateSegment(segment.id, { text: text.trim() })}
        aria-label={tp.segmentText(segment.position + 1)}
        className="min-h-16 text-sm"
        maxLength={5000}
      />

      <div className="grid gap-2 text-xs sm:grid-cols-4">
        <label className="space-y-0.5">
          <span className="text-muted-foreground">{tp.voice}</span>
          <select className={selectClass} value={segment.profile_id ?? ''} onChange={(e) => void updateSegment(segment.id, { profile_id: e.target.value || null, reference_id: null })}>
            <option value="">{tp.inherit}</option>
            {profiles.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>
        </label>
        <label className="space-y-0.5">
          <span className="text-muted-foreground">{tp.emotion}</span>
          <select className={selectClass} value={segment.emotion ?? ''} onChange={(e) => void updateSegment(segment.id, { emotion: e.target.value || null })}>
            <option value="">{tp.noEmotion}</option>
            {emotions.map((em) => (
              <option key={em.id} value={em.id}>
                {em.label}
              </option>
            ))}
          </select>
        </label>
        <label className="space-y-0.5">
          <span className="text-muted-foreground">{tp.pauseAfter}</span>
          <input
            type="number"
            min={0}
            max={10000}
            step={50}
            className={selectClass}
            placeholder={String(project.settings.default_pause_ms)}
            defaultValue={segment.pause_after_ms ?? ''}
            key={`pause-${segment.id}-${segment.pause_after_ms}`}
            onBlur={(e) => {
              const raw = e.target.value.trim()
              const value = raw === '' ? null : Math.max(0, Math.min(10000, parseInt(raw, 10) || 0))
              if (value !== segment.pause_after_ms) void updateSegment(segment.id, { pause_after_ms: value })
            }}
          />
        </label>
        <label className="space-y-0.5">
          <span className="text-muted-foreground">{tp.seed}</span>
          <input
            inputMode="numeric"
            className={cn(selectClass, 'font-mono')}
            placeholder={tp.seedRandom}
            defaultValue={segment.seed ?? ''}
            key={`seed-${segment.id}-${segment.seed}`}
            onBlur={(e) => {
              const raw = e.target.value.replace(/[^\d]/g, '')
              const value = raw === '' ? null : parseInt(raw, 10)
              if (value !== segment.seed) void updateSegment(segment.id, { seed: value })
            }}
          />
        </label>
      </div>

      {gen?.audio_url && (segment.status === 'ready' || segment.status === 'stale') && <AudioPlayer url={gen.audio_url} duration={gen.duration_s} />}
      {segment.status === 'stale' && <p className="text-[11px] text-warning">{tp.staleHint}</p>}
      {segment.status === 'failed' && gen?.message && (
        <p role="alert" className="flex items-start gap-1.5 text-xs text-destructive">
          <AlertTriangle className="mt-px size-3.5 shrink-0" />
          {gen.message}
        </p>
      )}
      {gen?.warnings.map((w) => (
        <p key={w} className="flex items-start gap-1.5 text-[11px] text-warning">
          <AlertTriangle className="mt-px size-3 shrink-0" />
          {w}
        </p>
      ))}
    </article>
  )
}

function ProjectDetail() {
  const { detail: project, busy, error, rename, generate, remove, refresh, addSegment, clearError } = useProjectsStore()
  const [name, setName] = useState(project?.name ?? '')
  const [preview, setPreview] = useState<string | null>(null)
  const formats = useExportFormats()
  const running = isProjectRunning(project)

  useEffect(() => {
    setName(project?.name ?? '')
    setPreview(null)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project?.id])

  useEffect(() => {
    if (!running) return
    const id = window.setInterval(() => void refresh(), POLL_MS)
    return () => window.clearInterval(id)
  }, [running, refresh])

  if (!project) return <Loader2 className="mx-auto mt-10 animate-spin text-muted-foreground" />

  const counts = project.segments.reduce<Record<string, number>>((acc, s) => ({ ...acc, [s.status]: (acc[s.status] ?? 0) + 1 }), {})
  const pending = project.segments.filter((s) => !['ready', 'queued', 'generating'].includes(s.status)).length
  const ready = counts.ready ?? 0
  const canExport = project.segments.length > 0 && ready === project.segments.length
  // Same length as the exported file: generated audio plus the pauses between segments (not after the last one).
  const pauses = project.segments.slice(0, -1).reduce((sum, s) => sum + s.effective_pause_ms, 0) / 1000

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-3">
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          onBlur={() => name.trim() && name !== project.name && void rename(name.trim())}
          aria-label={tp.name}
          className="min-w-0 flex-1 rounded-md border border-transparent bg-transparent px-1 text-xl font-semibold hover:border-border focus:border-border"
        />
        <Button size="sm" variant="ghost" className="hover:text-destructive" onClick={() => window.confirm(tp.confirmDelete) && void remove(project.id)}>
          <Trash2 />
          {tp.delete}
        </Button>
      </div>

      {error && (
        <p role="alert" className="flex items-start gap-1.5 text-xs text-destructive">
          <AlertTriangle className="mt-px size-3.5 shrink-0" />
          <span className="flex-1">{error}</span>
          <button type="button" onClick={clearError} aria-label={t.references.dismiss}>
            <X className="size-3.5" />
          </button>
        </p>
      )}

      <SettingsCard project={project} />
      <ImportCard hasSegments={project.segments.length > 0} />

      {project.segments.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>{tp.segments(project.segments.length)}</CardTitle>
            <span className="text-xs text-muted-foreground">
              {tp.progress(ready, project.segments.length)} · {formatDuration(project.total_duration_s + (ready ? pauses : 0))}
            </span>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex flex-wrap items-center gap-2">
              <Button size="sm" disabled={pending === 0 || busy === 'generate' || !project.settings.engine} onClick={() => void generate(null)} title={project.settings.engine ? undefined : tp.chooseEngine}>
                {busy === 'generate' ? <Loader2 className="animate-spin" /> : <Sparkles />}
                {tp.generatePending(pending)}
              </Button>
              {running && (
                <span className="flex items-center gap-1 text-xs text-muted-foreground">
                  <Loader2 className="size-3.5 animate-spin" />
                  {tp.running}
                </span>
              )}
              <span className="ml-auto flex flex-wrap gap-2">
                <Button size="sm" variant="outline" disabled={ready === 0} onClick={() => setPreview(projectsApi.exportUrl(project.id, 'wav', { download: false, partial: !canExport, bust: Date.now() }))}>
                  <Headphones />
                  {tp.listenAll}
                </Button>
                <DownloadMenu
                  label={tp.exportAudio}
                  disabled={!canExport}
                  disabledTitle={tp.exportNeedsAll}
                  items={formats.map((f) => ({
                    key: f.id,
                    label: f.label,
                    href: projectsApi.exportUrl(project.id, f.id),
                    disabled: !f.available,
                    title: f.reason ?? undefined,
                  }))}
                />
                <DownloadMenu
                  label={tp.subtitles}
                  icon={<Captions className="size-4" />}
                  disabled={!canExport}
                  disabledTitle={tp.exportNeedsAll}
                  items={(['srt', 'vtt'] as const).map((f) => ({ key: f, label: tp.subtitleFormats[f], href: projectsApi.subtitlesUrl(project.id, f), title: tp.subtitlesHint }))}
                />
                {canExport ? (
                  <Button asChild size="sm" variant="outline">
                    <a href={projectsApi.exportUrl(project.id, 'zip')}>
                      <FileArchive />
                      {tp.exportZip}
                    </a>
                  </Button>
                ) : (
                  <Button size="sm" variant="outline" disabled title={tp.exportNeedsAll}>
                    <FileArchive />
                    {tp.exportZip}
                  </Button>
                )}
              </span>
            </div>
            {!canExport && ready > 0 && <p className="text-[11px] text-muted-foreground">{tp.exportNeedsAll}</p>}
            {preview && (
              <div className="rounded-md border border-dashed p-2">
                <p className="mb-1 text-[11px] text-muted-foreground">{canExport ? tp.previewAll : tp.previewPartial}</p>
                <AudioPlayer url={preview} />
              </div>
            )}
            {project.segments.map((segment) => (
              <SegmentRow key={segment.id} project={project} segment={segment} count={project.segments.length} />
            ))}
            <Button size="sm" variant="ghost" onClick={() => void addSegment(tp.newSegmentText)} disabled={busy === 'add'}>
              <Plus />
              {tp.addSegment}
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

function NewProject() {
  const { create, busy } = useProjectsStore()
  const [name, setName] = useState('')
  const copy = () => {
    const { engineId, variantId, valuesByKey } = useEngineStore.getState()
    const generation = useGenerationStore.getState()
    return {
      ...EMPTY_SETTINGS,
      engine: engineId,
      variant: variantId,
      params: engineId ? (valuesByKey[keyOf(engineId, variantId)] ?? {}) : {},
      profile_id: generation.profileId,
      reference_id: generation.referenceId,
      markup: generation.markup,
    }
  }
  return (
    <form
      className="flex gap-2"
      onSubmit={(e) => {
        e.preventDefault()
        if (name.trim()) void create(name.trim(), copy()).then((ok) => ok && setName(''))
      }}
    >
      <input value={name} onChange={(e) => setName(e.target.value)} placeholder={tp.newName} aria-label={tp.newName} maxLength={120} className="h-8 min-w-0 flex-1 rounded-md border bg-background px-2 text-xs" />
      <Button size="sm" className="h-8" type="submit" disabled={!name.trim() || busy === 'create'}>
        {busy === 'create' ? <Loader2 className="animate-spin" /> : <Plus />}
        {tp.create}
      </Button>
    </form>
  )
}

export function ProjectsPage() {
  const { items, selected, load, open } = useProjectsStore()
  const loadEngines = useEngineStore((s) => s.loadEngines)
  const { load: loadProfiles, loadEmotions } = useProfilesStore()

  useEffect(() => {
    void load()
    void loadEngines()
    void loadProfiles()
    void loadEmotions()
  }, [load, loadEngines, loadProfiles, loadEmotions])

  return (
    <div className="flex h-full min-h-0 flex-col overflow-y-auto lg:flex-row lg:overflow-hidden">
      <aside className="w-full shrink-0 space-y-3 border-b bg-panel p-4 lg:w-72 lg:overflow-y-auto lg:border-r lg:border-b-0">
        <div className="flex items-center gap-2">
          <Clapperboard className="size-4" />
          <h1 className="text-sm font-semibold">{t.nav.projects}</h1>
        </div>
        <NewProject />
        {items.length === 0 && <p className="text-xs text-muted-foreground">{tp.empty}</p>}
        <ul className="space-y-1.5">
          {items.map((p) => (
            <li key={p.id}>
              <button
                type="button"
                onClick={() => void open(p.id)}
                aria-current={selected === p.id}
                className={cn('w-full space-y-1 rounded-md border p-2 text-left text-xs hover:bg-muted/40', selected === p.id && 'border-primary bg-primary/5')}
              >
                <span className="block truncate font-medium">{p.name}</span>
                <span className="flex flex-wrap gap-1">
                  <Badge variant="outline">{tp.progress(p.ready, p.segments)}</Badge>
                  {p.running > 0 && <Badge variant="warning">{tp.runningCount(p.running)}</Badge>}
                  {p.failed > 0 && <Badge variant="destructive">{tp.failedCount(p.failed)}</Badge>}
                </span>
              </button>
            </li>
          ))}
        </ul>
      </aside>
      <main className="min-w-0 flex-1 lg:overflow-y-auto">
        <div className="mx-auto max-w-5xl space-y-5 p-4 sm:p-6">
          {selected ? (
            <ProjectDetail />
          ) : (
            <header className="space-y-2 py-10 text-center">
              <Clapperboard className="mx-auto size-8 text-muted-foreground" />
              <h2 className="text-lg font-semibold">{t.nav.projects}</h2>
              <p className="mx-auto max-w-md text-sm text-muted-foreground">{tp.intro}</p>
            </header>
          )}
        </div>
      </main>
    </div>
  )
}
