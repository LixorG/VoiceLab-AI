import { FlaskConical, Gauge, Loader2, Plus, Trash2, X } from 'lucide-react'
import { useEffect, useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'
import { SyncedPlayers } from '@/features/audio/SyncedPlayers'
import { ComparisonTable } from '@/features/comparison/ComparisonTable'
import { type ArmDraft, ArmsEditor, armParams, newArm } from '@/features/experiments/ArmsEditor'
import { ExperimentForm } from '@/features/experiments/ExperimentForm'
import { t } from '@/i18n/es'
import { cn } from '@/lib/utils'
import { useEngineStore } from '@/stores/engine'
import { isRunning, useExperimentsStore } from '@/stores/experiments'
import { useProfilesStore } from '@/stores/profiles'

const tx = t.experiments
const POLL_MS = 1500

function ExperimentDetail() {
  const { detail, busy, error, update, remove, evaluateAll, addArms, refresh, replaceGeneration, clearError } = useExperimentsStore()
  const { engines, valuesByKey } = useEngineStore()
  const profile = useProfilesStore((s) => s.items.find((p) => p.id === detail?.profile_id) ?? null)
  const [name, setName] = useState(detail?.name ?? '')
  const [notes, setNotes] = useState(detail?.notes ?? '')
  const [adding, setAdding] = useState<ArmDraft[] | null>(null)
  const running = isRunning(detail)

  useEffect(() => {
    setName(detail?.name ?? '')
    setNotes(detail?.notes ?? '')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [detail?.id])

  useEffect(() => {
    if (!running) return
    const id = window.setInterval(() => void refresh(), POLL_MS)
    return () => window.clearInterval(id)
  }, [running, refresh])

  if (!detail) return <Loader2 className="mx-auto mt-10 animate-spin text-muted-foreground" />

  const completed = detail.generations.filter((g) => g.status === 'COMPLETED' && g.audio_url)
  const tracks = completed.map((g) => ({ id: g.id, label: g.label ?? g.engine, url: g.audio_url! }))

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-start gap-3">
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          onBlur={() => name.trim() && name !== detail.name && void update({ name: name.trim() })}
          aria-label={tx.name}
          className="min-w-0 flex-1 rounded-md border border-transparent bg-transparent px-1 text-xl font-semibold hover:border-border focus:border-border"
        />
        <Button size="sm" variant="outline" disabled={completed.length === 0 || busy === 'evaluate'} onClick={() => void evaluateAll()} title={tx.evaluateAllHint}>
          {busy === 'evaluate' ? <Loader2 className="animate-spin" /> : <Gauge />}
          {tx.evaluateAll}
        </Button>
        <Button size="sm" variant="ghost" className="hover:text-destructive" onClick={() => window.confirm(tx.confirmDelete) && void remove(detail.id)}>
          <Trash2 />
          {tx.delete}
        </Button>
      </div>
      <p className="rounded-md border bg-muted/30 p-3 text-sm whitespace-pre-wrap">{detail.text}</p>

      {error && (
        <p role="alert" className="flex items-start gap-1.5 text-xs text-destructive">
          <span className="flex-1">{error}</span>
          <button type="button" onClick={clearError} aria-label={t.references.dismiss}>
            <X className="size-3.5" />
          </button>
        </p>
      )}

      <Card>
        <CardHeader>
          <CardTitle>{tx.listen}</CardTitle>
          {running && (
            <span className="flex items-center gap-1 text-xs text-muted-foreground">
              <Loader2 className="size-3.5 animate-spin" />
              {tx.running}
            </span>
          )}
        </CardHeader>
        <CardContent>{tracks.length >= 1 ? <SyncedPlayers tracks={tracks} /> : <p className="text-sm text-muted-foreground">{tx.waiting}</p>}</CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>{tx.comparison}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <p className="text-[11px] text-muted-foreground">{t.compare.disclaimer}</p>
          <ComparisonTable generations={detail.generations} onUpdate={replaceGeneration} />
          {detail.generations
            .filter((g) => g.status === 'FAILED')
            .map((g) => (
              <p key={g.id} className="text-xs text-destructive">
                {g.label}: {g.message ?? t.generation.failed}
              </p>
            ))}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>{tx.moreArms}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {adding ? (
            <>
              <ArmsEditor arms={adding} onChange={setAdding} profile={profile} />
              <div className="flex justify-end gap-2">
                <Button size="sm" variant="ghost" onClick={() => setAdding(null)}>
                  {t.postprocess.cancel}
                </Button>
                <Button
                  size="sm"
                  disabled={busy === 'arms'}
                  onClick={async () => {
                    const ok = await addArms(adding.map((a) => ({ engine: a.engine, variant: a.variant, params: armParams(a, profile, valuesByKey) })))
                    if (ok) setAdding(null)
                  }}
                >
                  {busy === 'arms' && <Loader2 className="animate-spin" />}
                  {tx.run(adding.length)}
                </Button>
              </div>
            </>
          ) : (
            <Button size="sm" variant="outline" disabled={detail.generations.length >= 12 || !engines.length} onClick={() => setAdding([newArm(engines[0].id, engines[0].default_variant)])}>
              <Plus />
              {tx.addArm}
            </Button>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>{tx.notes}</CardTitle>
        </CardHeader>
        <CardContent>
          <Textarea value={notes} onChange={(e) => setNotes(e.target.value)} onBlur={() => notes !== (detail.notes ?? '') && void update({ notes })} placeholder={tx.notesPlaceholder} maxLength={5000} className="min-h-20" />
        </CardContent>
      </Card>
    </div>
  )
}

export function ExperimentsPage() {
  const { items, selected, load, open } = useExperimentsStore()
  const loadEngines = useEngineStore((s) => s.loadEngines)
  const loadProfiles = useProfilesStore((s) => s.load)

  useEffect(() => {
    void load()
    void loadEngines()
    void loadProfiles()
  }, [load, loadEngines, loadProfiles])

  return (
    <div className="flex h-full min-h-0 flex-col overflow-y-auto lg:flex-row lg:overflow-hidden">
      <aside className="w-full shrink-0 space-y-3 border-b bg-panel p-4 lg:w-72 lg:overflow-y-auto lg:border-r lg:border-b-0">
        <div className="flex items-center gap-2">
          <FlaskConical className="size-4" />
          <h1 className="text-sm font-semibold">{t.nav.experiments}</h1>
          <Button size="sm" className="ml-auto h-7" onClick={() => void open('new')}>
            <Plus />
            {tx.new}
          </Button>
        </div>
        {items.length === 0 && <p className="text-xs text-muted-foreground">{tx.empty}</p>}
        <ul className="space-y-1.5">
          {items.map((e) => (
            <li key={e.id}>
              <button
                type="button"
                onClick={() => void open(e.id)}
                aria-current={selected === e.id}
                className={cn('w-full space-y-1 rounded-md border p-2 text-left text-xs hover:bg-muted/40', selected === e.id && 'border-primary bg-primary/5')}
              >
                <span className="block truncate font-medium">{e.name}</span>
                <span className="line-clamp-1 block text-muted-foreground">{e.text}</span>
                <span className="flex flex-wrap gap-1">
                  <Badge variant="outline">{tx.armsCount(e.arms)}</Badge>
                  {e.running > 0 && <Badge variant="warning">{tx.runningCount(e.running)}</Badge>}
                  {e.failed > 0 && <Badge variant="destructive">{tx.failedCount(e.failed)}</Badge>}
                </span>
              </button>
            </li>
          ))}
        </ul>
      </aside>
      <main className="min-w-0 flex-1 lg:overflow-y-auto">
        <div className="mx-auto max-w-5xl space-y-5 p-4 sm:p-6">
          {selected === 'new' ? (
            <ExperimentForm />
          ) : selected ? (
            <ExperimentDetail />
          ) : (
            <header className="space-y-2 py-10 text-center">
              <FlaskConical className="mx-auto size-8 text-muted-foreground" />
              <h2 className="text-lg font-semibold">{t.nav.experiments}</h2>
              <p className="mx-auto max-w-md text-sm text-muted-foreground">{tx.intro}</p>
              <Button onClick={() => void open('new')}>
                <Plus />
                {tx.new}
              </Button>
            </header>
          )}
        </div>
      </main>
    </div>
  )
}
