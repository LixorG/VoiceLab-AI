import { Plus, Trash2 } from 'lucide-react'
import { useEffect, useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { t } from '@/i18n/es'
import { ApiError } from '@/services/api'
import { checkpointsApi } from '@/services/checkpoints'
import { useEngineStore } from '@/stores/engine'
import type { CustomCheckpoint } from '@/types/checkpoints'

const tc = t.checkpoints
const field = 'h-8 w-full rounded-md border bg-background px-2 text-sm'

const EMPTY = { name: '', base_variant: '', repo_id: '', ckpt_file: '', local_path: '', vocab_path: '', languages: '', notes: '' }

/** Add community or self-trained checkpoints (e.g. a Spanish fine-tune of F5) as extra engine variants. */
export function CustomCheckpoints({ onChanged }: { onChanged?: () => void }) {
  const engines = useEngineStore((s) => s.engines)
  const loadEngines = useEngineStore((s) => s.loadEngines)
  const supported = engines.filter((e) => e.supports_custom_checkpoints)
  const [items, setItems] = useState<CustomCheckpoint[]>([])
  const [engineId, setEngineId] = useState('')
  const [source, setSource] = useState<'hub' | 'local'>('hub')
  const [draft, setDraft] = useState(EMPTY)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const reload = async () => {
    try {
      const list = await checkpointsApi.list()
      setItems(Array.isArray(list) ? list : [])
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err))
    }
  }

  useEffect(() => {
    void reload()
  }, [])

  const engine = supported.find((e) => e.id === (engineId || supported[0]?.id))
  const baseVariants = engine?.variants.filter((v) => v.source !== 'custom') ?? []

  const submit = async () => {
    if (!engine) return
    setError(null)
    setBusy(true)
    try {
      await checkpointsApi.create(engine.id, {
        name: draft.name,
        base_variant: draft.base_variant || baseVariants[0]?.id || '',
        repo_id: source === 'hub' ? draft.repo_id : null,
        ckpt_file: source === 'hub' ? draft.ckpt_file : null,
        local_path: source === 'local' ? draft.local_path : null,
        vocab_file: source === 'hub' ? draft.vocab_path || null : null,
        vocab_path: source === 'local' ? draft.vocab_path || null : null,
        languages: draft.languages.split(',').map((l) => l.trim()).filter(Boolean),
        notes: draft.notes || null,
      })
      setDraft(EMPTY)
      await reload()
      await loadEngines()
      onChanged?.()
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  const remove = async (item: CustomCheckpoint) => {
    if (!confirm(tc.confirmRemove)) return
    try {
      await checkpointsApi.remove(item.id)
      await reload()
      await loadEngines()
      onChanged?.()
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err))
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>{tc.title}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-sm text-muted-foreground">{tc.intro}</p>
        <p className="rounded-md border border-dashed px-3 py-2 text-xs text-muted-foreground">{tc.warning}</p>

        {error && (
          <p role="alert" className="text-xs text-destructive">
            {error}
          </p>
        )}

        {items.length === 0 ? (
          <p className="text-sm text-muted-foreground">{tc.empty}</p>
        ) : (
          <ul className="divide-y rounded-md border" data-testid="checkpoint-list">
            {items.map((item) => (
              <li key={item.id} className="flex items-center gap-2 px-3 py-2 text-sm">
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-1.5">
                    <span className="truncate">{item.name}</span>
                    <Badge variant="outline">{item.engine}</Badge>
                    {item.languages.map((l) => (
                      <Badge key={l} variant="accent">
                        {l}
                      </Badge>
                    ))}
                    <Badge variant={item.weights_installed ? 'success' : 'outline'}>{item.weights_installed ? tc.ready : tc.notDownloaded}</Badge>
                  </div>
                  <p className="truncate font-mono text-[10px] text-muted-foreground">
                    {item.repo_id ? `${item.repo_id}/${item.ckpt_file}` : item.local_path} · {item.base_variant}
                  </p>
                </div>
                <Button size="icon" variant="ghost" aria-label={`${tc.remove}: ${item.name}`} onClick={() => void remove(item)}>
                  <Trash2 className="size-4" />
                </Button>
              </li>
            ))}
          </ul>
        )}

        <div className="grid gap-2 sm:grid-cols-2">
          <label className="flex flex-col gap-1 text-xs text-muted-foreground">
            {tc.engine}
            <select aria-label={tc.engine} className={field} value={engine?.id ?? ''} onChange={(e) => setEngineId(e.target.value)}>
              {supported.map((e) => (
                <option key={e.id} value={e.id}>
                  {e.name}
                </option>
              ))}
            </select>
          </label>
          <label className="flex flex-col gap-1 text-xs text-muted-foreground">
            {tc.baseVariant}
            <select aria-label={tc.baseVariant} className={field} value={draft.base_variant || baseVariants[0]?.id || ''} onChange={(e) => setDraft({ ...draft, base_variant: e.target.value })}>
              {baseVariants.map((v) => (
                <option key={v.id} value={v.id}>
                  {v.label}
                </option>
              ))}
            </select>
          </label>
          <label className="flex flex-col gap-1 text-xs text-muted-foreground">
            {tc.name}
            <input aria-label={tc.name} className={field} placeholder={tc.namePlaceholder} value={draft.name} onChange={(e) => setDraft({ ...draft, name: e.target.value })} />
          </label>
          <label className="flex flex-col gap-1 text-xs text-muted-foreground">
            {tc.source}
            <select aria-label={tc.source} className={field} value={source} onChange={(e) => setSource(e.target.value as 'hub' | 'local')}>
              <option value="hub">{tc.sourceHub}</option>
              <option value="local">{tc.sourceLocal}</option>
            </select>
          </label>
          {source === 'hub' ? (
            <>
              <label className="flex flex-col gap-1 text-xs text-muted-foreground">
                {tc.repo}
                <input aria-label={tc.repo} className={field} placeholder={tc.repoPlaceholder} value={draft.repo_id} onChange={(e) => setDraft({ ...draft, repo_id: e.target.value })} />
              </label>
              <label className="flex flex-col gap-1 text-xs text-muted-foreground">
                {tc.file}
                <input aria-label={tc.file} className={field} placeholder={tc.filePlaceholder} value={draft.ckpt_file} onChange={(e) => setDraft({ ...draft, ckpt_file: e.target.value })} />
              </label>
            </>
          ) : (
            <label className="flex flex-col gap-1 text-xs text-muted-foreground sm:col-span-2">
              {tc.localPath}
              <input aria-label={tc.localPath} className={field} placeholder={tc.localPathPlaceholder} value={draft.local_path} onChange={(e) => setDraft({ ...draft, local_path: e.target.value })} />
            </label>
          )}
          <label className="flex flex-col gap-1 text-xs text-muted-foreground">
            {tc.vocab}
            <input aria-label={tc.vocab} className={field}
                   placeholder={source === 'hub' ? tc.vocabHubPlaceholder : tc.vocabPlaceholder} value={draft.vocab_path} onChange={(e) => setDraft({ ...draft, vocab_path: e.target.value })} />
          </label>
          <label className="flex flex-col gap-1 text-xs text-muted-foreground">
            {tc.languages}
            <input aria-label={tc.languages} className={field} placeholder={tc.languagesPlaceholder} value={draft.languages} onChange={(e) => setDraft({ ...draft, languages: e.target.value })} />
          </label>
        </div>
        <div className="flex items-center gap-2">
          <Button size="sm" disabled={busy || !draft.name.trim()} onClick={() => void submit()}>
            <Plus />
            {tc.add}
          </Button>
          <span className="text-xs text-muted-foreground">{tc.afterAdd}</span>
        </div>
      </CardContent>
    </Card>
  )
}
