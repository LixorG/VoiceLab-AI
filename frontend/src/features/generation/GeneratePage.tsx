import { AlertTriangle, Info, Layers, Loader2, Play, Sparkles, Wand2, X } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { GenerationList } from '@/features/generation/GenerationList'
import { MarkupToolbar, PlanPreview } from '@/features/generation/MarkupTools'
import { ParamsClipboardBar } from '@/features/generation/ParamsClipboard'
import { BestTakeControl } from '@/features/generation/BestTakePanel'
import { DialoguePanel } from '@/features/generation/DialoguePanel'
import { BatchPanel, QueuePanel } from '@/features/generation/QueuePanel'
import { ScriptPrep } from '@/features/generation/ScriptPrep'
import { generationBlockers } from '@/features/generation/readiness'
import { EngineSettingsPanel } from '@/features/models/EngineSettingsPanel'
import { PostProcessPanel } from '@/features/postprocess/PostProcessPanel'
import { ReferenceLibrary } from '@/features/voices/ReferenceLibrary'
import { VoiceSelector } from '@/features/voices/VoiceSelector'
import { t } from '@/i18n/es'
import { keyOf, useEngineStore } from '@/stores/engine'
import { useGenerationStore } from '@/stores/generation'
import { useReferencesStore } from '@/stores/references'
import { useUiStore } from '@/stores/ui'

const tg = t.generation
const NO_VALUES: Record<string, never> = {}

export function GeneratePage() {
  const mode = useUiStore((s) => s.mode)
  const { text, setText, referenceId, profileId, emotion, intensity, markup, speakers, turnPauseMs, submitting, error, runtime, generate, generateVariations, loadPlan, clearError } =
    useGenerationStore()
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const [variationCount, setVariationCount] = useState(3)
  const { config, engineId, variantId, valuesByKey } = useEngineStore()
  const references = useReferencesStore((s) => s.items)

  const reference = references.find((r) => r.id === referenceId)
  const values = valuesByKey[keyOf(engineId, variantId)] ?? NO_VALUES
  const blockers = useMemo(
    () =>
      generationBlockers({
        config,
        runtime: runtime?.engine === engineId && runtime.variant === variantId ? runtime : null,
        values,
        text,
        reference,
      }),
    [config, runtime, engineId, variantId, values, text, reference],
  )
  const ready = blockers.length === 0 && !submitting

  // Live plan preview (debounced): shows segments, how emotions are applied and markup errors before generating.
  useEffect(() => {
    if (!engineId) return
    const id = window.setTimeout(() => void loadPlan(), 450)
    return () => window.clearTimeout(id)
  }, [text, emotion, intensity, markup, engineId, variantId, values, profileId, referenceId, speakers, turnPauseMs, loadPlan])

  return (
    <div className="flex h-full min-h-0 flex-col overflow-y-auto xl:flex-row xl:overflow-hidden">
      {/* The text and the settings panel read as one block: the column grows and the panel widens with the
          screen, so they stay side by side instead of drifting apart on a wide monitor. */}
      <div className="relative min-w-0 flex-1 xl:overflow-y-auto">
        {/* With the settings panel beside it (xl) the column fills its space, so both read as one block; when the
            panel stacks underneath (narrow screens) the text is centred and capped for comfortable reading. */}
        <div className="mx-auto max-w-5xl space-y-5 p-4 sm:p-6 xl:max-w-none">
          <header>
            <h1 className="text-2xl font-semibold tracking-tight">{t.generate.title}</h1>
            <p className="text-sm text-muted-foreground">{t.generate.subtitle}</p>
          </header>

          <VoiceSelector />
          <ReferenceLibrary />

          <Card>
            <CardHeader>
              <CardTitle>{t.generate.text}</CardTitle>
              <span className="font-mono text-xs text-muted-foreground">{t.generate.characters(text.length)}</span>
            </CardHeader>
            <CardContent className="space-y-3">
              <MarkupToolbar textareaRef={textareaRef} />
              <Textarea
                ref={textareaRef}
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder={t.generate.textPlaceholder}
                aria-label={t.generate.text}
                maxLength={5000}
              />
              <ScriptPrep />
              <PlanPreview />
            </CardContent>
          </Card>

          <DialoguePanel />
          <BatchPanel />
          <GenerationList />
        </div>
      </div>

      <aside className="relative flex w-full shrink-0 flex-col border-t bg-panel xl:w-80 xl:border-t-0 xl:border-l 2xl:w-[22rem]">
        <div className="flex-1 space-y-4 p-5 xl:overflow-y-auto">
          <QueuePanel />
          <ParamsClipboardBar />
          <EngineSettingsPanel />
          <PostProcessPanel />
        </div>

        <div className="space-y-2 border-t p-4">
          <BestTakeControl />
          {blockers.length > 0 && (
            <div className="space-y-1 text-xs text-muted-foreground" aria-live="polite">
              <p className="font-medium text-foreground">{tg.blockersTitle}</p>
              <ul className="list-disc space-y-0.5 pl-4">
                {blockers.map((b) => (
                  <li key={b}>{b}</li>
                ))}
              </ul>
            </div>
          )}
          {error && (
            <p role="alert" className="flex items-start gap-1.5 text-xs text-destructive">
              <AlertTriangle className="mt-px size-3.5 shrink-0" />
              <span className="flex-1">{error}</span>
              <button type="button" onClick={clearError} aria-label={t.references.dismiss}>
                <X className="size-3.5" />
              </button>
            </p>
          )}
          <div className="flex gap-2">
            <Button variant="outline" className="flex-1" disabled={!ready} onClick={() => void generate(true)}>
              {submitting === 'preview' ? <Loader2 className="animate-spin" /> : <Play />}
              {tg.preview}
            </Button>
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="grid place-items-center text-muted-foreground" tabIndex={0} aria-label={tg.previewHint}>
                  <Info className="size-4" />
                </span>
              </TooltipTrigger>
              <TooltipContent>{tg.previewHint}</TooltipContent>
            </Tooltip>
          </div>
          <div className="flex gap-2">
            <Button variant="secondary" className="flex-1" disabled={!ready} onClick={() => void generateVariations(variationCount)} title={t.variations.hint}>
              {submitting === 'variations' ? <Loader2 className="animate-spin" /> : <Layers />}
              {t.variations.button}
            </Button>
            <select
              aria-label={t.variations.count}
              value={variationCount}
              onChange={(e) => setVariationCount(parseInt(e.target.value, 10))}
              className="h-9 rounded-md border bg-background px-2 text-sm"
            >
              {[2, 3, 4, 5, 6, 8].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </div>
          <Button size="lg" className="w-full" disabled={!ready} onClick={() => void generate(false)}>
            {submitting === 'full' ? <Loader2 className="animate-spin" /> : mode === 'advanced' ? <Wand2 /> : <Sparkles />}
            {submitting ? tg.submitting : tg.generate}
          </Button>
        </div>
      </aside>
    </div>
  )
}
