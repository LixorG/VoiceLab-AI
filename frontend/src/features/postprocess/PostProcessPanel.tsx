import { ChevronDown, SlidersHorizontal } from 'lucide-react'
import { useEffect, useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { PostProcessEditor } from '@/features/postprocess/PostProcessEditor'
import { t } from '@/i18n/es'
import { useEngineStore } from '@/stores/engine'
import { useGenerationStore } from '@/stores/generation'
import { BASIC_MASTERING, DEFAULT_POSTPROCESS, isPostprocessActive } from '@/types/postprocess'

const tp = t.postprocess

/** Sidebar section: post-processing applied to the next generations (optional, off by default). */
export function PostProcessPanel() {
  const { postprocess, setPostprocess, postprocessCaps, loadPostprocessCaps, plan } = useGenerationStore()
  const speed = useEngineStore((s) => s.config?.capabilities.controls.speed)
  const active = isPostprocessActive(postprocess)
  const [open, setOpen] = useState(active)

  useEffect(() => {
    void loadPostprocessCaps()
  }, [loadPostprocessCaps])

  return (
    <section className="space-y-3 border-t pt-4" aria-labelledby="postprocess-title">
      <button type="button" className="flex w-full items-center gap-2 text-left" onClick={() => setOpen((v) => !v)} aria-expanded={open}>
        <SlidersHorizontal className="size-4" />
        <h2 id="postprocess-title" className="text-sm font-semibold">
          {tp.title}
        </h2>
        <Badge variant={active ? 'warning' : 'outline'}>{active ? tp.active : tp.off}</Badge>
        <ChevronDown className={`ml-auto size-4 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      {open && (
        <>
          <p className="text-[11px] text-muted-foreground">{tp.intro}</p>
          <div className="flex gap-2">
            <Button size="sm" variant="outline" className="h-7 flex-1 text-[11px]" onClick={() => setPostprocess(BASIC_MASTERING)} title={tp.basicHint}>
              {tp.basic}
            </Button>
            <Button size="sm" variant="ghost" className="h-7 text-[11px]" disabled={!active} onClick={() => setPostprocess(DEFAULT_POSTPROCESS)}>
              {tp.reset}
            </Button>
          </div>
          <PostProcessEditor
            value={postprocess}
            onChange={setPostprocess}
            capabilities={postprocessCaps}
            nativeSpeedParameter={speed?.source === 'native' ? speed.parameter : null}
            segmented={plan ? plan.segmented && plan.segments.length > 1 : undefined}
            idPrefix="next"
          />
        </>
      )}
    </section>
  )
}
