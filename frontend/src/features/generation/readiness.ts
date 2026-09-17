import { t } from '@/i18n/es'
import { formatDuration } from '@/lib/utils'
import type { EngineConfig, ParamValue } from '@/types/engines'
import type { EngineRuntimeStatus } from '@/types/generation'
import type { ReferenceRead } from '@/types/api'

const tb = t.generation.blockers
const TOLERANCE_S = 0.5

interface ReadinessInput {
  config: EngineConfig | null
  runtime: EngineRuntimeStatus | null
  values: Record<string, ParamValue>
  text: string
  reference: ReferenceRead | undefined
}

/** Human-readable reasons why generation can't start yet. The backend enforces the same rules. */
export function generationBlockers({ config, runtime, values, text, reference }: ReadinessInput): string[] {
  if (!config) return [tb.noEngine]
  const { engine, capabilities: caps } = config
  const blockers: string[] = []

  if (!engine.implemented && engine.implementation_phase != null) return [tb.notImplemented(engine.name, engine.implementation_phase)]
  if (!engine.installed) blockers.push(tb.notInstalled(engine.name))
  else if (runtime && runtime.download_state === 'downloading') blockers.push(tb.downloading)
  else if (runtime && !runtime.weights_installed) blockers.push(tb.noWeights)

  if (!text.trim()) blockers.push(tb.noText)

  for (const spec of config.parameters) {
    if (spec.type === 'text' && !spec.nullable && !String(values[spec.id] ?? '').trim()) blockers.push(tb.requiredParam(spec.label))
  }

  if (caps.requires_reference_audio) {
    if (!reference) blockers.push(tb.noReference)
    else if (reference.status !== 'ANALYZED') blockers.push(tb.referenceNotReady)
    else {
      const hasSegment = reference.segment_start_s != null && reference.segment_end_s != null
      const exempt = caps.reference_text_not_required_when
      const textOptional = !!exempt && Object.entries(exempt).every(([k, v]) => values[k] === v)
      if (caps.requires_reference_text && !textOptional) {
        const transcript = hasSegment ? reference.segment_transcript : reference.transcript
        if (!transcript?.text.trim()) blockers.push(hasSegment ? tb.noSegmentText : tb.noReferenceText)
      }
      const duration = hasSegment ? reference.segment_end_s! - reference.segment_start_s! : (reference.duration_s ?? 0)
      const max = caps.reference_duration_s?.[1]
      if (max != null && duration > max + TOLERANCE_S) blockers.push(tb.referenceTooLong(formatDuration(duration), max))
    }
  }
  return blockers
}
