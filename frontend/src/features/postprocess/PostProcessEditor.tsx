import { AlertTriangle } from 'lucide-react'
import type { ReactNode } from 'react'

import { t } from '@/i18n/es'
import { cn } from '@/lib/utils'
import type { PostProcessCapabilities, PostProcessConfig, ProcessorId } from '@/types/postprocess'

const tp = t.postprocess

interface Props {
  value: PostProcessConfig
  onChange: (value: PostProcessConfig) => void
  capabilities: PostProcessCapabilities | null
  /** Engine parameter that controls speed natively, if any (DSP stretch is then discouraged). */
  nativeSpeedParameter?: string | null
  /** Whether crossfade can apply (segmented generation); undefined = unknown before generating. */
  segmented?: boolean
  idPrefix?: string
}

function Range({ id, label, value, min, max, step, format, onChange, disabled }: {
  id: string
  label: string
  value: number
  min: number
  max: number
  step: number
  format: (v: number) => string
  onChange: (v: number) => void
  disabled?: boolean
}) {
  return (
    <div className="space-y-1">
      <div className="flex items-center text-[11px]">
        <label htmlFor={id} className="text-muted-foreground">
          {label}
        </label>
        <span className="ml-auto font-mono tabular-nums">{format(value)}</span>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="h-1.5 w-full cursor-pointer accent-[var(--primary)] disabled:cursor-not-allowed disabled:opacity-50"
      />
    </div>
  )
}

/** Controlled editor for the optional post-processing chain. Every processor starts disabled. */
export function PostProcessEditor({ value, onChange, capabilities, nativeSpeedParameter, segmented, idPrefix = 'pp' }: Props) {
  const patch = <K extends ProcessorId>(key: K, update: Partial<PostProcessConfig[K]>) => onChange({ ...value, [key]: { ...value[key], ...update } })
  const available = (key: ProcessorId) => capabilities?.processors[key] ?? true

  const section = (key: ProcessorId, body: ReactNode, warning?: string | null) => {
    const usable = available(key)
    const enabled = value[key].enabled && usable
    return (
      <div className="space-y-1.5" data-processor={key}>
        <label className={cn('flex items-start gap-2 text-xs', !usable && 'text-muted-foreground')} title={usable ? tp.hints[key] : capabilities?.reasons[key]}>
          <input
            type="checkbox"
            className="mt-0.5 accent-[var(--primary)]"
            checked={enabled}
            disabled={!usable}
            onChange={(e) => patch(key, { enabled: e.target.checked } as Partial<PostProcessConfig[typeof key]>)}
          />
          <span className="flex-1">
            <span className="font-medium">{tp.labels[key]}</span>
            <span className="block text-[11px] text-muted-foreground">{usable ? tp.hints[key] : capabilities?.reasons[key]}</span>
          </span>
        </label>
        {enabled && <div className="space-y-2 pl-5">{body}</div>}
        {enabled && warning && (
          <p className="flex items-start gap-1.5 pl-5 text-[11px] text-warning">
            <AlertTriangle className="mt-px size-3 shrink-0" />
            {warning}
          </p>
        )}
      </div>
    )
  }

  const crossfadeOn = value.crossfade_ms != null

  return (
    <div className="space-y-3">
      <div className="space-y-1.5" data-processor="crossfade">
        <label className={cn('flex items-start gap-2 text-xs', segmented === false && 'text-muted-foreground')}>
          <input
            type="checkbox"
            className="mt-0.5 accent-[var(--primary)]"
            checked={crossfadeOn}
            disabled={segmented === false}
            onChange={(e) => onChange({ ...value, crossfade_ms: e.target.checked ? 40 : null })}
          />
          <span className="flex-1">
            <span className="font-medium">{tp.labels.crossfade}</span>
            <span className="block text-[11px] text-muted-foreground">{segmented === false ? tp.crossfadeNotSegmented : tp.hints.crossfade}</span>
          </span>
        </label>
        {crossfadeOn && (
          <div className="pl-5">
            <Range id={`${idPrefix}-crossfade`} label={tp.fields.duration} value={value.crossfade_ms ?? 40} min={0} max={300} step={10} format={(v) => `${v} ms`} onChange={(v) => onChange({ ...value, crossfade_ms: v })} />
          </div>
        )}
      </div>

      {section(
        'denoise',
        <div className="flex gap-1" role="radiogroup" aria-label={tp.fields.strength}>
          {(['light', 'medium', 'strong'] as const).map((s) => (
            <button
              key={s}
              type="button"
              role="radio"
              aria-checked={value.denoise.strength === s}
              onClick={() => patch('denoise', { strength: s })}
              className={cn('flex-1 rounded-md border px-2 py-1 text-[11px]', value.denoise.strength === s ? 'border-primary bg-primary/10' : 'text-muted-foreground')}
            >
              {tp.strength[s]}
            </button>
          ))}
        </div>,
        value.denoise.strength === 'strong' ? tp.warnings.denoiseStrong : null,
      )}

      {section(
        'trim_silence',
        <>
          <Range id={`${idPrefix}-trim-th`} label={tp.fields.threshold} value={value.trim_silence.threshold_db} min={-70} max={-20} step={1} format={(v) => `${v} dBFS`} onChange={(v) => patch('trim_silence', { threshold_db: v })} />
          <Range id={`${idPrefix}-trim-pad`} label={tp.fields.padding} value={value.trim_silence.padding_ms} min={0} max={1000} step={10} format={(v) => `${v} ms`} onChange={(v) => patch('trim_silence', { padding_ms: v })} />
        </>,
      )}

      {section(
        'time_stretch',
        <Range id={`${idPrefix}-rate`} label={tp.fields.rate} value={value.time_stretch.rate} min={0.5} max={2} step={0.05} format={(v) => `×${v.toFixed(2)}`} onChange={(v) => patch('time_stretch', { rate: v })} />,
        nativeSpeedParameter ? tp.warnings.nativeSpeed(nativeSpeedParameter) : Math.abs(value.time_stretch.rate - 1) > 0.25 ? tp.warnings.largeStretch : null,
      )}

      {section(
        'pitch_shift',
        <>
          <Range id={`${idPrefix}-semitones`} label={tp.fields.semitones} value={value.pitch_shift.semitones} min={-12} max={12} step={0.5} format={(v) => `${v > 0 ? '+' : ''}${v} st`} onChange={(v) => patch('pitch_shift', { semitones: v })} />
          <label className="flex items-center gap-2 text-[11px]">
            <input type="checkbox" className="accent-[var(--primary)]" checked={value.pitch_shift.preserve_formants} onChange={(e) => patch('pitch_shift', { preserve_formants: e.target.checked })} />
            {tp.fields.preserveFormants}
          </label>
        </>,
        Math.abs(value.pitch_shift.semitones) > 4 ? tp.warnings.largePitch : null,
      )}

      {section(
        'loudness',
        <Range id={`${idPrefix}-lufs`} label={tp.fields.target} value={value.loudness.target_lufs} min={-35} max={-10} step={1} format={(v) => `${v} LUFS`} onChange={(v) => patch('loudness', { target_lufs: v })} />,
      )}

      {section(
        'peak',
        <Range id={`${idPrefix}-peak`} label={value.loudness.enabled ? tp.fields.ceiling : tp.fields.target} value={value.peak.target_dbfs} min={-12} max={0} step={0.5} format={(v) => `${v} dBFS`} onChange={(v) => patch('peak', { target_dbfs: v })} />,
      )}

      {section(
        'fades',
        <>
          <Range id={`${idPrefix}-fin`} label={tp.fields.fadeIn} value={value.fades.fade_in_ms} min={0} max={2000} step={5} format={(v) => `${v} ms`} onChange={(v) => patch('fades', { fade_in_ms: v })} />
          <Range id={`${idPrefix}-fout`} label={tp.fields.fadeOut} value={value.fades.fade_out_ms} min={0} max={2000} step={5} format={(v) => `${v} ms`} onChange={(v) => patch('fades', { fade_out_ms: v })} />
        </>,
      )}
    </div>
  )
}
