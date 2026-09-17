import { Info } from 'lucide-react'
import { useId } from 'react'

import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { t } from '@/i18n/es'
import { cn } from '@/lib/utils'
import type { ParamValue, ParameterSpec } from '@/types/engines'

const te = t.engines

export function ParameterTooltip({ spec }: { spec: ParameterSpec }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button type="button" aria-label={`${spec.label}: ${spec.tooltip.what}`} className="text-muted-foreground hover:text-foreground">
          <Info className="size-3.5" />
        </button>
      </TooltipTrigger>
      <TooltipContent side="left" className="max-w-80">
        <dl className="space-y-1.5">
          {(['what', 'effect', 'cost', 'typical'] as const).map((k) => (
            <div key={k}>
              <dt className="text-[10px] font-semibold tracking-wider text-muted-foreground uppercase">{te.tooltip[k]}</dt>
              <dd>{spec.tooltip[k]}</dd>
            </div>
          ))}
        </dl>
      </TooltipContent>
    </Tooltip>
  )
}

function formatValue(spec: ParameterSpec, value: ParamValue): string {
  if (value == null) return te.auto
  if (typeof value === 'number') {
    const decimals = spec.value_type === 'int' ? 0 : Math.max(0, -Math.floor(Math.log10(spec.step ?? 0.01)))
    return `${value.toFixed(decimals)}${spec.unit ? ` ${spec.unit}` : ''}`
  }
  return String(value)
}

interface ParameterControlProps {
  spec: ParameterSpec
  value: ParamValue
  onChange: (value: ParamValue) => void
  disabled?: boolean
}

/** Renders one backend-described parameter. The frontend never hardcodes engine parameters. */
export function ParameterControl({ spec, value, onChange, disabled }: ParameterControlProps) {
  const id = useId()
  const isDefault = value === spec.default

  const header = (
    <div className="flex items-center gap-1.5 text-xs">
      <label htmlFor={id} className="truncate">
        {spec.label}
      </label>
      <ParameterTooltip spec={spec} />
      {spec.type !== 'toggle' && spec.type !== 'text' && spec.type !== 'select' && (
        <span className={cn('ml-auto font-mono tabular-nums', isDefault ? 'text-muted-foreground' : 'text-primary')}>
          {spec.type === 'seed' ? (value == null ? te.seedRandom : String(value)) : formatValue(spec, value)}
        </span>
      )}
    </div>
  )

  switch (spec.type) {
    case 'slider':
      return (
        <div className="space-y-1.5" data-param={spec.id}>
          {header}
          <input
            id={id}
            type="range"
            min={spec.min ?? 0}
            max={spec.max ?? 1}
            step={spec.step ?? 'any'}
            value={typeof value === 'number' ? value : Number(spec.default ?? spec.min ?? 0)}
            disabled={disabled}
            onChange={(e) => onChange(spec.value_type === 'int' ? parseInt(e.target.value, 10) : parseFloat(e.target.value))}
            onDoubleClick={() => onChange(spec.default)}
            className="h-1.5 w-full cursor-pointer accent-[var(--primary)] disabled:cursor-not-allowed"
          />
        </div>
      )
    case 'number':
      return (
        <div className="space-y-1.5" data-param={spec.id}>
          {header}
          <input
            id={id}
            type="number"
            min={spec.min ?? undefined}
            max={spec.max ?? undefined}
            step={spec.step ?? 'any'}
            value={value == null ? '' : String(value)}
            placeholder={spec.nullable ? te.auto : undefined}
            disabled={disabled}
            onChange={(e) => onChange(e.target.value === '' ? null : Number(e.target.value))}
            className="h-8 w-full rounded-md border bg-background px-2 font-mono text-xs"
          />
        </div>
      )
    case 'seed': {
      const random = value == null
      return (
        <div className="space-y-1.5" data-param={spec.id}>
          {header}
          <div className="flex items-center gap-2">
            <input
              id={id}
              type="number"
              min={0}
              step={1}
              value={random ? '' : String(value)}
              placeholder="123456789"
              disabled={disabled || random}
              onChange={(e) => onChange(e.target.value === '' ? null : parseInt(e.target.value, 10))}
              className="h-8 min-w-0 flex-1 rounded-md border bg-background px-2 font-mono text-xs disabled:opacity-50"
            />
            <label className="flex items-center gap-1.5 text-xs">
              <input
                type="checkbox"
                checked={random}
                disabled={disabled}
                onChange={(e) => onChange(e.target.checked ? null : Math.floor(Math.random() * 2 ** 31))}
                className="accent-[var(--primary)]"
              />
              {te.seedRandom}
            </label>
          </div>
        </div>
      )
    }
    case 'toggle':
      return (
        <div className="flex items-center gap-2" data-param={spec.id}>
          <input
            id={id}
            type="checkbox"
            role="switch"
            aria-checked={Boolean(value)}
            checked={Boolean(value)}
            disabled={disabled}
            onChange={(e) => onChange(e.target.checked)}
            className="accent-[var(--primary)]"
          />
          {header}
        </div>
      )
    case 'select': {
      const selected = spec.options?.find((o) => o.value === value)
      return (
        <div className="space-y-1.5" data-param={spec.id}>
          {header}
          <select
            id={id}
            value={value == null ? '' : String(value)}
            disabled={disabled}
            onChange={(e) => {
              const option = spec.options?.find((o) => String(o.value) === e.target.value)
              onChange(option ? option.value : e.target.value)
            }}
            className="h-8 w-full rounded-md border bg-background px-2 text-xs"
          >
            {spec.options?.map((o) => (
              <option key={String(o.value)} value={String(o.value)}>
                {o.label}
              </option>
            ))}
          </select>
          {selected?.description && <p className="text-[11px] text-muted-foreground">{selected.description}</p>}
        </div>
      )
    }
    case 'text':
      return (
        <div className="space-y-1.5" data-param={spec.id}>
          {header}
          <textarea
            id={id}
            rows={3}
            maxLength={spec.max_length ?? undefined}
            value={value == null ? '' : String(value)}
            disabled={disabled}
            placeholder={spec.tooltip.typical}
            onChange={(e) => onChange(e.target.value === '' && spec.nullable ? null : e.target.value)}
            className="w-full resize-y rounded-md border bg-background px-2 py-1.5 text-xs"
          />
        </div>
      )
  }
}
