import { Badge } from '@/components/ui/badge'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { ParameterControl } from '@/features/models/ParameterControl'
import { t } from '@/i18n/es'
import type { ControlCapability, ControlSource, GenericControl, ParamValue, ParameterSpec } from '@/types/engines'

const te = t.engines

const SOURCE_VARIANT: Record<ControlSource, 'success' | 'accent' | 'warning' | 'default' | 'outline'> = {
  native: 'success',
  instruction: 'accent',
  segmentation: 'warning',
  dsp: 'warning',
  unavailable: 'outline',
}

export function SourceBadge({ capability }: { capability: ControlCapability }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Badge variant={SOURCE_VARIANT[capability.source]} tabIndex={0} className="cursor-help">
          {te.source[capability.source]}
        </Badge>
      </TooltipTrigger>
      <TooltipContent className="space-y-1">
        <p>{capability.reason ?? te.sourceHint[capability.source]}</p>
        {capability.available_from_phase != null && <p className="text-muted-foreground">{te.fromPhase(capability.available_from_phase)}</p>}
      </TooltipContent>
    </Tooltip>
  )
}

interface GenericControlRowProps {
  control: GenericControl
  capability: ControlCapability | undefined
  spec: ParameterSpec | undefined
  value: ParamValue
  onChange: (value: ParamValue) => void
}

/**
 * Engine-independent control. When the engine implements it with a real parameter the parameter is rendered;
 * otherwise the control is shown disabled with the engine's own explanation — never simulated.
 */
export function GenericControlRow({ control, capability, spec, value, onChange }: GenericControlRowProps) {
  if (!capability) return null
  if (spec && capability.parameter === spec.id) {
    return (
      <div className="space-y-1">
        <ParameterControl spec={spec} value={value} onChange={onChange} />
        {capability.source !== 'native' || capability.reason ? (
          <p className="text-[11px] text-muted-foreground">{capability.reason}</p>
        ) : null}
      </div>
    )
  }
  return (
    <div className="flex items-center gap-2 text-xs" data-control={control} aria-disabled>
      <span className="text-muted-foreground">{te.controls[control]}</span>
      <span className="ml-auto">
        <SourceBadge capability={capability} />
      </span>
    </div>
  )
}
