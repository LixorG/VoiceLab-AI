import { fireEvent, render, screen } from '@testing-library/react'
import { useState } from 'react'
import { describe, expect, it } from 'vitest'

import { PostProcessEditor } from '@/features/postprocess/PostProcessEditor'
import { DEFAULT_POSTPROCESS, type PostProcessConfig } from '@/types/postprocess'

let latest: PostProcessConfig = DEFAULT_POSTPROCESS

function Harness({ segmented }: { segmented?: boolean }) {
  const [value, setValue] = useState<PostProcessConfig>(DEFAULT_POSTPROCESS)
  return (
    <PostProcessEditor
      value={value}
      onChange={(v) => {
        latest = v
        setValue(v)
      }}
      capabilities={null}
      segmented={segmented}
    />
  )
}

const toggle = (label: string) => fireEvent.click(screen.getByText(label).closest('label')!.querySelector('input')!)
const slide = (label: string, value: number) => fireEvent.change(screen.getByLabelText(label), { target: { value: String(value) } })

describe('PostProcessEditor', () => {
  it('edits every processor and only shows controls for the enabled ones', () => {
    render(<Harness segmented />)
    expect(screen.queryByLabelText('Umbral de silencio')).not.toBeInTheDocument()

    toggle('Crossfade entre segmentos')
    slide('Duración', 120)
    expect(latest.crossfade_ms).toBe(120)

    toggle('Limpieza de ruido')
    fireEvent.click(screen.getByRole('radio', { name: 'Fuerte' }))
    expect(latest.denoise).toEqual({ enabled: true, strength: 'strong' })
    expect(screen.getByText(/artefactos metálicos/)).toBeInTheDocument()

    toggle('Recortar silencios')
    slide('Umbral de silencio', -50)
    slide('Margen conservado', 200)
    expect(latest.trim_silence).toEqual({ enabled: true, threshold_db: -50, padding_ms: 200 })

    toggle('Velocidad (DSP)')
    slide('Factor de velocidad', 1.5)
    expect(latest.time_stretch).toEqual({ enabled: true, rate: 1.5 })
    expect(screen.getByText(/pueden sonar artificiales/)).toBeInTheDocument()

    toggle('Tono (DSP)')
    slide('Semitonos', -7)
    toggle('Conservar formantes (timbre más natural)')
    expect(latest.pitch_shift).toEqual({ enabled: true, semitones: -7, preserve_formants: false })
    expect(screen.getByText(/alteran notablemente el timbre/)).toBeInTheDocument()

    toggle('Normalizar pico')
    expect(screen.getByLabelText('Objetivo')).toBeInTheDocument() // without loudness the peak is a target…
    toggle('Normalizar sonoridad')
    slide('Objetivo', -18)
    expect(screen.getByLabelText('Techo')).toBeInTheDocument() // …with loudness it becomes a ceiling
    slide('Techo', -2)
    expect(latest.loudness).toEqual({ enabled: true, target_lufs: -18 })
    expect(latest.peak).toEqual({ enabled: true, target_dbfs: -2 })

    toggle('Fundidos de entrada y salida')
    slide('Entrada', 50)
    slide('Salida', 400)
    expect(latest.fades).toEqual({ enabled: true, fade_in_ms: 50, fade_out_ms: 400 })

    toggle('Crossfade entre segmentos')
    expect(latest.crossfade_ms).toBeNull()
  })

  it('explains that crossfade does not apply to a single-segment generation', () => {
    render(<Harness segmented={false} />)
    expect(screen.getByText(/no tiene varios segmentos/)).toBeInTheDocument()
    expect(screen.getByText('Crossfade entre segmentos').closest('label')!.querySelector('input')).toBeDisabled()
  })
})
