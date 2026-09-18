/** Browser-side helpers for recorded references: mono mix, 16-bit WAV encoding and a quick quality check. */

export interface RecordingStats {
  durationS: number
  peakDb: number
  rmsDb: number
  clippedRatio: number
}

export type RecordingIssue = 'tooShort' | 'tooLong' | 'tooQuiet' | 'clipping' | 'silent'

/** Recommended length for a cloning reference (F5/E2 use up to ~12 s; longer ones can be cut into segments). */
export const MIN_SECONDS = 5
export const MAX_SECONDS = 30
const QUIET_RMS_DB = -35
const CLIP_LEVEL = 0.999
const CLIP_RATIO = 0.001

export const toDb = (value: number) => (value > 0 ? 20 * Math.log10(value) : -Infinity)

export function mixToMono(channels: Float32Array[]): Float32Array {
  if (channels.length === 1) return channels[0]
  const out = new Float32Array(channels[0].length)
  for (const channel of channels) for (let i = 0; i < out.length; i++) out[i] += channel[i] / channels.length
  return out
}

export function analyse(samples: Float32Array, sampleRate: number): RecordingStats {
  let peak = 0
  let sum = 0
  let clipped = 0
  for (const s of samples) {
    const a = Math.abs(s)
    if (a > peak) peak = a
    if (a >= CLIP_LEVEL) clipped++
    sum += s * s
  }
  const n = Math.max(1, samples.length)
  return { durationS: samples.length / sampleRate, peakDb: toDb(peak), rmsDb: toDb(Math.sqrt(sum / n)), clippedRatio: clipped / n }
}

export function issuesOf(stats: RecordingStats): RecordingIssue[] {
  if (!Number.isFinite(stats.peakDb)) return ['silent']
  const issues: RecordingIssue[] = []
  if (stats.durationS < MIN_SECONDS) issues.push('tooShort')
  if (stats.durationS > MAX_SECONDS) issues.push('tooLong')
  if (stats.rmsDb < QUIET_RMS_DB) issues.push('tooQuiet')
  if (stats.clippedRatio > CLIP_RATIO) issues.push('clipping')
  return issues
}

/** PCM 16-bit little-endian mono WAV (what the backend pipeline expects best; no lossy step). */
export function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2)
  const view = new DataView(buffer)
  const write = (offset: number, text: string) => [...text].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)))
  write(0, 'RIFF')
  view.setUint32(4, 36 + samples.length * 2, true)
  write(8, 'WAVE')
  write(12, 'fmt ')
  view.setUint32(16, 16, true) // fmt chunk size
  view.setUint16(20, 1, true) // PCM
  view.setUint16(22, 1, true) // mono
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true) // byte rate
  view.setUint16(32, 2, true) // block align
  view.setUint16(34, 16, true) // bits per sample
  write(36, 'data')
  view.setUint32(40, samples.length * 2, true)
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true)
  }
  return new Blob([buffer], { type: 'audio/wav' })
}

export function recordingName(date = new Date()): string {
  const pad = (n: number) => String(n).padStart(2, '0')
  return `grabacion-${date.getFullYear()}${pad(date.getMonth() + 1)}${pad(date.getDate())}-${pad(date.getHours())}${pad(date.getMinutes())}${pad(date.getSeconds())}.wav`
}
