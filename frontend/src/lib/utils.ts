import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatMegabytes(mb: number | null | undefined): string {
  if (mb == null) return '—'
  return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb} MB`
}

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return '—'
  const total = Math.max(0, seconds)
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = total % 60
  const ss = h || m >= 1 || total >= 10 ? Math.floor(s).toString().padStart(2, '0') : s.toFixed(1).padStart(4, '0')
  return h ? `${h}:${m.toString().padStart(2, '0')}:${ss}` : `${m}:${ss}`
}

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes == null) return '—'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(0)} KB`
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`
}

export function formatDb(value: number | null | undefined, unit = 'dB'): string {
  if (value == null || !Number.isFinite(value)) return '—'
  return `${value.toFixed(1)} ${unit}`
}
