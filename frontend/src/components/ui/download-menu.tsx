import { ChevronDown, Download } from 'lucide-react'
import { type ReactNode, useEffect, useState } from 'react'

import { t } from '@/i18n/es'
import { request } from '@/services/api'
import { cn } from '@/lib/utils'

export type AudioFormat = 'wav' | 'mp3' | 'ogg' | 'flac'

interface FormatInfo {
  id: AudioFormat
  label: string
  available: boolean
  reason: string | null
}

const FALLBACK: FormatInfo[] = (['wav', 'mp3', 'ogg', 'flac'] as const).map((id) => ({ id, label: t.download.formats[id], available: true, reason: null }))
let cached: Promise<FormatInfo[]> | null = null

/** Which formats this installation can deliver (MP3/OGG/FLAC need FFmpeg). Fetched once per page load. */
export function useExportFormats(): FormatInfo[] {
  const [formats, setFormats] = useState<FormatInfo[]>(FALLBACK)
  useEffect(() => {
    let active = true
    cached ??= request<FormatInfo[]>('/generation/export/formats').catch(() => FALLBACK)
    void cached.then((list) => active && Array.isArray(list) && setFormats(list))
    return () => {
      active = false
    }
  }, [])
  return formats
}

export interface MenuItem {
  key: string
  label: string
  href: string
  disabled?: boolean
  title?: string
}

/** A small disclosure menu of download links (native <details>: keyboard and screen-reader friendly). */
export function DownloadMenu({ items, label, icon, compact = false, disabled = false, disabledTitle }: {
  items: MenuItem[]
  label: string
  icon?: ReactNode
  compact?: boolean
  disabled?: boolean
  disabledTitle?: string
}) {
  if (disabled) {
    return (
      <span className="inline-flex h-8 cursor-not-allowed items-center gap-1.5 rounded-md border px-3 text-sm opacity-50" title={disabledTitle} aria-disabled="true">
        {icon ?? <Download className="size-4" />}
        {!compact && label}
      </span>
    )
  }
  return (
    <details className="group relative inline-block" data-testid="download-menu">
      <summary
        aria-label={label}
        title={label}
        className={cn(
          'flex cursor-pointer list-none items-center gap-1 rounded-md text-sm text-muted-foreground hover:text-foreground [&::-webkit-details-marker]:hidden',
          compact ? 'size-7 justify-center' : 'h-8 border px-3 text-foreground hover:bg-muted/50',
        )}
      >
        {icon ?? <Download className="size-4" />}
        {!compact && (
          <>
            {label}
            <ChevronDown className="size-3.5 transition-transform group-open:rotate-180" />
          </>
        )}
      </summary>
      <ul className="absolute right-0 z-20 mt-1 min-w-48 rounded-md border bg-panel p-1 shadow-lg">
        {items.map((item) => (
          <li key={item.key}>
            {item.disabled ? (
              <span className="block cursor-not-allowed rounded px-2 py-1.5 text-xs opacity-50" title={item.title}>
                {item.label}
              </span>
            ) : (
              <a href={item.href} className="block rounded px-2 py-1.5 text-xs hover:bg-muted" title={item.title}>
                {item.label}
              </a>
            )}
          </li>
        ))}
      </ul>
    </details>
  )
}

/** Download menu for a generated audio URL in every available format. */
export function AudioDownloadMenu({ url, compact = true }: { url: string; compact?: boolean }) {
  const formats = useExportFormats()
  const join = url.includes('?') ? '&' : '?'
  return (
    <DownloadMenu
      compact={compact}
      label={t.generation.download}
      items={formats.map((f) => ({
        key: f.id,
        label: f.label,
        href: `${url}${join}download=true${f.id === 'wav' ? '' : `&format=${f.id}`}`,
        disabled: !f.available,
        title: f.reason ?? undefined,
      }))}
    />
  )
}
