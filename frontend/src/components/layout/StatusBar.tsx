import { Cpu } from 'lucide-react'

import { t } from '@/i18n/es'
import { cn, formatMegabytes } from '@/lib/utils'
import { useSystemStore } from '@/stores/system'

export function StatusBar() {
  const { connection, version, environment } = useSystemStore()
  const gpu = environment?.gpu.devices[0]

  const dot = {
    connected: 'bg-success',
    disconnected: 'bg-destructive',
    checking: 'bg-warning animate-pulse',
  }[connection]
  const label = {
    connected: t.status.connected,
    disconnected: t.status.disconnected,
    checking: t.status.checking,
  }[connection]

  return (
    <footer className="flex h-7 shrink-0 items-center gap-4 border-t bg-panel px-3 font-mono text-[11px] text-muted-foreground">
      <span className="flex items-center gap-1.5" role="status">
        <span className={cn('size-1.5 rounded-full', dot)} />
        {label}
      </span>
      {environment && (
        <span className="flex items-center gap-1.5">
          <Cpu className="size-3" />
          {gpu
            ? `${gpu.name} · ${formatMegabytes(gpu.free_memory_mb)} ${t.status.free} / ${formatMegabytes(gpu.total_memory_mb)}`
            : t.status.noGpu}
        </span>
      )}
      {version && <span className="ml-auto">v{version}</span>}
    </footer>
  )
}
