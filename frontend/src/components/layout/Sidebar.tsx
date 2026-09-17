import {
  AudioWaveform,
  Boxes,
  FlaskConical,
  FolderKanban,
  Keyboard,
  Library,
  Mic2,
  Moon,
  PanelLeftClose,
  PanelLeftOpen,
  Settings2,
  Sun,
  type LucideIcon,
} from 'lucide-react'

import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { t } from '@/i18n/es'
import { cn } from '@/lib/utils'
import { type Section, type UiMode, useUiStore } from '@/stores/ui'

const NAV: { id: Section; label: string; icon: LucideIcon }[] = [
  { id: 'generate', label: t.nav.generate, icon: AudioWaveform },
  { id: 'library', label: t.nav.library, icon: Library },
  { id: 'voices', label: t.nav.voices, icon: Mic2 },
  { id: 'experiments', label: t.nav.experiments, icon: FlaskConical },
  { id: 'projects', label: t.nav.projects, icon: FolderKanban },
  { id: 'models', label: t.nav.models, icon: Boxes },
]

function NavButton({ id, label, icon: Icon }: { id: Section; label: string; icon: LucideIcon }) {
  const { section, setSection, sidebarCollapsed } = useUiStore()
  const active = section === id
  const button = (
    <button
      type="button"
      onClick={() => setSection(id)}
      aria-current={active ? 'page' : undefined}
      className={cn(
        'group relative flex h-10 w-full items-center gap-3 rounded-lg px-3 text-sm transition-colors',
        active ? 'bg-muted text-foreground' : 'text-muted-foreground hover:bg-muted/60 hover:text-foreground',
        sidebarCollapsed && 'justify-center px-0',
      )}
    >
      {active && <span className="absolute top-2 bottom-2 left-0 w-0.5 rounded-full bg-primary" />}
      <Icon className={cn('size-[18px] shrink-0', active && 'text-primary')} />
      {!sidebarCollapsed && <span>{label}</span>}
    </button>
  )
  if (!sidebarCollapsed) return button
  return (
    <Tooltip>
      <TooltipTrigger asChild>{button}</TooltipTrigger>
      <TooltipContent side="right">{label}</TooltipContent>
    </Tooltip>
  )
}

export function Sidebar() {
  const { sidebarCollapsed, toggleSidebar, mode, setMode, theme, toggleTheme } = useUiStore()

  return (
    <aside
      className={cn(
        'flex h-full shrink-0 flex-col border-r bg-panel transition-[width] duration-200',
        sidebarCollapsed ? 'w-16' : 'w-56',
      )}
    >
      <div className={cn('flex h-14 items-center gap-2.5 px-4', sidebarCollapsed && 'justify-center px-0')}>
        <img src="/favicon.svg" alt="" className="size-7" />
        {!sidebarCollapsed && (
          <div className="leading-tight">
            <div className="text-[15px] font-semibold tracking-tight">
              {t.app.name} <span className="text-primary">AI</span>
            </div>
            <div className="text-[11px] text-muted-foreground">{t.app.tagline}</div>
          </div>
        )}
      </div>

      <nav className="flex flex-1 flex-col gap-1 px-2 pt-3">
        {NAV.map((item) => (
          <NavButton key={item.id} {...item} />
        ))}
        <div className="mt-auto space-y-1 pb-2">
          <button
            type="button"
            aria-label={t.shortcuts.open}
            title={t.shortcuts.open}
            onClick={() => window.dispatchEvent(new KeyboardEvent('keydown', { key: '?' }))}
            className={cn('flex h-10 w-full items-center gap-3 rounded-lg px-3 text-sm text-muted-foreground hover:bg-muted/60 hover:text-foreground',
              sidebarCollapsed && 'justify-center px-0')}
          >
            <Keyboard className="size-[18px] shrink-0" />
            {!sidebarCollapsed && <span>{t.shortcuts.title}</span>}
          </button>
          <NavButton id="settings" label={t.nav.settings} icon={Settings2} />
        </div>
      </nav>

      <div className="space-y-2 border-t p-2">
        {!sidebarCollapsed && (
          <div role="radiogroup" aria-label={t.mode.label} className="grid grid-cols-2 rounded-lg bg-muted p-0.5 text-xs">
            {(['simple', 'advanced'] as UiMode[]).map((m) => (
              <Tooltip key={m}>
                <TooltipTrigger asChild>
                  <button
                    type="button"
                    role="radio"
                    aria-checked={mode === m}
                    onClick={() => setMode(m)}
                    className={cn(
                      'h-7 rounded-md transition-colors',
                      mode === m ? 'bg-card text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground',
                    )}
                  >
                    {m === 'simple' ? t.mode.simple : t.mode.advanced}
                  </button>
                </TooltipTrigger>
                <TooltipContent side="top">{m === 'simple' ? t.mode.simpleHint : t.mode.advancedHint}</TooltipContent>
              </Tooltip>
            ))}
          </div>
        )}
        <div className={cn('flex gap-1', sidebarCollapsed ? 'flex-col items-center' : 'justify-between')}>
          <button
            type="button"
            onClick={toggleTheme}
            aria-label={t.theme.toggle}
            className="grid size-8 place-items-center rounded-md text-muted-foreground hover:bg-muted hover:text-foreground"
          >
            {theme === 'dark' ? <Sun className="size-4" /> : <Moon className="size-4" />}
          </button>
          <button
            type="button"
            onClick={toggleSidebar}
            aria-label={sidebarCollapsed ? t.nav.expand : t.nav.collapse}
            className="grid size-8 place-items-center rounded-md text-muted-foreground hover:bg-muted hover:text-foreground"
          >
            {sidebarCollapsed ? <PanelLeftOpen className="size-4" /> : <PanelLeftClose className="size-4" />}
          </button>
        </div>
      </div>
    </aside>
  )
}
