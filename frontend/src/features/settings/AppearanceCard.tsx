import { Monitor, Moon, Sun } from 'lucide-react'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { t } from '@/i18n/es'
import { cn } from '@/lib/utils'
import { type Theme, resolveTheme, useUiStore } from '@/stores/ui'

const OPTIONS: { id: Theme; icon: typeof Sun }[] = [
  { id: 'light', icon: Sun },
  { id: 'dark', icon: Moon },
  { id: 'system', icon: Monitor },
]

/** Day / night / follow the system, with the same switch available from the sidebar icon. */
export function AppearanceCard() {
  const { theme, setTheme } = useUiStore()
  const resolved = resolveTheme(theme)

  return (
    <Card>
      <CardHeader>
        <CardTitle>{t.theme.title}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        <div className="flex gap-2" role="radiogroup" aria-label={t.theme.title}>
          {OPTIONS.map(({ id, icon: Icon }) => (
            <button
              key={id}
              type="button"
              role="radio"
              aria-checked={theme === id}
              onClick={() => setTheme(id)}
              className={cn(
                'flex flex-1 items-center justify-center gap-2 rounded-md border px-3 py-2 text-sm transition-colors',
                theme === id ? 'border-primary bg-primary/10 text-foreground' : 'text-muted-foreground hover:bg-muted',
              )}
            >
              <Icon className="size-4" />
              {t.theme.options[id]}
            </button>
          ))}
        </div>
        <p className="text-xs text-muted-foreground">
          {theme === 'system' ? t.theme.systemHint(t.theme.options[resolved]) : t.theme.hint}
        </p>
      </CardContent>
    </Card>
  )
}
