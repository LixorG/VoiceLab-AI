import { Suspense, lazy, useEffect } from 'react'

import { Sidebar } from '@/components/layout/Sidebar'
import { StatusBar } from '@/components/layout/StatusBar'
import { TooltipProvider } from '@/components/ui/tooltip'
import { Skeleton } from '@/components/ui/skeleton'
import { GeneratePage } from '@/features/generation/GeneratePage'
import { LiveStatus, Shortcuts } from '@/features/shortcuts/Shortcuts'
import { t } from '@/i18n/es'

// The other sections load on demand: the first paint only pays for «Generar».
const LibraryPage = lazy(() => import('@/features/library/LibraryPage').then((m) => ({ default: m.LibraryPage })))
const VoicesPage = lazy(() => import('@/features/voices/VoicesPage').then((m) => ({ default: m.VoicesPage })))
const ExperimentsPage = lazy(() => import('@/features/experiments/ExperimentsPage').then((m) => ({ default: m.ExperimentsPage })))
const ProjectsPage = lazy(() => import('@/features/projects/ProjectsPage').then((m) => ({ default: m.ProjectsPage })))
const ModelsPage = lazy(() => import('@/features/models/ModelsPage').then((m) => ({ default: m.ModelsPage })))
const SettingsPage = lazy(() => import('@/features/settings/SettingsPage').then((m) => ({ default: m.SettingsPage })))
import { useSystemStore } from '@/stores/system'
import { type Section, resolveTheme, useUiStore } from '@/stores/ui'

function SectionView({ section }: { section: Section }) {
  switch (section) {
    case 'generate':
      return <GeneratePage />
    case 'library':
      return <LibraryPage />
    case 'voices':
      return <VoicesPage />
    case 'experiments':
      return <ExperimentsPage />
    case 'projects':
      return <ProjectsPage />
    case 'models':
      return <ModelsPage />
    case 'settings':
      return <SettingsPage />
  }
}

export default function App() {
  const section = useUiStore((s) => s.section)
  const theme = useUiStore((s) => s.theme)
  const { checkHealth, loadEnvironment } = useSystemStore()

  // `color-scheme` makes the browser paint its own widgets (native selects, date pickers, scrollbars) to match.
  useEffect(() => {
    const media = window.matchMedia?.('(prefers-color-scheme: dark)')
    const apply = () => {
      const resolved = resolveTheme(theme)
      document.documentElement.classList.toggle('dark', resolved === 'dark')
      document.documentElement.style.colorScheme = resolved
    }
    apply()
    if (!media) return  // a browser (or test environment) without media queries: the chosen theme still applies
    if (theme !== 'system') return
    media.addEventListener('change', apply)
    return () => media.removeEventListener('change', apply)
  }, [theme])

  useEffect(() => {
    void checkHealth().then(() => loadEnvironment())
    const id = window.setInterval(() => void checkHealth(), 15_000)
    return () => window.clearInterval(id)
  }, [checkHealth, loadEnvironment])

  return (
    <TooltipProvider delayDuration={300}>
      <div className="flex h-full flex-col">
        <a
          href="#contenido"
          className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 focus:rounded-md focus:border focus:bg-panel focus:px-3 focus:py-2 focus:text-sm"
        >
          {t.a11y.skipToContent}
        </a>
        <div className="flex min-h-0 flex-1">
          <Sidebar />
          {/* `relative`: anything positioned inside a section belongs to the section, never to the page */}
          <main id="contenido" className="relative min-h-0 min-w-0 flex-1">
            <Suspense fallback={<div className="p-6"><Skeleton className="h-64" /></div>}>
              <SectionView section={section} />
            </Suspense>
          </main>
        </div>
        <StatusBar />
        <Shortcuts />
        <LiveStatus />
      </div>
    </TooltipProvider>
  )
}
