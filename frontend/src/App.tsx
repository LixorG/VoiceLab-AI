import { useEffect } from 'react'

import { Sidebar } from '@/components/layout/Sidebar'
import { StatusBar } from '@/components/layout/StatusBar'
import { TooltipProvider } from '@/components/ui/tooltip'
import { ExperimentsPage } from '@/features/experiments/ExperimentsPage'
import { GeneratePage } from '@/features/generation/GeneratePage'
import { LibraryPage } from '@/features/library/LibraryPage'
import { ProjectsPage } from '@/features/projects/ProjectsPage'
import { ModelsPage } from '@/features/models/ModelsPage'
import { SettingsPage } from '@/features/settings/SettingsPage'
import { VoicesPage } from '@/features/voices/VoicesPage'
import { useSystemStore } from '@/stores/system'
import { type Section, useUiStore } from '@/stores/ui'

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

  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark')
  }, [theme])

  useEffect(() => {
    void checkHealth().then(() => loadEnvironment())
    const id = window.setInterval(() => void checkHealth(), 15_000)
    return () => window.clearInterval(id)
  }, [checkHealth, loadEnvironment])

  return (
    <TooltipProvider delayDuration={300}>
      <div className="flex h-full flex-col">
        <div className="flex min-h-0 flex-1">
          <Sidebar />
          <main className="min-w-0 flex-1">
            <SectionView section={section} />
          </main>
        </div>
        <StatusBar />
      </div>
    </TooltipProvider>
  )
}
