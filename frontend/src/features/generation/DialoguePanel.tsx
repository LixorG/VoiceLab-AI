import { MessagesSquare } from 'lucide-react'

import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { t } from '@/i18n/es'
import { useGenerationStore } from '@/stores/generation'
import { useProfilesStore } from '@/stores/profiles'

const td = t.dialogue

/** Appears on its own when the text has characters («Ana: …»): one voice per character and the turn pause. */
export function DialoguePanel() {
  const { plan, speakers, setSpeaker, turnPauseMs, setTurnPause } = useGenerationStore()
  const profiles = useProfilesStore((s) => s.items)
  const detected = plan?.detected_speakers ?? []
  if (detected.length === 0) return null

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <MessagesSquare className="size-4 text-muted-foreground" />
          {td.title}
        </CardTitle>
        <span className="text-xs text-muted-foreground">{td.format}</span>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-xs text-muted-foreground">{td.hint}</p>
        <ul className="space-y-1.5">
          {detected.map((speaker) => (
            <li key={speaker} className="flex flex-wrap items-center gap-2 text-sm">
              <Badge variant="accent">{speaker}</Badge>
              <select
                aria-label={td.voice(speaker)}
                value={speakers[speaker] ?? ''}
                onChange={(e) => setSpeaker(speaker, e.target.value || null)}
                className="h-8 min-w-40 rounded-md border bg-background px-2 text-sm"
              >
                <option value="">{td.choose}</option>
                {profiles.map((profile) => (
                  <option key={profile.id} value={profile.id}>
                    {profile.name}
                  </option>
                ))}
              </select>
              {!speakers[speaker] && <span className="text-xs text-warning">{td.missing}</span>}
            </li>
          ))}
        </ul>
        <label className="flex items-center gap-2 text-xs text-muted-foreground" title={td.turnPauseHint}>
          {td.turnPause}
          <input
            type="number"
            min={0}
            max={5000}
            step={50}
            value={turnPauseMs}
            onChange={(e) => setTurnPause(Number(e.target.value))}
            aria-label={td.turnPause}
            className="h-8 w-24 rounded-md border bg-background px-2 font-mono text-sm"
          />
          ms
        </label>
      </CardContent>
    </Card>
  )
}
