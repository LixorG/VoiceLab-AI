export interface ScriptChange {
  kind: string
  before: string
  after: string
  reason: string
}

export interface ScriptAlert {
  kind: string
  message: string
  excerpt: string | null
}

export interface ScriptSuggestion {
  before: string
  after: string
  reason: string
}

export interface ScriptStats {
  words: number
  sentences: number
  paragraphs: number
  average_words: number
  longest_words: number
  seconds: number
}

export interface ScriptPrepared {
  text: string
  changed: boolean
  language: string | null
  changes: ScriptChange[]
  alerts: ScriptAlert[]
  suggestions: ScriptSuggestion[]
  stats: ScriptStats
}

export interface ScriptPrepareInput {
  text: string
  language?: string | null
  params?: Record<string, unknown>
  profile_id?: string | null
}
