export interface PronunciationEntry {
  id: string
  term: string
  replacement: string
  case_sensitive: boolean
  profile_id: string | null
  created_at: string
  updated_at: string
}

export interface PronunciationInput {
  term: string
  replacement: string
  case_sensitive: boolean
  profile_id: string | null
}

export interface TextChange {
  original: string
  replacement: string
  kind: string
}

export interface NormalizePreview {
  text: string
  changes: TextChange[]
  language: string | null
  numbers_supported: boolean
}
