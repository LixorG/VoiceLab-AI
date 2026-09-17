export interface CustomCheckpoint {
  id: string
  engine: string
  variant: string
  name: string
  base_variant: string
  repo_id: string | null
  ckpt_file: string | null
  local_path: string | null
  vocab_file: string | null
  vocab_path: string | null
  languages: string[]
  notes: string | null
  weights_installed: boolean
  created_at: string
}

export interface CheckpointInput {
  name: string
  base_variant: string
  repo_id: string | null
  ckpt_file: string | null
  local_path: string | null
  vocab_file: string | null
  vocab_path: string | null
  languages: string[]
  notes: string | null
}
