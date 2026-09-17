import { describe, expect, it } from 'vitest'

import { formatBytes, formatDb, formatDuration } from '@/lib/utils'

describe('formatters', () => {
  it('formats durations', () => {
    expect(formatDuration(2.34)).toBe('0:02.3')
    expect(formatDuration(65.9)).toBe('1:05')
    expect(formatDuration(3725)).toBe('1:02:05')
    expect(formatDuration(null)).toBe('—')
  })

  it('formats bytes and dB', () => {
    expect(formatBytes(512)).toBe('512 B')
    expect(formatBytes(2048)).toBe('2 KB')
    expect(formatBytes(5 * 1024 ** 2)).toBe('5.0 MB')
    expect(formatDb(-23.456, 'LUFS')).toBe('-23.5 LUFS')
    expect(formatDb(null)).toBe('—')
  })
})
