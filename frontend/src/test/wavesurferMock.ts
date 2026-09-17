/**
 * In-memory WaveSurfer + Regions plugin for jsdom (no canvas, no media decoding).
 * Usage in a test file:
 *   vi.mock('wavesurfer.js', async () => (await import('@/test/wavesurferMock')).wavesurferModule)
 *   vi.mock('wavesurfer.js/dist/plugins/regions.esm.js', async () => (await import('@/test/wavesurferMock')).regionsModule)
 */
type Handler = (...args: never[]) => void

class Emitter {
  private handlers = new Map<string, Handler[]>()

  on(event: string, handler: Handler) {
    this.handlers.set(event, [...(this.handlers.get(event) ?? []), handler])
    return () => this.handlers.set(event, (this.handlers.get(event) ?? []).filter((h) => h !== handler))
  }

  emit(event: string, ...args: unknown[]) {
    for (const handler of this.handlers.get(event) ?? []) (handler as (...a: unknown[]) => void)(...args)
  }
}

export class FakeRegion {
  played = 0
  constructor(
    private owner: FakeRegions,
    public start: number,
    public end: number,
  ) {}

  play() {
    this.played++
  }

  remove() {
    this.owner.regions = this.owner.regions.filter((r) => r !== this)
  }
}

export class FakeRegions extends Emitter {
  static instances: FakeRegions[] = []
  regions: FakeRegion[] = []
  dragEnabled = false

  static create() {
    const plugin = new FakeRegions()
    FakeRegions.instances.push(plugin)
    return plugin
  }

  addRegion({ start, end }: { start: number; end: number }) {
    const region = new FakeRegion(this, start, end)
    this.regions.push(region)
    this.emit('region-created', region)
    return region
  }

  getRegions() {
    return this.regions
  }

  clearRegions() {
    this.regions = []
  }

  enableDragSelection() {
    this.dragEnabled = true
    return () => {
      this.dragEnabled = false
    }
  }

  /** Simulates the user dragging a new range. */
  userSelect(start: number, end: number) {
    this.addRegion({ start, end })
  }
}

export class FakeWaveSurfer extends Emitter {
  static instances: FakeWaveSurfer[] = []
  options: Record<string, unknown>
  time = 0
  playing = false
  muted = false
  zoomLevel = 0
  destroyed = false
  duration: number

  constructor(options: Record<string, unknown>) {
    super()
    this.options = options
    this.duration = (options.duration as number | undefined) ?? 0
  }

  static create(options: Record<string, unknown>) {
    const ws = new FakeWaveSurfer(options)
    FakeWaveSurfer.instances.push(ws)
    return ws
  }

  /** Simulates decoding finishing. */
  ready(duration = this.duration) {
    this.duration = duration
    this.emit('ready', duration)
  }

  play() {
    this.playing = true
    this.emit('play')
    return Promise.resolve()
  }

  pause() {
    if (!this.playing) return
    this.playing = false
    this.emit('pause')
  }

  playPause() {
    return this.playing ? this.pause() : this.play()
  }

  setTime(seconds: number) {
    this.time = seconds
    this.emit('timeupdate', seconds)
  }

  getCurrentTime() {
    return this.time
  }

  setMuted(muted: boolean) {
    this.muted = muted
  }

  zoom(pxPerSec: number) {
    this.zoomLevel = pxPerSec
  }

  destroy() {
    this.destroyed = true
  }
}

export function resetWaveSurferMock() {
  FakeWaveSurfer.instances = []
  FakeRegions.instances = []
}

export const wavesurferModule = { default: FakeWaveSurfer }
export const regionsModule = { default: FakeRegions }
