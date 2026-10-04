import { GRID, composite, compositeSideBySide } from '../lib/imaging'
import type { Plane } from '../types'

export interface DrawArgs {
  canvas: HTMLCanvasElement
  fixed: Float32Array
  moving: Float32Array
  plane: Plane
  sliceT: number
  level: number
  width: number
  blend: number
  viewMode: string
  showCrosshair: boolean
  showLandmarks: boolean
  status: string
  landmarks: Array<{
    label: string
    treMm: number
    pos: [number, number, number]
    movingPos: [number, number, number]
  }>
}

export function orientationMarks(plane: Plane) {
  if (plane === 'axial') return { top: 'A', bottom: 'P', left: 'L', right: 'R' }
  if (plane === 'coronal') return { top: 'S', bottom: 'I', left: 'L', right: 'R' }
  return { top: 'S', bottom: 'I', left: 'A', right: 'P' }
}

/** Composite the pixel buffers and paint the full viewer frame. */
export function drawViewer(a: DrawArgs) {
  const { canvas } = a
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const dpr = Math.min(window.devicePixelRatio || 1, 2)
  const cssW = canvas.clientWidth
  const cssH = canvas.clientHeight
  if (cssW === 0 || cssH === 0) return

  if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {
    canvas.width = Math.round(cssW * dpr)
    canvas.height = Math.round(cssH * dpr)
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, cssW, cssH)

  // Native-resolution offscreen composite, then scaled up.
  const off = document.createElement('canvas')
  off.width = GRID
  off.height = GRID
  const offCtx = off.getContext('2d')
  if (!offCtx) return
  const img = offCtx.createImageData(GRID, GRID)

  if (a.viewMode === 'side-by-side') {
    compositeSideBySide(a.fixed, a.moving, a.level, a.width, img.data)
  } else {
    composite(
      a.fixed,
      a.moving,
      {
        mode: a.viewMode as 'overlay' | 'checker' | 'difference',
        blend: a.blend,
        sliceIndex: Math.round(a.sliceT * 100),
        level: a.level,
        width: a.width,
      },
      img.data,
    )
  }
  offCtx.putImageData(img, 0, 0)

  const size = Math.min(cssW, cssH)
  const ox = (cssW - size) / 2
  const oy = (cssH - size) / 2

  // Glow behind the slice frame.
  ctx.save()
  ctx.shadowColor = 'rgba(53, 223, 208, 0.16)'
  ctx.shadowBlur = 28
  ctx.fillStyle = '#04060a'
  ctx.fillRect(ox, oy, size, size)
  ctx.restore()

  ctx.imageSmoothingEnabled = true
  ctx.imageSmoothingQuality = 'high'
  ctx.drawImage(off, ox, oy, size, size)

  ctx.strokeStyle = 'rgba(53, 223, 208, 0.3)'
  ctx.lineWidth = 1
  ctx.strokeRect(ox + 0.5, oy + 0.5, size - 1, size - 1)

  const px = (nx: number) => ox + ((nx + 1) / 2) * size
  const py = (ny: number) => oy + ((ny + 1) / 2) * size

  if (a.showCrosshair) {
    ctx.save()
    ctx.strokeStyle = 'rgba(53, 223, 208, 0.45)'
    ctx.setLineDash([4, 5])
    ctx.lineWidth = 1
    const cx = ox + size / 2
    const cy = oy + size / 2
    ctx.beginPath()
    ctx.moveTo(ox, cy)
    ctx.lineTo(ox + size, cy)
    ctx.moveTo(cx, oy)
    ctx.lineTo(cx, oy + size)
    ctx.stroke()
    ctx.restore()
  }

  if (a.showLandmarks && a.status === 'complete') drawLandmarks(ctx, a, px, py)

  ctx.save()
  ctx.font = '600 10px Inter, sans-serif'
  ctx.fillStyle = 'rgba(167, 178, 196, 0.9)'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  const m = orientationMarks(a.plane)
  const inset = 15
  ctx.fillText(m.top, ox + size / 2, oy + inset)
  ctx.fillText(m.bottom, ox + size / 2, oy + size - inset)
  ctx.fillText(m.left, ox + inset, oy + size / 2)
  ctx.fillText(m.right, ox + size - inset, oy + size / 2)
  ctx.restore()

  if (a.status === 'running') {
    const t = (performance.now() % 2400) / 2400
    const yPos = oy + t * size
    const grad = ctx.createLinearGradient(0, yPos - 28, 0, yPos + 28)
    grad.addColorStop(0, 'rgba(53, 223, 208, 0)')
    grad.addColorStop(0.5, 'rgba(53, 223, 208, 0.3)')
    grad.addColorStop(1, 'rgba(53, 223, 208, 0)')
    ctx.fillStyle = grad
    ctx.fillRect(ox, yPos - 28, size, 56)
  }
}

/**
 * Fixed landmarks (green rings) vs their warped moving counterparts
 * (orange crosses) joined by a segment whose length is the visible TRE.
 */
function drawLandmarks(
  ctx: CanvasRenderingContext2D,
  a: DrawArgs,
  px: (n: number) => number,
  py: (n: number) => number,
) {
  ctx.save()
  ctx.font = '600 9.5px "JetBrains Mono", monospace'

  a.landmarks.forEach((lm, i) => {
    const project = (p: [number, number, number]): [number, number] => {
      if (a.plane === 'axial') return [p[0], p[1]]
      if (a.plane === 'coronal') return [p[0], p[2]]
      return [p[1], p[2]]
    }

    const [fx, fy] = project(lm.pos)
    const [mx, my] = project(lm.movingPos)
    const X = px(fx)
    const Y = py(fy)
    const mX = px(mx)
    const mY = py(my)

    // Error segment
    ctx.strokeStyle = 'rgba(251, 146, 60, 0.9)'
    ctx.lineWidth = 1.25
    ctx.beginPath()
    ctx.moveTo(X, Y)
    ctx.lineTo(mX, mY)
    ctx.stroke()

    // Fixed landmark ring
    ctx.beginPath()
    ctx.arc(X, Y, 4, 0, Math.PI * 2)
    ctx.strokeStyle = '#34d399'
    ctx.lineWidth = 1.5
    ctx.stroke()

    // Warped moving landmark cross
    ctx.strokeStyle = '#fb923c'
    ctx.lineWidth = 1.5
    ctx.beginPath()
    ctx.moveTo(mX - 3.5, mY - 3.5)
    ctx.lineTo(mX + 3.5, mY + 3.5)
    ctx.moveTo(mX + 3.5, mY - 3.5)
    ctx.lineTo(mX - 3.5, mY + 3.5)
    ctx.stroke()

    // TRE callout on the first four landmarks — stagger vertically so
    // clustered targets do not overlap each other.
    if (i < 4) {
      const label = `${lm.treMm} mm`
      const tw = ctx.measureText(label).width + 6
      const lx = mX + 8 + (i % 2) * 4
      const ly = mY - 8 - (i % 3) * 13
      ctx.fillStyle = 'rgba(5, 7, 11, 0.85)'
      ctx.fillRect(lx, ly - 9, tw, 12)
      ctx.fillStyle = '#f5a524'
      ctx.fillText(label, lx + 3, ly)
    }
  })

  ctx.restore()
}

