import type { Plane, ViewMode } from '../types'

/**
 * Procedural mock image synthesis.
 *
 * Everything here exists so the viewer can render convincing MRI / US frames
 * with zero backend. When the real VoxelMorph service lands, replace the
 * *callers* of these functions with slice fetches — the compositing helpers
 * (`composite`) already work on plain `Uint8ClampedArray` pixel buffers.
 */

export const GRID = 256 // synthetic slice resolution (square)

/* ---------------------------------------------------------------- */
/* Deterministic noise                                               */
/* ---------------------------------------------------------------- */

function hash2(x: number, y: number, seed: number) {
  let h = Math.imul(x, 374761393) + Math.imul(y, 668265263) + Math.imul(seed, 1442695041)
  h = Math.imul(h ^ (h >>> 13), 1274126177)
  return ((h ^ (h >>> 16)) >>> 0) / 4294967295
}

function smoothstep(t: number) {
  return t * t * (3 - 2 * t)
}

/** Bilinear value noise on an integer lattice. */
function valueNoise(x: number, y: number, seed: number) {
  const xi = Math.floor(x)
  const yi = Math.floor(y)
  const xf = smoothstep(x - xi)
  const yf = smoothstep(y - yi)
  const a = hash2(xi, yi, seed)
  const b = hash2(xi + 1, yi, seed)
  const c = hash2(xi, yi + 1, seed)
  const d = hash2(xi + 1, yi + 1, seed)
  return a + (b - a) * xf + (c - a) * yf + (a - b - c + d) * xf * yf
}

/** Fractal Brownian motion — gives tissue its mottled appearance. */
function fbm(x: number, y: number, seed: number, octaves = 5) {
  let value = 0
  let amp = 0.5
  let freq = 1
  let norm = 0
  for (let i = 0; i < octaves; i++) {
    value += amp * valueNoise(x * freq, y * freq, seed + i * 101)
    norm += amp
    amp *= 0.5
    freq *= 2
  }
  return value / norm
}

/* ---------------------------------------------------------------- */
/* Geometry helpers                                                  */
/* ---------------------------------------------------------------- */

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v)

function ellipse(x: number, y: number, cx: number, cy: number, rx: number, ry: number) {
  const dx = (x - cx) / rx
  const dy = (y - cy) / ry
  return Math.sqrt(dx * dx + dy * dy)
}

/**
 * Brain-like head mask for a normalised coordinate in [-1, 1].
 * Returns 0 outside the skull, ramping up through skull → cortex.
 */
function headField(nx: number, ny: number, plane: Plane, sliceT: number) {
  // Slice position squashes the head anterior-posteriorly/superiorly.
  const taper = 1 - 0.35 * sliceT * sliceT
  const rx = plane === 'sagittal' ? 0.92 : 0.78 * taper
  const ry = plane === 'axial' ? 0.94 : 0.86 * taper
  const r = ellipse(nx, ny, 0, 0.02, rx, ry)
  if (r > 1) return 0
  // Outer scalp/skull ring is bright on T1, then a dark inner table.
  if (r > 0.97) return 0.95
  if (r > 0.9) return 0.08
  return clamp01(1 - r) // 1 deep in the brain, → 0 at the cortex
}

/* ---------------------------------------------------------------- */
/* Synthetic MRI (fixed image)                                       */
/* ---------------------------------------------------------------- */

export interface SliceSpec {
  plane: Plane
  /** 0 → 1 position along the slice axis. */
  sliceT: number
  /** Stable per-case seed so each subject looks different. */
  seed: number
  /** How far the pathology has shifted (brain-shift magnitude, 0–1). */
  shift: number
}

/**
 * Render a T1-post-contrast-like axial/coronal/sagittal slice.
 * Returns grayscale intensities in 0–1.
 */
export function synthMRI(spec: SliceSpec, out: Float32Array) {
  const { plane, sliceT, seed } = spec
  const t = (sliceT - 0.5) * 2 // -1 … 1

  for (let y = 0; y < GRID; y++) {
    const ny = (y / (GRID - 1)) * 2 - 1
    for (let x = 0; x < GRID; x++) {
      const nx = (x / (GRID - 1)) * 2 - 1
      const i = y * GRID + x

      const head = headField(nx, ny, plane, t)
      if (head <= 0) {
        // Background: faint coil noise so it is not pure black.
        out[i] = 0.03 + hash2(x, y, seed) * 0.025
        continue
      }

      // Gyral texture: warped fbm drives sulcal darkness.
      const warpX = nx + 0.12 * Math.sin(ny * 5 + seed)
      const warpY = ny + 0.12 * Math.sin(nx * 4 - seed)
      const tissue = fbm(warpX * 3.2 + 8, warpY * 3.2 + 8, seed, 5)
      const sulci = clamp01((tissue - 0.56) * 4.2) * head

      // White matter is brighter than grey on T1.
      let v = 0.42 + head * 0.3 + (tissue - 0.5) * 0.22
      v -= sulci * 0.34

      // Ventricles — dark CSF, butterfly shape on axial.
      const vent = ventricleField(nx, ny, plane, t)
      v = v * (1 - vent) + 0.1 * vent

      // Contrast-enhancing lesion: bright blob with a dark necrotic core.
      const lesion = lesionField(nx, ny, plane, t, spec.shift)
      v = v * (1 - lesion.core) + 0.16 * lesion.core
      v += lesion.enhance * 0.72
      v += lesion.edema * -0.1

      // Mild global bias field (coil inhomogeneity).
      v *= 0.9 + 0.18 * (0.5 + 0.5 * Math.cos((nx + ny) * 1.1 + seed))

      out[i] = clamp01(v)
    }
  }
  return out
}

/** 0–1 membership of the lateral ventricles. */
function ventricleField(nx: number, ny: number, plane: Plane, t: number) {
  if (Math.abs(t) > 0.72) return 0
  if (plane === 'sagittal') {
    // C-shaped ventricle profile.
    const cx = -0.05 + 0.25 * Math.cos(ny * 2.4)
    const cy = -0.02 + 0.3 * Math.sin(ny * 1.6)
    const r = ellipse(nx, ny, cx, cy, 0.09, 0.34)
    return clamp01((1 - r) * 2.4)
  }
  const r1 = ellipse(nx, ny, -0.17, -0.03, 0.1, 0.2)
  const r2 = ellipse(nx, ny, 0.17, -0.03, 0.1, 0.2)
  const r = Math.min(r1, r2)
  const fade = 1 - Math.abs(t) / 0.72
  return clamp01((1 - r) * 2.6) * fade
}

/** Lesion geometry shared by MRI and US so the two agree anatomically. */
function lesionField(nx: number, ny: number, plane: Plane, t: number, shift: number) {
  // The lesion sits left-anterior and drifts with `shift` (brain shift).
  const lx = -0.3 + shift * 0.1
  const ly = -0.12 + shift * 0.16
  const lz = 0.08 + shift * 0.05

  const dx = nx - lx
  const dy = ny - ly
  const dz = (plane === 'axial' ? t : t * 0.6) - lz

  const r = Math.sqrt(dx * dx * 1.15 + dy * dy * 1.35 + dz * dz * 1.7) / 0.34
  const inPlane = plane === 'sagittal' ? Math.abs(nx - lx) : r
  const base = clamp01((1 - r) * 1.9)
  const radialFade = plane === 'sagittal' ? clamp01((1 - Math.abs(dx) / 0.36) * 1.8) : base

  return {
    core: radialFade * (inPlane < 0.55 ? 1 : 0) * clamp01(1.4 - r),
    enhance: clamp01((radialFade - 0.35) * 2.4) * (r < 1.25 ? 1 : 0),
    edema: clamp01(1 - r / 1.9) * 0.5,
  }
}

/* ---------------------------------------------------------------- */
/* Synthetic ultrasound (moving image)                              */
/* ---------------------------------------------------------------- */

/**
 * Intraoperative B-mode US probe frame. Sector-shaped field of view,
 * heavy speckle, depth attenuation — deliberately *not* aligned with the
 * MRI, which is exactly the mis-registration the dashboard is about.
 */
export function synthUS(spec: SliceSpec, out: Float32Array) {
  const { plane, sliceT, seed } = spec
  const t = (sliceT - 0.5) * 2
  const probeAngle = 0.62 + 0.2 * ((seed % 7) / 7)

  for (let y = 0; y < GRID; y++) {
    const ny = (y / (GRID - 1)) * 2 - 1
    const depth = (ny + 1) / 2 // 0 at probe face → 1 at far field
    for (let x = 0; x < GRID; x++) {
      const nx = (x / (GRID - 1)) * 2 - 1
      const i = y * GRID + x

      // Sector (pie-slice) field of view opening downward from top centre.
      const sx = nx / Math.max(depth, 0.08)
      if (Math.abs(sx) > probeAngle || ny < -0.86) {
        out[i] = 0.02 + hash2(x, y, seed + 9) * 0.04
        continue
      }

      // Speckle: multiplicative high-frequency noise (Rayleigh-like).
      const speckle = fbm(nx * 46, ny * 46, seed + 3, 3)
      const coarse = fbm(nx * 9 + 3, ny * 9, seed + 5, 4)

      // Underlying anatomy echo intensity, sampled from the same fields
      // as the MRI but modulated to look sonographic.
      const head = headField(nx * 0.92, ny * 0.9, plane, t * 0.85)
      let v = head * 0.55 * (0.4 + coarse * 1.1)

      const lesion = lesionField(nx, ny, plane, t, spec.shift)
      // Hyperechoic rim + hypoechoic core, the classic US appearance.
      v += lesion.enhance * 0.85
      v = v * (1 - lesion.core * 0.7) + 0.14 * lesion.core * 0.7

      // Bright near-field (probe/skull contact) and posterior shadowing.
      v *= 0.35 + 0.65 * Math.exp(-depth * 2.1)
      v += Math.exp(-depth * 9) * 0.55
      if (lesion.core > 0.5 && depth > 0.55) v *= 0.45 // acoustic shadow

      // Distance gain compensation ramp.
      v *= 0.7 + depth * 0.6

      v *= 0.55 + speckle * 0.95
      out[i] = clamp01(v)
    }
  }
  return out
}

/* ---------------------------------------------------------------- */
/* Deformation & resampling                                          */
/* ---------------------------------------------------------------- */

/**
 * Smooth, divergence-light displacement field in normalised units.
 * `magnitude` scales the whole field (0 = perfect alignment).
 */
function displacement(
  nx: number,
  ny: number,
  seed: number,
  magnitude: number,
): [number, number] {
  const fx = fbm(nx * 1.6 + 21, ny * 1.6, seed + 17, 4) - 0.5
  const fy = fbm(nx * 1.6, ny * 1.6 + 13, seed + 31, 4) - 0.5
  // A global rotation-ish shear is what affine pre-alignment misses.
  const gx = -ny * 0.14
  const gy = nx * 0.09
  return [(fx * 0.9 + gx) * magnitude, (fy * 0.9 + gy) * magnitude]
}

function bilinear(src: Float32Array, x: number, y: number) {
  const x0 = Math.floor(x)
  const y0 = Math.floor(y)
  if (x0 < 0 || y0 < 0 || x0 >= GRID - 1 || y0 >= GRID - 1) return 0
  const fx = x - x0
  const fy = y - y0
  const i = y0 * GRID + x0
  const a = src[i]
  const b = src[i + 1]
  const c = src[i + GRID]
  const d = src[i + GRID + 1]
  return a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + c * (1 - fx) * fy + d * fx * fy
}

/** Resample `src` into `dst` applying a displacement of `magnitude`. */
export function warp(
  src: Float32Array,
  dst: Float32Array,
  seed: number,
  magnitude: number,
) {
  if (magnitude === 0) {
    dst.set(src)
    return dst
  }
  for (let y = 0; y < GRID; y++) {
    const ny = (y / (GRID - 1)) * 2 - 1
    for (let x = 0; x < GRID; x++) {
      const nx = (x / (GRID - 1)) * 2 - 1
      const [dx, dy] = displacement(nx, ny, seed, magnitude)
      const sx = (nx + dx + 1) * 0.5 * (GRID - 1)
      const sy = (ny + dy + 1) * 0.5 * (GRID - 1)
      dst[y * GRID + x] = bilinear(src, sx, sy)
    }
  }
  return dst
}

/* ---------------------------------------------------------------- */
/* Window / level & colour                                           */
/* ---------------------------------------------------------------- */

/** Apply a soft window (contrast) around `level` — mimics a W/L preset. */
function window(v: number, level: number, width: number) {
  const lo = level - width / 2
  return clamp01((v - lo) / width)
}

/**
 * Hot-metal ramp for the US fusion overlay:
 * transparent → deep red → orange → yellow → white.
 */
function usRamp(t: number): [number, number, number, number] {
  if (t <= 0.02) return [0, 0, 0, 0]
  const r = clamp01(t * 2.6)
  const g = clamp01((t - 0.35) * 2.1)
  const b = clamp01((t - 0.78) * 4.2)
  const a = clamp01(0.15 + t * 1.35)
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255), a]
}

export interface CompositeOpts {
  mode: ViewMode
  /** Blend of moving over fixed, 0–1 (overlay mode). */
  blend: number
  /** Slice position — used by the checkerboard so tiles swap per slice. */
  sliceIndex: number
  /** Window level 0–1 and width 0.2–1 applied to the MRI. */
  level: number
  width: number
}

/**
 * Compose fixed + moving into RGBA pixels for the current `ViewMode`.
 * Output length is GRID*GRID*4.
 */
export function composite(
  fixed: Float32Array,
  moving: Float32Array,
  opts: CompositeOpts,
  out: Uint8ClampedArray,
) {
  const { mode, blend, sliceIndex, level, width } = opts
  const checkerFlip = sliceIndex % 2 === 1

  for (let i = 0; i < GRID * GRID; i++) {
    const o = i * 4
    const f = window(fixed[i], level, width)
    const m = window(moving[i], level, width)
    const fi = Math.round(f * 255)

    if (mode === 'overlay') {
      const [r, g, b, a] = usRamp(m)
      const A = a * blend
      out[o] = fi * (1 - A) + r * A
      out[o + 1] = fi * (1 - A) + g * A
      out[o + 2] = fi * (1 - A) + b * A
      out[o + 3] = 255
    } else if (mode === 'checker') {
      // 16-px tiles (on the 256 grid) — the usual registration-QC size.
      const cx = (i % GRID) >> 4
      const cy = (i / GRID / 16) | 0
      const useMoving = (cx + cy + (checkerFlip ? 1 : 0)) % 2 === 0
      const v = useMoving ? m : f
      const vi = Math.round(v * 255)
      if (useMoving) {
        // Tint US tiles warm so the two modalities are separable.
        out[o] = vi
        out[o + 1] = Math.round(vi * 0.82)
        out[o + 2] = Math.round(vi * 0.6)
      } else {
        out[o] = vi
        out[o + 1] = vi
        out[o + 2] = Math.round(vi * 1.04)
      }
      out[o + 3] = 255
    } else if (mode === 'difference') {
      const d = Math.abs(f - m)
      const e = Math.round(clamp01(d * 2.4) * 255)
      out[o] = e
      out[o + 1] = Math.round(e * 0.25)
      out[o + 2] = Math.round(e * 0.18)
      out[o + 3] = 255
    } else {
      // 'side-by-side' — vertical split fallback (the dedicated
      // compositeSideBySide path is used by the viewer).
      const half = GRID >> 1
      const x = i % GRID
      if (x < half) {
        out[o] = fi
        out[o + 1] = fi
        out[o + 2] = fi
        out[o + 3] = 255
      } else {
        const [r, g, b] = usRamp(m)
        out[o] = r
        out[o + 1] = g
        out[o + 2] = b
        out[o + 3] = 255
      }
    }
  }
  return out
}

/** Split `fixed` / `moving` into two half-width panels for side-by-side. */
export function compositeSideBySide(
  fixed: Float32Array,
  moving: Float32Array,
  level: number,
  width: number,
  out: Uint8ClampedArray,
) {
  const half = GRID >> 1
  for (let y = 0; y < GRID; y++) {
    for (let x = 0; x < GRID; x++) {
      const o = (y * GRID + x) * 4
      const isRight = x >= half
      // Map each half onto the full source image (scaled 0.5 in x).
      const srcX = isRight ? (x - half) * 2 : x * 2
      const si = y * GRID + Math.min(srcX, GRID - 1)
      const v = window(isRight ? moving[si] : fixed[si], level, width)
      if (isRight) {
        const [r, g, b] = usRamp(v)
        out[o] = r
        out[o + 1] = g
        out[o + 2] = b
      } else {
        const vi = Math.round(v * 255)
        out[o] = vi
        out[o + 1] = vi
        out[o + 2] = vi
      }
      // Divider gutter.
      if (Math.abs(x - half) < 1) {
        out[o] = 24
        out[o + 1] = 30
        out[o + 2] = 42
      }
      out[o + 3] = 255
    }
  }
  return out
}



