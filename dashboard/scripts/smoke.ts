/**
 * Headless smoke test for the mock layers — no browser needed.
 * Run: node --experimental-strip-types scripts/smoke.ts
 */
import { GRID, composite, compositeSideBySide, synthMRI, synthUS, warp } from '../src/lib/imaging.ts'
import { CASES, METHODS, comparisonFor, makeResult, makeRunningResult } from '../src/data/mock.ts'

let failures = 0
function check(name: string, cond: boolean, detail = '') {
  if (cond) {
    console.log(`  ok   ${name}`)
  } else {
    failures++
    console.log(`  FAIL ${name} ${detail}`)
  }
}

console.log('· imaging synthesis')
const fixed = new Float32Array(GRID * GRID)
const movingRaw = new Float32Array(GRID * GRID)
const moving = new Float32Array(GRID * GRID)
synthMRI({ plane: 'axial', sliceT: 0.5, seed: 1234, shift: 0.35 }, fixed)
synthUS({ plane: 'axial', sliceT: 0.5, seed: 1234, shift: 0.35 }, movingRaw)

let nan = 0
let inRange = 0
for (let i = 0; i < fixed.length; i++) {
  if (Number.isNaN(fixed[i]) || Number.isNaN(movingRaw[i])) nan++
  if (fixed[i] >= 0 && fixed[i] <= 1 && movingRaw[i] >= 0 && movingRaw[i] <= 1) inRange++
}
check('no NaN intensities', nan === 0, `(${nan} found)`)
check('all intensities in [0,1]', inRange === fixed.length, `(${inRange}/${fixed.length})`)

// MRI should have contrast (not a flat image).
let min = 1
let max = 0
for (let i = 0; i < fixed.length; i++) {
  if (fixed[i] < min) min = fixed[i]
  if (fixed[i] > max) max = fixed[i]
}
check('MRI has dynamic range', max - min > 0.8, `(min=${min.toFixed(2)} max=${max.toFixed(2)})`)

// Warp
warp(movingRaw, moving, 1234, 0.9)
let warpNan = 0
for (let i = 0; i < moving.length; i++) if (Number.isNaN(moving[i])) warpNan++
check('warp produces no NaN', warpNan === 0)

// Composites
const out = new Uint8ClampedArray(GRID * GRID * 4)
for (const mode of ['overlay', 'checker', 'difference'] as const) {
  composite(fixed, moving, { mode, blend: 0.55, sliceIndex: 10, level: 0.5, width: 0.85 }, out)
  let alphaOk = true
  for (let i = 3; i < out.length; i += 4) if (out[i] !== 255) alphaOk = false
  check(`composite(${mode}) fills opaque pixels`, alphaOk)
}
compositeSideBySide(fixed, moving, 0.5, 0.85, out)
check('compositeSideBySide produces pixels', out.some((v) => v > 0))

console.log('· mock metrics')
for (const c of CASES) {
  for (const m of METHODS) {
    const r = makeResult('smoke', c, m)
    const diceOk = r.dice.every((d) => d.dice > 0 && d.dice <= 1 && d.prior <= d.dice)
    const treOk = r.treMeanMm > 0 && r.treMedianMm <= r.treMaxMm
    const lmOk = r.landmarks.length === 8 && r.landmarks.every((l) => l.treMm > 0)
    if (!diceOk || !treOk || !lmOk) {
      check(`${c.id}/${m.id} sane`, false, JSON.stringify({ diceOk, treOk, lmOk }))
    }
  }
}
check(`all ${CASES.length * METHODS.length} case×method results sane`, true)

const running = makeRunningResult('smoke-run', CASES[0], METHODS[0], 3, 0.42)
check('running result has zeroed metrics', running.treMeanMm === 0 && running.status === 'running')
const cmp = comparisonFor(CASES[0], METHODS[0].id)
check('comparison sorted by TRE', cmp.every((r, i) => i === 0 || cmp[i - 1].treMeanMm <= r.treMeanMm))

// Determinism
const a = makeResult('x', CASES[1], METHODS[1])
const b = makeResult('x', CASES[1], METHODS[1])
check('results are deterministic', a.treMeanMm === b.treMeanMm && a.dice[0].dice === b.dice[0].dice)

console.log(failures === 0 ? '\nAll smoke checks passed.' : `\n${failures} check(s) FAILED.`)
process.exit(failures === 0 ? 0 : 1)
