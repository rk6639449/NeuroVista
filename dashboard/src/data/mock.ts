import type {
  DiceScore,
  ImagingCase,
  LandmarkError,
  MethodComparison,
  RegistrationMethod,
  RegistrationResult,
  RunStage,
} from '../types'

/**
 * Mock dataset for the NeuroVista dashboard.
 *
 * Case IDs mirror the ReMIND NIfTI subjects vendored in `remind-nifti/`, so
 * swapping to a real backend later is a matter of pointing `api/client.ts`
 * at a service that lists the same subjects.
 */

export const CASES: ImagingCase[] = [
  {
    id: 'ReMIND-007',
    label: 'Left temporal GTR',
    subject: 'P-007',
    fixedModality: 'MRI',
    fixedSequence: '3D AX T1 POST',
    movingModality: 'US',
    dims: [256, 256, 176],
    spacing: [1.0, 1.0, 1.0],
    acquired: '2024-11-02',
    fovMm: 256,
    status: 'registered',
    notes: 'Suspected residual along medial temporal margin.',
  },
  {
    id: 'ReMIND-014',
    label: 'Right frontal deep',
    subject: 'P-014',
    fixedModality: 'MRI',
    fixedSequence: '3D AX T1 POST',
    movingModality: 'US',
    dims: [256, 256, 176],
    spacing: [1.0, 1.0, 1.0],
    acquired: '2024-11-09',
    fovMm: 256,
    status: 'pending',
    notes: 'Brain shift after dural opening — registration recommended.',
  },
  {
    id: 'ReMIND-022',
    label: 'Parietal meningioma',
    subject: 'P-022',
    fixedModality: 'MRI',
    fixedSequence: '3D AX T1 POST',
    movingModality: 'US',
    dims: [320, 320, 180],
    spacing: [0.8, 0.8, 1.0],
    acquired: '2024-11-15',
    fovMm: 256,
    status: 'registered',
    notes: 'Near eloquent cortex — sub-mm target desired.',
  },
  {
    id: 'ReMIND-046',
    label: 'Insular recurrence',
    subject: 'P-046',
    fixedModality: 'MRI',
    fixedSequence: '3D AX T1 POST',
    movingModality: 'US',
    dims: [256, 256, 176],
    spacing: [1.0, 1.0, 1.0],
    acquired: '2024-12-03',
    fovMm: 256,
    status: 'failed',
    notes: 'Large resection cavity — poor US echo window.',
  },
  {
    id: 'ReMIND-071',
    label: 'Occipital cavernoma',
    subject: 'P-071',
    fixedModality: 'MRI',
    fixedSequence: '3D AX T1 POST',
    movingModality: 'US',
    dims: [256, 256, 160],
    spacing: [1.0, 1.0, 1.0],
    acquired: '2024-12-18',
    fovMm: 240,
    status: 'pending',
    notes: 'Superficial lesion, favourable US access.',
  },
  {
    id: 'ReMIND-085',
    label: 'Sellar extension',
    subject: 'P-085',
    fixedModality: 'MRI',
    fixedSequence: '3D AX T1 POST',
    movingModality: 'US',
    dims: [256, 256, 192],
    spacing: [1.0, 1.0, 1.0],
    acquired: '2025-01-07',
    fovMm: 230,
    status: 'pending',
    notes: 'Skull-base approach — limited transcranial window.',
  },
]

export const METHODS: RegistrationMethod[] = [
  {
    id: 'vm-diffeo',
    name: 'VoxelMorph · Diffeomorphic',
    shortName: 'VM-Diffeo',
    family: 'diffeomorphic',
    citation: 'Voxelmorph, CVPR 2019',
    nominalRuntimeS: 8.4,
    blurb:
      'Stationary-velocity field integrated with scaling-and-squaring. Guarantees invertible transforms — preferred for longitudinal and navigation use.',
    params: [
      { label: 'Encoder', value: '7 enc / 7 dec' },
      { label: 'Features', value: '16–256–1024' },
      { label: 'Integration', value: 'scaling & squaring × 12' },
      { label: 'Loss', value: 'NCC + λ·|∇u|²' },
    ],
    quality: 1.0,
    baselineTRE: 2.4,
  },
  {
    id: 'vm-unsup',
    name: 'VoxelMorph · Unsupervised',
    shortName: 'VM-Unsup',
    family: 'unsupervised',
    citation: 'Voxelmorph, NeurIPS 2018',
    nominalRuntimeS: 3.1,
    blurb:
      'Single forward pass, no labels required. Fastest option in the pool — good for quick intraoperative checks.',
    params: [
      { label: 'Encoder', value: '7 enc / 7 dec' },
      { label: 'Features', value: '16–256–1024' },
      { label: 'Similarity', value: 'Local NCC (w=9)' },
      { label: 'Regulariser', value: 'Gradient L2, σ=1.0' },
    ],
    quality: 0.86,
    baselineTRE: 3.6,
  },
  {
    id: 'vm-svf',
    name: 'Symmetric VM · SVF',
    shortName: 'SVF-Sym',
    family: 'symmetric',
    citation: 'Symmetric VoxelMorph, MICCAI 2020',
    nominalRuntimeS: 11.6,
    blurb:
      'Inverse-consistent pairwise registration; halves direction-dependent bias when moving US → MRI.',
    params: [
      { label: 'Formulation', value: 'Symmetric pair' },
      { label: 'Velocity', value: 'Stationary (SVF)' },
      { label: 'Similarity', value: 'Mutual information' },
      { label: 'Regulariser', value: 'Grad. L2, λ=0.01' },
    ],
    quality: 0.94,
    baselineTRE: 2.9,
  },
  {
    id: 'affine',
    name: 'Affine · 12-DoF baseline',
    shortName: 'Affine',
    family: 'baseline',
    citation: 'ITK/ANTs mutual information',
    nominalRuntimeS: 12.0,
    blurb:
      'Rigid + affine intensity matching. Shown as the pre-alignment reference every deformable method must beat.',
    params: [
      { label: 'DoF', value: '12 (affine)' },
      { label: 'Metric', value: 'Mattes MI, 32 bins' },
      { label: 'Optimizer', value: 'Regular step gradient' },
      { label: 'Interp', value: 'Linear / BSinc' },
    ],
    quality: 0.5,
    baselineTRE: 8.7,
  },
]

export const PIPELINE: RunStage[] = [
  { key: 'load', label: 'Load volumes', detail: 'Stream fixed/moving NIfTI → GPU' },
  { key: 'preproc', label: 'Preprocess', detail: 'N4 bias field · resample 1 mm³ · skull strip' },
  { key: 'affine', label: 'Affine pre-alignment', detail: '12-DoF MI, 60×40 pyramid' },
  { key: 'infer', label: 'VoxelMorph inference', detail: 'Encoder→decoder → dense velocity field' },
  { key: 'warp', label: 'Warp & resample', detail: 'Scaling-and-squaring, trilinear sampler' },
  { key: 'metrics', label: 'Quantify', detail: 'TRE · Dice · HD95 · Jacobian' },
  { key: 'review', label: 'AI review', detail: 'NeuroVista assistant summarises run' },
]

const STRUCTURES = [
  'Resection cavity',
  'Enhancing tumour',
  'Ventricles',
  'Cortical surface',
  'Brain mask',
]

const LANDMARK_SEED: Array<
  Pick<LandmarkError, 'id' | 'label' | 'group' | 'pos' | 'movingPos'>
> = [
  { id: 'lm1', label: 'Tumour centre', group: 'Target', pos: [0.42, 0.44, 0.5], movingPos: [0.38, 0.5, 0.55] },
  { id: 'lm2', label: 'Anterior margin', group: 'Target', pos: [0.35, 0.36, 0.5], movingPos: [0.3, 0.43, 0.56] },
  { id: 'lm3', label: 'Lateral margin', group: 'Target', pos: [0.52, 0.45, 0.5], movingPos: [0.47, 0.52, 0.55] },
  { id: 'lm4', label: 'Posterior margin', group: 'Target', pos: [0.44, 0.55, 0.5], movingPos: [0.4, 0.61, 0.54] },
  { id: 'lm5', label: 'Ventricle horn', group: 'Safety', pos: [0.55, 0.5, 0.48], movingPos: [0.51, 0.57, 0.53] },
  { id: 'lm6', label: 'Sylvian fissure', group: 'Safety', pos: [0.36, 0.5, 0.52], movingPos: [0.32, 0.57, 0.57] },
  { id: 'lm7', label: 'Midline (falx)', group: 'Safety', pos: [0.5, 0.4, 0.5], movingPos: [0.47, 0.47, 0.54] },
  { id: 'lm8', label: 'Skull base', group: 'Safety', pos: [0.5, 0.68, 0.42], movingPos: [0.47, 0.74, 0.47] },
]

/** Small deterministic PRNG so mock metrics are stable across renders. */
function makeRng(seedStr: string) {
  let h = 1779033703 ^ seedStr.length
  for (let i = 0; i < seedStr.length; i++) {
    h = Math.imul(h ^ seedStr.charCodeAt(i), 3432918353)
    h = (h << 13) | (h >>> 19)
  }
  let a = h >>> 0
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function round(v: number, dp = 1) {
  const f = 10 ** dp
  return Math.round(v * f) / f
}

function meanDiceFn(scores: DiceScore[]) {
  return scores.reduce((s, d) => s + d.dice, 0) / scores.length
}

/**
 * Deterministically build a completed `RegistrationResult` for a
 * (case, method) pair. Quality is derived from the method's `quality`
 * prior plus a stable per-case difficulty term.
 */
export function makeResult(
  runId: string,
  caseItem: ImagingCase,
  method: RegistrationMethod,
): RegistrationResult {
  const rng = makeRng(`${caseItem.id}::${method.id}`)
  const difficulty = rng() // 0 = easy, 1 = hard

  // Hard cases (poor US window, skull base) punish deformable methods more.
  const penalty = difficulty * 2.6 * (method.family === 'baseline' ? 0.6 : 1)
  const treMean = round(
    Math.max(1.1, method.baselineTRE * (0.75 + difficulty * 0.55) + penalty * 0.35),
    1,
  )
  const treStd = round(0.35 + difficulty * 1.15, 2)
  const treMedian = round(Math.max(0.9, treMean - treStd * 0.35), 1)
  const treMax = round(treMean + treStd * 1.9 + rng() * 1.2, 1)
  const hd95 = round(treMax * 1.28 + 1.4, 1)

  const baseDice = 0.62 + method.quality * 0.34 - difficulty * 0.14
  const dice: DiceScore[] = STRUCTURES.map((structure, i) => {
    const prior = Math.max(0.35, baseDice - 0.12 - rng() * 0.05 + (i === 4 ? 0.06 : 0))
    const value = Math.min(0.985, prior + 0.08 + method.quality * 0.06 + rng() * 0.03)
    return { structure, dice: round(value, 3), prior: round(prior, 3) }
  })

  // Landmark errors: target landmarks track TRE, safety landmarks are looser.
  const landmarks: LandmarkError[] = LANDMARK_SEED.map((l) => {
    const spread = l.group === 'Target' ? 1 : 1.35
    const treMm = round(Math.max(0.6, treMean * spread * (0.72 + rng() * 0.75)), 1)
    return { ...l, id: `${runId}-${l.id}`, treMm }
  })

  const jacobianNegativePct = round(
    method.family === 'baseline'
      ? 0
      : (method.family === 'unsupervised' ? 0.9 : 0.35) * (0.5 + difficulty),
    2,
  )
  const ncc = round(0.78 + method.quality * 0.17 - difficulty * 0.09, 3)
  const runtimeMs = Math.round(
    method.nominalRuntimeS * 1000 * (0.9 + rng() * 0.35) * (caseItem.dims[0] / 256),
  )

  const diceMean = meanDiceFn(dice)
  const verdict =
    treMean < 3 && diceMean > 0.85
      ? 'within the sub-3 mm navigation threshold'
      : treMean < 5
        ? 'acceptable for verification but short of sub-3 mm navigation tolerance'
        : 'outside the sub-3 mm navigation threshold — refine or re-acquire'

  const summary = [
    `${method.shortName} converged in ${(runtimeMs / 1000).toFixed(1)} s with mean TRE ${treMean.toFixed(1)} mm ` +
      `(median ${treMedian.toFixed(1)} mm, max ${treMax.toFixed(1)} mm, HD95 ${hd95.toFixed(1)} mm).`,
    `Mean Dice across ${dice.length} structures is ${diceMean.toFixed(3)} — strongest on ${dice
      .reduce((a, b) => (a.dice > b.dice ? a : b))
      .structure.toLowerCase()}.`,
    `Deformation field shows ${jacobianNegativePct}% folding ${
      jacobianNegativePct < 1
        ? '(diffeomorphic — safe to invert).'
        : '(non-diffeomorphic — do not invert).'
    }`,
    `Composite accuracy is ${verdict}.`,
  ].join(' ')

  return {
    runId,
    caseId: caseItem.id,
    methodId: method.id,
    status: 'complete',
    progress: 1,
    stageIndex: PIPELINE.length - 1,
    pipeline: PIPELINE,
    finishedAt: new Date().toISOString(),
    runtimeMs,
    treMeanMm: treMean,
    treMedianMm: treMedian,
    treMaxMm: treMax,
    treStdMm: treStd,
    hd95Mm: hd95,
    dice,
    jacobianNegativePct,
    ncc,
    landmarks,
    summary,
  }
}

/** A result that is still mid-flight — used to drive the "running" UI state. */
export function makeRunningResult(
  runId: string,
  caseItem: ImagingCase,
  method: RegistrationMethod,
  stageIndex: number,
  progress: number,
): RegistrationResult {
  const done = makeResult(runId, caseItem, method)
  const isDone = progress >= 1
  const emptyDice = done.dice.map((d) => ({ ...d, dice: isDone ? d.dice : 0, prior: 0 }))
  return {
    ...done,
    status: isDone ? 'complete' : 'running',
    progress,
    stageIndex,
    finishedAt: isDone ? done.finishedAt : null,
    treMeanMm: isDone ? done.treMeanMm : 0,
    treMedianMm: isDone ? done.treMedianMm : 0,
    treMaxMm: isDone ? done.treMaxMm : 0,
    treStdMm: isDone ? done.treStdMm : 0,
    hd95Mm: isDone ? done.hd95Mm : 0,
    dice: emptyDice,
    jacobianNegativePct: isDone ? done.jacobianNegativePct : 0,
    ncc: isDone ? done.ncc : 0,
    landmarks: isDone ? done.landmarks : [],
    summary: isDone ? done.summary : '',
  }
}

export function makeFailedResult(
  runId: string,
  caseItem: ImagingCase,
  method: RegistrationMethod,
): RegistrationResult {
  const base = makeRunningResult(runId, caseItem, method, 3, 0.42)
  return {
    ...base,
    status: 'failed',
    error:
      'Registration aborted: mutual-information optimizer diverged (ΔMI < 1e-5 for 12 iterations). ' +
      'The resection cavity removes most shared anatomy between MRI and US.',
  }
}

/** Head-to-head table shown in the assistant panel. */
export function comparisonFor(
  caseItem: ImagingCase,
  currentMethodId: string,
): MethodComparison[] {
  const rows = METHODS.map((m) => {
    const r = makeResult(`cmp-${caseItem.id}-${m.id}`, caseItem, m)
    return {
      methodId: m.id,
      methodName: m.shortName,
      treMeanMm: r.treMeanMm,
      diceMean: round(meanDiceFn(r.dice), 3),
      runtimeS: round(r.runtimeMs / 1000, 1),
      isCurrent: m.id === currentMethodId,
    }
  })
  return rows.sort((a, b) => a.treMeanMm - b.treMeanMm)
}

export const round1 = round
export const meanOf = meanDiceFn




