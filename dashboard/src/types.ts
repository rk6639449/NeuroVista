/**
 * NeuroVista domain model.
 *
 * Every shape in here is deliberately backend-agnostic: the mock API in
 * `src/api/client.ts` produces these types, and a real VoxelMorph service
 * should be adapted to return exactly the same shapes.
 */

export type Plane = 'axial' | 'coronal' | 'sagittal'

export type ViewMode = 'overlay' | 'checker' | 'side-by-side' | 'difference'

export type RunStatus = 'idle' | 'running' | 'complete' | 'failed'

export type CaseStatus = 'registered' | 'pending' | 'running' | 'failed'

/** One 3-D volume pair in the study list. */
export interface ImagingCase {
  id: string
  /** Short display label, e.g. "Left temporal GTR". */
  label: string
  /** De-identified subject token shown in the UI. */
  subject: string
  fixedModality: string
  fixedSequence: string
  movingModality: string
  /** [i, j, k] voxel counts of the fixed volume. */
  dims: [number, number, number]
  /** mm per voxel along each axis of the fixed volume. */
  spacing: [number, number, number]
  acquired: string
  /** Approximate FOV in mm along X — used to convert mm TRE to pixels. */
  fovMm: number
  status: CaseStatus
  notes: string
}

export type MethodFamily = 'baseline' | 'unsupervised' | 'diffeomorphic' | 'symmetric'

/** A selectable registration algorithm / configuration. */
export interface RegistrationMethod {
  id: string
  name: string
  shortName: string
  family: MethodFamily
  citation: string
  /** Nominal runtime for one pair at 1 mm³, in seconds. */
  nominalRuntimeS: number
  blurb: string
  params: { label: string; value: string }[]
  /** Quality prior used by the mock scorer — higher is better. */
  quality: number
  baselineTRE: number
}

export interface DiceScore {
  structure: string
  /** 0 – 1 */
  dice: number
  /** Dice of the pre-registration (affine-only) state, for the delta arrow. */
  prior: number
}

export interface LandmarkError {
  id: string
  label: string
  group: 'Target' | 'Safety'
  /** Target registration error in millimetres. */
  treMm: number
  /** Position on the fixed volume, normalised to 0–1 per axis. */
  pos: [number, number, number]
  /** Position on the *misaligned* moving volume, normalised 0–1. */
  movingPos: [number, number, number]
}

export interface RunStage {
  key: string
  label: string
  detail: string
}

/** Full output of one registration run — the contract your backend must meet. */
export interface RegistrationResult {
  runId: string
  caseId: string
  methodId: string
  status: RunStatus
  /** 0 – 1 */
  progress: number
  /** Index into `pipeline` of the stage currently executing. */
  stageIndex: number
  pipeline: RunStage[]
  finishedAt: string | null
  runtimeMs: number

  treMeanMm: number
  treMedianMm: number
  treMaxMm: number
  treStdMm: number
  hd95Mm: number

  dice: DiceScore[]
  /** % of the deformation field with Jacobian determinant < 0 (folding). */
  jacobianNegativePct: number
  /** Normalised cross-correlation between fixed and warped moving. */
  ncc: number

  landmarks: LandmarkError[]
  /** Free-text read-out that the assistant panel renders. */
  summary: string
  /** Set to a string when `status === 'failed'`. */
  error?: string
}

export type ChatRole = 'user' | 'assistant'

export interface ChatMessage {
  id: string
  role: ChatRole
  text: string
  /** Epoch ms. */
  at: number
  /** True while the mock assistant is "typing". */
  pending?: boolean
}

export interface MethodComparison {
  methodId: string
  methodName: string
  treMeanMm: number
  diceMean: number
  runtimeS: number
  isCurrent: boolean
}
