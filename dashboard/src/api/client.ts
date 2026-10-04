import {
  CASES,
  METHODS,
  PIPELINE,
  comparisonFor,
  makeFailedResult,
  makeResult,
  makeRunningResult,
} from '../data/mock'
import type {
  ImagingCase,
  MethodComparison,
  RegistrationMethod,
  RegistrationResult,
} from '../types'

/**
 * ============================================================================
 * NeuroVista API layer
 * ============================================================================
 *
 * The dashboard talks ONLY to this module. Everything under `src/data/mock.ts`
 * is swappable by flipping `USE_MOCK` to `false` and pointing `API_BASE` at
 * your existing VoxelMorph service.
 *
 * Expected backend contract (Flask / FastAPI both fine):
 *
 *   GET  {API_BASE}/api/cases
 *        → { cases: ImagingCase[] }
 *
 *   GET  {API_BASE}/api/methods
 *        → { methods: RegistrationMethod[] }
 *
 *   POST {API_BASE}/api/register
 *        body: { caseId: string, methodId: string, runId: string }
 *        → { result: RegistrationResult }            // synchronous, or
 *        → { runId: string } + SSE at GET /api/runs/{runId}/events
 *           emitting { progress, stageIndex, ...partial RegistrationResult }
 *
 *   GET  {API_BASE}/api/runs/latest?caseId=...&methodId=...
 *        → { result: RegistrationResult | null }
 *
 * A minimal FastAPI sketch:
 *
 *   @app.post("/api/register")
 *   def register(body: RegisterBody):
 *       fixed  = load_nifti(cases[body.caseId].fixed_path)
 *       moving = load_nifti(cases[body.caseId].moving_path)
 *       disp, stats = vm_infer(model, fixed, moving, method=body.methodId)
 *       return {"result": to_result_shape(stats)}   # mirror src/types.ts
 *
 * ============================================================================
 */

export const USE_MOCK = true
export const API_BASE =
  (import.meta.env.VITE_API_BASE as string | undefined) ?? 'http://localhost:8000'

/** Simulated network latency for mock reads, in ms. */
const MOCK_LATENCY = 260

function delay(ms: number) {
  return new Promise<void>((resolve) => setTimeout(resolve, ms))
}

export async function fetchCases(): Promise<ImagingCase[]> {
  if (!USE_MOCK) {
    const res = await fetch(`${API_BASE}/api/cases`)
    if (!res.ok) throw new Error(`GET /api/cases → ${res.status}`)
    const data = (await res.json()) as { cases: ImagingCase[] }
    return data.cases
  }
  await delay(MOCK_LATENCY)
  return CASES.map((c) => ({ ...c }))
}

export async function fetchMethods(): Promise<RegistrationMethod[]> {
  if (!USE_MOCK) {
    const res = await fetch(`${API_BASE}/api/methods`)
    if (!res.ok) throw new Error(`GET /api/methods → ${res.status}`)
    const data = (await res.json()) as { methods: RegistrationMethod[] }
    return data.methods
  }
  await delay(MOCK_LATENCY)
  return METHODS.map((m) => ({ ...m }))
}

/**
 * Seed result for a freshly selected case — what the UI shows before the
 * surgeon presses "Run".
 */
export function seedResult(
  caseItem: ImagingCase,
  method: RegistrationMethod,
): RegistrationResult {
  if (caseItem.status === 'failed') {
    return makeFailedResult(`seed-${caseItem.id}-${method.id}`, caseItem, method)
  }
  if (caseItem.status === 'pending') {
    return makeRunningResult(`seed-${caseItem.id}-${method.id}`, caseItem, method, -1, 0)
  }
  return makeResult(`seed-${caseItem.id}-${method.id}`, caseItem, method)
}

/** Head-to-head comparison rows for the assistant panel. */
export function comparison(
  caseItem: ImagingCase,
  methodId: string,
): MethodComparison[] {
  if (!USE_MOCK) {
    // Real backends: GET /api/compare?caseId=...&methodId=...
    // Until then fall back to the mock scorer so the panel is never empty.
    return comparisonFor(caseItem, methodId)
  }
  return comparisonFor(caseItem, methodId)
}

/**
 * Progress ticks a mock run emits. Returned so `Session` can replay them on
 * a timer — a real backend would push the same values over SSE/WebSocket.
 */
export const RUN_TICKS = PIPELINE.length

export function runProgressTick(
  runId: string,
  caseItem: ImagingCase,
  method: RegistrationMethod,
  tick: number,
): RegistrationResult {
  const total = PIPELINE.length
  const clamped = Math.max(0, Math.min(total, tick))
  const progress = clamped / total

  if (clamped >= total) {
    // ~8% of runs fail — deterministic per (case, method) so it is stable.
    const fails = caseItem.status === 'failed'
    return fails
      ? makeFailedResult(runId, caseItem, method)
      : makeResult(runId, caseItem, method)
  }
  return makeRunningResult(runId, caseItem, method, clamped - 1, progress)
}
