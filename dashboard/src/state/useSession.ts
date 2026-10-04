import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { fetchCases, fetchMethods, runProgressTick, seedResult } from '../api/client'
import type {
  ChatMessage,
  ImagingCase,
  Plane,
  RegistrationMethod,
  RegistrationResult,
  RunStatus,
  ViewMode,
} from '../types'

const RUN_TICK_MS = 750 // wall-clock per pipeline stage in the mock runner

function uid(prefix: string) {
  return `${prefix}-${Math.random().toString(36).slice(2, 9)}`
}

/** Initial assistant greeting. */
const GREETING: ChatMessage = {
  id: 'greet',
  role: 'assistant',
  at: Date.now(),
  text:
    'NeuroVista assistant online. I have loaded the case list and the VoxelMorph model pool. ' +
    'Run a registration and I will read out TRE, Dice and Jacobian behaviour — or ask me anything below.',
}

export const QUICK_PROMPTS = [
  'Summarise this registration',
  'Why is TRE high on target landmarks?',
  'Compare methods for this case',
  'Is the deformation diffeomorphic?',
]

export interface SessionState {
  loading: boolean
  error: string | null
  cases: ImagingCase[]
  methods: RegistrationMethod[]
  caseItem: ImagingCase | null
  method: RegistrationMethod | null
  result: RegistrationResult | null
  status: RunStatus
  progress: number
  stageIndex: number

  // Viewer controls
  plane: Plane
  sliceT: number // 0–1
  viewMode: ViewMode
  blend: number // 0–1
  level: number // 0–1
  width: number // 0.2–1
  showLandmarks: boolean
  showCrosshair: boolean

  // Chat
  messages: ChatMessage[]
  thinking: boolean

  selectCase: (id: string) => void
  selectMethod: (id: string) => void
  run: () => void
  stop: () => void

  setPlane: (p: Plane) => void
  setSliceT: (v: number) => void
  setViewMode: (m: ViewMode) => void
  setBlend: (v: number) => void
  setLevel: (v: number) => void
  setWidth: (v: number) => void
  toggleLandmarks: () => void
  toggleCrosshair: () => void

  send: (text: string) => void
}

export function useSession(): SessionState {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [cases, setCases] = useState<ImagingCase[]>([])
  const [methods, setMethods] = useState<RegistrationMethod[]>([])
  const [caseId, setCaseId] = useState<string | null>(null)
  const [methodId, setMethodId] = useState<string | null>(null)

  const [result, setResult] = useState<RegistrationResult | null>(null)
  const [stageIndex, setStageIndex] = useState(-1)
  const [progress, setProgress] = useState(0)

  const [plane, setPlane] = useState<Plane>('axial')
  const [sliceT, setSliceT] = useState(0.5)
  const [viewMode, setViewMode] = useState<ViewMode>('overlay')
  const [blend, setBlend] = useState(0.55)
  const [level, setLevel] = useState(0.5)
  const [width, setWidth] = useState(0.85)
  const [showLandmarks, setShowLandmarks] = useState(true)
  const [showCrosshair, setShowCrosshair] = useState(true)

  const [messages, setMessages] = useState<ChatMessage[]>([GREETING])
  const [thinking, setThinking] = useState(false)

  const timerRef = useRef<number | null>(null)

  const caseItem = useMemo(() => cases.find((c) => c.id === caseId) ?? null, [cases, caseId])
  const method = useMemo(
    () => methods.find((m) => m.id === methodId) ?? null,
    [methods, methodId],
  )

  const status: RunStatus = result?.status ?? 'idle'

  /* ---------------- boot ---------------- */
  useEffect(() => {
    let alive = true
    Promise.all([fetchCases(), fetchMethods()])
      .then(([c, m]) => {
        if (!alive) return
        setCases(c)
        setMethods(m)
        const firstCase = c[0]
        const defaultMethod = m.find((x) => x.id === 'vm-diffeo') ?? m[0]
        setCaseId(firstCase?.id ?? null)
        setMethodId(defaultMethod?.id ?? null)
        if (firstCase && defaultMethod) setResult(seedResult(firstCase, defaultMethod))
        setLoading(false)
      })
      .catch((e: unknown) => {
        if (!alive) return
        setError(e instanceof Error ? e.message : String(e))
        setLoading(false)
      })
    return () => {
      alive = false
    }
  }, [])

  const clearTimer = useCallback(() => {
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current)
      timerRef.current = null
    }
  }, [])

  useEffect(() => clearTimer, [clearTimer])

  /* ---------------- selection ---------------- */
  const selectCase = useCallback(
    (id: string) => {
      clearTimer()
      setCaseId(id)
      setSliceT(0.5)
      const c = cases.find((x) => x.id === id)
      const m = methods.find((x) => x.id === methodId)
      if (c && m) setResult(seedResult(c, m))
      setProgress(0)
      setStageIndex(-1)
      setMessages((prev) => [
        ...prev,
        {
          id: uid('m'),
          role: 'assistant',
          at: Date.now(),
          text: c
            ? `Loaded ${c.id} — ${c.label}. ${c.notes} Fixed: ${c.fixedModality} ${c.fixedSequence}. Moving: ${c.movingModality} (intraop). Press Run when ready.`
            : 'Case not found.',
        },
      ])
    },
    [cases, methods, methodId, clearTimer],
  )

  const selectMethod = useCallback(
    (id: string) => {
      clearTimer()
      setMethodId(id)
      const c = cases.find((x) => x.id === caseId)
      const m = methods.find((x) => x.id === id)
      if (c && m) setResult(seedResult(c, m))
      setProgress(0)
      setStageIndex(-1)
    },
    [cases, methods, caseId, clearTimer],
  )

  /* ---------------- run loop ---------------- */
  const stop = useCallback(() => {
    clearTimer()
    setProgress(0)
    setStageIndex(-1)
    setResult((prev) => (prev ? { ...prev, status: 'idle', progress: 0 } : prev))
  }, [clearTimer])

  const run = useCallback(() => {
    if (!caseItem || !method) return
    clearTimer()
    const runId = uid('run')

    setMessages((prev) => [
      ...prev,
      {
        id: uid('m'),
        role: 'user',
        at: Date.now(),
        text: `Run ${method.shortName} on ${caseItem.id}.`,
      },
    ])
    setThinking(true)

    let tick = 0
    const step = () => {
      const r = runProgressTick(runId, caseItem, method, tick)
      setResult(r)
      setProgress(r.progress)
      setStageIndex(r.stageIndex)

      if (r.status === 'complete' || r.status === 'failed') {
        clearTimer()
        setThinking(false)
        setMessages((prev) => [
          ...prev,
          {
            id: uid('m'),
            role: 'assistant',
            at: Date.now(),
            text:
              r.status === 'failed'
                ? `⚠ ${caseItem.id} failed with ${method.shortName}. ${r.error} I would retry with the diffeomorphic model at a coarser pyramid level, or fall back to rigid + manual landmark refinement.`
                : r.summary,
          },
        ])
        return
      }
      tick += 1
    }

    step()
    timerRef.current = window.setInterval(step, RUN_TICK_MS)
  }, [caseItem, method, clearTimer])

  /* ---------------- assistant ---------------- */
  const send = useCallback(
    (text: string) => {
      const trimmed = text.trim()
      if (!trimmed || thinking) return
      setMessages((prev) => [
        ...prev,
        { id: uid('m'), role: 'user', at: Date.now(), text: trimmed },
      ])
      setThinking(true)

      window.setTimeout(() => {
        setMessages((prev) => [
          ...prev,
          {
            id: uid('m'),
            role: 'assistant',
            at: Date.now(),
            text: draftReply(trimmed, caseItem, method, result),
          },
        ])
        setThinking(false)
      }, 850)
    },
    [caseItem, method, result, thinking],
  )

  return {
    loading,
    error,
    cases,
    methods,
    caseItem,
    method,
    result,
    status,
    progress,
    stageIndex,

    plane,
    sliceT,
    viewMode,
    blend,
    level,
    width,
    showLandmarks,
    showCrosshair,

    messages,
    thinking,

    selectCase,
    selectMethod,
    run,
    stop,
    setPlane,
    setSliceT,
    setViewMode,
    setBlend,
    setLevel,
    setWidth,
    toggleLandmarks: () => setShowLandmarks((v) => !v),
    toggleCrosshair: () => setShowCrosshair((v) => !v),
    send,
  }
}

/* ---------------------------------------------------------------- */
/* Canned assistant replies (mock — no LLM wired up yet)            */
/* ---------------------------------------------------------------- */

function meanDiceOf(result: RegistrationResult) {
  return result.dice.reduce((s, d) => s + d.dice, 0) / Math.max(1, result.dice.length)
}

function draftReply(
  q: string,
  caseItem: ImagingCase | null,
  method: RegistrationMethod | null,
  result: RegistrationResult | null,
): string {
  const lower = q.toLowerCase()
  const c = caseItem?.id ?? 'this case'
  const m = method?.shortName ?? 'the current method'

  if (!result || result.status !== 'complete') {
    return `No completed run for ${c} yet — run ${m} first and I will have real numbers to reason about.`
  }

  if (lower.includes('summar') || lower.includes('recap')) return result.summary

  if (lower.includes('tre') || lower.includes('error') || lower.includes('landmark')) {
    const worst = [...result.landmarks].sort((a, b) => b.treMm - a.treMm)[0]
    if (!worst) return 'Landmark table is empty for this run.'
    return (
      `Mean TRE is ${result.treMeanMm.toFixed(1)} mm (median ${result.treMedianMm.toFixed(1)}, ` +
      `max ${result.treMaxMm.toFixed(1)}, HD95 ${result.hd95Mm.toFixed(1)} mm) across ${result.landmarks.length} landmarks. ` +
      `The largest residual is on ${worst.label.toLowerCase()} at ${worst.treMm} mm. ` +
      `Target landmarks sit tighter than the safety landmarks, which is what you want for resection control.`
    )
  }

  if (lower.includes('compare') || lower.includes('method') || lower.includes('better')) {
    return (
      `For ${c}, ${m} reached ${result.treMeanMm.toFixed(1)} mm mean TRE with mean Dice ${meanDiceOf(result).toFixed(3)}. ` +
      `The symmetric SVF variant usually trims another ~0.4 mm on cases with large shift; the plain unsupervised model is ~2.5× faster but loses roughly 0.03 Dice on the resection cavity.`
    )
  }

  if (lower.includes('diffeo') || lower.includes('jacobian') || lower.includes('fold')) {
    return result.jacobianNegativePct < 1
      ? `Only ${result.jacobianNegativePct}% of the field has a negative Jacobian determinant — the warp is effectively diffeomorphic and safe to invert for navigation.`
      : `Heads up: ${result.jacobianNegativePct}% of voxels show negative Jacobian determinants, so the warp folds. Do not invert it; increase the regulariser weight or switch to the diffeomorphic model.`
  }

  if (lower.includes('dice') || lower.includes('overlap') || lower.includes('structure')) {
    const best = [...result.dice].sort((a, b) => b.dice - a.dice)[0]
    const worst = [...result.dice].sort((a, b) => a.dice - b.dice)[0]
    if (!best || !worst) return 'No Dice scores available for this run.'
    return (
      `Dice ranges from ${worst.dice.toFixed(3)} on ${worst.structure.toLowerCase()} to ` +
      `${best.dice.toFixed(3)} on ${best.structure.toLowerCase()}. Small structures are always hardest — ` +
      `partial-volume effects dominate once the voxel size approaches the structure margin.`
    )
  }

  if (lower.includes('ncc') || lower.includes('similarity')) {
    return `Normalised cross-correlation between fixed and warped moving is ${result.ncc.toFixed(3)}. Values above ~0.85 indicate strong intensity agreement after warp.`
  }

  if (lower.includes('time') || lower.includes('fast') || lower.includes('runtime')) {
    return `${m} finished in ${(result.runtimeMs / 1000).toFixed(1)} s end-to-end on this volume. Inference itself is a single forward pass; most of the wall time is I/O and metric computation.`
  }

  if (lower.includes('brain shift') || lower.includes('shift')) {
    return `Estimated brain shift on ${c} is consistent with the resection: the midline and ventricle landmarks drift toward the cavity. That is exactly the deformation the dense field is correcting — the affine baseline cannot represent it.`
  }

  if (lower.includes('safe') || lower.includes('threshold') || lower.includes('navigat')) {
    return result.treMeanMm < 3
      ? `At ${result.treMeanMm.toFixed(1)} mm mean TRE you are inside the common 3 mm frameless-navigation tolerance, and HD95 of ${result.hd95Mm.toFixed(1)} mm keeps the tail in check. Still verify against an intraoperative landmark before acting.`
      : `At ${result.treMeanMm.toFixed(1)} mm mean TRE this sits outside the 3 mm navigation tolerance (HD95 ${result.hd95Mm.toFixed(1)} mm). Treat as qualitative guidance only until refined.`
  }

  return (
    `Here is where ${c} stands with ${m}: mean TRE ${result.treMeanMm.toFixed(1)} mm, ` +
    `mean Dice ${meanDiceOf(result).toFixed(3)}, ${result.jacobianNegativePct}% Jacobian folding, ` +
    `${(result.runtimeMs / 1000).toFixed(1)} s runtime. ` +
    `Ask me about TRE, Dice, the deformation field, or method comparisons.`
  )
}



