import { useEffect, useMemo, useRef } from 'react'
import { Crosshair, Eye, ScanLine } from 'lucide-react'
import { GRID, synthMRI, synthUS, warp } from '../lib/imaging'
import { drawViewer } from './viewerDraw'
import type { SessionState } from '../state/useSession'
import type { Plane, ViewMode } from '../types'
import { Pill } from './ui'

const VIEW_MODES: { id: ViewMode; label: string }[] = [
  { id: 'overlay', label: 'Fusion' },
  { id: 'checker', label: 'Checker' },
  { id: 'side-by-side', label: 'Split' },
  { id: 'difference', label: 'Δ residual' },
]

const PLANES: { id: Plane; label: string }[] = [
  { id: 'axial', label: 'AX' },
  { id: 'coronal', label: 'COR' },
  { id: 'sagittal', label: 'SAG' },
]

/** Stable per-case seed so a case always looks the same. */
function caseSeed(id: string) {
  let h = 0
  for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) | 0
  return Math.abs(h) % 100000
}

export function Viewer({ s }: { s: SessionState }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const seed = caseSeed(s.caseItem?.id ?? 'none')

  // Heavy pixel work, memoised per inputs.
  const buffers = useMemo(() => {
    const spec = { plane: s.plane, sliceT: s.sliceT, seed, shift: 0.35 }
    const fixed = new Float32Array(GRID * GRID)
    const movingRaw = new Float32Array(GRID * GRID)
    const moving = new Float32Array(GRID * GRID)
    synthMRI(spec, fixed)
    synthUS(spec, movingRaw)
    // Mis-registration magnitude collapses once a run completes.
    warp(movingRaw, moving, seed, s.status === 'complete' ? 0.12 : 0.9)
    return { fixed, moving }
  }, [s.plane, s.sliceT, seed, s.status])

  const paint = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    drawViewer({
      canvas,
      fixed: buffers.fixed,
      moving: buffers.moving,
      plane: s.plane,
      sliceT: s.sliceT,
      level: s.level,
      width: s.width,
      blend: s.blend,
      viewMode: s.viewMode,
      showCrosshair: s.showCrosshair,
      showLandmarks: s.showLandmarks,
      status: s.status,
      landmarks: s.result?.landmarks ?? [],
    })
  }

  // Draw on mount and after every control/state change.
  useEffect(paint)

  // Keep the scan line animating while a run is in flight.
  useEffect(() => {
    if (s.status !== 'running') return
    const id = window.setInterval(paint, 80)
    return () => window.clearInterval(id)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [buffers, s])

  return (
    <div className="flex h-full min-w-0 flex-1 flex-col bg-void">
      <Toolbar s={s} />

      <div className="stage-grid relative min-h-0 flex-1 overflow-hidden">
        <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />

        {/* Case / slice read-out ------------------------------ */}
        <div className="pointer-events-none absolute top-3 left-3">
          <div className="rounded-md border border-line/80 bg-void/75 px-2.5 py-1.5 backdrop-blur-sm">
            <div className="font-mono text-[11px] font-medium text-mist">
              {s.caseItem?.id ?? '—'} <span className="text-mist-3">/</span>{' '}
              {s.plane.toUpperCase()}
            </div>
            <div className="font-mono text-[10px] text-mist-3">
              slice {Math.round(s.sliceT * ((s.caseItem?.dims[2] ?? 176) - 1) + 1)}/
              {s.caseItem?.dims[2] ?? 176} · {s.caseItem?.spacing.join('·') ?? '—'} mm
            </div>
          </div>
        </div>

        {/* Status chip ------------------------------------------ */}
        <div className="pointer-events-none absolute top-3 right-3">
          {s.status === 'running' && (
            <Pill tone="scan" className="animate-nv-pulse">
              Registering… {Math.round(s.progress * 100)}%
            </Pill>
          )}
          {s.status === 'complete' && s.result && (
            <Pill tone="good">Aligned · mean TRE {s.result.treMeanMm} mm</Pill>
          )}
          {s.status === 'failed' && <Pill tone="bad">Run failed</Pill>}
          {s.status === 'idle' && <Pill tone="warn">Awaiting run</Pill>}
        </div>

        {/* Modality legend -------------------------------------- */}
        <div className="pointer-events-none absolute bottom-3 left-3 rounded-md border border-line/80 bg-void/75 px-2.5 py-1.5 backdrop-blur-sm">
          <div className="flex items-center gap-3 text-[10px] text-mist-2">
            <span className="flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full bg-fixed" /> Fixed · MRI T1c
            </span>
            <span className="flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full bg-moving" /> Moving · US
            </span>
            {s.status === 'complete' && s.showLandmarks && (
              <>
                <span className="flex items-center gap-1.5">
                  <span className="h-2 w-2 rounded-full border border-good" /> Landmark
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="text-moving">✕</span> Warped
                </span>
              </>
            )}
          </div>
        </div>
      </div>

      <Footer s={s} />
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* Toolbar — plane, view mode, overlays, blend, slice, window/level    */
/* ------------------------------------------------------------------ */

function Toolbar({ s }: { s: SessionState }) {
  return (
    <div className="flex flex-wrap items-center gap-2 border-b border-line bg-panel px-3 py-2">
      <div className="flex items-center rounded-md border border-line bg-panel-2 p-0.5">
        {PLANES.map((p) => (
          <button
            key={p.id}
            onClick={() => s.setPlane(p.id)}
            className={`rounded px-2.5 py-1 text-[10.5px] font-semibold tracking-wide transition-colors ${
              s.plane === p.id ? 'bg-scan/15 text-scan' : 'text-mist-3 hover:text-mist-2'
            }`}
          >
            {p.label}
          </button>
        ))}
      </div>

      <span className="h-4 w-px bg-line" />

      <div className="flex items-center gap-0.5">
        {VIEW_MODES.map((m) => (
          <button
            key={m.id}
            onClick={() => s.setViewMode(m.id)}
            className={`rounded px-2.5 py-1 text-[11px] transition-colors ${
              s.viewMode === m.id
                ? 'bg-card-2 text-mist ring-1 ring-line-2'
                : 'text-mist-3 hover:text-mist-2'
            }`}
          >
            {m.label}
          </button>
        ))}
      </div>

      <span className="h-4 w-px bg-line" />

      <button
        onClick={s.toggleCrosshair}
        className={`flex items-center gap-1.5 rounded px-2 py-1 text-[11px] transition-colors ${
          s.showCrosshair ? 'text-scan' : 'text-mist-3 hover:text-mist-2'
        }`}
        title="Toggle crosshair"
      >
        <Crosshair className="h-3.5 w-3.5" /> Crosshair
      </button>
      <button
        onClick={s.toggleLandmarks}
        className={`flex items-center gap-1.5 rounded px-2 py-1 text-[11px] transition-colors ${
          s.showLandmarks ? 'text-scan' : 'text-mist-3 hover:text-mist-2'
        }`}
        title="Toggle landmark overlay"
      >
        <Eye className="h-3.5 w-3.5" /> Landmarks
      </button>

      <div className="ml-auto flex items-center gap-3">
        <label
          className={`flex items-center gap-2 text-[10.5px] ${s.viewMode === 'overlay' ? 'text-mist-3' : 'text-mist-3/40'}`}
        >
          <span className="tracking-wide">US α</span>
          <input
            type="range"
            min={0}
            max={100}
            value={Math.round(s.blend * 100)}
            onChange={(e) => s.setBlend(Number(e.target.value) / 100)}
            disabled={s.viewMode !== 'overlay'}
            className="nv-range w-20 disabled:opacity-40"
            style={{ ['--fill' as string]: `${Math.round(s.blend * 100)}%` }}
          />
          <span className="w-6 font-mono tnum text-mist-2">{Math.round(s.blend * 100)}</span>
        </label>

        <span className="h-4 w-px bg-line" />

        <label className="flex items-center gap-2 text-[10.5px] text-mist-3">
          <ScanLine className="h-3.5 w-3.5" />
          <input
            type="range"
            min={0}
            max={1000}
            value={Math.round(s.sliceT * 1000)}
            onChange={(e) => s.setSliceT(Number(e.target.value) / 1000)}
            className="nv-range w-36"
            style={{ ['--fill' as string]: `${Math.round(s.sliceT * 100)}%` }}
          />
          <span className="w-8 font-mono tnum text-mist-2">{(s.sliceT * 100).toFixed(0)}%</span>
        </label>

        <span className="h-4 w-px bg-line" />

        <label className="flex items-center gap-2 text-[10.5px] text-mist-3">
          W/L
          <input
            type="range"
            min={20}
            max={200}
            value={Math.round(s.width * 100)}
            onChange={(e) => s.setWidth(Number(e.target.value) / 100)}
            className="nv-range w-20"
            style={{ ['--fill' as string]: `${((s.width * 100 - 20) / 180) * 100}%` }}
          />
          <input
            type="range"
            min={0}
            max={100}
            value={Math.round(s.level * 100)}
            onChange={(e) => s.setLevel(Number(e.target.value) / 100)}
            className="nv-range w-20"
            style={{ ['--fill' as string]: `${Math.round(s.level * 100)}%` }}
          />
        </label>
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* Footer — pipeline stage strip shown while a run is in flight        */
/* ------------------------------------------------------------------ */

function Footer({ s }: { s: SessionState }) {
  const stages = s.result?.pipeline ?? []
  const failed = s.status === 'failed'
  const done = s.status === 'complete'

  return (
    <div className="border-t border-line bg-panel px-3 py-2">
      <div className="flex items-center gap-1 overflow-x-auto scroll-slim">
        {stages.map((st, i) => {
          const state = failed
            ? i <= (s.stageIndex ?? 0)
              ? i === s.stageIndex
                ? 'failed'
                : 'done'
              : 'todo'
            : done
              ? 'done'
              : i < s.stageIndex
                ? 'done'
                : i === s.stageIndex
                  ? 'active'
                  : 'todo'

          return (
            <div key={st.key} className="flex shrink-0 items-center">
              <div
                className={`flex items-center gap-1.5 rounded px-2 py-1 text-[10.5px] transition-colors ${
                  state === 'active'
                    ? 'bg-scan/12 text-scan'
                    : state === 'done'
                      ? 'text-good'
                      : state === 'failed'
                        ? 'bg-bad/12 text-bad'
                        : 'text-mist-3/60'
                }`}
                title={st.detail}
              >
                <span
                  className={`grid h-3.5 w-3.5 place-items-center rounded-full border text-[8px] ${
                    state === 'active'
                      ? 'animate-nv-pulse border-scan'
                      : state === 'done'
                        ? 'border-good'
                        : state === 'failed'
                          ? 'border-bad'
                          : 'border-line-2'
                  }`}
                >
                  {state === 'done' && '✓'}
                </span>
                <span className="font-medium whitespace-nowrap">{st.label}</span>
              </div>
              {i < stages.length - 1 && <span className="mx-0.5 h-px w-4 bg-line-2" />}
            </div>
          )
        })}

        {s.status === 'idle' && (
          <span className="ml-auto shrink-0 pl-3 text-[10.5px] whitespace-nowrap text-mist-3">
            Idle — configure a case and method, then run.
          </span>
        )}
        {done && s.result && (
          <span className="ml-auto shrink-0 pl-3 font-mono text-[10.5px] whitespace-nowrap text-mist-3 tnum">
            {(s.result.runtimeMs / 1000).toFixed(2)} s · {s.result.runId}
          </span>
        )}
        {failed && (
          <span className="ml-auto shrink-0 pl-3 text-[10.5px] whitespace-nowrap text-bad">
            Run aborted — see assistant
          </span>
        )}
      </div>
    </div>
  )
}

