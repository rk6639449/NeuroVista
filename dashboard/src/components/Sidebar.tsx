import { useMemo, useState } from 'react'
import {
  Activity,
  Boxes,
  ChevronRight,
  CircleAlert,
  CircleCheck,
  Clock3,
  Cpu,
  Play,
  Search,
  Square,
} from 'lucide-react'
import type { SessionState } from '../state/useSession'
import { KV, Pill, Section } from './ui'

const FAMILY_TONE = {
  diffeomorphic: 'scan',
  unsupervised: 'ai',
  symmetric: 'good',
  baseline: 'neutral',
} as const

function CaseStatusDot({ status }: { status: string }) {
  if (status === 'registered') return <CircleCheck className="h-3.5 w-3.5 text-good" />
  if (status === 'failed') return <CircleAlert className="h-3.5 w-3.5 text-bad" />
  if (status === 'running') return <Activity className="h-3.5 w-3.5 text-scan animate-nv-pulse" />
  return <Clock3 className="h-3.5 w-3.5 text-mist-3" />
}

export function Sidebar({ s }: { s: SessionState }) {
  const [query, setQuery] = useState('')

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return s.cases
    return s.cases.filter(
      (c) =>
        c.id.toLowerCase().includes(q) ||
        c.label.toLowerCase().includes(q) ||
        c.subject.toLowerCase().includes(q),
    )
  }, [s.cases, query])

  const running = s.status === 'running'

  return (
    <aside className="flex h-full w-[300px] shrink-0 flex-col border-r border-line bg-panel">
      {/* Brand ------------------------------------------------- */}
      <div className="flex items-center gap-2.5 border-b border-line px-4 pt-4 pb-3.5 panel-noise">
        <div className="grid h-8 w-8 place-items-center rounded-lg border border-scan/40 bg-scan/10">
          <Boxes className="h-4 w-4 text-scan" />
        </div>
        <div className="leading-tight">
          <div className="text-[13.5px] font-semibold tracking-tight text-mist">
            Neuro<span className="text-scan">Vista</span>
          </div>
          <div className="text-[10px] uppercase tracking-[0.16em] text-mist-3">
            Registration console
          </div>
        </div>
        <Pill tone="ai" className="ml-auto">
          <Cpu className="h-2.5 w-2.5" /> v0.1
        </Pill>
      </div>

      {/* Scrollable body --------------------------------------- */}
      <div className="scroll-slim flex-1 overflow-y-auto">
        <Section
          title="Cases"
          hint={`${filtered.length}/${s.cases.length}`}
          action={<span className="font-mono text-[10px] text-mist-3/70">ReMIND</span>}
        >
          <div className="relative mb-2.5">
            <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-mist-3" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Filter by ID, label, subject…"
              className="w-full rounded-md border border-line bg-panel-2 py-1.5 pr-2.5 pl-8 text-[11.5px] text-mist placeholder:text-mist-3/70 focus:border-scan/50 focus:outline-none"
            />
          </div>

          <ul className="space-y-1">
            {filtered.map((c) => {
              const active = c.id === s.caseItem?.id
              return (
                <li key={c.id}>
                  <button
                    onClick={() => s.selectCase(c.id)}
                    className={`group flex w-full items-center gap-2 rounded-md border px-2.5 py-2 text-left transition-colors ${
                      active
                        ? 'border-scan/45 bg-scan/8'
                        : 'border-transparent hover:border-line-2 hover:bg-card/70'
                    }`}
                  >
                    <CaseStatusDot status={c.status} />
                    <div className="min-w-0 flex-1">
                      <div
                        className={`truncate text-[12px] font-medium ${active ? 'text-mist' : 'text-mist-2'}`}
                      >
                        {c.id}
                      </div>
                      <div className="truncate text-[10.5px] text-mist-3">{c.label}</div>
                    </div>
                    <ChevronRight
                      className={`h-3.5 w-3.5 shrink-0 transition-transform ${
                        active ? 'text-scan' : 'text-mist-3/50 group-hover:translate-x-0.5'
                      }`}
                    />
                  </button>
                </li>
              )
            })}
            {filtered.length === 0 && (
              <li className="rounded-md border border-dashed border-line-2 px-3 py-4 text-center text-[11px] text-mist-3">
                No cases match “{query}”.
              </li>
            )}
          </ul>
        </Section>

        {/* Case detail ------------------------------------------ */}
        {s.caseItem && (
          <Section title="Volume" hint={s.caseItem.id}>
            <div className="rounded-lg border border-line bg-card/60 px-3 py-2">
              <KV
                k="Fixed"
                v={`${s.caseItem.fixedModality} · ${s.caseItem.fixedSequence}`}
                mono={false}
              />
              <KV k="Moving" v={`${s.caseItem.movingModality} · intraop probe`} mono={false} />
              <KV k="Matrix" v={s.caseItem.dims.join(' × ')} />
              <KV k="Spacing" v={`${s.caseItem.spacing.join(' × ')} mm`} />
              <KV k="Acquired" v={s.caseItem.acquired} />
            </div>
            <p className="mt-2 text-[11px] leading-relaxed text-mist-3">{s.caseItem.notes}</p>
          </Section>
        )}

        {/* Method picker ---------------------------------------- */}
        <Section title="Method" hint={`${s.methods.length} available`}>
          <ul className="space-y-1.5">
            {s.methods.map((m) => {
              const active = m.id === s.method?.id
              return (
                <li key={m.id}>
                  <button
                    onClick={() => s.selectMethod(m.id)}
                    className={`w-full rounded-lg border px-3 py-2.5 text-left transition-colors ${
                      active
                        ? 'border-scan/45 bg-scan/8'
                        : 'border-line bg-card/50 hover:border-line-2 hover:bg-card'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <span
                        className={`grid h-3.5 w-3.5 shrink-0 place-items-center rounded-full border ${
                          active ? 'border-scan' : 'border-line-2'
                        }`}
                      >
                        {active && <span className="h-1.5 w-1.5 rounded-full bg-scan" />}
                      </span>
                      <span
                        className={`flex-1 truncate text-[12px] font-medium ${active ? 'text-mist' : 'text-mist-2'}`}
                      >
                        {m.name}
                      </span>
                      <Pill tone={FAMILY_TONE[m.family]}>{m.shortName}</Pill>
                    </div>
                    <div className="mt-1 pl-5.5 text-[10.5px] text-mist-3">{m.citation}</div>
                  </button>
                </li>
              )
            })}
          </ul>

          {s.method && (
            <div className="mt-2.5 rounded-lg border border-line bg-card/60 px-3 py-2">
              <p className="mb-2 text-[11px] leading-relaxed text-mist-3">{s.method.blurb}</p>
              <div className="grid grid-cols-2 gap-x-4">
                {s.method.params.map((p) => (
                  <div key={p.label} className="py-[3px]">
                    <div className="text-[9.5px] tracking-[0.1em] text-mist-3/80 uppercase">
                      {p.label}
                    </div>
                    <div className="truncate font-mono text-[11px] text-mist-2">{p.value}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </Section>
      </div>

      {/* Run controls (pinned) --------------------------------- */}
      <RunControls s={s} running={running} />
    </aside>
  )
}

function RunControls({ s, running }: { s: SessionState; running: boolean }) {
  return (
    <div className="border-t border-line bg-panel-2 px-4 py-3">
      <div className="mb-2 flex items-center justify-between text-[10.5px]">
        <span className="tracking-[0.12em] text-mist-3 uppercase">Pipeline</span>
        <span className="font-mono tnum text-mist-2">
          {running
            ? `${Math.round(s.progress * 100)}%`
            : s.status === 'complete'
              ? 'ready'
              : s.status === 'failed'
                ? 'failed'
                : 'idle'}
        </span>
      </div>
      <div className="mb-2.5 h-1 w-full overflow-hidden rounded-full bg-line">
        <div
          className="h-full rounded-full bg-gradient-to-r from-scan-2 to-scan transition-[width] duration-500"
          style={{ width: `${(running ? s.progress : s.status === 'complete' ? 1 : 0) * 100}%` }}
        />
      </div>

      <button
        onClick={running ? s.stop : s.run}
        disabled={!s.caseItem || !s.method || s.loading}
        className={`flex w-full items-center justify-center gap-1.5 rounded-md px-3 py-2 text-[12px] font-semibold transition-all disabled:cursor-not-allowed disabled:opacity-40 ${
          running
            ? 'bg-bad/15 text-bad ring-1 ring-bad/40 hover:bg-bad/25'
            : 'bg-scan text-void shadow-[0_6px_18px_-8px_rgba(53,223,208,0.9)] hover:bg-scan-2'
        }`}
      >
        {running ? (
          <>
            <Square className="h-3.5 w-3.5" /> Abort run
          </>
        ) : (
          <>
            <Play className="h-3.5 w-3.5" /> Run registration
          </>
        )}
      </button>

      <div className="mt-2 flex items-center justify-between text-[10px] text-mist-3">
        <span className="flex items-center gap-1">
          <span className="h-1.5 w-1.5 rounded-full bg-warn" /> Mock data — backend offline
        </span>
        <span className="font-mono">{s.method?.nominalRuntimeS ?? '—'}s nominal</span>
      </div>
    </div>
  )
}

