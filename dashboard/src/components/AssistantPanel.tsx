import { useEffect, useRef, useState } from 'react'
import {
  Activity,
  CircleAlert,
  CircleCheck,
  GitCompareArrows,
  Info,
  Loader2,
  Sparkles,
  TrendingDown,
  TrendingUp,
} from 'lucide-react'
import type { SessionState } from '../state/useSession'
import { QUICK_PROMPTS } from '../state/useSession'
import { comparison } from '../api/client'
import type { DiceScore, RegistrationResult } from '../types'
import { Meter, Pill, Section, Stat } from './ui'

function diceTone(d: number): 'good' | 'scan' | 'warn' | 'bad' {
  if (d >= 0.9) return 'good'
  if (d >= 0.8) return 'scan'
  if (d >= 0.7) return 'warn'
  return 'bad'
}

function treTone(v: number): 'good' | 'warn' | 'bad' {
  if (v < 3) return 'good'
  if (v < 5) return 'warn'
  return 'bad'
}

const TONE_TEXT = { good: 'text-good', warn: 'text-warn', bad: 'text-bad' } as const

/* ------------------------------------------------------------------ */
/* Status header                                                       */
/* ------------------------------------------------------------------ */

function StatusBanner({ s }: { s: SessionState }) {
  const r = s.result
  const running = s.status === 'running'
  const failed = s.status === 'failed'
  const complete = s.status === 'complete'

  return (
    <div className="border-b border-line bg-panel-2 px-4 py-3 panel-noise">
      <div className="flex items-center gap-2">
        <span
          className={`grid h-6 w-6 place-items-center rounded-md border ${
            failed
              ? 'border-bad/40 bg-bad/12 text-bad'
              : running
                ? 'border-scan/40 bg-scan/12 text-scan'
                : complete
                  ? 'border-good/40 bg-good/12 text-good'
                  : 'border-line-2 bg-card text-mist-3'
          }`}
        >
          {running ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : failed ? (
            <CircleAlert className="h-3.5 w-3.5" />
          ) : complete ? (
            <CircleCheck className="h-3.5 w-3.5" />
          ) : (
            <Activity className="h-3.5 w-3.5" />
          )}
        </span>
        <div className="min-w-0 flex-1">
          <div className="text-[12px] font-semibold text-mist">
            {failed
              ? 'Registration failed'
              : running
                ? 'Registering…'
                : complete
                  ? 'Registration complete'
                  : 'Awaiting run'}
          </div>
          <div className="truncate font-mono text-[10.5px] text-mist-3">
            {s.caseItem?.id ?? '—'} → {s.method?.shortName ?? '—'}
          </div>
        </div>
        {complete && r && <Pill tone={treTone(r.treMeanMm)}>{r.treMeanMm} mm</Pill>}
        {running && <Pill tone="scan">{Math.round(s.progress * 100)}%</Pill>}
        {failed && <Pill tone="bad">error</Pill>}
      </div>

      <div className="mt-2.5 h-1 w-full overflow-hidden rounded-full bg-line">
        <div
          className={`h-full rounded-full transition-[width] duration-500 ${
            failed ? 'bg-bad' : 'bg-gradient-to-r from-scan-2 to-scan'
          }`}
          style={{ width: `${(failed ? 0.42 : s.progress) * 100}%` }}
        />
      </div>

      {failed && r?.error && (
        <p className="mt-2 rounded-md border border-bad/30 bg-bad/8 px-2.5 py-2 text-[11px] leading-relaxed text-bad/90">
          {r.error}
        </p>
      )}

      {running && r && (
        <div className="mt-2 text-[11px] text-mist-3">
          <span className="text-scan">{r.pipeline[Math.max(0, s.stageIndex)]?.label}</span>
          {' — '}
          {r.pipeline[Math.max(0, s.stageIndex)]?.detail}
        </div>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* TRE / Dice metrics                                                   */
/* ------------------------------------------------------------------ */

function TreCard({ r }: { r: RegistrationResult }) {
  const tone = treTone(r.treMeanMm)
  const within3 = Math.max(0, Math.min(1, 1 - r.treMeanMm / 6))
  return (
    <div className="rounded-lg border border-line bg-card/60 p-3">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-[10px] tracking-[0.12em] text-mist-3 uppercase">
          Target registration error
        </span>
        <Pill tone={tone}>
          {tone === 'good' ? '< 3 mm tol.' : tone === 'warn' ? 'borderline' : 'above tol.'}
        </Pill>
      </div>

      <div className="grid grid-cols-3 gap-2">
        <div>
          <div className={`font-mono text-[26px] leading-none font-semibold tnum ${TONE_TEXT[tone]}`}>
            {r.treMeanMm.toFixed(1)}
          </div>
          <div className="mt-1 text-[9.5px] tracking-wide text-mist-3 uppercase">mean mm</div>
        </div>
        <div className="border-l border-line pl-3">
          <div className="font-mono text-[16px] leading-none font-semibold tnum text-mist-2">
            {r.treMedianMm.toFixed(1)}
          </div>
          <div className="mt-1 text-[9.5px] tracking-wide text-mist-3 uppercase">median</div>
        </div>
        <div className="border-l border-line pl-3">
          <div className="font-mono text-[16px] leading-none font-semibold tnum text-mist-2">
            {r.treMaxMm.toFixed(1)}
          </div>
          <div className="mt-1 text-[9.5px] tracking-wide text-mist-3 uppercase">max</div>
        </div>
      </div>

      <div className="mt-3">
        <Meter
          value={within3}
          tone={tone === 'good' ? 'good' : tone === 'warn' ? 'warn' : 'bad'}
        />
        <div className="mt-1 flex justify-between font-mono text-[9.5px] text-mist-3">
          <span>0 mm</span>
          <span>HD95 {r.hd95Mm} mm</span>
          <span>6 mm</span>
        </div>
      </div>
    </div>
  )
}

function DiceRow({ d }: { d: DiceScore }) {
  const tone = diceTone(d.dice)
  const delta = d.dice - d.prior
  const up = delta >= 0
  return (
    <div className="py-1.5">
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <span className="truncate text-[11px] text-mist-2">{d.structure}</span>
        <span className="flex items-baseline gap-1.5">
          <span className="font-mono text-[11.5px] font-medium tnum text-mist">
            {d.dice > 0 ? d.dice.toFixed(3) : '—.———'}
          </span>
          {d.prior > 0 && (
            <span className={`flex items-center text-[9.5px] ${up ? 'text-good' : 'text-bad'}`}>
              {up ? <TrendingUp className="h-2.5 w-2.5" /> : <TrendingDown className="h-2.5 w-2.5" />}
              {Math.abs(delta).toFixed(3)}
            </span>
          )}
        </span>
      </div>
      <Meter value={d.dice} tone={tone} />
    </div>
  )
}
/* ------------------------------------------------------------------ */
/* Method comparison table                                             */
/* ------------------------------------------------------------------ */

function Comparison({ s }: { s: SessionState }) {
  if (!s.caseItem || !s.method) return null
  const rows = comparison(s.caseItem, s.method.id)
  const best = rows[0]

  return (
    <Section
      title="Method comparison"
      hint="mock scorer"
      action={<GitCompareArrows className="h-3.5 w-3.5 text-mist-3" />}
    >
      <table className="w-full text-[11px]">
        <thead>
          <tr className="text-[9.5px] tracking-[0.1em] text-mist-3 uppercase">
            <th className="pb-1 text-left font-medium">Method</th>
            <th className="pb-1 text-right font-medium">TRE mm</th>
            <th className="pb-1 text-right font-medium">Dice</th>
            <th className="pb-1 text-right font-medium">s</th>
          </tr>
        </thead>
        <tbody className="font-mono tnum">
          {rows.map((r) => (
            <tr
              key={r.methodId}
              className={`border-t border-line/70 ${r.isCurrent ? 'bg-scan/8 text-mist' : 'text-mist-2'}`}
            >
              <td className="py-1.5 pr-2 font-sans">
                <span className="flex items-center gap-1.5">
                  {r.isCurrent && <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-scan" />}
                  <span className={r.isCurrent ? 'font-medium' : ''}>{r.methodName}</span>
                  {r.methodId === best.methodId && (
                    <span className="text-[9px] text-good">best</span>
                  )}
                </span>
              </td>
              <td
                className={`py-1.5 text-right ${r.methodId === best.methodId ? 'text-good' : ''}`}
              >
                {r.treMeanMm.toFixed(1)}
              </td>
              <td className="py-1.5 text-right">{r.diceMean.toFixed(3)}</td>
              <td className="py-1.5 text-right text-mist-3">{r.runtimeS.toFixed(1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Section>
  )
}

/* ------------------------------------------------------------------ */
/* Chat                                                                */
/* ------------------------------------------------------------------ */

function Chat({ s }: { s: SessionState }) {
  const [draft, setDraft] = useState('')
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [s.messages, s.thinking])

  const submit = (text: string) => {
    if (!text.trim()) return
    s.send(text)
    setDraft('')
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="scroll-slim min-h-0 flex-1 space-y-3 overflow-y-auto px-4 py-3">
        {s.messages.map((m) => (
          <div
            key={m.id}
            className={`animate-nv-rise flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[92%] rounded-lg px-2.5 py-2 text-[11.5px] leading-relaxed ${
                m.role === 'user'
                  ? 'bg-scan/12 text-mist ring-1 ring-scan/25'
                  : 'border border-line bg-card/70 text-mist-2'
              }`}
            >
              {m.role === 'assistant' && (
                <div className="mb-1 flex items-center gap-1 text-[9.5px] tracking-[0.1em] text-ai uppercase">
                  <Sparkles className="h-2.5 w-2.5" /> NeuroVista AI
                </div>
              )}
              <span className="whitespace-pre-wrap">{m.text}</span>
            </div>
          </div>
        ))}

        {s.thinking && (
          <div className="flex justify-start">
            <div className="flex items-center gap-1.5 rounded-lg border border-line bg-card/70 px-2.5 py-2">
              <span className="h-1.5 w-1.5 animate-nv-pulse rounded-full bg-ai" />
              <span className="h-1.5 w-1.5 animate-nv-pulse rounded-full bg-ai [animation-delay:0.2s]" />
              <span className="h-1.5 w-1.5 animate-nv-pulse rounded-full bg-ai [animation-delay:0.4s]" />
            </div>
          </div>
        )}
        <div ref={endRef} />
      </div>

      {/* Quick prompts */}
      <div className="flex flex-wrap gap-1.5 border-t border-line px-4 pt-2.5">
        {QUICK_PROMPTS.map((q) => (
          <button
            key={q}
            onClick={() => submit(q)}
            disabled={s.thinking}
            className="rounded-full border border-line-2 bg-card/60 px-2 py-[3px] text-[10px] text-mist-3 transition-colors hover:border-ai/50 hover:text-ai disabled:opacity-40"
          >
            {q}
          </button>
        ))}
      </div>

      {/* Composer */}
      <form
        onSubmit={(e) => {
          e.preventDefault()
          submit(draft)
        }}
        className="flex items-center gap-2 px-4 py-3"
      >
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder={s.thinking ? 'Assistant is thinking…' : 'Ask about TRE, Dice, Jacobian…'}
          className="min-w-0 flex-1 rounded-md border border-line bg-panel-2 px-2.5 py-2 text-[11.5px] text-mist placeholder:text-mist-3/70 focus:border-ai/60 focus:outline-none"
        />
        <button
          type="submit"
          disabled={!draft.trim() || s.thinking}
          className="rounded-md bg-ai px-3 py-2 text-[11.5px] font-semibold text-void transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-35"
        >
          Send
        </button>
      </form>
    </div>
  )
}



/* ------------------------------------------------------------------ */
/* Panel shell                                                         */
/* ------------------------------------------------------------------ */

export function AssistantPanel({ s }: { s: SessionState }) {
  const r = s.result
  const complete = s.status === 'complete' && r

  return (
    <aside className="flex h-full w-[360px] shrink-0 flex-col border-l border-line bg-panel">
      {/* Header ------------------------------------------------- */}
      <div className="flex items-center gap-2.5 border-b border-line px-4 py-3.5 panel-noise">
        <div className="relative grid h-8 w-8 place-items-center rounded-lg border border-ai/40 bg-ai/10">
          <Sparkles className="h-4 w-4 text-ai" />
          <span className="absolute -right-0.5 -bottom-0.5 h-2 w-2 rounded-full border-2 border-panel bg-good" />
        </div>
        <div className="leading-tight">
          <div className="text-[13px] font-semibold text-mist">NeuroVista AI</div>
          <div className="text-[10px] text-mist-3">registration assistant · mock LLM</div>
        </div>
        <Pill tone="ai" className="ml-auto">
          nv-mock-1
        </Pill>
      </div>

      <StatusBanner s={s} />

      {/* Metrics ------------------------------------------------ */}
      <div className="scroll-slim min-h-0 flex-1 overflow-y-auto">
        {r && (
          <Section title="Metrics" hint={complete ? r.runId : 'last run'}>
            {complete ? (
              <div className="space-y-3">
                <TreCard r={r} />

                <div className="grid grid-cols-3 gap-2">
                  <Stat
                    value={r.hd95Mm.toFixed(1)}
                    unit="mm"
                    caption="HD95"
                    tone={r.hd95Mm < 5 ? 'good' : r.hd95Mm < 8 ? 'warn' : 'bad'}
                  />
                  <Stat
                    value={r.ncc.toFixed(2)}
                    caption="NCC"
                    tone={r.ncc > 0.85 ? 'good' : r.ncc > 0.75 ? 'warn' : 'bad'}
                  />
                  <Stat
                    value={r.jacobianNegativePct.toFixed(1)}
                    unit="%"
                    caption="Jac. < 0"
                    tone={r.jacobianNegativePct < 1 ? 'good' : 'warn'}
                  />
                </div>

                <div className="rounded-lg border border-line bg-card/60 px-3 pt-2 pb-1">
                  <div className="mb-1 flex items-center justify-between">
                    <span className="text-[10px] tracking-[0.12em] text-mist-3 uppercase">
                      Dice by structure
                    </span>
                    <span className="text-[9.5px] text-mist-3">vs affine prior</span>
                  </div>
                  {r.dice.map((d) => (
                    <DiceRow key={d.structure} d={d} />
                  ))}
                </div>

                <div className="grid grid-cols-2 gap-2">
                  <div className="rounded-lg border border-line bg-card/60 px-3 py-2">
                    <div className="text-[9.5px] tracking-[0.1em] text-mist-3 uppercase">
                      Landmarks
                    </div>
                    <div className="mt-1 font-mono text-[15px] font-semibold tnum text-mist">
                      {r.landmarks.length}
                      <span className="ml-1 text-[10px] font-normal text-mist-3">
                        ({r.landmarks.filter((l) => l.group === 'Target').length} target /{' '}
                        {r.landmarks.filter((l) => l.group === 'Safety').length} safety)
                      </span>
                    </div>
                  </div>
                  <div className="rounded-lg border border-line bg-card/60 px-3 py-2">
                    <div className="text-[9.5px] tracking-[0.1em] text-mist-3 uppercase">
                      Runtime
                    </div>
                    <div className="mt-1 font-mono text-[15px] font-semibold tnum text-mist">
                      {(r.runtimeMs / 1000).toFixed(2)}
                      <span className="ml-1 text-[10px] font-normal text-mist-3">s e2e</span>
                    </div>
                  </div>
                </div>

                <div className="rounded-lg border border-ai/25 bg-ai/6 px-3 py-2.5">
                  <div className="mb-1 flex items-center gap-1.5 text-[9.5px] tracking-[0.1em] text-ai uppercase">
                    <Sparkles className="h-2.5 w-2.5" /> Read-out
                  </div>
                  <p className="text-[11.5px] leading-relaxed text-mist-2">{r.summary}</p>
                </div>
              </div>
            ) : (
              <div className="rounded-lg border border-dashed border-line-2 px-3 py-6 text-center">
                <Info className="mx-auto mb-2 h-4 w-4 text-mist-3" />
                <p className="text-[11.5px] text-mist-3">
                  {s.status === 'running'
                    ? 'Metrics stream in as each pipeline stage completes…'
                    : 'No completed run yet for this pair. Press Run registration.'}
                </p>
              </div>
            )}
          </Section>
        )}

        <Comparison s={s} />
      </div>

      <Chat s={s} />
    </aside>
  )
}
