import type { ReactNode } from 'react'

/** A labelled card section used across the side panels. */
export function Section({
  title,
  hint,
  action,
  children,
  className = '',
}: {
  title: string
  hint?: string
  action?: ReactNode
  children: ReactNode
  className?: string
}) {
  return (
    <section className={`px-4 py-4 border-b border-line ${className}`}>
      <header className="flex items-baseline justify-between mb-3">
        <div className="flex items-baseline gap-2">
          <h2 className="text-[10.5px] font-semibold uppercase tracking-[0.14em] text-mist-3">
            {title}
          </h2>
          {hint && <span className="text-[10px] text-mist-3/70">{hint}</span>}
        </div>
        {action}
      </header>
      {children}
    </section>
  )
}

/** Small pill. `tone` drives the colour. */
export function Pill({
  children,
  tone = 'neutral',
  className = '',
}: {
  children: ReactNode
  tone?: 'neutral' | 'scan' | 'ai' | 'good' | 'warn' | 'bad' | 'ember'
  className?: string
}) {
  const tones: Record<string, string> = {
    neutral: 'bg-card text-mist-2 border-line-2',
    scan: 'bg-scan/12 text-scan border-scan/35',
    ai: 'bg-ai/12 text-ai border-ai/35',
    good: 'bg-good/12 text-good border-good/35',
    warn: 'bg-warn/12 text-warn border-warn/35',
    bad: 'bg-bad/12 text-bad border-bad/35',
    ember: 'bg-ember/12 text-ember border-ember/35',
  }
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-1.5 py-[1px] text-[10px] font-medium leading-[15px] ${tones[tone]} ${className}`}
    >
      {children}
    </span>
  )
}

/** Label → value row used in the "volume" read-outs. */
export function KV({ k, v, mono = true }: { k: string; v: ReactNode; mono?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-[3px]">
      <span className="text-[11px] text-mist-3">{k}</span>
      <span className={`text-[11px] text-mist-2 ${mono ? 'font-mono tnum' : ''}`}>{v}</span>
    </div>
  )
}

/** Big metric number with a caption and optional delta. */
export function Stat({
  value,
  unit,
  caption,
  tone = 'default',
  delta,
}: {
  value: string
  unit?: string
  caption: string
  tone?: 'default' | 'good' | 'warn' | 'bad'
  delta?: string
}) {
  const toneCls =
    tone === 'good' ? 'text-good' : tone === 'warn' ? 'text-warn' : tone === 'bad' ? 'text-bad' : 'text-mist'
  return (
    <div className="rounded-lg border border-line bg-card/70 px-3 py-2.5">
      <div className="flex items-baseline gap-1">
        <span className={`font-mono text-[22px] font-semibold leading-none tnum ${toneCls}`}>
          {value}
        </span>
        {unit && <span className="text-[11px] text-mist-3">{unit}</span>}
        {delta && <span className="ml-auto text-[10px] text-mist-3">{delta}</span>}
      </div>
      <div className="mt-1.5 text-[10px] uppercase tracking-[0.1em] text-mist-3">{caption}</div>
    </div>
  )
}

/** Horizontal meter with a 0–1 value. */
export function Meter({
  value,
  tone = 'scan',
  height = 4,
}: {
  value: number
  tone?: 'scan' | 'ai' | 'good' | 'warn' | 'bad'
  height?: number
}) {
  const bg =
    tone === 'good'
      ? 'bg-good'
      : tone === 'warn'
        ? 'bg-warn'
        : tone === 'bad'
          ? 'bg-bad'
          : tone === 'ai'
            ? 'bg-ai'
            : 'bg-scan'
  return (
    <div className="w-full rounded-full bg-line" style={{ height }}>
      <div
        className={`rounded-full ${bg} transition-[width] duration-700 ease-out`}
        style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%`, height }}
      />
    </div>
  )
}
