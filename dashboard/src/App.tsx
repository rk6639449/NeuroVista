import { Boxes, Database, GitBranch, Settings2 } from 'lucide-react'
import { AssistantPanel } from './components/AssistantPanel'
import { Sidebar } from './components/Sidebar'
import { Viewer } from './components/Viewer'
import { Pill } from './components/ui'
import { USE_MOCK } from './api/client'
import { useSession } from './state/useSession'

export default function App() {
  const s = useSession()

  if (s.error) {
    return (
      <div className="grid h-full place-items-center bg-void px-8 text-center">
        <div>
          <Boxes className="mx-auto mb-3 h-8 w-8 text-bad" />
          <h1 className="text-[15px] font-semibold text-mist">Failed to reach the API</h1>
          <p className="mt-1 font-mono text-[11.5px] text-mist-3">{s.error}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-full flex-col bg-void">
      {/* Top bar ---------------------------------------------------- */}
      <header className="flex h-11 shrink-0 items-center gap-3 border-b border-line bg-panel px-3">
        <div className="flex items-center gap-2 text-[11.5px] text-mist-2">
          <Database className="h-3.5 w-3.5 text-mist-3" />
          <span className="font-mono">NeuroVista</span>
          <span className="text-mist-3">/</span>
          <span className="font-mono">{s.caseItem?.id ?? 'no case'}</span>
          <span className="text-mist-3">/</span>
          <span className="font-mono text-mist-3">{s.method?.shortName ?? 'no method'}</span>
        </div>

        <div className="ml-auto flex items-center gap-2">
          <Pill tone={USE_MOCK ? 'warn' : 'good'}>
            <span className={`h-1.5 w-1.5 rounded-full ${USE_MOCK ? 'bg-warn' : 'bg-good'}`} />
            {USE_MOCK ? 'mock data' : 'live backend'}
          </Pill>
          <span className="hidden text-[10.5px] text-mist-3 sm:inline">
            MRI → US registration
          </span>
          <button
            className="grid h-7 w-7 place-items-center rounded-md text-mist-3 transition-colors hover:bg-card hover:text-mist-2"
            title="Settings"
          >
            <Settings2 className="h-3.5 w-3.5" />
          </button>
          <a
            href="https://github.com/rk6639449/NeuroVista"
            target="_blank"
            rel="noreferrer"
            className="grid h-7 w-7 place-items-center rounded-md text-mist-3 transition-colors hover:bg-card hover:text-mist-2"
            title="Repository"
          >
            <GitBranch className="h-3.5 w-3.5" />
          </a>
        </div>
      </header>

      {/* Three-column body ------------------------------------------ */}
      <div className="flex min-h-0 flex-1">
        {s.loading ? (
          <div className="grid w-full place-items-center">
            <div className="flex items-center gap-2 text-[12px] text-mist-3">
              <span className="h-3 w-3 animate-spin rounded-full border-2 border-line-2 border-t-scan" />
              Loading cases…
            </div>
          </div>
        ) : (
          <>
            <Sidebar s={s} />
            <Viewer s={s} />
            <AssistantPanel s={s} />
          </>
        )}
      </div>
    </div>
  )
}
