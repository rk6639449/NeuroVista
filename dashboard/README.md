# NeuroVista Dashboard

A clean, medical-imaging-focused registration console:

| Left | Center | Right |
| --- | --- | --- |
| Case list (ReMIND subjects), volume details, registration method picker, run controls | MRI–US registration viewer — canvas-rendered slices, fusion/checker/split/residual modes, plane + slice + W/L controls, landmark TRE overlay | **NeuroVista AI** assistant — run status pipeline, TRE / Dice / HD95 / NCC / Jacobian metrics, method comparison, chat |

Everything runs on **mock data** — no backend required.

## Quick start

```bash
cd dashboard
npm install
npm run dev        # http://localhost:5173
```

Other commands:

```bash
npm run build      # tsc + vite production build → dist/
npm run lint       # eslint
node --experimental-strip-types scripts/smoke.ts   # headless logic checks
node scripts/e2e.mjs                               # full UI flow (needs dev server + Chrome)
```

## Layout

```
src/
├── types.ts                 # domain model — the contract with your backend
├── data/mock.ts             # cases, methods, pipeline, deterministic metric scorer
├── api/client.ts            # ← THE BACKEND SEAM (see below)
├── state/useSession.ts      # app state: selection, run loop, chat
├── lib/imaging.ts           # synthetic MRI/US synthesis + compositing (mock only)
└── components/
    ├── Sidebar.tsx          # left: cases + methods + run
    ├── Viewer.tsx           # center: toolbar, canvas stage, pipeline footer
    ├── viewerDraw.ts        # canvas paint: fusion, landmarks, orientation marks
    ├── AssistantPanel.tsx   # right: NeuroVista AI
    └── ui.tsx               # Section / Pill / Stat / Meter primitives
```

## Connecting your VoxelMorph backend

All I/O goes through `src/api/client.ts`. To go live:

1. Set `USE_MOCK = false` (or wire it to an env flag).
2. Point `API_BASE` at your service — default `http://localhost:8000`,
   override with `VITE_API_BASE=https://... npm run dev`.
3. Implement the endpoints below, returning the **exact shapes in
   `src/types.ts`**. Nothing else in the app needs to change.

```
GET  /api/cases                       → { cases: ImagingCase[] }
GET  /api/methods                     → { methods: RegistrationMethod[] }
POST /api/register                    → { result: RegistrationResult }
GET  /api/runs/latest?caseId=&methodId= → { result: RegistrationResult | null }
GET  /api/compare?caseId=&methodId=     → { rows: MethodComparison[] }   (optional)
```

A minimal FastAPI sketch:

```python
from fastapi import FastAPI
app = FastAPI()

@app.post("/api/register")
def register(body: RegisterBody):
    fixed  = load_nifti(cases[body.caseId].fixed_path)
    moving = load_nifti(cases[body.caseId].moving_path)
    disp, stats = vm_infer(model, fixed, moving, method=body.methodId)
    # stats must mirror RegistrationResult in src/types.ts:
    # treMeanMm, treMedianMm, treMaxMm, hd95Mm, dice[], jacobianNegativePct,
    # ncc, landmarks[], runtimeMs, summary
    return {"result": to_result_shape(stats)}
```

**Streaming progress (recommended):** the mock runner ticks once per stage of
`PIPELINE` (`src/data/mock.ts`) via `runProgressTick()`. A real backend can push
the same `{ progress, stageIndex }` increments over SSE or a WebSocket and the
status banner / footer stage strip will animate identically — just replace the
`window.setInterval` loop in `useSession.ts#run()` with an `EventSource`.

### Slice data

`src/lib/imaging.ts` generates synthetic MRI/US frames so the viewer works with
zero data. When your backend serves slices, replace the `synthMRI`/`synthUS`/
`warp` calls in `Viewer.tsx` (`buffers` memo) with fetches (PNG/JPEG per slice,
or a shared WebGL texture) — `drawViewer()` already composites plain
`Uint8ClampedArray` buffers and draws landmarks from `result.landmarks`, so the
overlay logic stays unchanged.

## Mock behaviours worth knowing

- Metrics are **deterministic** per `(case, method)` — the same pair always
  gives the same numbers, so screenshots and demos are stable.
- `ReMIND-046` is seeded as a failed case (poor US window) so the failure UI
  path is reachable; running it always fails by design.
- The assistant replies are keyword-canned (`useSession.ts#draftReply`) — swap
  for a real LLM call when ready.
- UI targets a ≥1280 px desktop viewport (reading-room layout, fixed sidebars).

## Notes

- Stack: Vite · React 19 · TypeScript · Tailwind v4 · lucide-react.
- Two validation scripts are included: `scripts/smoke.ts` (pure logic) and
  `scripts/e2e.mjs` (real browser flow, uses system Chrome via `puppeteer-core`).
