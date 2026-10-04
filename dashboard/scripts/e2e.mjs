/**
 * Interactive E2E check against the running dev server.
 *
 *   npm run dev            # terminal 1
 *   node scripts/e2e.mjs   # terminal 2
 *
 * Uses the system Chrome via puppeteer-core (no browser download).
 */
import puppeteer from 'puppeteer-core'

const CHROME =
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
const URL = process.env.NV_URL ?? 'http://localhost:5199/'

let failures = 0
const check = (name, cond, detail = '') => {
  console.log(`  ${cond ? 'ok  ' : 'FAIL'} ${name}${cond ? '' : ' ' + detail}`)
  if (!cond) failures++
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: 'new',
  args: ['--no-sandbox', '--hide-scrollbars', '--window-size=1680,1000'],
})

const page = await browser.newPage()
await page.setViewport({ width: 1680, height: 1000 })

const consoleErrors = []
page.on('pageerror', (e) => consoleErrors.push(String(e)))
page.on('console', (m) => {
  if (m.type() === 'error') consoleErrors.push(m.text())
})

await page.goto(URL, { waitUntil: 'networkidle0', timeout: 30000 })
await page.waitForFunction(
  () => document.body.innerText.includes('Run registration'),
  { timeout: 15000 },
)

const text = () => page.evaluate(() => document.body.innerText)
/** Case-insensitive containment — several labels use CSS text-transform. */
const has = (s, sub) => s.toLowerCase().includes(sub.toLowerCase())

console.log('· initial render')
let t = await text()
check('left sidebar present', has(t, 'Registration console'))
check('case list present', has(t, 'ReMIND-007') && has(t, 'ReMIND-085'))
check('method list present', has(t, 'VoxelMorph · Diffeomorphic'))
check('viewer present', has(t, 'Fusion') && has(t, 'Checker'))
check('assistant present', has(t, 'NeuroVista AI'))
check('metrics visible after seed', has(t, 'Target registration error'))
check('comparison table present', has(t, 'Method comparison'))
check('canvas rendered', await page.$('canvas') !== null)

// Canvas is actually painting (non-blank).
const painted = await page.evaluate(() => {
  const c = document.querySelector('canvas')
  const ctx = c.getContext('2d')
  const d = ctx.getImageData(0, 0, c.width, c.height).data
  let bright = 0
  for (let i = 0; i < d.length; i += 400) if (d[i] + d[i + 1] + d[i + 2] > 90) bright++
  return bright
})
check('canvas has visible pixels', painted > 100, `(bright samples: ${painted})`)

console.log('· select case + method')
const buttons = await page.$$('aside button')
for (const b of buttons) {
  const label = await b.evaluate((el) => el.innerText)
  if (label.includes('ReMIND-022')) {
    await b.click()
    break
  }
}
await sleep(400)
t = await text()
check('case switch updates header', t.includes('ReMIND-022'))
check('case switch posts assistant note', t.includes('Loaded ReMIND-022'))

for (const b of await page.$$('aside button')) {
  const label = await b.evaluate((el) => el.innerText)
  if (label.includes('Symmetric VM')) {
    await b.click()
    break
  }
}
await sleep(400)
t = await text()
check('method switch registered', t.includes('SVF-Sym'))

console.log('· run registration pipeline')
await page.evaluate(() => {
  const btn = [...document.querySelectorAll('button')].find((b) =>
    b.innerText.includes('Run registration'),
  )
  btn.click()
})
await sleep(900)
t = await text()
check('pipeline advances (running state)', t.includes('Registering') || t.includes('%'), t.slice(0, 200))
check('abort button appears', t.includes('Abort run'))

// 7 stages × 750 ms ≈ 5.3 s; allow headroom.
await page.waitForFunction(
  () => document.body.innerText.includes('Registration complete'),
  { timeout: 20000 },
)
t = await text()
check('run completes', t.includes('Registration complete'))
check('TRE metric populated', /\d+\.\d mm/.test(t) || t.includes('mean mm'))
check('Dice values populated', /\b0\.\d{3}\b/.test(t))
check('assistant posts read-out', t.includes('converged in'))
check('pipeline shows runtime', /s · run-/.test(t))

console.log('· viewer interactions')
for (const b of await page.$$('button')) {
  const label = await b.evaluate((el) => el.innerText.trim())
  if (label === 'Checker') {
    await b.click()
    break
  }
}
await sleep(500)
check('checker mode active', await page.evaluate(() => {
  const b = [...document.querySelectorAll('button')].find((x) => x.innerText.trim() === 'Checker')
  return b.className.includes('ring-1')
}))

for (const b of await page.$$('button')) {
  const label = await b.evaluate((el) => el.innerText.trim())
  if (label === 'COR') {
    await b.click()
    break
  }
}
await sleep(500)
check('coronal plane switches', (await text()).includes('/ COR'))

// Landmarks drawn after a completed run.
const lm = await page.evaluate(() => {
  const c = document.querySelector('canvas')
  const ctx = c.getContext('2d')
  const d = ctx.getImageData(0, 0, c.width, c.height).data
  let green = 0
  let orange = 0
  for (let i = 0; i < d.length; i += 4) {
    const r = d[i], g = d[i + 1], b = d[i + 2]
    if (g > 180 && r < 120 && b > 100 && b < 190) green++
    if (r > 220 && g > 120 && g < 190 && b < 110) orange++
  }
  return { green, orange }
})
check('landmark rings drawn', lm.green > 50, JSON.stringify(lm))
check('warped landmark crosses drawn', lm.orange > 50, JSON.stringify(lm))

console.log('· assistant chat')
const input = await page.$('form input')
await input.type('Is the deformation diffeomorphic?')
await page.evaluate(() => {
  const btn = [...document.querySelectorAll('button')].find((b) => b.innerText === 'Send')
  btn.click()
})
await page.waitForFunction(
  () => document.body.innerText.includes('Jacobian determinant'),
  { timeout: 8000 },
)
check('chat answers Jacobian question', true)

console.log('· screenshots')
await page.screenshot({ path: '/tmp/nv-final.png' })

check('no page/console errors', consoleErrors.length === 0, consoleErrors.join(' | '))

await browser.close()
console.log(failures === 0 ? '\nAll E2E checks passed.' : `\n${failures} check(s) FAILED.`)
process.exit(failures === 0 ? 0 : 1)
