/* NeuroVista web app */
'use strict';

const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

const state = {
  sid: null,
  metrics: null,
  step: 1,
  view: { base: 'fixed', ovls: new Set(['ovl_now']), plane: 'axial', idx: 88 },
  imgCache: new Map(),
  points: [],
};

/* ---------------- utilities ---------------- */
function toast(msg, kind = '') {
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  el.textContent = msg;
  $('#toasts').appendChild(el);
  setTimeout(() => el.remove(), 5200);
}

function log(msg, cls = '') {
  const line = document.createElement('div');
  line.className = `console-line ${cls}`;
  const t = new Date().toLocaleTimeString();
  line.textContent = `[${t}] ${msg}`;
  const c = $('#console');
  c.appendChild(line);
  c.scrollTop = c.scrollHeight;
}

async function api(path, opts = {}) {
  const res = await fetch(`/api${path}`, {
    headers: opts.body && !(opts.body instanceof FormData)
      ? { 'Content-Type': 'application/json' } : undefined,
    ...opts,
    body: opts.body instanceof FormData ? opts.body
      : opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) { /* ignore */ }
    throw new Error(detail);
  }
  const ct = res.headers.get('content-type') || '';
  return ct.includes('application/json') ? res.json() : res;
}

function busy(btn, on) { btn.classList.toggle('busy', on); }

/* ---------------- stepper / navigation ---------------- */
function gotoStep(n) {
  if (n > 1 && !state.metrics?.stages?.upload) { toast('Upload both scans first.', 'err'); return; }
  if (n > 2 && !state.metrics?.stages?.register) { toast('Run the pipeline first.', 'err'); return; }
  if (n > 3 && !state.metrics?.stages?.relocate) { toast('Compute the tumour shift first.', 'err'); return; }
  state.step = n;
  $$('.panel').forEach((p) => p.classList.toggle('active', p.id === `panel-${n}`));
  $$('.step').forEach((s) => {
    const sn = Number(s.dataset.step);
    s.classList.toggle('active', sn === n);
    s.classList.toggle('done', sn < n);
  });
  if (n === 3) enterVisualization();
  if (n === 4) ensurePlanDefaults();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

/* ---------------- uploads ---------------- */
const uploadState = { fixed: false, moving: false, mask: false };

function wireDrop(zoneId, role) {
  const zone = $(zoneId);
  const input = zone.querySelector('input[type=file]');
  const fileLabel = zone.querySelector('[data-file]');

  const setFile = async (file) => {
    if (!file) return;
    if (!/\.nii(\.gz)?$/i.test(file.name)) {
      toast('Please choose a .nii / .nii.gz file.', 'err');
      return;
    }
    fileLabel.textContent = `Uploading ${file.name}…`;
    try {
      const fd = new FormData();
      fd.append('file', file);
      const endpoint = role === 'mask'
        ? `/session/${state.sid}/mask`
        : `/session/${state.sid}/upload?role=${role}`;
      const info = await api(endpoint, { method: 'POST', body: fd });
      if (role === 'mask') {
        uploadState.mask = true;
        toast(`Mask accepted (${info.voxels} voxels).`, 'ok');
      } else {
        uploadState[role] = true;
        fileLabel.textContent = `${file.name} — ${info.shape.join('×')} · ` +
          `${info.zooms_mm.join('×')} mm`;
        toast(`${role === 'fixed' ? 'Fixed' : 'Moving'} image uploaded.`, 'ok');
      }
      zone.classList.add('ok');
      await refreshMetrics();
      updateUploadGate();
    } catch (err) {
      fileLabel.textContent = 'Drag & drop or click to browse';
      zone.classList.remove('ok');
      toast(`Upload failed: ${err.message}`, 'err');
    }
  };

  input.addEventListener('change', () => setFile(input.files[0]));
  zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('drag'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag'));
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('drag');
    setFile(e.dataTransfer.files[0]);
  });
}

function updateUploadGate() {
  const ready = uploadState.fixed && uploadState.moving;
  $('#toStep2').disabled = !ready;
  $('#uploadHint').textContent = ready
    ? 'Both scans loaded — continue to the pipeline.'
    : `Fixed: ${uploadState.fixed ? '✓' : '✗'} · Moving: ` +
      `${uploadState.moving ? '✓' : '✗'} — both scans are required.`;
}
/* ---------------- metrics ---------------- */
async function refreshMetrics() {
  try {
    state.metrics = await api(`/session/${state.sid}/metrics`);
  } catch (err) {
    toast(err.message, 'err');
    return;
  }
  const m = state.metrics;

  setStage('Segment', m.stages.segment,
    m.stages.segment ? `engine: ${m.segment.engine}` : 'pending');
  setStage('Register', m.stages.register,
    m.stages.register ? `NCC ${m.ncc ?? '—'} · ${m.runtime_s}s` : 'pending');
  setStage('Relocate', m.stages.relocate,
    m.stages.relocate ? `shift ${m.reloc.shift_magnitude_mm.toFixed(1)} mm` : 'pending');

  $('#toStep3').disabled = !m.stages.register;
  $('#toStep4').disabled = !m.stages.relocate;
  $$('.step').forEach((s) => {
    const sn = Number(s.dataset.step);
    if (sn === 3) s.classList.toggle('locked', !m.stages.register);
    if (sn === 4) s.classList.toggle('locked', !m.stages.relocate);
  });

  if (m.reloc) {
    $('#mShift').textContent = m.reloc.shift_magnitude_mm.toFixed(1);
    $('#mShiftDir').textContent = `mm ${m.reloc.shift_direction}`;
    $('#mP95').textContent = m.reloc.field_magnitude_mm.p95.toFixed(1);
    $('#mJac').textContent = (m.reloc.jacobian.pct_nonpositive ?? 0).toFixed(2);
    renderPosTable(m.reloc);
  }
  if (m.ncc != null) $('#mNcc').textContent = m.ncc.toFixed(3);
  if (m.runtime_s) $('#mTime').textContent = m.runtime_s.toFixed(1);
  if (m.stages.segment) $('#segEngineNote').textContent = m.segment.detail;
  if (m.has_segmentation_model === false) {
    $('#segModelPill').textContent = 'segmentation: demo engine';
    $('#segModelPill').className = 'pill warn';
  }

  const href = `/api/session/${state.sid}/export`;
  $('#exportBtn').href = href;
  $('#exportBtn2').href = href;
}

function setStage(name, done, badgeText) {
  const card = $(`#stage${name}`);
  const badge = $(`#badge${name}`);
  card.classList.toggle('done', !!done);
  badge.className = `badge ${done ? 'ok' : ''}`;
  badge.textContent = badgeText;
}

function renderPosTable(reloc) {
  const rows = [
    ['Preop centre', reloc.preop.mm, 't-green'],
    ['Affine-aligned', reloc.affine_aligned.mm, 't-cyan'],
    ['Current (shifted)', reloc.shifted.mm, 't-yellow'],
  ];
  $('#posTable').innerHTML = rows.map(([label, mm, cls]) =>
    `<tr><td class="tag ${cls}">${label}</td>` +
    `<td>${mm.map((v) => v.toFixed(1)).join(', ')}</td><td>mm</td></tr>`).join('') +
    `<tr><td class="delta">Net shift</td><td class="delta" colspan="2">` +
    `${reloc.shift_magnitude_mm.toFixed(2)} mm → ${reloc.shift_direction}</td></tr>`;
}

/* ---------------- pipeline ---------------- */
async function runStage(name, btn) {
  const card = $(`#stage${name[0].toUpperCase()}${name.slice(1)}`);
  const badge = $(`#badge${name[0].toUpperCase()}${name.slice(1)}`);
  card.classList.add('running');
  badge.className = 'badge run';
  badge.textContent = 'running…';
  if (btn) busy(btn, true);
  const t0 = performance.now();
  log(`${name}: started…`);
  try {
    const res = await api(`/session/${state.sid}/${name}`, { method: 'POST' });
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    if (name === 'register') {
      log(`register: done in ${res.runtime_s}s (wall ${secs}s), NCC=${res.ncc}, ` +
          `weights=${res.weights} on ${res.device}`, '');
    } else if (name === 'segment') {
      log(`segment: engine=${res.engine}, ${res.volume_cc} cc — ${res.detail}`);
    } else {
      log(`relocate: shift ${res.shift_magnitude_mm.toFixed(2)} mm ` +
          `toward the ${res.shift_direction}`);
    }
    await refreshMetrics();
    return true;
  } catch (err) {
    card.classList.remove('running');
    badge.className = 'badge err';
    badge.textContent = 'failed';
    log(`${name}: FAILED — ${err.message}`, 'err');
    toast(`${name} failed: ${err.message}`, 'err');
    return false;
  } finally {
    if (btn) busy(btn, false);
    card.classList.remove('running');
  }
}

async function runAll(btn) {
  busy(btn, true);
  log('full pipeline: segment → register → relocate');
  const s1 = await runStage('segment');
  const s2 = s1 && await runStage('register');
  const s3 = s2 && await runStage('relocate');
  busy(btn, false);
  if (s3) { toast('Pipeline complete — open the visualization.', 'ok'); gotoStep(3); }
}
/* ---------------- slice viewer ---------------- */
const PLANE_AXIS = { axial: 2, coronal: 1, sagittal: 0 };

function orientationLetters(affine, plane) {
  const ax = PLANE_AXIS[plane];
  const others = [0, 1, 2].filter((a) => a !== ax);
  const [col, row] = others;
  const names = [['L', 'R'], ['P', 'A'], ['I', 'S']];
  const colSign = Math.sign(affine[col][col]) || 1;
  const rowSign = Math.sign(affine[row][row]) || 1;
  const c = names[col];
  const r = names[row];
  return {
    left: colSign > 0 ? c[0] : c[1],
    right: colSign > 0 ? c[1] : c[0],
    bottom: rowSign > 0 ? r[0] : r[1],
    top: rowSign > 0 ? r[1] : r[0],
  };
}

async function fetchImage(kind) {
  const { plane, idx } = state.view;
  const key = `${kind}|${plane}|${idx}`;
  if (state.imgCache.has(key)) return state.imgCache.get(key);
  const res = await fetch(
    `/api/session/${state.sid}/view/${kind}?plane=${plane}&idx=${idx}`);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) { /* ignore */ }
    throw new Error(detail);
  }
  const bmp = await createImageBitmap(await res.blob());
  state.imgCache.set(key, bmp);
  if (state.imgCache.size > 360) {
    state.imgCache.delete(state.imgCache.keys().next().value);
  }
  return bmp;
}

function drawInto(canvas, img, fitBase) {
  const w = fitBase ? fitBase.width : img.width;
  const h = fitBase ? fitBase.height : img.height;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(img, 0, 0, w, h);
}

async function renderView() {
  const loading = $('#vpLoading');
  loading.hidden = false;
  try {
    const base = await fetchImage(state.view.base);
    drawInto($('#cvBase'), base);
    const overlayOrder = ['ovl_pre', 'ovl_affine', 'ovl_now']
      .filter((k) => state.view.ovls.has(k));
    const canvases = [$('#cvOvl1'), $('#cvOvl2'), $('#cvOvl3')];
    canvases.forEach((c) => {
      c.getContext('2d').clearRect(0, 0, c.width, c.height);
    });
    for (let i = 0; i < overlayOrder.length; i++) {
      try {
        const ovl = await fetchImage(overlayOrder[i]);
        drawInto(canvases[i], ovl, base);
      } catch (_) { /* overlay not ready — silently skip */ }
    }
    $('#vpBadge').textContent =
      `${state.view.base} · ${state.view.plane} · #${state.view.idx}`;
    updateOrientation();
  } catch (err) {
    $('#vpBadge').textContent = 'unavailable';
    if (!renderView._warned) {
      toast(err.message, 'err');
      renderView._warned = true;
      setTimeout(() => { renderView._warned = false; }, 8000);
    }
  } finally {
    loading.hidden = true;
  }
}

function updateOrientation() {
  const aff = state.metrics?.affines?.fixed;
  if (!aff) return;
  const o = orientationLetters(aff, state.view.plane);
  $('#oriTop').textContent = o.top;
  $('#oriBottom').textContent = o.bottom;
  $('#oriLeft').textContent = o.left;
  $('#oriRight').textContent = o.right;
}

function planeMax() {
  const dims = state.metrics?.dims || [176, 176, 176];
  return dims[PLANE_AXIS[state.view.plane]] - 1;
}

function setSlice(idx) {
  state.view.idx = Math.max(0, Math.min(planeMax(), idx));
  $('#sliceRange').value = state.view.idx;
  $('#sliceReadout').textContent = `${state.view.idx} / ${planeMax()}`;
  renderView();
}

function enterVisualization() {
  $('#sliceRange').max = planeMax();
  $('#viewport').classList.add('bright');
  applyBrightness();
  setSlice(Math.min(state.view.idx, planeMax()));
  loadMesh();
}

function applyBrightness() {
  const b = Number($('#bright').value) / 100;
  const c = Number($('#contrast').value) / 100;
  $('#viewport').style.setProperty('--b', b);
  $('#viewport').style.setProperty('--c', c);
}
/* ---------------- 3D tumour view (three.js via CDN, graceful fallback) -------- */
const three = { ready: false, group: null, renderer: null, scene: null, camera: null,
  rot: { x: 0.4, y: 0.6 }, drag: null };

function meshToGeometry(mesh, center) {
  const pos = new Float32Array(mesh.vertices.length * 3);
  mesh.vertices.forEach((v, i) => {
    pos[i * 3] = v[0] - center[0];
    pos[i * 3 + 1] = v[1] - center[1];
    pos[i * 3 + 2] = v[2] - center[2];
  });
  const idx = new Uint32Array(mesh.faces.length * 3);
  mesh.faces.forEach((f, i) => {
    idx[i * 3] = f[0];
    idx[i * 3 + 1] = f[1];
    idx[i * 3 + 2] = f[2];
  });
  const geo = new three.THREE.BufferGeometry();
  geo.setAttribute('position', new three.THREE.BufferAttribute(pos, 3));
  geo.setIndex(new three.THREE.BufferAttribute(idx, 1));
  geo.computeVertexNormals();
  return geo;
}

async function loadMesh() {
  if (three.loading || three.loaded) return;
  three.loading = true;
  try {
    const data = await api(`/session/${state.sid}/mesh`);
    if (!data.now) { three.loading = false; return; }
    if (!three.ready) await initThree();
    if (!three.ready) return;

    const anchor = data.now || data.affine;
    const center = anchor.vertices.reduce(
      (acc, v) => [acc[0] + v[0], acc[1] + v[1], acc[2] + v[2]],
      [0, 0, 0]).map((s) => s / anchor.vertices.length);

    three.group.clear();
    if (data.preop) {
      const ghost = new three.THREE.Mesh(
        meshToGeometry(data.preop, center),
        new three.THREE.MeshPhongMaterial({
          color: 0x3cdc78, transparent: true, opacity: 0.28,
          side: three.THREE.DoubleSide,
        }));
      three.group.add(ghost);
    }
    const nowMesh = new three.THREE.Mesh(
      meshToGeometry(data.now, center),
      new three.THREE.MeshPhongMaterial({
        color: 0xff5454, side: three.THREE.DoubleSide, shininess: 60,
      }));
    three.group.add(nowMesh);
    three.loaded = true;
    animateThree();
  } catch (err) {
    if (!three.ready) $('#threeFallback').hidden = false;
  } finally {
    three.loading = false;
  }
}

async function initThree() {
  try {
    three.THREE = await import('https://unpkg.com/three@0.160.0/build/three.module.js');
  } catch (_) {
    $('#threeFallback').hidden = false;
    three.ready = false;
    return;
  }
  const canvas = $('#threeCanvas');
  three.renderer = new three.THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  three.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  three.scene = new three.THREE.Scene();
  three.camera = new three.THREE.PerspectiveCamera(45, 1, 1, 5000);
  three.camera.position.set(0, 0, 320);
  three.scene.add(new three.THREE.AmbientLight(0xffffff, 0.55));
  const key = new three.THREE.DirectionalLight(0xffffff, 1.0);
  key.position.set(120, 160, 200);
  three.scene.add(key);
  const rim = new three.THREE.DirectionalLight(0x6fa8ff, 0.5);
  rim.position.set(-140, -80, -120);
  three.scene.add(rim);
  three.group = new three.THREE.Group();
  three.scene.add(three.group);
  three.ready = true;

  const resize = () => {
    const r = canvas.parentElement.getBoundingClientRect();
    three.renderer.setSize(r.width, r.height, false);
    three.camera.aspect = r.width / Math.max(1, r.height);
    three.camera.updateProjectionMatrix();
  };
  resize();
  window.addEventListener('resize', resize);

  canvas.addEventListener('pointerdown', (e) => {
    three.drag = { x: e.clientX, y: e.clientY, rx: three.rot.x, ry: three.rot.y };
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!three.drag) return;
    three.rot.y = three.drag.ry + (e.clientX - three.drag.x) * 0.01;
    three.rot.x = Math.max(-1.4, Math.min(1.4,
      three.drag.rx + (e.clientY - three.drag.y) * 0.01));
  });
  canvas.addEventListener('pointerup', () => { three.drag = null; });
  canvas.addEventListener('pointerleave', () => { three.drag = null; });
}

function animateThree() {
  if (!three.ready) return;
  requestAnimationFrame(animateThree);
  if (!three.drag && $('#autoRotate').checked) three.rot.y += 0.006;
  three.group.rotation.x = three.rot.x;
  three.group.rotation.y = three.rot.y;
  three.renderer.render(three.scene, three.camera);
}
/* ---------------- planning (Cline SDK) ---------------- */
function addPointRow(label = '', x = '', y = '', z = '') {
  const tr = document.createElement('tr');
  tr.innerHTML =
    `<td><input value="${label}" placeholder="label" data-f="label"></td>` +
    `<td><input value="${x}" placeholder="x" data-f="x" inputmode="decimal"></td>` +
    `<td><input value="${y}" placeholder="y" data-f="y" inputmode="decimal"></td>` +
    `<td><input value="${z}" placeholder="z" data-f="z" inputmode="decimal"></td>` +
    '<td><button class="del" title="remove">✕</button></td>';
  tr.querySelector('.del').addEventListener('click', () => tr.remove());
  $('#pointsTable tbody').appendChild(tr);
}

function ensurePlanDefaults() {
  if ($('#pointsTable tbody').children.length) return;
  const r = state.metrics?.reloc;
  if (r) {
    addPointRow('Tumour centre',
      r.preop.mm[0].toFixed(1), r.preop.mm[1].toFixed(1), r.preop.mm[2].toFixed(1));
    addPointRow('Entry point',
      (r.preop.mm[0] + 25).toFixed(1), (r.preop.mm[1] - 15).toFixed(1),
      (r.preop.mm[2] + 30).toFixed(1));
  } else {
    addPointRow('Tumour centre', '', '', '');
  }
}

function collectPlan() {
  const points = [];
  $$('#pointsTable tbody tr').forEach((tr) => {
    const get = (f) => tr.querySelector(`[data-f="${f}"]`).value.trim();
    const p = { label: get('label'), x: parseFloat(get('x')),
      y: parseFloat(get('y')), z: parseFloat(get('z')) };
    if ([p.x, p.y, p.z].every((v) => Number.isFinite(v))) points.push(p);
  });
  return {
    points,
    steps: $('#planSteps').value.split('\n').filter((l) => l.trim()),
    notes: $('#planNotes').value,
  };
}

async function updatePlan(btn) {
  busy(btn, true);
  try {
    const res = await api(`/session/${state.sid}/plan/update`,
      { method: 'POST', body: collectPlan() });
    $('#planOut').textContent = res.plan_text;
    $('#planMeta').textContent = `engine: ${res.engine} · ${res.generated_at}`;
    $('#engineBadge').textContent = res.engine.startsWith('claude')
      ? `Cline SDK · ${res.engine}` : 'Cline SDK · offline engine';
    toast('Plan updated through the deformation field.', 'ok');
  } catch (err) {
    toast(`Plan update failed: ${err.message}`, 'err');
  } finally {
    busy(btn, false);
  }
}

async function askCline(question) {
  const box = $('#chatLog');
  box.insertAdjacentHTML('beforeend',
    `<div class="msg user">${question.replace(/</g, '&lt;')}</div>`);
  const thinking = document.createElement('div');
  thinking.className = 'msg bot';
  thinking.textContent = '…';
  box.appendChild(thinking);
  box.scrollTop = box.scrollHeight;
  try {
    const res = await api(`/session/${state.sid}/plan/ask`,
      { method: 'POST', body: { question } });
    thinking.textContent = res.answer;
  } catch (err) {
    thinking.textContent = `Error: ${err.message}`;
  }
  box.scrollTop = box.scrollHeight;
}
/* ---------------- event wiring ---------------- */
function wireEvents() {
  $$('.step').forEach((s) => s.addEventListener('click',
    () => gotoStep(Number(s.dataset.step))));
  $('#toStep2').addEventListener('click', () => gotoStep(2));
  $('#backTo1').addEventListener('click', () => gotoStep(1));
  $('#toStep3').addEventListener('click', () => gotoStep(3));
  $('#backTo2').addEventListener('click', () => gotoStep(2));
  $('#toStep4').addEventListener('click', () => gotoStep(4));
  $('#backTo3').addEventListener('click', () => gotoStep(3));

  wireDrop('#dropFixed', 'fixed');
  wireDrop('#dropMoving', 'moving');
  wireDrop('#dropMask', 'mask');

  $$('[data-run]').forEach((b) => b.addEventListener('click',
    () => runStage(b.dataset.run, b)));
  $('#runAll').addEventListener('click', (e) => runAll(e.currentTarget));

  $('#baseChips').addEventListener('click', (e) => {
    const btn = e.target.closest('[data-base]');
    if (!btn) return;
    $$('#baseChips .chip').forEach((c) => c.classList.remove('active'));
    btn.classList.add('active');
    state.view.base = btn.dataset.base;
    renderView();
  });
  $('#ovlChips').addEventListener('click', (e) => {
    const btn = e.target.closest('[data-ovl]');
    if (!btn) return;
    btn.classList.toggle('active');
    const key = btn.dataset.ovl;
    if (btn.classList.contains('active')) state.view.ovls.add(key);
    else state.view.ovls.delete(key);
    renderView();
  });
  $('#planeChips').addEventListener('click', (e) => {
    const btn = e.target.closest('[data-plane]');
    if (!btn) return;
    $$('#planeChips .chip').forEach((c) => c.classList.remove('active'));
    btn.classList.add('active');
    state.view.plane = btn.dataset.plane;
    $('#sliceRange').max = planeMax();
    setSlice(Math.min(state.view.idx, planeMax()));
  });

  $('#sliceRange').addEventListener('input', (e) => setSlice(Number(e.target.value)));
  $('#slicePrev').addEventListener('click', () => setSlice(state.view.idx - 1));
  $('#sliceNext').addEventListener('click', () => setSlice(state.view.idx + 1));
  $('#viewport').addEventListener('wheel', (e) => {
    e.preventDefault();
    setSlice(state.view.idx + (e.deltaY > 0 ? 1 : -1));
  }, { passive: false });
  window.addEventListener('keydown', (e) => {
    if (state.step !== 3) return;
    if (e.key === 'ArrowUp' || e.key === 'ArrowRight') setSlice(state.view.idx + 1);
    if (e.key === 'ArrowDown' || e.key === 'ArrowLeft') setSlice(state.view.idx - 1);
  });

  $('#bright').addEventListener('input', applyBrightness);
  $('#contrast').addEventListener('input', applyBrightness);

  $('#addPoint').addEventListener('click', () => addPointRow());
  $('#updatePlan').addEventListener('click', (e) => updatePlan(e.currentTarget));
  $('#chatForm').addEventListener('submit', (e) => {
    e.preventDefault();
    const q = $('#chatInput').value.trim();
    if (!q) return;
    $('#chatInput').value = '';
    askCline(q);
  });
}

/* ---------------- init ---------------- */
async function init() {
  wireEvents();
  try {
    const health = await api('/health');
    $('#weightsPill').textContent = health.weights.exists
      ? 'weights: final.pt ✓' : 'weights: MISSING';
    $('#weightsPill').className = `pill ${health.weights.exists ? 'ok' : 'warn'}`;
    if (health.segmentation_model.exists) {
      $('#segModelPill').textContent = 'segmentation: model ✓';
      $('#segModelPill').className = 'pill ok';
    } else {
      $('#segModelPill').textContent = 'segmentation: demo engine';
      $('#segModelPill').className = 'pill warn';
    }
    const s = await api('/session', { method: 'POST' });
    state.sid = s.session_id;
    $('#sessionPill').textContent = `session: ${state.sid.slice(0, 8)}`;
    log(`session ${state.sid} created — upload the case to begin.`);
  } catch (err) {
    toast(`Could not reach the server: ${err.message}`, 'err');
    log(`init failed: ${err.message}`, 'err');
  }
}

document.addEventListener('DOMContentLoaded', init);
