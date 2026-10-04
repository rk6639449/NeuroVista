"""NeuroVista surgical planning web server (FastAPI).

Pipeline: upload fixed/moving -> segment tumour -> VoxelMorph registration ->
deformation field -> tumour relocation -> visualisation + 3D Slicer export ->
Cline SDK plan update.
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from pipeline import cline_sdk as cline
from pipeline import export as exp
from pipeline import preprocess as pp
from pipeline import registration as reg
from pipeline import relocate as rel
from pipeline import segmentation as seg
from pipeline import visualize as vis

ROOT = Path(__file__).resolve().parent
WORK = ROOT / 'work'
STATIC = ROOT / 'static'
MAX_UPLOAD = 600 * 1024 * 1024  # 600 MB per file

app = FastAPI(title='NeuroVista Surgical Planning', version='1.0')
app.add_middleware(
    CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'],
)


@dataclass
class Session:
    """All state for one uploaded case (kept in memory + on disk)."""

    id: str
    dir: Path
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    fixed: pp.Volume | None = None
    moving: pp.Volume | None = None
    mask_manual: pp.Volume | None = None
    mask_moving: np.ndarray | None = None       # bool, moving (preop) grid
    segment_engine: str = ''
    segment_detail: str = ''
    moving_resampled: np.ndarray | None = None  # affine-aligned onto fixed grid
    disp: np.ndarray | None = None              # (3, D, H, W) voxel units
    warped: np.ndarray | None = None
    mask_affine: np.ndarray | None = None       # preop mask, affine-aligned
    mask_now: np.ndarray | None = None          # preop mask after deformation
    reloc: dict | None = None
    ncc: float | None = None
    runtime_s: float = 0.0
    plan_text: str = ''
    u8: dict = field(default_factory=dict)      # cached windowed grayscale


SESSIONS: dict[str, Session] = {}
_REGISTRAR = None
_REG_LOCK = threading.Lock()


def _get_registrar() -> reg.Registrar:
    """Lazy singleton for the trained VoxelMorph model."""
    global _REGISTRAR
    with _REG_LOCK:
        if _REGISTRAR is None:
            _REGISTRAR = reg.Registrar()
        return _REGISTRAR


def _session(sid: str) -> Session:
    sess = SESSIONS.get(sid)
    if sess is None:
        raise HTTPException(404, f'Unknown session {sid}')
    return sess


def _require(cond: bool, detail: str, status: int = 400) -> None:
    if not cond:
        raise HTTPException(status, detail)


async def _save_upload(upload: UploadFile, dest: Path) -> dict:
    size = 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('wb') as fh:
        while chunk := await upload.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD:
                fh.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, 'File exceeds the 600 MB upload limit.')
            fh.write(chunk)
    if size == 0:
        raise HTTPException(400, 'Uploaded file is empty.')
    return {'name': upload.filename, 'bytes': size}


def _suffix(filename: str | None) -> str:
    name = (filename or '').lower()
    return '.nii.gz' if name.endswith('.nii.gz') else '.nii'


def _meta(vol: pp.Volume, name: str) -> dict:
    return {'name': name, 'shape': list(vol.shape),
            'zooms_mm': [round(z, 3) for z in vol.zooms]}


def _cache_u8(sess: Session) -> None:
    """Pre-window grayscale volumes once so slice requests stay fast."""
    candidates = {'fixed': sess.fixed.data if sess.fixed else None,
                  'moving': sess.moving.data if sess.moving else None,
                  'resampled': sess.moving_resampled,
                  'warped': sess.warped}
    for key, data in candidates.items():
        if data is not None and key not in sess.u8:
            sess.u8[key], _, _ = vis.window(data)


def _sync_masks(sess: Session) -> None:
    """(Re)derive affine-aligned + deformed masks and relocation metrics."""
    if sess.mask_moving is None or sess.disp is None or sess.fixed is None:
        return
    sess.mask_affine = pp.resample_to_grid(
        pp.Volume(sess.mask_moving.astype(np.float32), sess.moving.affine,
                  sess.moving.zooms),
        sess.fixed.shape, sess.fixed.affine, order=0) > 0.5
    sess.mask_now = reg.warp_volume(
        sess.mask_affine.astype(np.float32), sess.disp, method='nearest') > 0.5
    sess.reloc = rel.relocate(
        sess.mask_moving, sess.disp, sess.moving.affine, sess.fixed.affine,
        sess.moving.zooms, sess.mask_affine, sess.mask_now)


# --------------------------------------------------------------------------
# Session / upload / segmentation
# --------------------------------------------------------------------------
@app.post('/api/session')
def create_session() -> dict:
    sid = uuid.uuid4().hex[:12]
    sess = Session(id=sid, dir=WORK / sid)
    sess.dir.mkdir(parents=True, exist_ok=True)
    SESSIONS[sid] = sess
    return {'session_id': sid}


@app.post('/api/session/{sid}/upload')
async def upload(sid: str, role: str = 'fixed', file: UploadFile = File(...)) -> dict:
    sess = _session(sid)
    _require(role in ('fixed', 'moving'), 'role must be "fixed" or "moving"')
    with sess.lock:
        dest = sess.dir / f'{role}{_suffix(file.filename)}'
        info = await _save_upload(file, dest)
        try:
            vol = pp.Volume.load(dest)
        except Exception as exc:
            dest.unlink(missing_ok=True)
            raise HTTPException(422, f'Could not read NIfTI: {exc}')
        if role == 'fixed':
            sess.fixed = vol
        else:
            sess.moving = vol
            sess.mask_moving = None
            sess.mask_affine = sess.mask_now = sess.reloc = None
            sess.moving_resampled = sess.warped = sess.disp = None
            sess.u8.pop('moving', None)
            sess.u8.pop('resampled', None)
            sess.u8.pop('warped', None)
        return {'role': role, **_meta(vol, info['name']), 'bytes': info['bytes']}


@app.post('/api/session/{sid}/mask')
async def upload_mask(sid: str, file: UploadFile = File(...)) -> dict:
    """Optional clinician-provided tumour mask (preoperative / moving space)."""
    sess = _session(sid)
    _require(sess.moving is not None, 'Upload the moving (preoperative) image first.')
    with sess.lock:
        dest = sess.dir / f'mask_manual{_suffix(file.filename)}'
        await _save_upload(file, dest)
        try:
            vol = pp.Volume.load(dest)
        except Exception as exc:
            raise HTTPException(422, f'Could not read mask NIfTI: {exc}')
        if vol.shape != sess.moving.shape:
            data = pp.resample_to_grid(vol, sess.moving.shape, sess.moving.affine,
                                       order=0) > 0.5
        else:
            data = vol.data > 0
        sess.mask_manual = pp.Volume(data, sess.moving.affine, sess.moving.zooms)
        return {'name': file.filename, 'shape': list(data.shape),
                'voxels': int(data.sum())}


@app.post('/api/session/{sid}/segment')
def segment_tumour(sid: str) -> dict:
    """Segment the tumour on the preoperative (moving) image."""
    sess = _session(sid)
    _require(sess.moving is not None, 'Upload the moving (preoperative) image first.')
    with sess.lock:
        manual = sess.mask_manual.data if sess.mask_manual is not None else None
        result = seg.segment(sess.moving.data, manual_mask=manual)
        sess.mask_moving = result['mask']
        sess.segment_engine = result['engine']
        sess.segment_detail = result['detail']
        vox_mm3 = float(sess.mask_moving.sum() * np.prod(sess.moving.zooms))
        _sync_masks(sess)
        return {'engine': result['engine'], 'detail': result['detail'],
                'voxels': int(sess.mask_moving.sum()),
                'volume_cc': round(vox_mm3 / 1000.0, 2)}


# --------------------------------------------------------------------------
# Registration / relocation / metrics
# --------------------------------------------------------------------------
@app.post('/api/session/{sid}/register')
def register(sid: str) -> dict:
    """Run VoxelMorph: resample moving to fixed grid, infer warp, cache results."""
    sess = _session(sid)
    _require(sess.fixed is not None, 'Upload the fixed (intraoperative) image first.')
    _require(sess.moving is not None, 'Upload the moving (preoperative) image first.')
    with sess.lock:
        sess.moving_resampled = pp.resample_to_grid(
            sess.moving, sess.fixed.shape, sess.fixed.affine)
        norm_fixed = pp.normalize(sess.fixed.data)
        norm_moving = pp.normalize(sess.moving_resampled)
        registrar = _get_registrar()
        disp, warped, runtime = registrar.run_padded(norm_moving, norm_fixed)
        sess.disp, sess.warped, sess.runtime_s = disp, warped, runtime
        sess.ncc = reg.ncc(warped, norm_fixed)
        sess.u8.pop('resampled', None)
        sess.u8.pop('warped', None)
        _cache_u8(sess)
        _sync_masks(sess)
        return {'runtime_s': round(runtime, 2),
                'ncc': round(sess.ncc, 4),
                'weights': Path(registrar.weights).name,
                'device': registrar.device,
                'relocated': sess.reloc is not None}


@app.post('/api/session/{sid}/relocate')
def relocate(sid: str) -> dict:
    """Recompute tumour relocation (requires registration + segmentation)."""
    sess = _session(sid)
    _require(sess.disp is not None, 'Run registration first.', 409)
    _require(sess.mask_moving is not None, 'Run segmentation first.', 409)
    with sess.lock:
        _sync_masks(sess)
        return sess.reloc


@app.get('/api/session/{sid}/metrics')
def metrics(sid: str) -> dict:
    sess = _session(sid)
    ref = sess.fixed or sess.moving
    dims = list(ref.shape) if ref else [0, 0, 0]
    stages = {
        'upload': sess.fixed is not None and sess.moving is not None,
        'segment': sess.mask_moving is not None,
        'register': sess.disp is not None,
        'relocate': sess.reloc is not None,
    }
    markers = None
    if sess.reloc:
        markers = {
            'preop': {'mm': sess.reloc['preop']['mm'], 'color': 'green',
                      'label': 'Preop centre'},
            'affine': {'mm': sess.reloc['affine_aligned']['mm'], 'color': 'cyan',
                       'label': 'Affine-aligned'},
            'shifted': {'mm': sess.reloc['shifted']['mm'], 'color': 'yellow',
                        'label': 'Current centre'},
        }
    return {
        'session_id': sid,
        'stages': stages,
        'fixed': _meta(sess.fixed, Path(sess.fixed.path).name) if sess.fixed else None,
        'moving': _meta(sess.moving, Path(sess.moving.path).name) if sess.moving else None,
        'dims': dims,
        'affines': {'fixed': sess.fixed.affine.tolist() if sess.fixed else None,
                    'moving': sess.moving.affine.tolist() if sess.moving else None},
        'segment': {'engine': sess.segment_engine, 'detail': sess.segment_detail},
        'ncc': sess.ncc,
        'runtime_s': sess.runtime_s,
        'reloc': sess.reloc,
        'markers': markers,
        'views': {'warped': sess.warped is not None, 'flow': sess.disp is not None,
                  'checker': sess.warped is not None,
                  'ovl_pre': sess.mask_moving is not None,
                  'ovl_affine': sess.mask_affine is not None,
                  'ovl_now': sess.mask_now is not None},
        'has_segmentation_model': seg.MODEL_PATH.is_file(),
        'plan_text': sess.plan_text,
    }


_PLANE_MAX = {'axial': 2, 'coronal': 1, 'sagittal': 0}


@app.get('/api/session/{sid}/view/{kind}')
def view(sid: str, kind: str, plane: str = 'axial', idx: int = 0) -> Response:
    """One rendered slice as PNG (grayscale base or transparent overlay)."""
    sess = _session(sid)
    _require(plane in _PLANE_MAX, f'plane must be one of {list(_PLANE_MAX)}')
    _require(sess.fixed is not None or sess.moving is not None,
             'Nothing uploaded yet.', 409)
    _cache_u8(sess)
    idx = max(0, idx)

    def _n(key: str) -> int:
        data = {'fixed': sess.fixed.data if sess.fixed else None,
                'moving': sess.moving.data if sess.moving else None,
                'resampled': sess.moving_resampled, 'warped': sess.warped}[key]
        return data.shape[_PLANE_MAX[plane]]

    if kind in ('fixed', 'moving', 'resampled', 'warped'):
        _require(sess.u8.get(kind) is not None,
                 f'"{kind}" is not available yet (upload/register first).', 409)
        png = vis.render_gray(sess.u8[kind], plane, min(idx, _n(kind) - 1))
    elif kind == 'flow':
        _require(sess.disp is not None, 'Run registration first.', 409)
        png = vis.render_flow(sess.disp, sess.u8['fixed'], plane,
                              min(idx, _n('fixed') - 1))
    elif kind == 'checker':
        _require(sess.warped is not None, 'Run registration first.', 409)
        i = min(idx, min(_n('fixed'), _n('warped')) - 1)
        png = vis.render_checker(sess.u8['fixed'], sess.u8['warped'], plane, i)
    elif kind == 'ovl_pre':
        _require(sess.mask_moving is not None, 'Run segmentation first.', 409)
        marker = [{'mm': sess.reloc['preop']['mm'], 'color': 'green',
                   'label': 'Preop centre'}] if sess.reloc else None
        png = vis.render_overlay(sess.mask_moving, plane, min(idx, _n('moving') - 1),
                                 'green', marker, sess.moving.affine)
    elif kind == 'ovl_affine':
        _require(sess.mask_affine is not None,
                 'Run registration + segmentation first.', 409)
        marker = [{'mm': sess.reloc['affine_aligned']['mm'], 'color': 'cyan',
                   'label': 'Affine-aligned'}] if sess.reloc else None
        png = vis.render_overlay(sess.mask_affine, plane, min(idx, _n('fixed') - 1),
                                 'cyan', marker, sess.fixed.affine)
    elif kind == 'ovl_now':
        _require(sess.mask_now is not None,
                 'Run registration + segmentation first.', 409)
        marker = [{'mm': sess.reloc['shifted']['mm'], 'color': 'yellow',
                   'label': 'Current centre'}] if sess.reloc else None
        png = vis.render_overlay(sess.mask_now, plane, min(idx, _n('fixed') - 1),
                                 'red', marker, sess.fixed.affine)
    else:
        raise HTTPException(404, f'Unknown view "{kind}"')

    return Response(content=png, media_type='image/png',
                    headers={'Cache-Control': 'public, max-age=3600'})


# --------------------------------------------------------------------------
# 3D mesh / Cline SDK planning / export / health
# --------------------------------------------------------------------------
@app.get('/api/session/{sid}/mesh')
def mesh(sid: str) -> dict:
    """Tumour surfaces (marching cubes, RAS mm) for the in-browser 3D view."""
    sess = _session(sid)
    _require(sess.mask_now is not None, 'Run registration + segmentation first.', 409)
    preop = vis.tumor_mesh(sess.mask_moving, sess.moving.affine)
    affine = vis.tumor_mesh(sess.mask_affine, sess.fixed.affine)
    now = vis.tumor_mesh(sess.mask_now, sess.fixed.affine)
    return {'preop': preop, 'affine': affine, 'now': now}


@app.post('/api/session/{sid}/plan/update')
def plan_update(sid: str, payload: dict) -> dict:
    """Cline SDK: map the initial plan's coordinates through the deformation field."""
    sess = _session(sid)
    _require(sess.disp is not None, 'Run registration first.', 409)
    _require(sess.reloc is not None, 'Run segmentation + registration first.', 409)
    plan = {'points': payload.get('points', []),
            'notes': payload.get('notes', ''),
            'steps': payload.get('steps', [])}
    result = cline.update_plan(plan, sess.disp, sess.moving.affine,
                               sess.fixed.affine, sess.reloc, case=sid)
    sess.plan_text = result['plan_text']
    return result


@app.post('/api/session/{sid}/plan/ask')
def plan_ask(sid: str, payload: dict) -> dict:
    """Cline SDK assistant: case-aware Q&A for the surgeon."""
    sess = _session(sid)
    _require(sess.reloc is not None,
             'Run segmentation + registration first so I have case metrics.', 409)
    question = (payload.get('question') or '').strip()
    _require(bool(question), 'Question is empty.')
    return {'answer': cline.ask(question, sess.reloc, case=sid),
            'engine': 'claude' if cline.os.environ.get('ANTHROPIC_API_KEY')
            else 'deterministic'}


@app.get('/api/session/{sid}/export')
def export_bundle(sid: str) -> FileResponse:
    """Download the full 3D Slicer review bundle (NIfTI + masks + field + plan)."""
    sess = _session(sid)
    _require(sess.disp is not None, 'Run registration first.', 409)
    _require(sess.mask_now is not None, 'Run segmentation first.', 409)
    dest = sess.dir / 'slicer_bundle.zip'
    exp.build_bundle(
        dest,
        fixed=sess.fixed,
        moving_resampled=pp.Volume(sess.moving_resampled, sess.fixed.affine,
                                   sess.fixed.zooms),
        warped=pp.Volume(sess.warped, sess.fixed.affine, sess.fixed.zooms),
        disp=sess.disp,
        mask_pre=pp.Volume(sess.mask_moving.astype(np.uint8), sess.moving.affine,
                           sess.moving.zooms),
        mask_now=pp.Volume(sess.mask_now.astype(np.uint8), sess.fixed.affine,
                           sess.fixed.zooms),
        metrics={'session_id': sid, 'ncc': sess.ncc, 'runtime_s': sess.runtime_s,
                 'segment_engine': sess.segment_engine, 'reloc': sess.reloc},
        plan_text=sess.plan_text,
    )
    return FileResponse(dest, media_type='application/zip',
                        filename=f'neurovista_case_{sid}.zip')


@app.get('/api/health')
def health() -> dict:
    weights = reg.DEFAULT_WEIGHTS
    return {'status': 'ok',
            'weights': {'path': str(weights), 'exists': weights.is_file()},
            'segmentation_model': {'path': str(seg.MODEL_PATH),
                                   'exists': seg.MODEL_PATH.is_file()},
            'sessions': len(SESSIONS)}


# --------------------------------------------------------------------------
# Static frontend
# --------------------------------------------------------------------------
@app.get('/')
def index() -> FileResponse:
    return FileResponse(STATIC / 'index.html')


app.mount('/static', StaticFiles(directory=str(STATIC)), name='static')


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='127.0.0.1', port=8000)
