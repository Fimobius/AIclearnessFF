"""HTTP API + static web UI.

Run:  .venv/Scripts/python -m uvicorn app.main:app --port 8000
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import csv
import io
import json
import logging
import re
import shutil
import tempfile
import time
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError
from starlette.background import BackgroundTask

from . import config, db
from .imaging import io as imgio
from .imaging import sizing, textguard
from .imaging.pipeline import TIERS, ProcessSettings
from .imaging.sr import available_models
from .worker import Worker, job_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("aiclear")

VERSION = "1.1.0"
worker: Worker | None = None


def purge_old_jobs() -> None:
    if config.RETENTION_DAYS <= 0:
        return
    cutoff = time.time() - config.RETENTION_DAYS * 86400
    for r in db.q("""SELECT id FROM jobs WHERE created_at < ? AND NOT EXISTS
                     (SELECT 1 FROM items WHERE job_id=jobs.id AND status IN ('queued','processing'))""", (cutoff,)):
        db.ex("DELETE FROM jobs WHERE id=?", (r["id"],))
        shutil.rmtree(job_dir(r["id"]), ignore_errors=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global worker
    db.connect()
    purge_old_jobs()
    worker = Worker()
    worker.start()
    log.info("device: %s", worker.engine.device_info())
    yield
    worker.stop()
    db.close()


app = FastAPI(title="AI 图片高清化 & 尺寸标准化", version=VERSION, lifespan=lifespan,
              description="批量上传 → AI 高清化 → 按模板输出标准尺寸。所有接口返回 JSON，错误信息在 `detail` 字段。")


@app.middleware("http")
async def api_key_guard(request: Request, call_next):
    if config.API_KEY and request.url.path.startswith("/api/") and request.url.path != "/api/health":
        key = request.headers.get("x-api-key") or request.query_params.get("key")
        if key != config.API_KEY:
            return JSONResponse({"detail": "缺少或错误的 API Key（请求头 X-API-Key）"}, status_code=401)
    return await call_next(request)


# ---------------------------------------------------------------------------
# helpers

STATUS_ORDER = ("uploaded", "invalid", "skipped", "queued", "processing", "done", "failed", "canceled")


def _sanitize_stem(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem).strip(" .")
    return stem[:100] or "image"


def _assign_output_names(job_id: str, fmt: str, suffix: str) -> None:
    used: set[str] = set()
    for r in db.q("SELECT id, original_name FROM items WHERE job_id=? AND status IN ('queued','processing','done','failed','canceled') ORDER BY idx", (job_id,)):
        base = f"{_sanitize_stem(r['original_name'])}{_sanitize_suffix(suffix)}"
        name, n = f"{base}.{fmt}", 2
        while name.lower() in used:
            name, n = f"{base} ({n}).{fmt}", n + 1
        used.add(name.lower())
        db.ex("UPDATE items SET output_name=? WHERE id=?", (name, r["id"]))


def _sanitize_suffix(s: str) -> str:
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", s or "")


def _item_json(r: dict) -> dict:
    d = dict(r)
    rep = d.pop("report", None)
    d.pop("input_file", None)
    d.pop("output_file", None)
    live = worker.live.get(d["id"]) if worker else None
    d["stage"] = live["stage"] if live else None
    d["progress"] = live["progress"] if live else (1.0 if d["status"] == "done" else 0.0)
    if rep:
        d["result"] = {
            "width": rep["target"]["width"], "height": rep["target"]["height"],
            "bytes": rep["output"]["bytes"], "format": rep["output"]["format"],
            "qa": rep["qa"], "flags": rep["flags"], "warnings": rep["warnings"],
            "seconds": rep["timings"]["total"], "model": rep["model"]["name"],
            "magnification": rep["geometry"]["magnification"],
            "text": rep.get("text"),
        }
    base = f"/api/items/{d['id']}"
    d["urls"] = {"thumb": f"{base}/thumb", "original": f"{base}/original"}
    if d["status"] == "done":
        d["urls"].update(result=f"{base}/result", baseline=f"{base}/baseline", download=f"{base}/result?download=1")
    return d


def _job_json(job: dict, with_items: bool = True) -> dict:
    items = [dict(r) for r in db.q("SELECT * FROM items WHERE job_id=? ORDER BY idx", (job["id"],))]
    counts = {s: 0 for s in STATUS_ORDER}
    for it in items:
        counts[it["status"]] += 1
    active = counts["queued"] + counts["processing"]
    work = [it for it in items if it["status"] in ("queued", "processing", "done", "failed")]
    finished = counts["done"] + counts["failed"]
    live_frac = sum(worker.live.get(it["id"], {}).get("progress", 0) for it in items if it["status"] == "processing") if worker else 0
    progress = (finished + live_frac) / len(work) if work else 0.0
    eta = sum((it["est_seconds"] or 0) for it in items if it["status"] == "queued")
    for it in items:
        if it["status"] == "processing" and it["est_seconds"]:
            frac = worker.live.get(it["id"], {}).get("progress", 0) if worker else 0
            eta += it["est_seconds"] * (1 - frac)
    if job["state"] == "draft":
        status = "draft"
    elif active:
        status = "running" if counts["processing"] or counts["done"] or counts["failed"] else "queued"
    elif counts["failed"]:
        status = "done_with_errors"
    elif counts["done"]:
        status = "done"
    else:
        status = "canceled"
    out = {k: job[k] for k in ("id", "name", "created_at", "started_at", "state", "settings")}
    out.update(status=status, counts=counts, total=len(items), progress=round(progress, 4),
               eta_seconds=round(eta, 1), needs_review=sum(1 for it in items if it["report"] and '"needs_review": true' in it["report"]))
    if with_items:
        out["items"] = [_item_json(db.row_to_dict(it)) for it in items]
    return out


def _get_job(job_id: str) -> dict:
    job = db.row_to_dict(db.q1("SELECT * FROM jobs WHERE id=?", (job_id,)))
    if not job:
        raise HTTPException(404, "任务不存在")
    return job


def _get_item(item_id: str) -> dict:
    it = db.row_to_dict(db.q1("SELECT * FROM items WHERE id=?", (item_id,)))
    if not it:
        raise HTTPException(404, "图片不存在")
    return it


def _parse_settings(raw: str | dict | None) -> ProcessSettings:
    try:
        if raw is None or raw == "":
            return ProcessSettings()
        return ProcessSettings(**(json.loads(raw) if isinstance(raw, str) else raw))
    except ValidationError as e:
        msgs = "; ".join(f"{'.'.join(map(str, er['loc'])) or '设置'}: {er['msg']}" for er in e.errors())
        raise HTTPException(422, f"处理参数无效 — {msgs}")
    except json.JSONDecodeError:
        raise HTTPException(422, "settings 不是合法的 JSON")


async def _store_uploads(job_id: str, files: list[UploadFile]) -> None:
    jd = job_dir(job_id)
    for sub in ("in", "thumb"):
        (jd / sub).mkdir(parents=True, exist_ok=True)
    existing = db.q1("SELECT COUNT(*) n, COALESCE(SUM(size_bytes),0) s FROM items WHERE job_id=?", (job_id,))
    count, total = existing["n"], existing["s"]
    if count + len(files) > config.MAX_FILES_PER_JOB:
        raise HTTPException(413, f"单个任务最多 {config.MAX_FILES_PER_JOB} 张图片")
    limit = config.MAX_FILE_MB * 1024 * 1024
    for f in files:
        iid = db.new_id()
        name = Path(f.filename or "image").name
        data = await f.read(limit + 1)
        status, err, fmt, w, h = "uploaded", None, None, None, None
        if len(data) > limit:
            status, err = "invalid", f"文件超过 {config.MAX_FILE_MB} MB"
            data = b""
        elif total + len(data) > config.MAX_JOB_MB * 1024 * 1024:
            status, err = "invalid", f"批量总大小超过 {config.MAX_JOB_MB} MB"
            data = b""
        else:
            try:
                fmt, w, h = imgio.probe(data)
            except imgio.ImageError as e:
                status, err = "invalid", str(e)
        input_file = None
        if status == "uploaded":
            input_file = f"{iid}.{imgio.ALLOWED_FORMATS[fmt]}"
            (jd / "in" / input_file).write_bytes(data)
            try:
                (jd / "thumb" / f"{iid}.webp").write_bytes(imgio.make_thumbnail(data))
            except Exception as e:  # decodes in probe but fails later (exotic colour profile etc.)
                status, err = "invalid", f"无法生成预览：{e}"
            total += len(data)
        db.ex("""INSERT INTO items(id,job_id,idx,original_name,input_file,size_bytes,width,height,format,status,error)
                 VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
              (iid, job_id, count, name, input_file, len(data), w, h, fmt, status, err))
        count += 1


def _start(job: dict, settings: ProcessSettings, item_ids: list[str] | None) -> None:
    if job["state"] != "draft":
        raise HTTPException(409, "任务已经开始，不能重复启动")
    now = time.time()
    rows = db.q("SELECT id, width, height FROM items WHERE job_id=? AND status='uploaded' ORDER BY idx", (job["id"],))
    chosen = {r["id"] for r in rows} if item_ids is None else set(item_ids)
    if not chosen & {r["id"] for r in rows}:
        raise HTTPException(422, "没有选中任何可处理的图片")
    with db.tx() as c:
        c.execute("UPDATE jobs SET state='started', started_at=?, settings=? WHERE id=?",
                  (now, settings.model_dump_json(), job["id"]))
        for r in rows:
            if r["id"] in chosen:
                est = worker.estimate(r["width"], r["height"], settings) if worker else None
                c.execute("UPDATE items SET status='queued', queued_at=?, est_seconds=? WHERE id=?", (now, est, r["id"]))
            else:
                c.execute("UPDATE items SET status='skipped' WHERE id=?", (r["id"],))
    _assign_output_names(job["id"], settings.format, settings.name_suffix)
    if worker:
        worker.notify()


# ---------------------------------------------------------------------------
# meta

@app.get("/api/health", tags=["meta"])
def health():
    return {
        "app": "AIClear", "version": VERSION, "device": worker.engine.device_info() if worker else None,
        "tiers": {k: {"label": v["label"], "desc": v["desc"], "model": v["spec"].name} for k, v in TIERS.items()},
        "models": {k: {"available": v["available"], "license": v["license"], "desc": v["desc"]}
                   for k, v in available_models().items()},
        "allow_stretch": config.ALLOW_STRETCH, "text_protect_available": textguard.available(), "auth_required": bool(config.API_KEY),
        "limits": {"max_file_mb": config.MAX_FILE_MB, "max_files_per_job": config.MAX_FILES_PER_JOB,
                   "max_job_mb": config.MAX_JOB_MB, "min_side": config.MIN_SIDE, "max_side": config.MAX_SIDE,
                   "max_output_mp": config.MAX_OUTPUT_PIXELS / 1e6},
        "defaults": ProcessSettings().model_dump(),
        "rates": worker.rates if worker else None,
    }


@app.get("/api/size", tags=["meta"], summary="按宽度 + 宽高比计算输出尺寸")
def size(width: int, ratio_w: float | None = None, ratio_h: float | None = None, height: int | None = None):
    try:
        t = sizing.compute_target(width, ratio_w, ratio_h, height)
    except sizing.SizeError as e:
        raise HTTPException(422, str(e))
    return {"width": t.width, "height": t.height, "exact_ratio": t.exact_ratio,
            "ratio": f"10:{t.actual_ratio_h:g}", "suggested_widths": list(t.suggested_widths)}


# ---------------------------------------------------------------------------
# presets

class PresetIn(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    width: int
    ratio_w: float | None = None
    ratio_h: float | None = None
    height: int | None = None


def _validate_preset(p: PresetIn) -> None:
    try:
        sizing.compute_target(p.width, p.ratio_w, p.ratio_h, p.height)
    except sizing.SizeError as e:
        raise HTTPException(422, str(e))


def _preset_json(r) -> dict:
    d = dict(r)
    t = sizing.compute_target(d["width"], d["ratio_w"], d["ratio_h"], d["height"])
    d.update(out_width=t.width, out_height=t.height, exact_ratio=t.exact_ratio, builtin=bool(d["builtin"]))
    return d


@app.get("/api/presets", tags=["presets"])
def list_presets():
    return [_preset_json(r) for r in db.q("SELECT * FROM presets ORDER BY builtin DESC, id")]


@app.post("/api/presets", tags=["presets"], status_code=201)
def create_preset(p: PresetIn):
    _validate_preset(p)
    try:
        db.ex("INSERT INTO presets(name,width,ratio_w,ratio_h,height,created_at) VALUES(?,?,?,?,?,?)",
              (p.name.strip(), p.width, p.ratio_w, p.ratio_h, p.height, time.time()))
    except Exception:
        raise HTTPException(409, "模板名称已存在")
    return _preset_json(db.q1("SELECT * FROM presets WHERE name=?", (p.name.strip(),)))


@app.put("/api/presets/{pid}", tags=["presets"])
def update_preset(pid: int, p: PresetIn):
    r = db.q1("SELECT * FROM presets WHERE id=?", (pid,))
    if not r:
        raise HTTPException(404, "模板不存在")
    if r["builtin"]:
        raise HTTPException(403, "内置模板不能修改，可另存为新模板")
    _validate_preset(p)
    try:
        db.ex("UPDATE presets SET name=?, width=?, ratio_w=?, ratio_h=?, height=? WHERE id=?",
              (p.name.strip(), p.width, p.ratio_w, p.ratio_h, p.height, pid))
    except Exception:
        raise HTTPException(409, "模板名称已存在")
    return _preset_json(db.q1("SELECT * FROM presets WHERE id=?", (pid,)))


@app.delete("/api/presets/{pid}", tags=["presets"], status_code=204)
def delete_preset(pid: int):
    r = db.q1("SELECT * FROM presets WHERE id=?", (pid,))
    if not r:
        raise HTTPException(404, "模板不存在")
    if r["builtin"]:
        raise HTTPException(403, "内置模板不能删除")
    db.ex("DELETE FROM presets WHERE id=?", (pid,))
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# jobs

@app.post("/api/jobs", tags=["jobs"], status_code=201,
          summary="上传图片创建任务（start=true 时直接开始处理）")
async def create_job(files: Annotated[list[UploadFile], File(description="JPG / PNG / WEBP，可多选")],
                     name: Annotated[str | None, Form()] = None,
                     settings: Annotated[str | None, Form(description="处理参数 JSON，见 /api/health 的 defaults")] = None,
                     start: Annotated[bool, Form()] = False):
    parsed = _parse_settings(settings) if (start or settings) else None
    jid = db.new_id()
    db.ex("INSERT INTO jobs(id,name,created_at,state) VALUES(?,?,?, 'draft')",
          (jid, (name or time.strftime("任务 %m-%d %H:%M:%S")).strip()[:80], time.time()))
    await _store_uploads(jid, files)
    job = _get_job(jid)
    if start:
        _start(job, parsed, None)
        job = _get_job(jid)
    return _job_json(job)


@app.post("/api/jobs/{job_id}/files", tags=["jobs"], summary="向未开始的任务追加图片")
async def add_files(job_id: str, files: Annotated[list[UploadFile], File()]):
    job = _get_job(job_id)
    if job["state"] != "draft":
        raise HTTPException(409, "任务已开始，不能再添加图片")
    await _store_uploads(job_id, files)
    return _job_json(job)


class StartIn(BaseModel):
    settings: dict = Field(default_factory=dict)
    item_ids: list[str] | None = Field(None, description="要处理的图片 id；不传则处理全部有效图片")


@app.post("/api/jobs/{job_id}/start", tags=["jobs"])
def start_job(job_id: str, body: StartIn):
    job = _get_job(job_id)
    _start(job, _parse_settings(body.settings), body.item_ids)
    return _job_json(_get_job(job_id))


@app.get("/api/jobs", tags=["jobs"], summary="历史任务")
def list_jobs(limit: int = Query(50, le=500)):
    return [_job_json(db.row_to_dict(r), with_items=False)
            for r in db.q("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))]


@app.get("/api/jobs/{job_id}", tags=["jobs"])
def get_job(job_id: str):
    return _job_json(_get_job(job_id))


@app.post("/api/jobs/{job_id}/cancel", tags=["jobs"], summary="取消排队中的图片")
def cancel_job(job_id: str):
    _get_job(job_id)
    db.ex("UPDATE items SET status='canceled' WHERE job_id=? AND status='queued'", (job_id,))
    return _job_json(_get_job(job_id))


@app.post("/api/jobs/{job_id}/retry", tags=["jobs"], summary="重新处理全部失败/已取消的图片")
def retry_job(job_id: str):
    _get_job(job_id)
    db.ex("UPDATE items SET status='queued', error=NULL, queued_at=? WHERE job_id=? AND status IN ('failed','canceled')",
          (time.time(), job_id))
    if worker:
        worker.notify()
    return _job_json(_get_job(job_id))


@app.delete("/api/jobs/{job_id}", tags=["jobs"], status_code=204)
def delete_job(job_id: str):
    _get_job(job_id)
    db.ex("DELETE FROM jobs WHERE id=?", (job_id,))
    shutil.rmtree(job_dir(job_id), ignore_errors=True)
    return Response(status_code=204)


def _content_disposition(name: str, inline: bool = False) -> str:
    ascii_name = re.sub(r"[^\x20-\x7e]", "_", name).replace('"', "")
    return f"{'inline' if inline else 'attachment'}; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(name)}"


REPORT_COLUMNS = ["原文件名", "输出文件名", "状态", "错误", "输出宽", "输出高", "格式", "文件大小KB", "增强档位", "模型",
                  "放大倍数", "SSIM", "平均色差ΔE", "清晰度提升倍数", "需人工复核", "文字保护处数", "耗时秒", "提示"]


STATUS_LABEL = {"uploaded": "未开始", "invalid": "无法读取", "skipped": "未选择", "queued": "排队中",
                "processing": "处理中", "done": "成功", "failed": "失败", "canceled": "已取消"}


def _report_rows(items: list[dict]) -> list[list]:
    rows = []
    for it in items:
        rep = it.get("report") or {}
        qa = rep.get("qa", {})
        rows.append([
            it["original_name"], it.get("output_name") or "", STATUS_LABEL[it["status"]], it.get("error") or "",
            rep.get("target", {}).get("width", ""), rep.get("target", {}).get("height", ""),
            rep.get("output", {}).get("format", ""),
            round(rep["output"]["bytes"] / 1024, 1) if rep else "",
            TIERS.get(rep.get("model", {}).get("tier"), {}).get("label", ""), rep.get("model", {}).get("name", ""),
            rep.get("geometry", {}).get("magnification", ""), qa.get("ssim", ""), qa.get("delta_e_mean", ""),
            qa.get("sharpness_gain", ""), "是" if qa.get("needs_review") else ("否" if qa else ""),
            (rep.get("text") or {}).get("protected", "") if rep else "",
            rep.get("timings", {}).get("total", ""), "；".join(rep.get("warnings", [])),
        ])
    return rows


@app.get("/api/jobs/{job_id}/download", tags=["jobs"], summary="打包下载全部已完成图片（含处理报告 CSV）")
def download_zip(job_id: str):
    job = _get_job(job_id)
    items = [db.row_to_dict(r) for r in db.q("SELECT * FROM items WHERE job_id=? ORDER BY idx", (job_id,))]
    done = [it for it in items if it["status"] == "done"]
    if not done:
        raise HTTPException(404, "还没有处理完成的图片")
    tmp_dir = config.DATA_DIR / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".zip", dir=tmp_dir)
    with open(fd, "wb") as fh, zipfile.ZipFile(fh, "w", zipfile.ZIP_STORED) as z:
        for it in done:
            z.write(job_dir(job_id) / "out" / it["output_file"], it["output_name"])
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(REPORT_COLUMNS)
        w.writerows(_report_rows(items))
        z.writestr("处理报告.csv", "﻿" + buf.getvalue())  # BOM so Excel detects UTF-8
    name = f"{_sanitize_stem(job['name'])}_高清.zip"
    return FileResponse(tmp, media_type="application/zip", headers={"Content-Disposition": _content_disposition(name)},
                        background=BackgroundTask(lambda: Path(tmp).unlink(missing_ok=True)))


@app.get("/api/jobs/{job_id}/report.csv", tags=["jobs"])
def report_csv(job_id: str):
    job = _get_job(job_id)
    items = [db.row_to_dict(r) for r in db.q("SELECT * FROM items WHERE job_id=? ORDER BY idx", (job_id,))]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(REPORT_COLUMNS)
    w.writerows(_report_rows(items))
    return Response("﻿" + buf.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": _content_disposition(f"{_sanitize_stem(job['name'])}_报告.csv")})


# ---------------------------------------------------------------------------
# items

@app.get("/api/items/{item_id}", tags=["items"], summary="单张图片详情（含完整处理报告）")
def get_item(item_id: str):
    it = _get_item(item_id)
    out = _item_json(it)
    out["report"] = it.get("report")
    return out


@app.post("/api/items/{item_id}/retry", tags=["items"])
def retry_item(item_id: str):
    it = _get_item(item_id)
    if it["status"] not in ("failed", "canceled"):
        raise HTTPException(409, "只有失败或已取消的图片可以重新处理")
    db.ex("UPDATE items SET status='queued', error=NULL, queued_at=? WHERE id=?", (time.time(), item_id))
    if worker:
        worker.notify()
    return _item_json(_get_item(item_id))


def _file(path: Path, media: str | None = None, headers: dict | None = None) -> FileResponse:
    if not path.exists():
        raise HTTPException(404, "文件不存在")
    return FileResponse(path, media_type=media, headers={"Cache-Control": "private, max-age=86400", **(headers or {})})


@app.get("/api/items/{item_id}/thumb", tags=["items"])
def item_thumb(item_id: str):
    it = _get_item(item_id)
    return _file(job_dir(it["job_id"]) / "thumb" / f"{item_id}.webp", "image/webp")


@app.get("/api/items/{item_id}/original", tags=["items"])
def item_original(item_id: str):
    it = _get_item(item_id)
    if not it["input_file"]:
        raise HTTPException(404, "原图不可用")
    return _file(job_dir(it["job_id"]) / "in" / it["input_file"])


@app.get("/api/items/{item_id}/baseline", tags=["items"], summary="相同构图的普通缩放图（对比用）")
def item_baseline(item_id: str):
    it = _get_item(item_id)
    return _file(job_dir(it["job_id"]) / "out" / f"{item_id}.baseline.webp", "image/webp")


@app.get("/api/items/{item_id}/result", tags=["items"])
def item_result(item_id: str, download: bool = False):
    it = _get_item(item_id)
    if it["status"] != "done":
        raise HTTPException(404, "尚未处理完成")
    return _file(job_dir(it["job_id"]) / "out" / it["output_file"],
                 headers={"Content-Disposition": _content_disposition(it["output_name"], inline=not download)})


# ---------------------------------------------------------------------------
# static web UI

if config.FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=config.FRONTEND_DIST, html=True), name="ui")
