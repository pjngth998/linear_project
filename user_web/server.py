"""
Rewrite Reborn — เว็บแอปสำหรับผู้ใช้ (FastAPI)

    python -m webapp                 # เปิด http://localhost:8000
    python -m webapp --port 8080 --no-browser

API
    POST /api/files                         อัปโหลดหลายไฟล์ → รายการไฟล์ + จำนวนหน้า
    GET  /api/files/{fid}/pages/{n}/thumb   ภาพย่อของหน้า (เริ่มนับ 1)
    DELETE /api/files/{fid}                 ลบไฟล์ออกจากรายการ
    POST /api/jobs                          เริ่มจัดลายมือหน้าที่เลือก + การตั้งค่า
    GET  /api/jobs/{jid}                    สถานะ / ผลลัพธ์
    GET  /api/jobs/{jid}/img/{i}/{which}    ภาพก่อน (before) / หลัง (after) ของหน้าที่ i  (?w=ความกว้าง)
    GET  /api/jobs/{jid}/download.pdf       รวมผลทุกหน้าเป็น PDF
    GET  /api/jobs/{jid}/download.zip       ภาพผลทุกหน้า (PNG) + ตัวชี้วัด
"""
from __future__ import annotations

import io
import json
import os
import sys
import threading
import time
import traceback
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional   # ใช้ typing แทน X | None ให้รองรับ Python 3.8–3.9

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_rr() -> Optional[str]:
    """หาโฟลเดอร์ที่มี rr/ (โค้ด pipeline): ตัวแปร RR_HOME → โฟลเดอร์นี้ → โฟลเดอร์แม่ขึ้นไปทีละชั้น"""
    cands = [os.environ.get("RR_HOME", "")]
    d = HERE
    for _ in range(4):
        cands.append(d)
        d = os.path.dirname(d)
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "rr", "pipeline.py")):
            return c
    return None


_root = _find_rr()
if _root is None:
    raise SystemExit(
        "หาโฟลเดอร์ rr/ (โค้ด pipeline) ไม่เจอ\n"
        f"  - วางโฟลเดอร์ {os.path.basename(HERE)}/ ไว้ข้าง ๆ โฟลเดอร์ rr/ ในโปรเจกต์ หรือ\n"
        "  - ตั้งตัวแปร RR_HOME ให้ชี้ไปที่โฟลเดอร์ที่มี rr/ เช่น  RR_HOME=/path/to/rewrite_reborn python server.py")
sys.path.insert(0, _root)
from rr.config import Params            # noqa: E402
from rr.io_loader import load_bytes     # noqa: E402
from rr.pipeline import run_page        # noqa: E402

# ไฟล์หน้าเว็บ: อยู่ใน static/ หรือวางไว้โฟลเดอร์เดียวกับ server.py ก็ได้
STATIC = os.path.join(HERE, "static") if os.path.isfile(os.path.join(HERE, "static", "index.html")) else HERE
ALLOWED = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".goodnotes"}
MAX_MB = 80

app = FastAPI(title="Rewrite Reborn")
FILES: Dict[str, dict] = {}
JOBS: Dict[str, dict] = {}
LOCK = threading.Lock()
POOL = ThreadPoolExecutor(max_workers=max(1, min(2, (os.cpu_count() or 2) // 2)))


# ------------------------------------------------------------------ helpers
def png(img: np.ndarray, width: Optional[int] = None) -> bytes:
    if width and img.shape[1] > width:
        s = width / img.shape[1]
        img = cv2.resize(img, (width, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def jpg(img: np.ndarray, width: Optional[int] = None, q: int = 86) -> bytes:
    if width and img.shape[1] > width:
        s = width / img.shape[1]
        img = cv2.resize(img, (width, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes()


def page_count(data: bytes, name: str) -> tuple[int, str]:
    ext = os.path.splitext(name)[1].lower()
    if ext == ".pdf":
        import pymupdf
        doc = pymupdf.open(stream=data, filetype="pdf")
        return doc.page_count, "pdf"
    return 1, "image"


def thumbnail(data: bytes, name: str, n: int, width: int = 360) -> bytes:
    ext = os.path.splitext(name)[1].lower()
    if ext == ".pdf":
        import pymupdf
        doc = pymupdf.open(stream=data, filetype="pdf")
        pg = doc[n - 1]
        zoom = width / max(pg.rect.width, 1)
        pix = pg.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
        img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    else:
        img = load_bytes(data, name, "1")[0].image
    return jpg(img, width, 82)


class PageRef(BaseModel):
    fid: str
    page: int


class Settings(BaseModel):
    strength: str = "medium"          # light | medium | strong
    fix_baseline: bool = True
    fix_slant: bool = True
    fix_size: bool = True
    fix_spacing: bool = True
    fix_shape: bool = True
    fix_lines: bool = True
    slant_style: str = "keep"         # keep | upright
    output_bg: str = "paper"          # paper | white
    background: str = "auto"          # auto | light | dark
    light_ink: bool = False           # ปากกาสีอ่อน
    sensitivity: int = 50             # 0 = เก็บเส้นจาง ... 100 = ตัดจุดรบกวน
    keep_math: bool = True            # คงสูตร / กรอบ / ภาพไว้ตามเดิม


class JobIn(BaseModel):
    pages: List[PageRef]
    settings: Settings = Settings()


def to_params(s: Settings) -> Params:
    alpha = {"light": 0.5, "medium": 0.8, "strong": 1.0}.get(s.strength, 0.8)
    return Params(
        alpha=alpha,
        enable_baseline=s.fix_baseline, enable_slant=s.fix_slant, enable_scale=s.fix_size,
        enable_spacing=s.fix_spacing, shape_consistency=s.fix_shape,
        regularize_spacing=s.fix_lines, regularize_indent=s.fix_lines,
        slant_target="vertical" if s.slant_style == "upright" else "page_median",
        output_bg="white" if s.output_bg == "white" else "paper",
        background_mode=s.background if s.background in ("auto", "light", "dark") else "auto",
        ink_max=200 if s.light_ink else 160,
        thresh_C=int(round(6 + (max(0, min(100, s.sensitivity)) / 100) * 14)),     # 6 … 20
        freeze_regions=s.keep_math, freeze_math=s.keep_math, bind_fractions=True,
        protect_images=True, protect_headings=True,
    )


def summarize(metrics: dict) -> dict:
    rows = [r for r in metrics.get("rows", []) if r.get("improve_pct") == r.get("improve_pct")]
    imp = {r["metric"]: round(float(r["improve_pct"]), 1) for r in rows}
    core = [v for k, v in imp.items()]
    idc = metrics.get("identity_cos")
    if idc is None or idc != idc:
        identity = None
    elif idc >= 0.995:
        identity = "คงลายมือเดิมเกือบทั้งหมด"
    elif idc >= 0.985:
        identity = "ปรับรูปตัวอักษรเล็กน้อย"
    else:
        identity = "ปรับรูปตัวอักษรค่อนข้างมาก"
    return {"regularity_pct": round(float(np.mean(core)), 0) if core else None,
            "baseline_pct": imp.get("baseline_sigma"), "slant_pct": imp.get("slant_std_deg"),
            "size_pct": imp.get("height_cv"), "lines_pct": imp.get("indent_dev"),
            "identity_cos": None if idc is None or idc != idc else round(float(idc), 4),
            "identity_label": identity, "detail": imp}


def run_job(jid: str):
    job = JOBS[jid]
    p = job["params"]
    for i, it in enumerate(job["items"]):
        if job.get("cancel"):
            break
        it["status"] = "running"
        t0 = time.time()
        try:
            f = FILES[it["fid"]]
            page = load_bytes(f["data"], f["name"], str(it["page"]), p.dpi)[0]
            r = run_page(page, p)
            it["before"] = r.original
            it["after"] = r.output
            it["summary"] = summarize(r.metrics)
            # แสดงเฉพาะข้อความที่ผู้ใช้ทำอะไรต่อได้ (ข้อความทางเทคนิคสำหรับผู้พัฒนาไม่แสดง)
            it["notes"] = [n for n in r.notes if "vector" not in n and "pipeline" not in n]
            it["status"] = "done"
        except Exception as e:                     # หน้าเดียวพัง ไม่ทำให้ทั้งงานหยุด
            traceback.print_exc()
            it["status"] = "error"
            it["error"] = str(e)[:300]
        it["seconds"] = round(time.time() - t0, 1)
        job["done"] = i + 1
    job["status"] = "cancelled" if job.get("cancel") else "done"
    job["finished"] = time.time()


# ------------------------------------------------------------------ API
@app.post("/api/files")
async def upload(files: List[UploadFile] = File(...)):
    out = []
    for uf in files:
        name = uf.filename or "file"
        ext = os.path.splitext(name)[1].lower()
        data = await uf.read()
        item = {"name": name, "size": len(data)}
        if ext not in ALLOWED:
            item["error"] = "รองรับเฉพาะ PDF, PNG, JPG และ WEBP"
        elif ext == ".goodnotes":
            item["error"] = "ไฟล์ GoodNotes เปิดตรงไม่ได้ — ใน GoodNotes ให้กด Share → Export → PDF แล้วอัปโหลดไฟล์ PDF แทน"
        elif len(data) > MAX_MB * 1024 * 1024:
            item["error"] = f"ไฟล์ใหญ่เกิน {MAX_MB} MB"
        else:
            try:
                n, kind = page_count(data, name)
                if n == 0:
                    raise ValueError("ไม่มีหน้าในไฟล์")
                fid = uuid.uuid4().hex[:10]
                FILES[fid] = {"name": name, "data": data, "pages": n, "kind": kind}
                item.update({"fid": fid, "pages": n, "kind": kind})
            except Exception as e:
                item["error"] = f"เปิดไฟล์ไม่ได้ ({str(e)[:120]})"
        out.append(item)
    return {"files": out}


@app.get("/api/files/{fid}/pages/{n}/thumb")
def thumb(fid: str, n: int, w: int = 360):
    f = FILES.get(fid)
    if not f or not (1 <= n <= f["pages"]):
        raise HTTPException(404, "ไม่พบหน้านี้")
    key = ("thumb", n, w)
    cache = f.setdefault("cache", {})
    if key not in cache:
        cache[key] = thumbnail(f["data"], f["name"], n, min(max(w, 120), 1200))
    return Response(cache[key], media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


@app.delete("/api/files/{fid}")
def remove_file(fid: str):
    FILES.pop(fid, None)
    return {"ok": True}


@app.post("/api/jobs")
def create_job(body: JobIn):
    items = []
    for pr in body.pages:
        f = FILES.get(pr.fid)
        if not f or not (1 <= pr.page <= f["pages"]):
            continue
        stem = os.path.splitext(f["name"])[0]
        items.append({"fid": pr.fid, "page": pr.page, "file": f["name"],
                      "name": f"{stem}_p{pr.page}" if f["pages"] > 1 else stem, "status": "queued"})
    if not items:
        raise HTTPException(400, "ยังไม่ได้เลือกหน้าที่จะจัด")
    jid = uuid.uuid4().hex[:10]
    JOBS[jid] = {"items": items, "params": to_params(body.settings), "status": "running", "done": 0,
                 "created": time.time(), "settings": body.settings.model_dump()}
    POOL.submit(run_job, jid)
    return {"job": jid, "total": len(items)}


@app.get("/api/jobs/{jid}")
def job_status(jid: str):
    job = JOBS.get(jid)
    if not job:
        raise HTTPException(404, "ไม่พบงานนี้")
    items = [{k: v for k, v in it.items() if k not in ("before", "after")} for it in job["items"]]
    return {"status": job["status"], "done": job["done"], "total": len(items), "items": items}


@app.post("/api/jobs/{jid}/cancel")
def cancel(jid: str):
    if jid in JOBS:
        JOBS[jid]["cancel"] = True
    return {"ok": True}


@app.get("/api/jobs/{jid}/img/{i}/{which}")
def job_img(jid: str, i: int, which: str, w: Optional[int] = None, fmt: str = "jpg"):
    job = JOBS.get(jid)
    if not job or not (0 <= i < len(job["items"])) or which not in ("before", "after"):
        raise HTTPException(404, "ไม่พบภาพ")
    img = job["items"][i].get(which)
    if img is None:
        raise HTTPException(404, "ยังประมวลผลไม่เสร็จ")
    if fmt == "png":
        return Response(png(img, w), media_type="image/png")
    return Response(jpg(img, w), media_type="image/jpeg")


def _done(job):
    return [it for it in job["items"] if it.get("after") is not None]


@app.get("/api/jobs/{jid}/download.pdf")
def download_pdf(jid: str):
    job = JOBS.get(jid)
    if not job or not _done(job):
        raise HTTPException(404, "ยังไม่มีผลลัพธ์")
    from PIL import Image
    pages = [Image.fromarray(cv2.cvtColor(it["after"], cv2.COLOR_BGR2RGB)) for it in _done(job)]
    buf = io.BytesIO()
    pages[0].save(buf, "PDF", save_all=True, append_images=pages[1:], resolution=200.0)
    return Response(buf.getvalue(), media_type="application/pdf",
                    headers={"Content-Disposition": 'attachment; filename="rewrite_reborn.pdf"'})


@app.get("/api/jobs/{jid}/download.zip")
def download_zip(jid: str):
    job = JOBS.get(jid)
    if not job or not _done(job):
        raise HTTPException(404, "ยังไม่มีผลลัพธ์")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for it in _done(job):
            z.writestr(f"{it['name']}.png", png(it["after"]))
        z.writestr("metrics.json", json.dumps(
            [{"name": it["name"], **it.get("summary", {})} for it in _done(job)], ensure_ascii=False, indent=2))
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="rewrite_reborn.zip"'})


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/static/{name}")
def static_file(name: str):
    # เปิดให้เฉพาะไฟล์หน้าเว็บ (ไม่เปิดทั้งโฟลเดอร์ เผื่อวางไว้รวมกับไฟล์ .py)
    types = {"app.css": "text/css", "app.js": "application/javascript"}
    if name not in types or not os.path.isfile(os.path.join(STATIC, name)):
        raise HTTPException(404, "ไม่พบไฟล์")
    return FileResponse(os.path.join(STATIC, name), media_type=types[name])


if __name__ == "__main__":            # python server.py  (ไม่ต้องเป็น package)
    import argparse
    import webbrowser

    import uvicorn
    ap = argparse.ArgumentParser(description="Rewrite Reborn web app")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    url = f"http://localhost:{a.port}"
    print(f"Rewrite Reborn กำลังทำงานที่ {url}  (กด Ctrl+C เพื่อหยุด)")
    if not a.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")