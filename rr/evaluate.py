"""
E. ประเมินผล (วัดก่อน/หลังด้วยวิธีเดียวกัน ในพิกัดหลังแก้หน้าเอียงแล้ว)

ความเป็นระเบียบ (ยิ่งต่ำยิ่งดี)
    baseline_sigma   σ ของ residual จาก baseline (least squares) หารด้วย x-height
    slant_std_deg    ส่วนเบี่ยงเบนมาตรฐานของมุมเอียงรายตัว (องศา)
    height_cv        coefficient of variation ของความสูงในกลุ่มเดียวกัน
    gap_cv           CV ของช่องว่างระหว่างตัว (เฉพาะกลุ่ม 'small')
ความคงเอกลักษณ์ (ยิ่งใกล้ 1 ยิ่งดี)
    identity_cos     cosine similarity รูปทรงก่อน/หลัง (resample 64 จุด, จัดจุดเริ่มต้นแล้ว)
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile

import cv2
import numpy as np

from . import geometry as G
from .components import Glyph
from .config import Params
from .correction import glyph_segments
from .layout import Layout


def _ext(P: np.ndarray):
    return P[:, 0].min(), P[:, 1].min(), P[:, 0].max(), P[:, 1].max(), P[:, 0].mean()


def _cv(v):
    v = np.asarray(v, float)
    return float(np.std(v) / np.mean(v)) if len(v) >= 3 and np.mean(v) > 0 else float("nan")


def _refine_metrics(glyphs, lay, finals, p) -> tuple:
    """ตัวชี้วัดของการปรับรูปทรงและการจัดบรรทัด (วัดก่อน/หลังด้วยวิธีเดียวกัน)
    shape_dev      ในแต่ละกลุ่มตัวที่หน้าตาเหมือนกัน: ระยะ RMS ของแต่ละตัวจากรูปเฉลี่ยของกลุ่ม หารด้วยขนาด
    line_pitch_cv  CV ของระยะห่างระหว่างบรรทัดในแต่ละช่วง (block)
    indent_dev     SD ของจุดเริ่มบรรทัดในกลุ่มย่อหน้าเดียวกัน หารด้วย x-height"""
    from .refine import prototype
    gl = {g.gid: g for g in glyphs}
    lines = {L.lid: L for L in lay.lines}
    B, A = {}, {}
    # --- shape
    groups = {}
    for g in glyphs:
        if g.shape_cluster >= 0 and g.gid in finals:
            groups.setdefault(g.shape_cluster, []).append(g)
    for store, src in ((B, lambda g: g.outer), (A, lambda g: finals[g.gid][0])):
        devs = []
        for mem in groups.values():
            shapes = []
            for g in mem:
                X = G.resample_n(src(g), p.shape_points)
                X = X - X.mean(axis=0)
                shapes.append(X / (np.sqrt(np.mean(np.sum(X ** 2, axis=1))) + 1e-9))
            proto, shifts = prototype(shapes)
            for X, s_ in zip(shapes, shifts):
                devs.append(float(np.sqrt(np.mean(np.sum((np.roll(X, s_, axis=0) - proto) ** 2, axis=1)))))
        if devs:
            store["shape_dev"] = float(np.mean(devs))

    def base_y(L, src):
        ys = [src(i)[:, 1].max() for i in L.members if gl[i].cls in ("mid", "asc")]
        return float(np.median(ys)) if ys else None

    def start_x(L, src):
        return float(min(src(i)[:, 0].min() for i in L.members))

    for store, src in ((B, lambda i: gl[i].outer), (A, lambda i: finals[i][0])):
        cvs = []
        for blk in getattr(lay, "blocks", []):
            ys = [base_y(lines[l], src) for l in blk]
            if None in ys or len(ys) < 3:
                continue
            d = np.diff(ys)
            if np.mean(d) > 0:
                cvs.append(float(np.std(d) / np.mean(d)))
        if cvs:
            store["line_pitch_cv"] = float(np.mean(cvs))
        sds = []
        for grp in getattr(lay, "indent_groups", []):
            xs = [start_x(lines[l], src) for l in grp]
            xh = float(np.median([lines[l].xh for l in grp]))
            sds.append(float(np.std(xs)) / max(xh, 1e-6))
        if sds:
            store["indent_dev"] = float(np.mean(sds))
    return B, A


def evaluate(glyphs: list[Glyph], lay: Layout, finals: dict, p: Params) -> dict:
    gl = {g.gid: g for g in glyphs}
    before = {"baseline": [], "slant": [], "gaps": [], "h": {"mid": [], "asc": [], "desc": []}}
    after = {"baseline": [], "slant": [], "gaps": [], "h": {"mid": [], "asc": [], "desc": []}}
    cos = []

    for L in lay.lines:
        # บรรทัด/ตัวที่ freeze ไม่ถูกแก้ ไม่นับในตัวชี้วัด  เศษส่วน (หน่วยแข็ง) ก็ไม่นับรูปทรง/ความสูง
        ids = [i for i in L.members if not gl[i].frozen and gl[i].cls != "fraction"]
        if not ids or L.frozen:
            continue
        # baseline residual
        sel = [i for i in ids if gl[i].cls in ("mid", "asc")]
        if len(sel) >= 4:
            # ก่อน/หลังวัดจาก contour ด้วยวิธีเดียวกัน (x เฉลี่ย, y ล่างสุด)
            for store, src in ((before, lambda i: gl[i].outer), (after, lambda i: finals[i][0])):
                pts = np.array([(_ext(src(i))[4], _ext(src(i))[3]) for i in sel])
                a, b = G.fit_line_lsq(pts[:, 0], pts[:, 1])
                r = pts[:, 1] - (a * pts[:, 0] + b)
                store["baseline"].append(np.std(r) / L.xh)
        # gaps
        for store, ext in ((before, lambda i: (_ext(gl[i].outer)[0], _ext(gl[i].outer)[2])),
                           (after, lambda i: (_ext(finals[i][0])[0], _ext(finals[i][0])[2]))):
            right = None
            kinds = {r: k for (_, r, _, k) in L.gaps}
            for i in ids:
                x0, x1 = ext(i)
                if right is not None and kinds.get(i) == "small" and x0 - right > 0:
                    store["gaps"].append(x0 - right)
                right = x1 if right is None else max(right, x1)
        # slant, height, identity
        for i in ids:
            g = gl[i]
            fo = finals[i][0]
            sb = glyph_segments(g, L.xh, p)
            sa = glyph_segments(g, L.xh, p, P=fo)
            if len(sb) >= 6 and len(sa) >= 6:
                before["slant"].append(np.median(sb))
                after["slant"].append(np.median(sa))
            if g.cls in before["h"] and not g.merged:
                e0 = _ext(g.outer)
                before["h"][g.cls].append(e0[3] - e0[1])
                e = _ext(fo)
                after["h"][g.cls].append(e[3] - e[1])
            if len(cos) < 600 and len(g.outer) >= 12:
                cos.append(G.shape_cosine(g.outer, fo, p.cos_points))

    def hcv(d):
        num, den = 0.0, 0
        for v in d.values():
            c = _cv(v)
            if not np.isnan(c):
                num += c * len(v)
                den += len(v)
        return num / den if den else float("nan")

    def pack(d):
        return {
            "baseline_sigma": float(np.mean(d["baseline"])) if d["baseline"] else float("nan"),
            "slant_std_deg": float(np.std(d["slant"])) if len(d["slant"]) >= 3 else float("nan"),
            "height_cv": hcv(d["h"]),
            "gap_cv": _cv(d["gaps"]),
        }

    b, a = pack(before), pack(after)
    extra_b, extra_a = _refine_metrics(glyphs, lay, finals, p)
    b.update(extra_b)
    a.update(extra_a)
    rows = []
    names = {
        "baseline_sigma": "ความเบี่ยงจากเส้นบรรทัด (σ/x-height)",
        "slant_std_deg": "ความแปรปรวนของมุมเอียง (SD, องศา)",
        "height_cv": "ความไม่สม่ำเสมอของความสูง (CV)",
        "gap_cv": "ความไม่สม่ำเสมอของช่องไฟ (CV)",
        "shape_dev": "ความต่างของรูปทรงตัวเดียวกัน (RMS/ขนาด)",
        "line_pitch_cv": "ความไม่สม่ำเสมอของระยะบรรทัด (CV)",
        "indent_dev": "ความไม่ตรงของแนวหน้าบรรทัด (SD/x-height)",
    }
    for k, label in names.items():
        if k not in b:
            continue
        bv, av = b[k], a[k]
        imp = (bv - av) / bv * 100 if bv and not np.isnan(bv) and not np.isnan(av) else float("nan")
        rows.append({"metric": k, "label": label, "before": bv, "after": av, "improve_pct": imp})
    n_base = sum(len(L.members) for L in lay.lines)
    return {
        "rows": rows,
        "identity_cos": float(np.mean(cos)) if cos else float("nan"),
        "identity_cos_min": float(np.min(cos)) if cos else float("nan"),
        "counts": {
            "lines": len(lay.lines),
            "base": n_base,
            "marks": sum(1 for g in glyphs if g.role == "mark"),
            "free": sum(1 for g in glyphs if g.role == "free"),
            "nontext": sum(1 for g in glyphs if g.kind == "nontext"),
            "noise": sum(1 for g in glyphs if g.kind == "noise"),
            "merged": sum(1 for g in glyphs if g.merged),
            "fallback_lines": sum(1 for L in lay.lines if L.fallback),
            "frozen_lines": sum(1 for L in lay.lines if L.frozen),
            "frozen_glyphs": sum(1 for g in glyphs if g.frozen),
        },
    }


# ------------------------------------------------------------------ OCR (ไม่บังคับ)
def tesseract_langs() -> list[str]:
    if not shutil.which("tesseract"):
        return []
    out = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True).stdout
    return [l.strip() for l in out.splitlines()[1:] if l.strip()]


def ocr_confidence(img: np.ndarray, lang: str) -> dict:
    """ค่า confidence เฉลี่ยของคำที่ tesseract อ่านได้ (ใช้เป็น proxy ของความอ่านง่าย)"""
    have = tesseract_langs()
    if not have:
        return {"ok": False, "msg": "ไม่พบ tesseract"}
    want = [l for l in lang.split("+") if l in have]
    if not want:
        return {"ok": False, "msg": f"ไม่มีภาษา {lang} ใน tesseract (มี: {', '.join(have)})"}
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        cv2.imwrite(f.name, img)
        out = subprocess.run(["tesseract", f.name, "stdout", "-l", "+".join(want), "--psm", "6", "tsv"],
                             capture_output=True, text=True).stdout
    confs, words = [], 0
    for row in out.splitlines()[1:]:
        cols = row.split("\t")
        if len(cols) >= 12 and cols[11].strip():
            c = float(cols[10])
            if c >= 0:
                confs.append(c)
                words += 1
    return {"ok": True, "lang": "+".join(want), "mean_conf": float(np.mean(confs)) if confs else 0.0,
            "words": words, "missing": [l for l in lang.split("+") if l not in have]}