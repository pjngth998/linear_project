"""
ตัวประสานงาน: รันขั้น A → E กับหน้าเดียว แล้วคืนผลทั้งหมด (ภาพ + ตัวชี้วัด + ภาพระหว่างทาง)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .components import extract_glyphs
from .config import Params
from .correction import compute_corrections
from .evaluate import evaluate, ocr_confidence
from .io_loader import PageInput
from .layout import analyze_layout
from .nontext import freeze_regions
from .preprocess import preprocess
from .render import compose_output, debug_overlay, final_contours, render_text_layer

SOURCE_LABEL = {
    "pdf_vector": "PDF แบบ vector (มักมาจาก GoodNotes)",
    "pdf_digital": "PDF ดิจิทัล",
    "pdf_scan": "PDF สแกน",
    "img_digital": "ภาพดิจิทัล",
    "img_photo": "ภาพถ่าย/สแกน",
}


@dataclass
class PageResult:
    name: str
    source_type: str
    info: dict
    original: np.ndarray        # ภาพเข้า (ยังไม่ทำอะไร)
    input_aligned: np.ndarray   # ภาพที่ใช้เทียบกับ output (= ต้นฉบับ เพราะ output อยู่ในกรอบภาพต้นฉบับแล้ว)
    output: np.ndarray
    debug: np.ndarray
    steps: dict
    metrics: dict
    correction_info: dict
    page_stats: dict
    timings: dict
    notes: list = field(default_factory=list)


def run_page(page: PageInput, p: Params) -> PageResult:
    t = {}
    notes = []
    t0 = time.perf_counter()

    pre = preprocess(page, p)
    t["A_preprocess"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    glyphs, stats = extract_glyphs(pre.binary, pre.color, p, pre.h_rough, pre.dark)
    lay = analyze_layout(glyphs, stats, p)
    nt_info = freeze_regions(glyphs, lay, stats.h, p)          # B11 ข้อความในกรอบ / สูตร → คงไว้
    t["B_layout"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    cinfo = compute_corrections(glyphs, lay, stats, p)
    cinfo["nontext"] = {**nt_info, "by_reason": {
        r: sum(1 for g in glyphs if g.kind == "nontext" and g.nt_reason == r)
        for r in ("large", "frame", "bracket", "line", "sparse", "image", "in_image", "heading")},
        **stats.protected}
    t["C_correction"] = time.perf_counter() - t2

    t3 = time.perf_counter()
    finals = {g.gid: final_contours(g, p) for g in glyphs if g.kind == "text"}
    # สีพื้นของผลลัพธ์: ภาพถ่าย/สแกนพื้นสว่าง สีพื้นเทา ๆ มาจากแสง ไม่ใช่สีกระดาษจริง → ใช้ขาว
    # (พื้นมืดไม่เปลี่ยนเป็นขาว เพราะหมึกสว่างจะมองไม่เห็น)
    if not pre.dark and (p.output_bg == "white" or (p.output_bg == "auto"
                                                    and page.source_type in ("img_photo", "pdf_scan"))):
        bg_out = (255, 255, 255)
    else:
        bg_out = pre.bg_color
    # วิเคราะห์และจัดในกรอบที่แก้หน้าเอียงแล้ว แต่ผลลัพธ์ประกอบกลับในกรอบภาพต้นฉบับ
    # → กระดาษ กริด ไฮไลต์ ไม่ถูกหมุน, เทียบกับต้นฉบับได้ตรงตำแหน่ง
    layer = render_text_layer(pre.color.shape, glyphs, p, finals, alpha=pre.alpha)
    if p.straighten_page and abs(pre.angle) > 1e-6:
        # หมุนทั้งหน้าให้ตรง (กระดาษ + ตัวหนังสือ): ประกอบในกรอบที่แก้เอียงแล้ว
        out = compose_output(pre.color, np.float64([[1, 0, 0], [0, 1, 0]]), layer, glyphs, p, bg_out)
    else:
        # คงกรอบภาพต้นฉบับ: หมุนชั้นตัวหนังสือกลับด้วย R⁻¹ แล้ววางบนกระดาษเดิม
        out = compose_output(page.image, pre.R, layer, glyphs, p, bg_out)
    # debug วาดบนภาพแบบพื้นสว่างเสมอ (พื้นมืดจะกลับสี) ให้กรอบสีต่าง ๆ มองเห็นชัด แล้วหมุนกลับกรอบต้นฉบับ
    dbg = debug_overlay(255 - pre.color if pre.dark else pre.color, glyphs, lay, p)
    if abs(pre.angle) > 1e-6 and not p.straighten_page:
        Hh, Ww = page.image.shape[:2]
        dbg = cv2.warpAffine(dbg, cv2.invertAffineTransform(pre.R), (Ww, Hh), borderValue=(255, 255, 255))
    t["D_render"] = time.perf_counter() - t3

    t4 = time.perf_counter()
    metrics = evaluate(glyphs, lay, finals, p) if lay.lines else {"rows": [], "counts": {}}
    if p.ocr:
        inv = (lambda im: 255 - im) if pre.dark else (lambda im: im)   # tesseract อ่านหมึกเข้มบนพื้นสว่างได้ดีกว่า
        metrics["ocr_before"] = ocr_confidence(inv(page.image), p.ocr_lang)
        metrics["ocr_after"] = ocr_confidence(inv(out), p.ocr_lang)
    t["E_evaluate"] = time.perf_counter() - t4
    t["total"] = time.perf_counter() - t0

    if page.source_type == "pdf_vector":
        notes.append("หน้านี้เป็น PDF แบบ vector: ตอนนี้ใช้วิธี render เป็นภาพคมชัดแล้วเข้า pipeline ปกติ (ไม่ blur) "
                     "ยังไม่ได้ดึงเส้นปากกาจาก vector path โดยตรง")
    if not lay.lines:
        notes.append("ไม่พบบรรทัดข้อความ ลองลด thresh_C หรือเช็กว่าภาพมีลายมือจริง")
    if stats.h < 12:
        notes.append(f"ตัวอักษรเล็กมาก (h ≈ {stats.h:.0f}px) ลองเพิ่ม dpi")

    return PageResult(
        name=page.name, source_type=page.source_type, info=page.info,
        original=page.image,
        input_aligned=pre.color if (p.straighten_page and abs(pre.angle) > 1e-6) else page.image,
        output=out, debug=dbg,
        steps=pre.steps, metrics=metrics, correction_info=cinfo,
        page_stats={"dark_background": bool(pre.dark), "bg_color": list(pre.bg_color),
                    "h": round(stats.h, 1), "stroke": round(stats.stroke, 2),
                    "deskew_deg": round(pre.angle, 2), "deskew_source": pre.steps.get("deskew_source", ""), "page_slope": round(lay.page_slope, 4),
                    "gap_threshold": None if np.isinf(lay.gap_threshold) else round(lay.gap_threshold, 1),
                    "gap_small": round(lay.gap_small, 1), "gap_large": round(lay.gap_large, 1)},
        timings={k: round(v, 2) for k, v in t.items()}, notes=notes,
    )


def side_by_side(a: np.ndarray, b: np.ndarray, labels=("INPUT", "OUTPUT")) -> np.ndarray:
    h = max(a.shape[0], b.shape[0])
    pad = lambda x: cv2.copyMakeBorder(x, 0, h - x.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    sep = np.full((h, 12, 3), 180, np.uint8)
    img = np.hstack([pad(a), sep, pad(b)])
    s = max(0.8, h / 900)
    for j, lab in enumerate(labels):
        x = 20 + j * (a.shape[1] + 12)
        cv2.putText(img, lab, (x, int(40 * s)), cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 220), max(2, int(2 * s)), cv2.LINE_AA)
    return img