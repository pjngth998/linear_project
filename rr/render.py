"""
D1–D4. แปลงพิกัด, ทำเส้นให้เรียบ, วาดกลับเป็นภาพ  +  ภาพ debug overlay
"""
from __future__ import annotations

import cv2
import numpy as np

from . import geometry as G
from .components import Glyph, PageStats
from .config import Params
from .layout import Layout

SHIFT = 4                         # sub-pixel bits ของ cv2.fillPoly (1/16 px)

CLS_COLOR = {                     # BGR สำหรับ debug
    "mid": (200, 120, 30),
    "asc": (30, 140, 240),
    "desc": (170, 50, 160),
    "tall": (40, 90, 140),
    "mark": (200, 40, 200),
    "punct": (90, 160, 90),
}

NT_COLOR = {                      # BGR ของ non-text แต่ละชนิด (debug)
    "large": (40, 40, 220),
    "frame": (30, 90, 240),
    "bracket": (130, 30, 110),
    "line": (20, 20, 150),
    "sparse": (120, 40, 220),
    "image": (160, 160, 0),
    "in_image": (160, 160, 0),
    "heading": (0, 140, 0),
}


def smooth_contour(P: np.ndarray, p: Params) -> np.ndarray:
    """D3: resample ตาม arc length → circulant moving average โดยคงจุดมุมแหลมไว้"""
    w = p.ma_window if p.ma_window % 2 else p.ma_window + 1
    Q = G.resample_closed(P, p.resample_step)
    if len(Q) < 4 * w:
        return Q
    corner = G.turning_angles(Q, k=max(2, w // 2 + 1)) > p.corner_deg
    corner = corner | np.roll(corner, 1) | np.roll(corner, -1)
    return G.moving_average_closed(Q, w, keep=corner)


def final_contours(g: Glyph, p: Params) -> list[np.ndarray]:
    out = []
    for c in g.all_contours():
        P = G.apply(g.T, c) if not G.is_identity(g.T) else c.astype(np.float64)
        if p.enable_smoothing and g.role in ("base", "mark"):
            P = smooth_contour(P, p)
        out.append(P)
    return out


class Layer:
    """ชั้นตัวหนังสือแบบมีความโปร่งใส (premultiplied): สี C (H,W,3) และความทึบ A (H,W)
    แยกจากพื้นหลัง เพื่อหมุนกลับเข้ากรอบภาพต้นฉบับได้ก่อนค่อยวางบนกระดาษ"""

    def __init__(self, H: int, W: int):
        self.C = np.zeros((H, W, 3), np.float32)
        self.A = np.zeros((H, W), np.float32)

    def blend(self, y0, y1, x0, x1, a: np.ndarray, color):
        """over-compositing: ของใหม่ทับของเดิม"""
        a3 = a[..., None]
        self.C[y0:y1, x0:x1] = self.C[y0:y1, x0:x1] * (1 - a3) + np.array(color, np.float32) * a3
        self.A[y0:y1, x0:x1] = self.A[y0:y1, x0:x1] * (1 - a) + a


def _composite(layer: Layer, contours: list[np.ndarray], color, scale: float):
    """วาด glyph เดียวลง mask ของตัวเอง (outer = 255, hole = 0, ขอบ anti-alias) แล้ว blend ลงชั้น"""
    Hc, Wc = layer.A.shape
    allp = np.vstack(contours) * scale
    x0 = int(max(np.floor(allp[:, 0].min()) - 2, 0))
    y0 = int(max(np.floor(allp[:, 1].min()) - 2, 0))
    x1 = int(min(np.ceil(allp[:, 0].max()) + 3, Wc))
    y1 = int(min(np.ceil(allp[:, 1].max()) + 3, Hc))
    if x1 <= x0 or y1 <= y0:
        return
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    off = np.array([x0, y0], float)
    f = float(1 << SHIFT)
    pts = lambda P: np.round((P * scale - off) * f).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(m, [pts(contours[0])], 255, cv2.LINE_AA, SHIFT)
    for hole in contours[1:]:
        cv2.fillPoly(m, [pts(hole)], 0, cv2.LINE_AA, SHIFT)
    layer.blend(y0, y1, x0, x1, m.astype(np.float32) / 255, color)


def _composite_warp(layer: Layer, g: Glyph, alpha_img: np.ndarray, color):
    """วาดแบบ warp: ยกพิกเซลหมึกจริงของตัวนี้ (ความเข้ม/ความหนา/ขอบนุ่มตามต้นฉบับ)
    แล้วแปลงด้วย affine T เดียวกับที่คำนวณไว้ → เส้นไม่หนาขึ้น ไม่เป็นก้อน รักษาลายมือเดิมที่สุด"""
    Hc, Wc = layer.A.shape
    pad = 3
    x0, y0 = max(int(g.x0) - pad, 0), max(int(g.y0) - pad, 0)
    x1, y1 = min(int(g.x1) + pad + 1, alpha_img.shape[1]), min(int(g.y1) + pad + 1, alpha_img.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    # mask ของตัวนี้เท่านั้น (ขยายเล็กน้อยให้ติดขอบนุ่ม) กันไม่ให้ยกพิกเซลของตัวข้าง ๆ มาด้วย
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    off = np.array([x0, y0], float)
    cv2.fillPoly(m, [np.round(g.outer - off).astype(np.int32)], 255)
    for hole in g.holes:
        cv2.fillPoly(m, [np.round(hole - off).astype(np.int32)], 0)
    m = cv2.dilate(m, np.ones((5, 5), np.uint8))
    a_src = alpha_img[y0:y1, x0:x1].astype(np.float32) * (m.astype(np.float32) / 255)

    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
    dc = G.apply(g.T, corners)
    dx0, dy0 = int(max(np.floor(dc[:, 0].min()) - 1, 0)), int(max(np.floor(dc[:, 1].min()) - 1, 0))
    dx1, dy1 = int(min(np.ceil(dc[:, 0].max()) + 2, Wc)), int(min(np.ceil(dc[:, 1].max()) + 2, Hc))
    if dx1 <= dx0 or dy1 <= dy0:
        return
    # พิกัด patch ต้นทาง → patch ปลายทาง:  M = Tr(−d0) · T · Tr(s0)
    M = G.T_translate(-dx0, -dy0) @ g.T @ G.T_translate(x0, y0)
    a = cv2.warpAffine(a_src, M[:2], (dx1 - dx0, dy1 - dy0), flags=cv2.INTER_LINEAR, borderValue=0)
    layer.blend(dy0, dy1, dx0, dx1, a, color)


def _glyph_mask(shape, glyphs, select) -> np.ndarray:
    """mask ของ glyph ที่เลือก (แต่ละชิ้นทำ mask ของตัวเองแล้ว OR รวม — ถ้าวาดลง mask เดียวกันตรง ๆ
    การเติมรูของกรอบ (ค่า 0) จะไปลบตัวอักษรที่อยู่ในกรอบซึ่งวาดไว้ก่อน)"""
    H, W = shape[:2]
    mask = np.zeros((H, W), np.uint8)
    for g in glyphs:
        if not select(g):
            continue
        x0, y0 = max(int(g.x0) - 1, 0), max(int(g.y0) - 1, 0)
        x1, y1 = min(int(g.x1) + 2, W), min(int(g.y1) + 2, H)
        if x1 <= x0 or y1 <= y0:
            continue
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        off = np.array([x0, y0])
        cv2.fillPoly(m, [np.round(g.outer - off).astype(np.int32)], 255)
        if g.nt_reason not in ("image", "in_image", "heading"):   # ภาพ/หัวเรื่อง: เอาทั้งก้อนรวมรู
            for hole in g.holes:
                cv2.fillPoly(m, [np.round(hole - off).astype(np.int32)], 0)
        np.maximum(mask[y0:y1, x0:x1], m, out=mask[y0:y1, x0:x1])
    return mask


def render_text_layer(shape, glyphs: list[Glyph], p: Params, finals: dict | None = None,
                      alpha: np.ndarray | None = None) -> Layer:
    """วาดตัวหนังสือที่จัดแล้วลงชั้นโปร่งใส (ในกรอบภาพที่แก้หน้าเอียงแล้ว)
    render_mode = "warp"    แปลงพิกเซลหมึกจริง (คงความหนา/ความนุ่มของเส้นตามต้นฉบับ)  ← ค่าเริ่มต้น
                  "contour" เติมสีจาก contour ที่ผ่าน moving average (เส้นเรียบ แต่หนาขึ้นได้)"""
    H, W = shape[:2]
    text = [g for g in glyphs if g.kind == "text" and not g.frozen]
    text.sort(key=lambda g: -g.area)
    if p.render_mode == "warp" and alpha is not None:
        layer = Layer(H, W)
        for g in text:
            _composite_warp(layer, g, alpha, g.color)
        return layer
    F = max(1, int(p.supersample))
    big = Layer(H * F, W * F)
    for g in text:
        cs = finals[g.gid] if finals and g.gid in finals else final_contours(g, p)
        _composite(big, cs, g.color, F)
    if F == 1:
        return big
    layer = Layer(H, W)
    layer.C = cv2.resize(big.C, (W, H), interpolation=cv2.INTER_AREA)
    layer.A = cv2.resize(big.A, (W, H), interpolation=cv2.INTER_AREA)
    return layer


def compose_output(original: np.ndarray, R: np.ndarray, layer: Layer, glyphs: list[Glyph],
                   p: Params, bg_color) -> np.ndarray:
    """ประกอบผลลัพธ์ "ในกรอบภาพต้นฉบับ" (ไม่หมุนหน้า):
    1) หมุนชั้นตัวหนังสือกลับด้วย R⁻¹ (R = เมทริกซ์ที่ใช้แก้หน้าเอียงตอนวิเคราะห์)
    2) พื้นหลัง: paper = ภาพต้นฉบับที่ลบตัวหนังสือที่ถูกย้ายออก (inpainting) → กริด/ไฮไลต์อยู่ตำแหน่งเดิมเป๊ะ
                 อื่น ๆ = สีเรียบ + วาง non-text / ตัวที่ freeze จากภาพต้นฉบับ
    3) out = พื้น·(1 − A) + C"""
    H, W = original.shape[:2]
    Rinv = cv2.invertAffineTransform(R.astype(np.float64))
    C = cv2.warpAffine(layer.C, Rinv, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
    A = cv2.warpAffine(layer.A, Rinv, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
    back = lambda m: cv2.warpAffine(m, Rinv, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)

    if p.output_bg == "paper":
        tm = back(_glyph_mask(layer.A.shape, glyphs, lambda g: g.kind == "text" and not g.frozen))
        bg = original.copy()
        if tm.any():
            bg = cv2.inpaint(original, cv2.dilate(tm, np.ones((5, 5), np.uint8)), 3, cv2.INPAINT_TELEA)
    else:
        bg = np.empty((H, W, 3), np.uint8)
        bg[:] = np.array(bg_color, np.uint8)
        fm = back(_glyph_mask(layer.A.shape, glyphs,
                              lambda g: g.kind == "nontext" or (g.kind == "text" and g.frozen)))
        if fm.any():
            fm = cv2.dilate(fm, np.ones((3, 3), np.uint8)) > 0
            bg[fm] = original[fm]
    out = bg.astype(np.float32) * (1 - A[..., None]) + C
    return np.clip(out, 0, 255).astype(np.uint8)


def debug_overlay(color_img: np.ndarray, glyphs: list[Glyph], lay: Layout, p: Params,
                  show_obb: bool = True) -> np.ndarray:
    """ภาพตรวจสอบ: baseline (เขียว), x-height (ฟ้า), กรอบตามกลุ่ม, เส้นโยง mark→เจ้าของ,
    non-text (แดง), free (เทา), ตัวติดกัน (กรอบหนา), Primary OBB (เหลือง)"""
    img = (0.45 * color_img + 0.55 * 255).astype(np.uint8)
    gl = {g.gid: g for g in glyphs}
    t = max(1, int(round(lay.H / 25)))
    for L in lay.lines:
        xs = np.array([L.x0, L.x1])
        yb = L.y_at(xs)
        cv2.line(img, (int(xs[0]), int(yb[0])), (int(xs[1]), int(yb[1])), (40, 170, 40), t + 1, cv2.LINE_AA)
        n = np.array([-L.a, 1.0]) / np.sqrt(1 + L.a ** 2)
        p0 = np.array([xs[0], yb[0]]) - n * L.xh
        p1 = np.array([xs[1], yb[1]]) - n * L.xh
        cv2.line(img, tuple(p0.astype(int)), tuple(p1.astype(int)), (200, 180, 40), t, cv2.LINE_AA)
        lab = f"L{L.lid}" + ("*" if L.fallback else "") + (" freeze" if L.frozen else "")
        cv2.putText(img, lab, (int(L.x0) - int(1.6 * lay.H), int(yb[0])), cv2.FONT_HERSHEY_SIMPLEX,
                    lay.H / 45, (40, 140, 40), t, cv2.LINE_AA)

    for g in glyphs:
        r = (int(g.x0), int(g.y0)), (int(g.x1), int(g.y1))
        if g.kind == "nontext":
            cv2.rectangle(img, *r, NT_COLOR.get(g.nt_reason, (40, 40, 220)), t + 1)
            cv2.putText(img, g.nt_reason, (int(g.x0) + 2, int(g.y0) - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.4, lay.H / 70), NT_COLOR.get(g.nt_reason, (40, 40, 220)), max(1, t - 1), cv2.LINE_AA)
        elif g.kind == "noise":
            continue
        elif g.frozen:
            cv2.rectangle(img, *r, (170, 160, 0), t)                  # ตัวที่ freeze (คงไว้ตามต้นฉบับ)
        elif g.role == "free":
            cv2.rectangle(img, *r, (150, 150, 150), t)
        elif g.role == "mark":
            cv2.rectangle(img, *r, CLS_COLOR["mark"], t)
            if g.owner >= 0:
                o = gl[g.owner]
                cv2.line(img, (int(g.cx), int(g.cy)), (int(o.cx), int(o.cy)), CLS_COLOR["mark"], 1, cv2.LINE_AA)
        elif g.role == "base":
            cv2.rectangle(img, *r, CLS_COLOR.get(g.cls, (0, 0, 0)), t * (3 if g.merged else 1))
            if show_obb and g.line >= 0:
                L = lay.lines[g.line]
                P = g.outer
                d = L.sdist(P[:, 0], P[:, 1])
                body = P[(d <= 0.15 * L.xh) & (d >= -1.15 * L.xh)]      # ส่วนหลัก (ไม่รวมหางขึ้น-ลง)
                if len(body) >= 5:
                    box, _, _ = G.oriented_bbox(body, p.obb_min_ratio)
                    cv2.polylines(img, [box.astype(np.int32).reshape(-1, 1, 2)], True, (0, 200, 230), 1, cv2.LINE_AA)
    return img


def legend_text() -> str:
    return ("เขียว = baseline, ฟ้า = เส้น x-height, กรอบน้ำเงิน = ตัวกลาง, ส้ม = หางขึ้น, ม่วง = หางลง, "
            "น้ำตาล = เต็มสูง, เขียวอ่อน = เครื่องหมายยืนเดี่ยว (= - .), ชมพู = สระ/วรรณยุกต์ (เส้นโยงไปตัวเจ้าของ), เหลือง = Primary OBB, "
            "non-text มีป้ายบอกชนิด: แดง = large, ส้มแดง = frame, ม่วงเข้ม = bracket, แดงเข้ม = line, ชมพูเข้ม = sparse, "
            "ฟ้าหม่น = image / in_image (ภาพที่แปะมา), เขียวเข้ม = heading (หัวเรื่องตัวใหญ่) · "
            "ฟ้าอมเขียว = ตัวที่ freeze (ในกรอบ/สูตร คงไว้ตามต้นฉบับ), เทา = ไม่มีเจ้าของ (ไม่แก้), กรอบหนา = ตัวติดกัน, "
            "L*n = บรรทัด fallback, 'freeze' = บรรทัดที่ไม่แก้")