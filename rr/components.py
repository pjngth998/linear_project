"""
B1–B3. ดึง contour เป็นกลุ่มตัวอักษร (glyph) และแยก text / non-text / noise

แต่ละ glyph = contour ด้านนอก 1 เส้น + contour รู (hole) ข้างใน
(ใช้ cv2.RETR_CCOMP ซึ่งให้ hierarchy 2 ระดับ: outer กับ hole)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import Params
from .nontext import absorb_protected, classify_component


@dataclass
class Glyph:
    gid: int
    outer: np.ndarray                      # (N, 2) float  จุดขอบนอก เรียงต่อเนื่อง
    holes: list                            # [(M, 2) float, ...]
    x0: float
    y0: float
    x1: float
    y1: float
    area: float
    cx: float
    cy: float
    color: tuple = (0, 0, 0)               # BGR สีหมึก
    kind: str = "text"                     # text | nontext | noise
    nt_reason: str = ""                    # เหตุผลที่เป็น non-text/noise: large | frame | bracket | line | sparse | tiny | border
    frozen: bool = False                   # True = คงรูปและตำแหน่งเดิม (อยู่ในกรอบ / ภาพวาด / สูตร)
    role: str = ""                         # base | mark | free
    line: int = -1
    owner: int = -1                        # glyph เจ้าของ (สำหรับ mark)
    zone: str = ""                         # upper | middle | lower
    cls: str = ""                          # mid | asc | desc | tall | mark
    merged: bool = False
    word: int = -1
    disp: tuple | None = None              # (จุดควบคุมบนตัวต้นฉบับ, เวกเตอร์เลื่อน) จากการบิดเข้าหาต้นแบบ
    shape_cluster: int = -1                # กลุ่มรูปทรง (ตัวที่หน้าตาเหมือนกัน) จาก refine.shape_consistency
    slant_deg: float = float("nan")        # มุมเอียงของตัวนี้ (ไว้ประเมินผล)
    T: np.ndarray = field(default_factory=lambda: np.eye(3))
    notes: list = field(default_factory=list)

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    def all_contours(self):
        return [self.outer] + list(self.holes)


@dataclass
class PageStats:
    h: float            # ความสูงอ้างอิงของตัวอักษร
    stroke: float       # ความหนาเส้นมัธยฐาน
    protected: dict = field(default_factory=dict)


def _ink_color(color_img: np.ndarray, g: Glyph, dark: bool = False) -> tuple:
    x0, y0 = int(max(g.x0, 0)), int(max(g.y0, 0))
    x1, y1 = int(min(g.x1 + 1, color_img.shape[1])), int(min(g.y1 + 1, color_img.shape[0]))
    if x1 <= x0 or y1 <= y0:
        return (0, 0, 0)
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    off = np.array([x0, y0])
    cv2.fillPoly(m, [np.round(g.outer - off).astype(np.int32)], 255)
    for hole in g.holes:
        cv2.fillPoly(m, [np.round(hole - off).astype(np.int32)], 0)
    m = cv2.erode(m, np.ones((3, 3), np.uint8)) if m.sum() > 255 * 30 else m
    px = color_img[y0:y1, x0:x1][m > 0]
    if len(px) == 0:
        return (0, 0, 0)
    # ใช้ครึ่งที่ต่างจากพื้นมากกว่า (แกนกลางเส้น) จะได้สีหมึกจริง ไม่ปนขอบ
    # กระดาษสว่าง → ครึ่งที่เข้มกว่า, พื้นมืด → ครึ่งที่สว่างกว่า
    lum = px.astype(np.int32).sum(axis=1)
    core = px[lum >= np.median(lum)] if dark else px[lum <= np.median(lum)]
    return tuple(int(v) for v in np.median(core, axis=0))


def extract_glyphs(binary: np.ndarray, color_img: np.ndarray, p: Params, h_hint: float, dark: bool = False):
    contours, hier = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    glyphs: list[Glyph] = []
    if hier is None:
        return glyphs, PageStats(h=h_hint, stroke=2.0)
    hier = hier[0]
    for i, c in enumerate(contours):
        if hier[i][3] != -1:            # เป็นรู ไม่ใช่ขอบนอก
            continue
        outer = c[:, 0, :].astype(np.float64)
        holes = []
        ch = hier[i][2]
        while ch != -1:
            hc = contours[ch][:, 0, :].astype(np.float64)
            if len(hc) >= 3:
                holes.append(hc)
            ch = hier[ch][0]
        area = cv2.contourArea(c)
        area -= sum(cv2.contourArea(hh.astype(np.float32)) for hh in holes)
        x, y, w, h = cv2.boundingRect(c)
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        else:
            cx, cy = x + w / 2, y + h / 2
        glyphs.append(Glyph(gid=len(glyphs), outer=outer, holes=holes,
                            x0=float(x), y0=float(y), x1=float(x + w), y1=float(y + h),
                            area=max(float(area), float(len(outer)) * 0.5), cx=float(cx), cy=float(cy)))

    # ---- B2 หน่วยอ้างอิงของหน้า
    hs = np.array([g.h for g in glyphs if g.area >= 8 and g.h >= 4])
    if len(hs):
        h0 = float(np.median(hs))
        sel = hs[(hs > 0.4 * h0) & (hs < 2.5 * h0)]
        H = float(np.median(sel)) if len(sel) else h0
    else:
        H = h_hint
    strokes = []
    for g in glyphs:
        if 0.5 * H <= g.h <= 2 * H:
            per = sum(cv2.arcLength(c.astype(np.float32), True) for c in g.all_contours())
            if per > 0:
                strokes.append(2 * g.area / per)
    stats = PageStats(h=H, stroke=float(np.median(strokes)) if strokes else 2.0)

    # ---- B3 text / non-text / noise (กฎทั้งหมดอยู่ใน nontext.py)
    Hi, Wi = binary.shape
    for g in glyphs:
        g.kind, g.nt_reason = classify_component(g, H, Wi, Hi, p)
    stats.protected = absorb_protected(glyphs, H, p, color_img)        # ภาพที่แปะมา + หัวเรื่องตัวใหญ่

    for g in glyphs:
        if g.kind != "noise":
            g.color = _ink_color(color_img, g, dark)
    return glyphs, stats