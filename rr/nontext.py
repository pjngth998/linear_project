"""
จับสิ่งที่ไม่ใช่ตัวอักษร (non-text) — 2 ระดับ

ระดับชิ้น (B3, เรียกจาก components.py) ดูจากรูปร่างของ connected component แต่ละชิ้น
    large    ใหญ่เกินตัวอักษร (สูง > 3h หรือกว้าง > 8h)
    frame    มีรูขนาดใหญ่ = กรอบสี่เหลี่ยม / วงกลมล้อมข้อความ
    bracket  สูงแต่แคบ = วงเล็บใหญ่ ปีกกา เครื่องหมาย ∫ เส้นตั้ง
    line     เส้นตรงยาว (λ1/λ2 ของ covariance สูงมาก) = ลูกศร เส้นใต้ เส้นแบ่ง
    sparse   ชิ้นใหญ่ที่หมึกน้อยเมื่อเทียบกับกรอบ = เส้นโค้ง กราฟ ภาพวาด

ระดับพื้นที่ (B11, เรียกจาก pipeline หลังจัดบรรทัด) ตัวอักษรที่ "ไม่ควรถูกจัดเป็นข้อความ"
จะถูก freeze (คงตำแหน่งและรูปเดิม วาดกลับด้วยพิกเซลต้นฉบับ)
    ในกรอบ     ศูนย์กลางอยู่ในรูของกรอบ (frame)
    ในภาพวาด   ศูนย์กลางอยู่ในขอบเขตของชิ้น large / sparse (เช่น ป้ายบนกราฟ)
    เศษส่วน    มีเส้นแนวนอนสั้นที่มีตัวอักษรทั้งด้านบนและด้านล่าง → เส้น + ตัวบน + ตัวล่าง
    ตัวซ้อน    บรรทัดที่มีตัวฐานซ้อนกันแนวตั้งหลายคู่ (โครงสร้างสูตร)
แล้วถ้าตัวในบรรทัดถูก freeze เกิน freeze_line_frac → freeze ทั้งบรรทัด (กันบรรทัดครึ่งจัดครึ่งไม่จัด)

โมดูลนี้ไม่ import components / layout เพื่อเลี่ยง import วน — ใช้ attribute ของ object ตรง ๆ
"""
from __future__ import annotations

import cv2
import numpy as np

from . import geometry as G
from .config import Params


# ------------------------------------------------------------------ ระดับชิ้น
def big_holes(g, H: float, p: Params) -> list:
    """รูที่ใหญ่พอจะเป็น "ภายในกรอบ" → [(x0, y0, x1, y1), ...]"""
    out = []
    for hole in g.holes:
        hx0, hy0 = hole.min(axis=0)
        hx1, hy1 = hole.max(axis=0)
        if hx1 - hx0 > p.frame_hole_w_h * H and hy1 - hy0 > p.frame_hole_h_h * H:
            out.append((float(hx0), float(hy0), float(hx1), float(hy1)))
    return out


def classify_component(g, H: float, img_w: int, img_h: int, p: Params) -> tuple[str, str]:
    """คืน (kind, เหตุผล)  kind = text | nontext | noise"""
    if g.area < p.noise_area_h2 * H * H:
        return "noise", "tiny"
    if g.x0 <= 1 or g.y0 <= 1 or g.x1 >= img_w - 1 or g.y1 >= img_h - 1:
        return "noise", "border"            # ติดขอบภาพ = ขอบกระดาษ/เงา/ขอบจากการหมุน

    box_area = max(g.w * g.h, 1.0)
    # ภาพที่แปะมา (รูปถ่าย/ภาพหน้าจอ/ตาราง): ชิ้นใหญ่ที่หมึกทึบเต็มพื้นที่ (กรอบเส้นบาง ๆ ทึบไม่ถึง)
    if g.w > p.image_min_w_h * H and g.h > p.image_min_h_h * H and g.area / box_area >= p.image_fill:
        return "nontext", "image"

    # กรอบล้อม: รูใหญ่และกินพื้นที่ส่วนมากของกรอบ (ตัว อ ข ฯลฯ มีรูเล็ก ไม่เข้าเงื่อนไข)
    for (hx0, hy0, hx1, hy1) in big_holes(g, H, p):
        if (hx1 - hx0) * (hy1 - hy0) >= 0.35 * box_area:
            return "nontext", "frame"

    if g.h > p.nontext_height_h * H or g.w > p.nontext_width_h * H:
        return "nontext", "large"

    if g.h > p.bracket_h_h * H and g.w < p.bracket_w_h * H:
        return "nontext", "bracket"

    if len(g.outer) >= 10:
        _, lam, _ = G.principal_axes(g.outer)
        ratio = lam[0] / max(lam[1], 1e-9)
        length = 4 * np.sqrt(lam[0])         # ≈ ความยาวแกนหลัก
        if ratio > p.nontext_line_ratio and length > 2.5 * H:
            return "nontext", "line"

    if (g.h > 2 * H or g.w > 3 * H) and g.area / box_area < p.sparse_fill:
        return "nontext", "sparse"

    return "text", ""


# ------------------------------------------------------------------ ระดับพื้นที่
def _overlap(a0, a1, b0, b1) -> float:
    return min(a1, b1) - max(a0, b0)


def detect_image_regions(color_img: np.ndarray, H: float, p: Params) -> list:
    """หาภาพที่แปะมา (รูปถ่าย/ภาพหน้าจอ/ตาราง) จากภาพสี: บริเวณสี่เหลี่ยมที่ "ต่างจากพื้นกระดาษ" เกือบเต็มพื้นที่
    ลายมือ/กรอบที่วาดเอง ต่างจากพื้นแค่ตามเส้น → สัดส่วนต่ำ ไม่โดน
    คืน [(x0, y0, x1, y1), ...]"""
    Hh, Ww = color_img.shape[:2]
    s = min(1.0, 1600 / max(Hh, Ww))
    small = cv2.resize(color_img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else color_img
    bg = np.median(small.reshape(-1, 3)[::7], axis=0)
    diff = (np.abs(small.astype(np.int16) - bg.astype(np.int16)).max(axis=2) > 40).astype(np.uint8)
    k = max(3, int(0.6 * H * s))
    closed = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if w < p.image_min_w_h * H * s or h < p.image_min_h_h * H * s:
            continue
        box = float(w * h)
        fill_closed = a / box                                  # รูปร่างเป็นสี่เหลี่ยมทึบไหม
        fill_raw = float(diff[y:y + h, x:x + w].mean())        # เนื้อในต่างจากพื้นจริงแค่ไหน
        if fill_closed >= 0.85 and fill_raw >= p.image_fill:
            out.append((x / s, y / s, (x + w) / s, (y + h) / s))
    return out


def absorb_protected(glyphs: list, H: float, p: Params, color_img: np.ndarray | None = None) -> dict:
    """เรียกหลังจัดชนิดรายชิ้น: กันพื้นที่ที่ "ไม่ใช่ลายมือ" ออกจากการจัดทั้งหมด
    1) ภาพที่แปะมา: ทุกชิ้นที่อยู่ในกรอบของภาพ (ตัวหนังสือในตาราง/ภาพหน้าจอ) → nontext "in_image"
    2) หัวเรื่องตัวใหญ่ (มักเป็นตัวพิมพ์): แถวของชิ้นที่สูงกว่าตัวหนังสือปกติมาก ฐานตรงกัน ≥ 3 ชิ้น
       → ทั้งแถว (รวมตัวเล็กที่อยู่ในแถวนั้น) เป็น nontext "heading" คงพิกเซลเดิมทุกจุด
    ทำแบบนี้แทนการ freeze ทีหลัง เพื่อไม่ให้ชิ้นพวกนี้ไปปนตอนหาบรรทัด/baseline ของลายมือรอบ ๆ"""
    info = {"image_regions": 0, "in_image": 0, "heading_rows": 0, "in_heading": 0}
    regions = []

    if p.protect_images:
        for g in glyphs:
            if g.kind == "nontext" and g.nt_reason == "image":
                regions.append(("in_image", g.x0, g.y0, g.x1, g.y1, g.gid))
        if color_img is not None:
            for (x0, y0, x1, y1) in detect_image_regions(color_img, H, p):
                regions.append(("in_image", x0 - 2, y0 - 2, x1 + 2, y1 + 2, -1))
        info["image_regions"] = len(regions)
    image_boxes = [r[1:5] for r in regions]

    if p.protect_headings:
        inside_img = lambda g: any(b[0] <= g.cx <= b[2] and b[1] <= g.cy <= b[3] for b in image_boxes)
        big = [g for g in glyphs if g.kind in ("text", "nontext") and g.nt_reason in ("", "large", "bracket", "line", "sparse")
               and p.heading_min_h * H <= g.h <= 10 * H and g.w < 8 * H and not inside_img(g)]
        big.sort(key=lambda g: g.x0)
        used = set()
        for a in big:
            if a.gid in used:
                continue
            row = [a]
            for b in big:
                if b.gid in used or b is a or b.x0 < row[-1].x0:
                    continue
                ref = float(np.median([r.y1 for r in row]))
                hh = float(np.median([r.h for r in row]))
                if abs(b.y1 - ref) <= 0.18 * hh and b.x0 - max(r.x1 for r in row) <= 1.2 * hh:
                    row.append(b)
            if len(row) >= 3:
                bottoms = np.array([r.y1 for r in row])
                hh = float(np.median([r.h for r in row]))
                hs = np.array([r.h for r in row])
                span = max(r.x1 for r in row) - min(r.x0 for r in row)
                # ฐานตรงกันมาก + ความสูงใกล้กัน + แถวยาวพอ (ตัวพิมพ์/หัวเรื่อง) — กันวงเล็บ/ลูกศรลายมือที่บังเอิญเรียงกัน
                if np.std(bottoms) <= 0.08 * hh and np.std(hs) <= 0.35 * hh and span >= 3 * hh and len(row) >= 4:
                    used.update(r.gid for r in row)
                    m = 0.15 * hh
                    regions.append(("heading", min(r.x0 for r in row) - m, min(r.y0 for r in row) - 0.6 * hh,
                                    max(r.x1 for r in row) + m, max(r.y1 for r in row) + 0.4 * hh, -1))
                    info["heading_rows"] += 1

    for (why, x0, y0, x1, y1, owner) in regions:
        for g in glyphs:
            if g.gid == owner or g.kind == "noise":
                continue
            if x0 <= g.cx <= x1 and y0 <= g.cy <= y1:
                # หัวเรื่อง: ไม่ทับชนิดของกรอบ/ภาพที่จับได้แล้ว (เปลี่ยนเฉพาะตัวอักษรและชิ้นตัวใหญ่)
                if g.kind == "text" or (why == "heading" and g.nt_reason in ("large", "bracket", "line", "sparse")) \
                        or (why == "in_image" and g.nt_reason != "image"):
                    g.kind, g.nt_reason = "nontext", why
                    info["in_" + ("image" if why == "in_image" else "heading")] += 1
    return info


def _fraction_groups(texts: list, H: float) -> list[list]:
    """หาเส้นเศษส่วน: แนวนอน บาง สั้น และมีตัวอักษรอยู่ทั้งเหนือและใต้ (ซ้อนในช่วงแกน x ของเส้น)
    คืนกลุ่ม [เส้น, ตัวบน..., ตัวล่าง...]
    เครื่องหมาย - และ = ไม่โดน เพราะไม่มีตัวอักษรครอบทั้งบนและล่าง"""
    groups = []
    for bar in texts:
        if not (bar.w > 0.7 * H and bar.h < 0.25 * H and bar.w > 4 * max(bar.h, 1.0)):
            continue
        above, below = [], []
        for o in texts:
            if o is bar or o.w <= 0:
                continue
            if _overlap(o.x0, o.x1, bar.x0, bar.x1) < 0.6 * o.w:
                continue
            if o.y1 <= bar.cy and bar.y0 - o.y1 < 0.8 * H:
                above.append(o)
            elif o.y0 >= bar.cy and o.y0 - bar.y1 < 0.8 * H:
                below.append(o)
        if above and below:
            groups.append([bar] + above + below)
    return groups


def _stacked_pairs(members: list, xh: float) -> int:
    """จำนวนคู่ตัวฐานในบรรทัดเดียวกันที่ซ้อนแนวตั้ง (ช่วง x ทับกันเกินครึ่ง และตัวหนึ่งอยู่เหนืออีกตัว)
    ไม่นับตัวบาง ๆ (เช่น สองขีดของ =)"""
    ms = [g for g in members if g.h >= 0.3 * xh]
    n = 0
    for a in range(len(ms)):
        for b in range(a + 1, len(ms)):
            g1, g2 = ms[a], ms[b]
            if _overlap(g1.x0, g1.x1, g2.x0, g2.x1) <= 0.5 * min(g1.w, g2.w):
                continue
            if g1.y1 <= g2.y0 + 0.1 * xh or g2.y1 <= g1.y0 + 0.1 * xh:
                n += 1
    return n


def freeze_regions(glyphs: list, lay, H: float, p: Params) -> dict:
    """ตั้ง g.frozen / L.frozen แล้วคืนสรุปจำนวน"""
    gl = {g.gid: g for g in glyphs}
    texts = [g for g in glyphs if g.kind == "text"]
    stats = {"regions": 0, "in_region": 0, "fraction_groups": 0, "math_lines": 0, "frozen_lines": 0}

    # 1) ข้อความในกรอบ / ในภาพวาด
    if p.freeze_regions:
        regions = []
        for g in glyphs:
            if g.kind != "nontext":
                continue
            regions.extend(big_holes(g, H, p))
            if g.nt_reason in ("large", "sparse") and g.w > 1.5 * H and g.h > 1.5 * H:
                m = 0.2 * H
                regions.append((g.x0 - m, g.y0 - m, g.x1 + m, g.y1 + m))
        stats["regions"] = len(regions)
        for g in texts:
            if any(r[0] < g.cx < r[2] and r[1] < g.cy < r[3] for r in regions):
                g.frozen = True
                g.notes.append("freeze:region")
                stats["in_region"] += 1

    # 2) เศษส่วน
    if p.freeze_math:
        for grp in _fraction_groups(texts, H):
            stats["fraction_groups"] += 1
            for g in grp:
                g.frozen = True
                g.notes.append("freeze:fraction")

    # 3) ต่อบรรทัด: ตัวซ้อน หรือ freeze เกินสัดส่วน → ทั้งบรรทัด, ไม่ถึง → ปลด freeze (จัดตามปกติ)
    for L in lay.lines:
        mem = [gl[i] for i in L.members]
        if not mem:
            continue
        math = p.freeze_math and _stacked_pairs(mem, L.xh) >= p.math_stack_min
        frac = sum(1 for g in mem if g.frozen) / len(mem)
        # บรรทัดที่เรียบร้อยอยู่แล้ว (ตัวพิมพ์ / ลายมือที่ตรงมาก): ฐานแทบไม่เบี่ยง และความสูงสม่ำเสมอ → ไม่แตะ
        neat = False
        if p.freeze_neat_lines:
            base = [g for g in mem if g.cls in ("mid", "asc")]
            mids = [g.h for g in mem if g.cls == "mid"]
            if len(base) >= 5 and len(mids) >= 3:
                r = np.array([g.y1 - float(L.y_at(g.cx)) for g in base]) / max(L.xh, 1.0)
                hcv = float(np.std(mids) / max(np.mean(mids), 1e-6))
                neat = float(np.std(r)) <= p.neat_resid_xh and hcv <= p.neat_height_cv
            if neat:
                stats["neat_lines"] = stats.get("neat_lines", 0) + 1
        if math or neat or frac >= p.freeze_line_frac:
            L.frozen = True
            stats["frozen_lines"] += 1
            stats["math_lines"] += int(math)
            for i in L.members + L.marks:
                gl[i].frozen = True
        else:
            L.frozen = False
            for g in mem:
                # ตัวที่อยู่ในกรอบ/ภาพวาดจริง ๆ ยังคงไว้ (เป็นสิ่งกีดขวางตอนจัดช่องไฟ)
                # ส่วนที่ freeze เพราะเหตุอื่นปลดออกให้จัดตามปกติ
                if "freeze:region" not in g.notes:
                    g.frozen = False

    # สระ/วรรณยุกต์ตามเจ้าของเสมอ
    for g in glyphs:
        if g.role == "mark" and g.owner >= 0:
            g.frozen = gl[g.owner].frozen
    stats["frozen_glyphs"] = sum(1 for g in glyphs if g.frozen)
    return stats