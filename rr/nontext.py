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


def _is_bar(g, H: float) -> bool:
    """เส้นเศษส่วน: บางและแนวนอน
    - เส้นตรงบางชัดเจน (สูงไม่ถึง 0.45h และกว้างกว่าสูง 1.4 เท่า) หรือ
    - เส้นที่ลากหนา/เอียงเล็กน้อยแบบลายมือ: covariance บอกว่าเป็นเส้น (λ₁/λ₂ สูง) และแกนหลักเกือบแนวนอน"""
    if g.w <= 0.4 * H or g.h > 0.8 * H:
        return False
    if g.h < 0.45 * H and g.w > 1.4 * max(g.h, 1.0):
        return True
    if len(g.outer) >= 8 and g.w > 0.8 * H and g.w > 1.6 * g.h:
        _, lam, E = G.principal_axes(g.outer)
        ratio = lam[0] / max(lam[1], 1e-9)
        ang = abs(np.degrees(np.arctan2(E[1, 0], E[0, 0])))
        ang = min(ang, 180 - ang)
        return ratio > 5 and ang < 25
    return False


def _fraction_groups(texts: list, H: float, bar_pool: list | None = None) -> list[list]:
    """หาเส้นเศษส่วน: แนวนอน บาง สั้น และมีตัวอักษรอยู่ทั้งเหนือและใต้ (ซ้อนในช่วงแกน x ของเส้น)
    คืนกลุ่ม [เส้น, ตัวบน..., ตัวล่าง...]
    เครื่องหมาย - และ = ไม่โดน เพราะไม่มีตัวอักษรครอบทั้งบนและล่าง"""
    groups = []
    for bar in (bar_pool if bar_pool is not None else texts):
        if not _is_bar(bar, H):
            continue
        above, below = [], []
        for o in texts:
            if o is bar or o.w <= 0 or o.kind != "text":
                continue
            # ตัวบน/ล่างของเศษส่วน: จุดกึ่งกลางอยู่ในช่วงของเส้น (เผื่อขอบเล็กน้อย) หรือทับกันเกินครึ่ง
            in_span = bar.x0 - 0.25 * H <= o.cx <= bar.x1 + 0.25 * H
            if not in_span and _overlap(o.x0, o.x1, bar.x0, bar.x1) < 0.5 * o.w:
                continue
            if o.h > 2.5 * H:
                continue
            if o.y1 <= bar.cy + 0.15 * H and bar.y0 - o.y1 < 1.0 * H and o.cy < bar.cy:
                above.append(o)
            elif o.y0 >= bar.cy - 0.15 * H and o.y0 - bar.y1 < 1.3 * H and o.cy > bar.cy:
                below.append(o)
        if above and below:
            # ขยายตัวเศษ/ตัวส่วนให้ครบ: ชิ้นที่อยู่แถวเดียวกันและต่อกันในแนวนอน (วงเล็บ, ตัวยก, ตัวห้อย)
            # เช่น (v₀ sinα)² ที่กว้างกว่าเส้นเศษส่วน ไม่อย่างนั้นส่วนที่ล้นจะถูกจัดแยกแล้วไปทับเส้น
            for part, side in ((above, -1), (below, 1)):
                grew = True
                while grew:
                    grew = False
                    px0 = min(g.x0 for g in part); px1 = max(g.x1 for g in part)
                    py0 = min(g.y0 for g in part); py1 = max(g.y1 for g in part)
                    for o in texts:
                        if o is bar or o in above or o in below or o.h > 2.5 * H or o.kind != "text":
                            continue
                        if (o.cy - bar.cy) * side <= 0:          # ต้องอยู่ฝั่งเดียวกับกลุ่มนี้
                            continue
                        gap = max(px0 - o.x1, o.x0 - px1)
                        vov = _overlap(o.y0, o.y1, py0 - 0.3 * H, py1 + 0.3 * H)
                        if gap <= 0.45 * H and vov >= 0.4 * o.h:
                            part.append(o)
                            grew = True
            groups.append([bar] + above + below)
    return groups


def bind_scripts(glyphs: list, lay, H: float, p: Params) -> int:
    """ตัวยก/ตัวห้อย (t², v₀, x₁): ชิ้นเล็กที่อยู่ติดขวาของตัวก่อนหน้า และสูง/ต่ำกว่าแนวตัวปกติ
    → เป็น mark ของตัวทางซ้าย เคลื่อนไปด้วยกัน (ไม่ถูกยกลงเส้นบรรทัดจนกลายเป็นตัวปกติ เช่น v₀ → vo)"""
    gl = {g.gid: g for g in glyphs}
    n = 0
    for L in lay.lines:
        mem = [gl[i] for i in L.members]
        mids = [g.h for g in mem if g.cls == "mid"]
        if len(mids) < 3:
            continue
        hm = float(np.median(mids))
        mem.sort(key=lambda g: g.x0)
        for k in range(1, len(mem)):
            g, left = mem[k], mem[k - 1]
            if g.cls in ("fraction", "enclosed") or left.cls in ("fraction",) or g.h > 0.75 * hm:
                continue
            if g.x0 - left.x1 > 0.35 * H or g.x0 < left.cx:
                continue
            dtop = float(L.sdist(g.cx, g.y0)) + L.xh       # 0 = ขอบบนแนว x-height
            dbot = float(L.sdist(g.cx, g.y1))               # 0 = baseline
            sub = dbot > 0.25 * L.xh and dtop > 0.15 * L.xh   # ห้อยต่ำกว่าเส้น
            sup = dtop < -0.25 * L.xh and dbot < -0.25 * L.xh  # ลอยเหนือแนวตัว
            if not (sub or sup):
                continue
            L.members.remove(g.gid)
            L.marks.append(g.gid)
            g.role, g.cls, g.owner, g.zone = "mark", "mark", left.gid, "middle"
            n += 1
    return n


def _merge_math_groups(groups: list, texts: list, H: float) -> list:
    """รวมเศษส่วนที่ซ้อนกัน/อยู่ติดกันเป็น "นิพจน์" เดียว แล้วดูดสิ่งที่อยู่ในแถวเดียวกันของนิพจน์เข้ามา
    เช่น  dr/ds = (dr/dt)/(ds/dt) = v/|v| = T  → ก้อนเดียว เคลื่อนแบบแข็งทั้งก้อน
    - เศษส่วนซ้อน (เส้นของตัวในอยู่ในตัวเศษของตัวนอก) จะมีชิ้นร่วมกัน → union
    - ชิ้นที่อยู่ระหว่าง/ติดกับนิพจน์ (=, +, ตัวแปร) ในแนวตั้งของนิพจน์ → ดูดเข้า
    เส้นที่ยาวที่สุดเป็นตัวหลัก (ตัวแรกของกลุ่ม)"""
    # 1) union กลุ่มที่มีชิ้นร่วมกันหรือกรอบทับกัน
    merged = []
    for grp in groups:
        gset = {g.gid: g for g in grp}
        for m in merged:
            mx0 = min(g.x0 for g in m.values()); mx1 = max(g.x1 for g in m.values())
            my0 = min(g.y0 for g in m.values()); my1 = max(g.y1 for g in m.values())
            gx0 = min(g.x0 for g in grp); gx1 = max(g.x1 for g in grp)
            gy0 = min(g.y0 for g in grp); gy1 = max(g.y1 for g in grp)
            if set(m) & set(gset) or (_overlap(mx0, mx1, gx0, gx1) > 0 and _overlap(my0, my1, gy0, gy1) > 0):
                m.update(gset)
                break
        else:
            merged.append(gset)
    # 2) ดูดชิ้นในแถวเดียวกันที่อยู่ติดกัน (ช่องไม่เกิน ~1.2h) และศูนย์กลางอยู่ในช่วงสูงของนิพจน์
    #    และเชื่อมนิพจน์ที่อยู่ติดกันให้เป็นก้อนเดียว
    changed = True
    while changed:
        changed = False
        for m in merged:
            if not m:
                continue
            x0 = min(g.x0 for g in m.values()); x1 = max(g.x1 for g in m.values())
            y0 = min(g.y0 for g in m.values()); y1 = max(g.y1 for g in m.values())
            for o in texts:
                if o.gid in m or o.h > 2.5 * H:
                    continue
                gap = max(x0 - o.x1, o.x0 - x1)
                if gap <= 1.8 * H and y0 - 0.2 * H <= o.cy <= y1 + 0.2 * H:
                    m[o.gid] = o
                    changed = True
            for m2 in merged:
                if m2 is m or not m2:
                    continue
                a0 = min(g.x0 for g in m2.values()); a1 = max(g.x1 for g in m2.values())
                b0 = min(g.y0 for g in m2.values()); b1 = max(g.y1 for g in m2.values())
                if max(x0 - a1, a0 - x1) <= 1.8 * H and _overlap(y0, y1, b0, b1) > 0.3 * min(y1 - y0, b1 - b0):
                    m.update(m2)
                    m2.clear()
                    changed = True
            if changed:
                break
    out = []
    for m in merged:
        if not m:
            continue
        items = list(m.values())
        bars = [g for g in items if _is_bar(g, H)]
        main = max(bars or items, key=lambda g: g.w)
        out.append([main] + [g for g in items if g is not main])
    return out


def bind_fractions(glyphs: list, lay, H: float, p: Params) -> int:
    """เศษส่วน (dT/ds, 1/|v|, dy/dx …) → ผูกเป็นหน่วยเดียวที่เคลื่อนแบบแข็ง (rigid)
    - เส้นเศษส่วนเป็น "ตัวฐาน" ของบรรทัด (cls = fraction) ตัวบน/ตัวล่างเป็น mark ของเส้น → ใช้ T เดียวกัน
    - รวมเครื่องหมายสูงแคบที่ครอบเศษส่วน (|  |, ( ), [ ]) ที่อยู่ติดกันเข้ากลุ่มด้วย
    - หน่วยนี้ไม่ถูกเอียง/ย่อขยาย แค่เลื่อนตามเส้นบรรทัดและช่องไฟ → รูปสูตรไม่บิด แต่ยังเข้าแถวกับข้อความรอบ ๆ"""
    texts = [g for g in glyphs if g.kind == "text"]
    # เส้นเศษส่วนยาว ๆ มักถูกจัดเป็น non-text "line" ไปก่อน → ให้เป็นผู้สมัครเส้นเศษส่วนด้วย
    bars = [g for g in glyphs if g.kind == "nontext" and g.nt_reason == "line" and g.h < 0.6 * H and g.w < 12 * H]
    groups = _fraction_groups(texts + bars, H, bar_pool=bars + texts)
    groups = _merge_math_groups(groups, texts, H)
    used = set()
    n = 0
    for grp in groups:
        bar = grp[0]
        if bar.gid in used or any(g.gid in used for g in grp):
            continue
        x0 = min(g.x0 for g in grp); x1 = max(g.x1 for g in grp)
        y0 = min(g.y0 for g in grp); y1 = max(g.y1 for g in grp)
        # เครื่องหมายครอบ: สูงอย่างน้อยครึ่งหนึ่งของกลุ่ม แคบ และอยู่ชิดซ้าย/ขวา
        for o in glyphs:
            if o in grp or o.kind == "noise" or o.gid in used:
                continue
            tall_thin = o.h >= 0.5 * (y1 - y0) and o.w <= 0.45 * o.h
            vov = _overlap(o.y0, o.y1, y0, y1) >= 0.6 * o.h
            near = (0 <= x0 - o.x1 <= 0.8 * H) or (0 <= o.x0 - x1 <= 0.8 * H)
            if tall_thin and vov and near:
                grp.append(o)
        x0 = min(g.x0 for g in grp); x1 = max(g.x1 for g in grp)
        # บรรทัดที่เศษส่วนนี้อยู่: เส้นเศษส่วนอยู่ราวกลาง x-height ของบรรทัดนั้น
        best, best_d = None, 1e18
        for L in lay.lines:
            if not L.members or not (L.x0 - 3 * H <= bar.cx <= L.x1 + 3 * H):
                continue
            d = abs(float(L.sdist(bar.cx, bar.cy)) + 0.5 * L.xh)
            if d < best_d and d < 1.5 * L.xh + 0.5 * H:
                best, best_d = L, d
        if best is None:
            continue
        for L in lay.lines:
            for g in grp:
                if g.gid in L.members:
                    L.members.remove(g.gid)
                if g.gid in L.marks:
                    L.marks.remove(g.gid)
        best.members.append(bar.gid)
        best.members.sort(key=lambda k: next(g for g in glyphs if g.gid == k).cx)
        bar.kind, bar.role, bar.cls, bar.line, bar.zone, bar.owner = "text", "base", "fraction", best.lid, "middle", -1
        bar.word = -1
        for g in grp[1:]:
            g.kind, g.role, g.cls, g.owner, g.line, g.zone = "text", "mark", "mark", bar.gid, best.lid, "middle"
            if g.gid not in best.marks:
                best.marks.append(g.gid)
        used.update(g.gid for g in grp)
        n += 1
    for L in lay.lines:                         # ช่วงของบรรทัดหลังย้ายสมาชิก
        if L.members:
            gl = {g.gid: g for g in glyphs}
            L.x0 = min(gl[i].x0 for i in L.members)
            L.x1 = max(gl[i].x1 for i in L.members)
    return n


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

    # 2) เศษส่วน: ผูกเป็นหน่วยแข็ง (ไม่ freeze) → เลื่อนตามบรรทัดได้ แต่ไม่บิดรูป
    if p.bind_fractions:
        stats["fraction_groups"] = bind_fractions(glyphs, lay, H, p)
        texts = [g for g in glyphs if g.kind == "text"]
    # 2.1) ตัวยก/ตัวห้อย ผูกกับตัวทางซ้าย
    if p.bind_scripts:
        stats["scripts"] = bind_scripts(glyphs, lay, H, p)

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