"""
B4–B10. วิเคราะห์เลย์เอาต์

B4  จัดบรรทัด (line tracking ตามแกน x → รองรับหลายคอลัมน์ / ย่อหน้าเยื้อง / บรรทัดโค้งเล็กน้อย)
B5  หา baseline ด้วย least squares + ตัด outlier, หา x-height
B6  แบ่งโซนด้วย vector projection (ระยะตั้งฉากจาก baseline)
B7  ผูกสระ/วรรณยุกต์/จุด เข้ากับตัวฐาน
B8  จัดกลุ่มหยาบ: mid / asc / desc / tall
B9  ตัวอักษรติดกัน
B10 แยกช่องว่างระหว่างตัว vs ระหว่างคำ (1D k-means, k=2)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import geometry as G
from .components import Glyph, PageStats
from .config import Params


@dataclass
class Line:
    lid: int
    members: list = field(default_factory=list)   # gid ของตัวฐาน เรียงซ้าย→ขวา
    marks: list = field(default_factory=list)
    a: float = 0.0                                 # baseline: y = a·x + b
    b: float = 0.0
    xh: float = 0.0                                # x-height ของบรรทัด
    fallback: bool = False                         # บรรทัดสั้น ใช้ความชันจากบรรทัดอื่น
    frozen: bool = False                           # บรรทัดสูตร / ในกรอบ → ไม่แก้ (ตั้งใน nontext.freeze_regions)
    gap_thr: float = float("inf")                  # ช่องกว้างกว่านี้ = ช่องระหว่างคำ
    gap_small_max: float = 0.0                     # ช่องแคบกว่านี้ = ช่องในคำแน่ ๆ
    gap_large_min: float = float("inf")            # ช่องกว้างกว่านี้ = ช่องคำแน่ ๆ (ระหว่างกลาง = ก้ำกึ่ง ไม่แตะ)
    inliers: list = field(default_factory=list)
    x0: float = 0.0
    x1: float = 0.0
    gaps: list = field(default_factory=list)       # [(gid_left, gid_right, gap_px, 'small'|'large')]

    def y_at(self, x):
        return self.a * np.asarray(x) + self.b

    def sdist(self, x, y):
        """ระยะตั้งฉาก (มีเครื่องหมาย) จากจุดถึง baseline ด้วย vector projection
        d = n·(p − p0),  n = (−a, 1)/‖(−a, 1)‖   บวก = อยู่ใต้ baseline"""
        return (np.asarray(y) - self.a * np.asarray(x) - self.b) / np.sqrt(1 + self.a ** 2)


@dataclass
class Layout:
    lines: list
    H: float
    page_slope: float
    gap_threshold: float
    gap_small: float
    gap_large: float
    blocks: list = field(default_factory=list)          # [[lid, ...]] ช่วงบรรทัดที่จัดระยะบรรทัด (refine)
    indent_groups: list = field(default_factory=list)   # [[lid, ...]] บรรทัดที่ระดับย่อหน้าเดียวกัน (refine)


# ------------------------------------------------------------------ B4
def build_lines(glyphs: list[Glyph], H: float, p: Params) -> list[Line]:
    cand = [g for g in glyphs if g.kind == "text" and g.h >= p.mark_height_h * H]
    cand.sort(key=lambda g: g.x0)
    tracks: list[dict] = []
    for g in cand:
        best, best_score = None, 1e18
        for t in tracks:
            dx = g.x0 - t["right"]
            if dx < -0.6 * H or dx > p.line_join_dx_h * H:
                continue
            # ทำนายตำแหน่งแนวตั้งจากแนวโน้มของตัวล่าสุด (รองรับบรรทัดที่เขียนเฉียงขึ้น/ลง)
            cx3, cy3 = t["cxs"][-3:], t["cys"][-3:]
            if len(cx3) >= 2 and cx3[-1] - cx3[0] > 0.5 * H:
                slope = float(np.clip((cy3[-1] - cy3[0]) / (cx3[-1] - cx3[0]), -0.6, 0.6))
                pred = float(np.mean(cy3)) + slope * (g.cx - float(np.mean(cx3)))
            else:
                pred = float(np.mean(cy3))
            dy = abs(g.cy - pred)
            if dy > p.line_join_dy_h * H:
                continue
            score = dy / H + 0.3 * max(dx, 0) / H
            if score < best_score:
                best, best_score = t, score
        if best is None:
            best = {"ids": [], "cys": [], "cxs": [], "right": -1e18}
            tracks.append(best)
        best["ids"].append(g.gid)
        best["cys"].append(g.cy)
        best["cxs"].append(g.cx)
        best["right"] = max(best["right"], g.x1)
    lines = []
    for t in tracks:
        L = Line(lid=len(lines), members=list(t["ids"]))
        lines.append(L)
    return lines


# ------------------------------------------------------------------ B5
def fit_baseline(L: Line, gl: dict, H: float, p: Params, page_slope: float | None):
    xs = np.array([gl[i].cx for i in L.members])
    ys = np.array([gl[i].y1 for i in L.members])
    ids = np.array(L.members)
    L.x0 = min(gl[i].x0 for i in L.members)
    L.x1 = max(gl[i].x1 for i in L.members)
    if len(L.members) >= p.baseline_min_glyphs and page_slope is None:
        keep = np.ones(len(xs), bool)
        for _ in range(p.baseline_iters):
            a, b = G.fit_line_lsq(xs[keep], ys[keep])
            r = ys - (a * xs + b)
            sigma = np.std(r[keep])
            new_keep = np.abs(r) <= max(p.baseline_outlier_k * sigma, 0.06 * H)
            if new_keep.sum() < 3 or (new_keep == keep).all():
                break
            keep = new_keep
        a, b = G.fit_line_lsq(xs[keep], ys[keep])
        L.a, L.b, L.fallback = a, b, False
        L.inliers = list(ids[keep])
    else:
        a = page_slope if page_slope is not None else 0.0
        L.a, L.b, L.fallback = a, float(np.median(ys - a * xs)), True
        r = ys - (L.a * xs + L.b)
        L.inliers = list(ids[np.abs(r) <= 0.25 * H])
    hts = [-L.sdist(gl[i].cx, gl[i].y0) for i in (L.inliers or L.members)]
    xh = float(np.percentile(hts, 40)) if hts else 0.7 * H
    L.xh = xh if xh > 0.3 * H else 0.7 * H


# ------------------------------------------------------------------ helpers
def _zone(L: Line, g: Glyph) -> str:
    dtop, dbot = L.sdist(g.cx, g.y0), L.sdist(g.cx, g.y1)
    if dbot < -0.7 * L.xh:
        return "upper"
    if dtop > -0.15 * L.xh:
        return "lower"
    return "middle"


def _pick_owner(L: Line, gl: dict, m: Glyph, H: float, prefer_left: bool = False) -> int:
    bases = [gl[i] for i in L.members]
    if not bases:
        return -1
    if not prefer_left:
        cover = [b for b in bases if b.x0 - 0.15 * H <= m.cx <= b.x1 + 0.15 * H]
        if cover:
            return min(cover, key=lambda b: abs(b.cx - m.cx)).gid
    def gap(b):
        if b.x0 <= m.cx <= b.x1:
            return 0.0
        return min(abs(m.cx - b.x0), abs(m.cx - b.x1))
    pool = [b for b in bases if b.cx <= m.cx] if prefer_left else bases
    pool = pool or bases
    nb = min(pool, key=gap)
    return nb.gid if gap(nb) < 1.0 * H else -1


def _kmeans_1d(v: np.ndarray):
    c = np.array([np.percentile(v, 25), np.percentile(v, 90)], float)
    for _ in range(30):
        lab = np.abs(v[:, None] - c[None, :]).argmin(axis=1)
        newc = np.array([v[lab == k].mean() if (lab == k).any() else c[k] for k in range(2)])
        if np.allclose(newc, c):
            break
        c = newc
    return np.sort(c)


def _fit_all(lines: list[Line], gl: dict, H: float, p: Params) -> float:
    for L in lines:
        if len(L.members) >= p.baseline_min_glyphs:
            fit_baseline(L, gl, H, p, None)
    good = [L.a for L in lines if len(L.members) >= p.baseline_min_glyphs]
    slope = float(np.median(good)) if good else 0.0
    for L in lines:
        if 0 < len(L.members) < p.baseline_min_glyphs:
            fit_baseline(L, gl, H, p, slope)
    return slope


def _merge_collinear(lines: list[Line], gl: dict, H: float, p: Params) -> list[Line]:
    """รวมบรรทัดที่ต่อกันตามแนวนอนและอยู่ "แถวเดียวกัน"
    ความคลาดเคลื่อนที่ยอมได้ = max(line_merge_tol·x-height, 0.4·ระยะห่างระหว่างบรรทัดของหน้า)
    → ลายมือที่ลอยขึ้นปลายบรรทัด (เช่น 'by x = f(t)' ที่สูงกว่าคำก่อนหน้า) ยังถูกรวมกลับ
    แต่จะไม่ข้ามไปรวมกับบรรทัดบน/ล่าง เพราะต้องห่างไม่ถึงครึ่งของระยะห่างบรรทัด
    รวมทีละคู่ที่ต่างกันน้อยที่สุดก่อน (greedy) จะได้ไม่จับคู่ผิดบรรทัด"""
    longs = sorted((L for L in lines if len(L.members) >= p.baseline_min_glyphs),
                   key=lambda L: float(L.y_at((L.x0 + L.x1) / 2)))
    ys = [float(L.y_at((L.x0 + L.x1) / 2)) for L in longs]
    dif = [d for d in np.diff(ys) if d > 1.2 * H]            # ตัดคู่ที่ซ้อนกัน (เศษ/ส่วนของสูตร) ออก
    pitch = float(np.median(dif)) if dif else 3.0 * H
    while True:
        best = None
        for A in lines:
            for B in lines:
                if A is B or B.x0 < A.x0:
                    continue
                gap = B.x0 - A.x1
                if gap > 2 * p.line_join_dx_h * H:
                    continue
                ref, oth = (A, B) if len(A.members) >= len(B.members) else (B, A)
                # วัดจากฐานของตัวอักษรจริงในบรรทัดสั้นเทียบกับ baseline ของบรรทัดยาว (ใช้มัธยฐาน
                # → ไม่พังเพราะตัวหางลงอย่าง y g) แทนการเทียบเส้น baseline สองเส้น ซึ่งบรรทัดสั้น
                # มักได้ความชันเพี้ยน
                d = abs(float(np.median([gl[k].y1 - float(ref.y_at(gl[k].cx)) for k in oth.members])))
                tol = max(p.line_merge_tol * ref.xh, min(0.4 * pitch, 1.3 * ref.xh))
                if d <= tol and (best is None or d < best[0]):
                    best = (d, A, B)
        if best is None:
            break
        _, A, B = best
        A.members = sorted(A.members + B.members, key=lambda k: gl[k].cx)
        fit_baseline(A, gl, H, p, None if len(A.members) >= p.baseline_min_glyphs else A.a)
        lines.remove(B)
    return lines


def _dissolve_short(lines: list[Line], gl: dict, H: float, p: Params) -> list[Line]:
    longs = [L for L in lines if len(L.members) >= p.baseline_min_glyphs]
    for S in lines:
        if len(S.members) >= p.baseline_min_glyphs:
            continue
        keep = []
        for i in S.members:
            g = gl[i]
            best, best_s = None, 1e18
            for L in longs:
                if not (L.x0 - H <= g.cx <= L.x1 + H):
                    continue
                dc = float(L.sdist(g.cx, g.cy))
                if -(L.xh + 1.4 * H) <= dc <= 1.0 * H and abs(dc + L.xh / 2) < best_s:
                    best, best_s = L, abs(dc + L.xh / 2)
            if best is None:
                keep.append(i)
            elif _zone(best, g) == "middle":
                best.members.append(i)       # ตัวฐานที่หลุดบรรทัด → กลับเข้าบรรทัด
            # else: ปล่อยให้เป็น mark candidate (ไม่อยู่บรรทัดไหน)
        S.members = keep
    out = [L for L in lines if L.members]
    for L in out:
        L.members.sort(key=lambda k: gl[k].cx)
    return out


# ------------------------------------------------------------------ main
def analyze_layout(glyphs: list[Glyph], stats: PageStats, p: Params) -> Layout:
    H = stats.h
    gl = {g.gid: g for g in glyphs}
    lines = build_lines(glyphs, H, p)

    # B5: บรรทัดยาวก่อน แล้วใช้ความชันมัธยฐานกับบรรทัดสั้น
    page_slope = _fit_all(lines, gl, H, p)

    # รวมบรรทัดที่อยู่บนเส้นเดียวกัน (ถูกตัดขาดเพราะช่องว่างกว้างหรือตัวกระโดด)
    lines = _merge_collinear(lines, gl, H, p)
    page_slope = _fit_all(lines, gl, H, p)

    # ยุบบรรทัดสั้นที่จริง ๆ เป็นสระ/วรรณยุกต์ของบรรทัดยาวข้างเคียง
    lines = _dissolve_short(lines, gl, H, p)
    page_slope = _fit_all(lines, gl, H, p)

    # ตัวฐานที่ลอยเหนือ/จมใต้ baseline ทั้งตัว = จริง ๆ เป็น mark (เช่นวรรณยุกต์ที่เขียนใหญ่)
    demoted = []
    for L in lines:
        keep = []
        for i in L.members:
            g = gl[i]
            dtop, dbot = L.sdist(g.cx, g.y0), L.sdist(g.cx, g.y1)
            # ลอยเหนือเส้นทั้งตัว (วรรณยุกต์ใหญ่) หรือห้อยใต้เส้นเกือบทั้งตัว (ุ ู ที่เขียนยาว)
            if dbot < -0.6 * L.xh or dtop > 0.3 * L.xh or (dtop > -0.2 * L.xh and dbot > 0.4 * L.xh):
                demoted.append(i)
            else:
                keep.append(i)
        if keep and len(keep) != len(L.members):
            L.members = keep
            fit_baseline(L, gl, H, p, None if len(keep) >= p.baseline_min_glyphs else page_slope)
        elif not keep:
            L.members = []          # ทั้งบรรทัดเป็น mark → ยุบบรรทัด ตัวพวกนี้ไปหาเจ้าของในบรรทัดอื่น
    lines = [L for L in lines if L.members]
    lines.sort(key=lambda L: (float(L.y_at((L.x0 + L.x1) / 2)), L.x0))   # เรียงบนลงล่าง
    for k, L in enumerate(lines):
        L.lid = k
        for i in L.members:
            gl[i].line, gl[i].role = k, "base"

    # B6 + B7: mark candidates
    in_line = {i for L in lines for i in L.members}
    cands = [g for g in glyphs if g.kind == "text" and g.gid not in in_line]
    for m in cands:
        best, best_s = None, 1e18
        for L in lines:
            if not (L.x0 - p.line_join_dx_h * H <= m.cx <= L.x1 + p.line_join_dx_h * H):   # จุด : ท้ายบรรทัดมักเว้นห่าง
                continue
            dc = float(L.sdist(m.cx, m.cy))
            if -(L.xh + 1.4 * H) <= dc <= 1.0 * H:
                s = abs(dc + L.xh / 2)
                if s < best_s:
                    best, best_s = L, s
        if best is None:
            m.role, m.cls = "free", "mark"
            continue
        L = best
        m.line = L.lid
        m.zone = _zone(L, m)
        if m.zone == "middle" and m.h >= 0.45 * L.xh and m.w >= 0.25 * L.xh:
            m.role = "base"                        # ตัวเล็กแต่อยู่โซนกลาง = ตัวอักษรจริง
            L.members.append(m.gid)
            continue
        m.owner = _pick_owner(L, gl, m, H, prefer_left=(m.zone == "middle"))
        if m.owner < 0:
            # อยู่ในบรรทัดแต่ไม่มีเจ้าของ (เช่น = - . ที่ยืนเดี่ยว) → เป็นหน่วยของตัวเอง
            # ร่วมจัดระยะและยกลงเส้นบรรทัด แต่ไม่ปรับขนาด
            m.role, m.cls = "base", "punct"
            L.members.append(m.gid)
            continue
        m.role, m.cls = "mark", "mark"
        L.marks.append(m.gid)

    # เครื่องหมายหลายชิ้นที่ซ้อนกันแนวตั้ง (: ; = ≤ ≥ ÷) → รวมเป็นหน่วยเดียว
    # ให้ชิ้นล่างสุดเป็นเจ้าของ ชิ้นอื่นเป็น mark ที่เคลื่อนตาม ไม่อย่างนั้นการจัดช่องไฟจะแยกชิ้นออกจากกัน
    small = [g for g in glyphs if g.kind == "text" and g.line >= 0 and g.h < p.mark_height_h * H
             and (g.cls in ("punct", "mark") or g.role == "free")]
    small.sort(key=lambda g: -g.y1)                     # ล่างสุดก่อน
    for low in small:
        if low.role != "base" or low.cls != "punct":
            continue
        L = lines[low.line]
        for up in small:
            if up is low or up.y1 > low.y0 + 0.2 * L.xh or low.y0 - up.y1 > 1.2 * L.xh:
                continue
            ov = min(up.x1, low.x1) - max(up.x0, low.x0)
            if ov < 0.5 * min(up.w, low.w):
                continue
            if up.role == "mark" and up.owner >= 0 and gl[up.owner].cls != "punct":
                o = gl[up.owner]                        # เป็นสระ/จุดของตัวอักษรจริงอยู่แล้ว (เช่นจุดของ i)
                if min(up.x1, o.x1) - max(up.x0, o.x0) > 0.5 * up.w:
                    continue
            if up.role == "base" and up.gid in L.members:
                L.members.remove(up.gid)
            if up.gid not in L.marks:
                L.marks.append(up.gid)
            up.role, up.cls, up.owner, up.line = "mark", "mark", low.gid, low.line

    # สัญลักษณ์ล้อมกรอบเล็ก (①, Ⓐ, ตัวเลขในวงกลม): ตัวในรูเป็น mark ของวงกลม
    # → เคลื่อนไปด้วยกัน, ไม่ปรับขนาด/เอียง, และถือว่าเว้นช่องคำทั้งก่อนและหลัง
    texts = [g for g in glyphs if g.kind == "text" and g.line >= 0]
    for L in lines:
        for i in list(L.members):
            g = gl[i]
            if not g.holes:
                continue
            boxes = []
            for hole in g.holes:
                hx0, hy0 = hole.min(axis=0)
                hx1, hy1 = hole.max(axis=0)
                if (hx1 - hx0) * (hy1 - hy0) > 0.25 * g.w * g.h:      # รูใหญ่พอจะมีของข้างใน
                    boxes.append((hx0, hy0, hx1, hy1))
            inner = [o for o in texts if o is not g and o.gid != g.owner
                     and any(b[0] < o.cx < b[2] and b[1] < o.cy < b[3] for b in boxes)]
            if not inner:
                continue
            g.cls = "enclosed"
            for o in inner:
                for LL in lines:
                    if o.gid in LL.members:
                        LL.members.remove(o.gid)
                    if o.gid in LL.marks:
                        LL.marks.remove(o.gid)
                L.marks.append(o.gid)
                o.role, o.cls, o.owner, o.line, o.zone = "mark", "mark", g.gid, L.lid, "inside"

    # B8 จัดกลุ่มหยาบ
    for L in lines:
        L.members.sort(key=lambda i: gl[i].cx)
        L.x0 = min(gl[i].x0 for i in L.members)
        L.x1 = max(gl[i].x1 for i in L.members)
        for i in L.members:
            g = gl[i]
            if g.cls in ("punct", "enclosed"):
                continue
            g.zone = "middle"
            asc = L.sdist(g.cx, g.y0) < -1.3 * L.xh
            desc = L.sdist(g.cx, g.y1) > 0.3 * L.xh
            g.cls = "tall" if (asc and desc) else "asc" if asc else "desc" if desc else "mid"

    # B9 ตัวติดกัน
    mids = [gl[i].w for L in lines for i in L.members if gl[i].cls == "mid"]
    medw = float(np.median(mids)) if mids else H
    for L in lines:
        for i in L.members:
            if gl[i].w > p.merged_width_ratio * medw and gl[i].w > 1.2 * H:
                gl[i].merged = True

    # B10 ช่องว่าง: แยก "ช่องระหว่างตัว" กับ "ช่องระหว่างคำ" ทีละบรรทัด
    # ขอบเขตแนวนอนของตัวฐานรวมเครื่องหมายข้างตัว (- , .) ที่ผูกอยู่
    side = {}
    for m in glyphs:
        if m.role == "mark" and m.owner >= 0 and m.zone == "middle":
            side.setdefault(m.owner, []).append(m)

    def ext(i):
        g = gl[i]
        xs0 = [g.x0] + [m.x0 for m in side.get(i, [])]
        xs1 = [g.x1] + [m.x1 for m in side.get(i, [])]
        return min(xs0), max(xs1)

    def raw_gaps(L):
        out, right = [], None
        for i in L.members:
            x0, x1 = ext(i)
            if right is not None:
                out.append(x0 - right)
            right = x1 if right is None else max(right, x1)
        return out

    page_small, page_large = [], []
    word = 0
    for L in lines:
        gaps = raw_gaps(L)
        pos = np.array([v for v in gaps if v > 0.02 * L.xh], float)
        # ค่าเริ่มต้นเมื่อบรรทัดแยกสองกลุ่มไม่ชัด: แคบมาก = ช่องในคำ, กว้างมาก = ช่องคำ, ตรงกลาง = ไม่แน่ใจ
        small_max, large_min = p.gap_small_xh * L.xh, p.gap_large_xh * L.xh
        if len(pos) >= 4:
            # k-means บน log(ช่อง) → แยกตาม "สัดส่วน" ไม่ใช่ระยะสัมบูรณ์ (คนเขียนห่าง/ชิดใช้ได้เหมือนกัน)
            c1, c2 = np.exp(_kmeans_1d(np.log(pos)))
            if c2 >= 2.2 * c1:
                gm = float(np.sqrt(c1 * c2))
                small_max = min(c1 * p.gap_amb_ratio, gm)
                large_min = max(c2 / p.gap_amb_ratio, gm)
        large_min = min(large_min, 1.3 * L.xh)       # กว้างเกิน 1.3 x-height = ช่องคำแน่ ๆ
        small_max = min(max(small_max, 0.2 * L.xh), large_min)
        L.gap_small_max, L.gap_large_min = float(small_max), float(large_min)
        L.gap_thr = float(large_min)
        right, prev = None, None
        for k, i in enumerate(L.members):
            g = gl[i]
            if prev is not None:
                gap = gaps[k - 1]
                # small = ช่องในคำ (จัดให้สม่ำเสมอ), large = ช่องคำ (จัดเบา ๆ),
                # amb = ก้ำกึ่ง → ไม่แตะเลย เพราะเดาผิดแล้วคำติดกัน/แยกกันจนความหมายเพี้ยน
                if gap >= large_min or "enclosed" in (gl[prev].cls, g.cls):
                    kind = "large"
                elif gap <= small_max:
                    kind = "small"
                else:
                    kind = "amb"
                if kind == "large":
                    word += 1
                    page_large.append(gap)
                elif kind == "small" and gap > 0:
                    page_small.append(gap)
                L.gaps.append((prev, i, float(gap), kind))
            g.word = word
            prev = i
        word += 1
    for m in glyphs:
        if m.role == "mark" and m.owner >= 0:
            m.word = gl[m.owner].word
    thr = float(np.median([L.gap_thr for L in lines])) if lines else float("inf")
    gs = float(np.median(page_small)) if page_small else 0.2 * H
    glg = float(np.median(page_large)) if page_large else 0.0

    return Layout(lines=lines, H=H, page_slope=page_slope, gap_threshold=thr, gap_small=gs, gap_large=glg)