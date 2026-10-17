"""
C1–C6. คำนวณเมทริกซ์ปรับแก้ของแต่ละตัวอักษร

ทุกตัวฐานได้ affine matrix 3×3 หนึ่งตัว:

    T0  = Tr(0, dy) · about(anchor, K · S)          (เอียง → ขนาด → ยกลงเส้นบรรทัด)
    T0α = I + α(T0 − I)                             (แก้บางส่วน)
    T   = Tr(dx, 0) · T0α                           (จัดระยะห่าง ซึ่งคิดจากผลของ T0α และใช้ α เช่นกัน)

anchor = จุดบน baseline ใต้ centroid ของตัวอักษร
S = shear (แก้เอียง), K = diag(s, s) (ปรับขนาด, s เท่ากันสองแกนเพื่อรักษาสัดส่วนเดิม)
สระ/วรรณยุกต์ใช้ T เดียวกับตัวฐานที่เป็นเจ้าของ → ตามไปถูกตำแหน่งเสมอ
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from . import geometry as G
from .components import Glyph, PageStats
from .config import Params
from .layout import Layout


def glyph_segments(g: Glyph, xh: float, p: Params, P: np.ndarray | None = None) -> np.ndarray:
    k = max(3, int(round(0.15 * xh)))
    Q = G.resample_closed(g.outer if P is None else P, 1.0)
    return G.slant_segments(Q, k, p.slant_max_dev_deg)


def compute_corrections(glyphs: list[Glyph], lay: Layout, stats: PageStats, p: Params) -> dict:
    gl = {g.gid: g for g in glyphs}
    for g in glyphs:
        g.T = np.eye(3)
    lines = {L.lid: L for L in lay.lines}

    # ---- C1 มุมเอียง (บรรทัดที่ freeze ไม่นำมาคิดค่าเป้าหมายของหน้า)
    segs: dict[int, np.ndarray] = {}
    active = [L for L in lay.lines if not L.frozen]
    for L in active:
        for i in L.members:
            s = glyph_segments(gl[i], L.xh, p)
            segs[i] = s
            if len(s) >= 6:
                gl[i].slant_deg = float(np.median(s))
    all_s = np.concatenate([v for v in segs.values()]) if segs else np.empty(0)
    page_slant = float(np.median(all_s)) if len(all_s) >= p.slant_min_segments else 0.0
    target = page_slant if p.slant_target == "page_median" else 0.0
    tan_t = np.tan(np.radians(target))

    word_slant: dict[int, float] = {}
    line_slant: dict[int, float] = {}
    for L in active:
        ls = np.concatenate([segs[i] for i in L.members]) if L.members else np.empty(0)
        line_slant[L.lid] = float(np.median(ls)) if len(ls) >= p.slant_min_segments else page_slant
        by_word = defaultdict(list)
        for i in L.members:
            by_word[gl[i].word].append(segs[i])
        for w, arrs in by_word.items():
            ws = np.concatenate(arrs)
            word_slant[w] = float(np.median(ws)) if len(ws) >= p.slant_min_segments else line_slant[L.lid]

    # ---- C3 ขนาดเป้าหมายของแต่ละกลุ่ม (มัธยฐานทั้งหน้า)
    by_cls = defaultdict(list)
    for L in active:
        for i in L.members:
            if not gl[i].merged:
                by_cls[gl[i].cls].append(gl[i].h)
    cls_target = {c: float(np.median(v)) for c, v in by_cls.items() if len(v) >= 3}

    owned = defaultdict(list)
    for g in glyphs:
        if g.role == "mark" and g.owner >= 0 and g.zone == "middle":   # เฉพาะเครื่องหมายข้างตัว (- , .)
            owned[g.owner].append(g.gid)                                # สระบน/ล่างไม่นับ อยู่ในแนวตัวอยู่แล้ว

    skipped_lines = 0
    for L in lay.lines:
        # ---- C5 ความมั่นใจต่ำ / บรรทัดสูตร → ไม่แก้ทั้งบรรทัด
        if L.frozen:
            continue
        if L.fallback and len(L.members) < p.baseline_min_glyphs:
            skipped_lines += 1
            for i in L.members:
                gl[i].notes.append("short_line")
            continue

        y_t = float(L.y_at((L.x0 + L.x1) / 2))      # baseline ใหม่: แนวนอนพอดี
        T0a: dict[int, np.ndarray] = {}
        for i in L.members:
            g = gl[i]
            ax, ay = g.cx, float(L.y_at(g.cx))

            # S: shear
            k = 0.0
            if p.enable_slant and g.cls not in ("enclosed", "fraction"):
                cur = word_slant.get(g.word, line_slant[L.lid])
                n = len(segs[i])
                if p.slant_level == "glyph" and n >= 6:
                    # shrinkage: ตัวที่มีเส้นตั้งเยอะ เชื่อมุมของตัวเองมาก, ตัวที่มีน้อย เชื่อมุมของคำ
                    w = n / (n + p.slant_shrink_n0)
                    cur = w * float(np.median(segs[i])) + (1 - w) * cur
                diff = np.clip(cur - target, -p.slant_clip_deg, p.slant_clip_deg)
                k = np.tan(np.radians(target + diff)) - tan_t
            # x' = x + k·(y − ay): จุดที่สูงกว่า anchor (y < ay) ถูกเลื่อนซ้ายเมื่อ k > 0
            # เส้นที่เอียง tan(cur) จึงเหลือ tan(cur) − k = tan(target)
            S = G.shear_matrix(k)

            # K: scale
            s = 1.0
            if p.enable_scale and g.cls in cls_target and g.h > 0:
                raw = cls_target[g.cls] / g.h
                if p.scale_skip_outliers and not (p.scale_min <= raw <= p.scale_max):
                    s = 1.0      # ต่างจากกลุ่มมากเกินไป = น่าจะไม่ใช่ตัวอักษรปกติ (เลขในวงกลม, ตัวใหญ่ตั้งใจ) → ไม่ย่อขยาย
                else:
                    s = float(np.clip(raw, p.scale_min, p.scale_max))
            K = G.scale_matrix(s, s)

            Lin = G.about(K @ S, ax, ay)

            # Tr: ยกตัวอักษรลงเส้นบรรทัดใหม่
            dy = 0.0
            if p.enable_baseline:
                dy = y_t - ay
                P1 = G.apply(Lin, g.outer)
                if g.cls in ("mid", "asc"):
                    r = P1[:, 1].max() - ay
                elif g.cls == "desc":
                    r = P1[:, 1].min() - (ay - L.xh)
                else:
                    r = 0.0
                if abs(r) < p.resid_max_xh * L.xh:
                    dy -= r
            T0 = G.T_translate(0, dy) @ Lin
            T0a[i] = G.blend(T0, p.alpha)

        # ตัวที่ freeze (เช่นอยู่ในกรอบ/เมฆ) แต่บรรทัดไม่ได้ freeze → ไม่ขยับ
        for i in L.members:
            if gl[i].frozen:
                T0a[i] = np.eye(3)

        # ---- C4 ระยะห่าง
        shift = {i: 0.0 for i in L.members}
        if p.enable_spacing and len(L.members) > 1:
            kinds = {right: kind for (_, right, _, kind) in L.gaps}
            orig_gap = {right: gap for (_, right, gap, _) in L.gaps}
            # เป้าหมายคิดจากในบรรทัดเอง ถ้าข้อมูลน้อยใช้ค่าของหน้า
            sm = [gp for (_, _, gp, kd) in L.gaps if kd == "small" and gp > 0]
            lg = [gp for (_, _, gp, kd) in L.gaps if kd == "large" and gp > 0]
            tgt_small = float(np.median(sm)) if len(sm) >= 3 else min(lay.gap_small, L.gap_small_max)
            tgt_large = float(np.median(lg)) if len(lg) >= 2 else None
            min_gap = max(1.0, 0.06 * lay.H)
            small_max, large_min = L.gap_small_max, L.gap_large_min

            # สิ่งกีดขวางในแถวนี้: non-text (เมฆ ลูกศร กรอบ) และตัวที่ freeze → ช่องไฟเริ่มนับใหม่หลังสิ่งกีดขวาง
            band0, band1 = y_t - 1.6 * L.xh, y_t + 0.6 * L.xh
            barriers = [(g.x0, g.x1) for g in glyphs
                        if g.kind == "nontext" and g.y1 > band0 and g.y0 < band1
                        and g.x1 > L.x0 - lay.H and g.x0 < L.x1 + lay.H]

            # ขอบเขตแนวนอนรวมเครื่องหมายข้างตัว (- , .) ไม่อย่างนั้นตัวถัดไปจะถูกเลื่อนมาทับ
            ex = {}
            for i in L.members:
                pts = [gl[i].outer] + [gl[m].outer for m in owned.get(i, [])]
                ex[i] = G.apply(T0a[i], np.vstack(pts))[:, 0]
            first = L.members[0]
            right_new = right_old = ex[first].max()
            for i in L.members[1:]:
                x0i, x1i = ex[i].min(), ex[i].max()
                # มีสิ่งกีดขวางคั่นระหว่างตัวก่อนหน้ากับตัวนี้ → ตั้งจุดเริ่มใหม่ที่ขอบขวาของสิ่งกีดขวาง
                for (b0, b1) in barriers:
                    if b0 >= right_old - 2 and b1 <= x0i + 2:
                        right_old = max(right_old, b1)
                        right_new = max(right_new, b1)
                if gl[i].frozen:
                    shift[i] = 0.0
                    right_new = max(right_new, x1i)
                    right_old = max(right_old, x1i)
                    continue
                g_cur = x0i - right_old
                kind = kinds.get(i, "amb")
                # หลักการ: "ช่องคำไม่แคบลง ช่องในคำไม่กว้างขึ้น"
                # → ความต่างระหว่างช่องคำกับช่องในคำมีแต่จะชัดขึ้น คำจะไม่ติดกันและไม่แตกเป็นชิ้น
                word_min = max(large_min, p.word_letter_ratio * tgt_small)
                if g_cur > 0:
                    if kind == "large":
                        # ช่องคำ: ไม่บีบเลย ขยายได้อย่างเดียว (เข้าหาค่ากลางของบรรทัด / ขั้นต่ำให้ห่างกว่าช่องในคำชัด ๆ)
                        tgt = max(tgt_large if tgt_large is not None else g_cur, word_min)
                        g_new = max(g_cur, g_cur + p.alpha * p.word_gap_alpha * (tgt - g_cur))
                    elif kind == "small":
                        # ช่องในคำ: บีบได้อย่างเดียว (ช่องที่ห่างเกินค่ากลางถูกดึงเข้า, ช่องที่ชิดอยู่แล้วคงไว้)
                        # ถ้าขยายช่องที่ชิด ตัวในคำเดียวกันจะแยกห่างจนดูเป็นคนละคำ เช่น "of" → "o f"
                        g_new = min(g_cur, g_cur + p.alpha * (tgt_small - g_cur))
                        g_new = min(g_new, small_max)
                    else:
                        g_new = g_cur      # ก้ำกึ่ง: คงไว้ตามที่ผู้เขียนเว้น
                elif orig_gap.get(i, 0) > 0:
                    g_new = min_gap        # เดิมไม่ชน แต่ชนเพราะการขยาย/เอียง → ดันออก
                else:
                    g_new = g_cur          # เดิมเขียนซ้อนกันอยู่แล้ว → คงไว้
                shift[i] = (right_new + g_new) - x0i
                right_new = max(right_new, x1i + shift[i])
                right_old = max(right_old, x1i)

        for i in L.members:
            gl[i].T = G.T_translate(shift[i], 0) @ T0a[i]

    # ---- marks ตามเจ้าของ
    for g in glyphs:
        if g.role == "mark" and g.owner >= 0:
            g.T = gl[g.owner].T.copy()
        if g.frozen:
            g.T = np.eye(3)

    return {
        "page_slant_deg": round(page_slant, 2),
        "target_slant_deg": round(target, 2),
        "class_target_height": {c: round(v, 1) for c, v in cls_target.items()},
        "skipped_short_lines": skipped_lines,
        "frozen_lines": sum(1 for L in lay.lines if L.frozen),
    }