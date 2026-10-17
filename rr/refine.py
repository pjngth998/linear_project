"""
C3.5 / C6. ปรับให้สวยขึ้นนอกเหนือจากการจัดให้ตรง

(1) shape_consistency — ตัวเดียวกันหน้าตาเหมือนกัน โดยใช้ต้นแบบจากลายมือของผู้เขียนเอง
    a) จัดกลุ่มตัวอักษรที่หน้าตาคล้ายกัน: แปลงแต่ละตัวเป็นเวกเตอร์ภาพขนาด 32×32 (1024 มิติ)
       แล้ววัด cosine similarity  cos θ = a·b / (‖a‖‖b‖)   ไม่ต้องรู้ว่าเป็นตัวอะไร (ไม่ใช้ OCR)
    b) ต้นแบบของกลุ่ม = ค่าเฉลี่ยของ contour ทุกตัวในกลุ่ม (resample เป็น N จุด, จัดศูนย์กลาง,
       จัดจุดเริ่มด้วย cyclic shift ที่ให้ระยะห่างน้อยสุด)
    c) แต่ละตัว หา affine A (2×2) ที่พาตัวนั้นไปใกล้ต้นแบบที่สุดด้วย least squares
           min ‖X Aᵀ − Y‖²   →   Aᵀ = (XᵀX)⁻¹ Xᵀ Y        (normal equation)
       แยก A ด้วย SVD:  A = U Σ Vᵀ  ตรวจว่าไม่หมุน/ยืด/บีบมากเกินไป (σ ใกล้ 1, det > 0)
    d) ใช้เพียงบางส่วน  A_β = I + β(A − I)   แล้วคูณซ้อนหน้าเมทริกซ์เดิม  T ← about(A_β, pivot) · T
    ผลคือรูปทรงแต่ละตัวเข้าหา "ค่าเฉลี่ยของตัวเอง" — ยังเป็นลายมือของคนนั้น เพราะต้นแบบมาจากตัวเขาเอง
    และเพราะใช้แค่ affine จึงเปลี่ยนตัวหนึ่งให้กลายเป็นอีกตัวไม่ได้

(2) regularize_lines — ระยะบรรทัดเท่ากัน และแนวหน้าบรรทัด (ย่อหน้า) ตรงกัน
    a) ต่อบรรทัดเป็นสาย (คอลัมน์) ตามแนวตั้ง: บรรทัดถัดไป = บรรทัดใต้ที่ใกล้ที่สุดและช่วง x ทับกัน
    b) ตัดสายเป็นช่วง (block) เมื่อช่องว่างใหญ่ผิดปกติ (ขึ้นย่อหน้าใหม่) หรือมีภาพ/กรอบคั่น
    c) ใน block: fit  y_k = c₀ + c₁·k  (k = ลำดับบรรทัด) ด้วย least squares → ตำแหน่งเป้าหมาย
       ที่ระยะบรรทัดเท่ากันทุกบรรทัด แล้วเลื่อนแต่ละบรรทัดเข้าหาเป้าหมายบางส่วน (translation matrix)
    d) ในสาย: จัดกลุ่มจุดเริ่มบรรทัดที่ใกล้กัน (ระดับย่อหน้าเดียวกัน) แล้วเลื่อนเข้าหาค่ามัธยฐานของกลุ่ม
"""
from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np

from . import geometry as G
from .config import Params

_SKIP_CLS = ("fraction", "enclosed", "punct", "mark")
RASTER = 32


# ------------------------------------------------------------------ helpers
def raster_vector(outer: np.ndarray, holes: list, size: int = RASTER) -> np.ndarray:
    """ภาพตัวอักษรย่อเป็น size×size (คงสัดส่วนกว้าง/ยาว, วางกลาง) → เวกเตอร์ 1 มิติ ความยาว 1"""
    pts = [outer] + list(holes)
    x0, y0 = outer.min(axis=0)
    x1, y1 = outer.max(axis=0)
    s = (size - 4) / max(x1 - x0, y1 - y0, 1e-6)
    off = np.array([(size - (x1 - x0) * s) / 2 - x0 * s, (size - (y1 - y0) * s) / 2 - y0 * s])
    m = np.zeros((size, size), np.uint8)
    cv2.fillPoly(m, [np.round(pts[0] * s + off).astype(np.int32)], 255)
    for h in pts[1:]:
        cv2.fillPoly(m, [np.round(h * s + off).astype(np.int32)], 0)
    # ขยายเส้นเล็กน้อยแล้ว blur: เส้นที่หนาต่างกันหรือเลื่อนกัน 1 px ยังถือว่าคล้าย
    m = cv2.GaussianBlur(cv2.dilate(m, np.ones((2, 2), np.uint8)), (3, 3), 0.8).astype(np.float64).ravel()
    return m / (np.linalg.norm(m) + 1e-12)


def best_shift(X: np.ndarray, Y: np.ndarray) -> int:
    """จุดเริ่ม (cyclic shift) ของ X ที่จับคู่จุดกับ Y ได้ระยะรวมน้อยสุด
    ‖X_s − Y‖² = ‖X‖² + ‖Y‖² − 2⟨X_s, Y⟩  → เลือก s ที่ dot product มากสุด"""
    return int(np.argmax([np.sum(np.roll(X, s, axis=0) * Y) for s in range(len(X))]))


def _rms(X: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(X ** 2, axis=1))))


# ------------------------------------------------------------------ (1) shape consistency
def cluster_glyphs(items: list, p: Params) -> list:
    """items: [(gid, cls, nholes, w, h, vec)] → list ของกลุ่ม [gid, ...]
    leader clustering: เทียบกับเวกเตอร์เฉลี่ยของแต่ละกลุ่ม เข้ากลุ่มที่ cos สูงสุดถ้า ≥ เกณฑ์"""
    clusters = []   # dict(cls, nh, w, h, sum_vec, members)
    for gid, cls, nh, w, h, v in sorted(items, key=lambda t: -t[4] * t[3]):
        best, bc = p.shape_cos_min, None
        for c in clusters:
            if c["cls"] != cls or c["nh"] != nh:
                continue
            if not (1 / 1.3 <= h / c["h"] <= 1.3) or not (1 / 1.35 <= (w / h) / c["ar"] <= 1.35):
                continue
            mu = c["sum"] / (np.linalg.norm(c["sum"]) + 1e-12)
            cs = float(v @ mu)
            if cs >= best:
                best, bc = cs, c
        if bc is None:
            clusters.append({"cls": cls, "nh": nh, "h": h, "ar": w / h, "sum": v.copy(), "members": [gid]})
        else:
            bc["sum"] += v
            bc["members"].append(gid)
            k = len(bc["members"])
            bc["h"] += (h - bc["h"]) / k
            bc["ar"] += (w / h - bc["ar"]) / k
    return [c["members"] for c in clusters]


def prototype(shapes: list, iters: int = 3) -> tuple:
    """ต้นแบบของกลุ่ม = ค่าเฉลี่ยของรูปทรงที่จับคู่จุดแล้ว (จุดศูนย์กลางที่ 0)
    คืน (ต้นแบบ, shift ของแต่ละตัว)"""
    shifts = [best_shift(X, shapes[0]) for X in shapes]
    for _ in range(iters):
        ref = np.mean([np.roll(X, s, axis=0) for X, s in zip(shapes, shifts)], axis=0)
        shifts = [best_shift(X, ref) for X in shapes]
    ref = np.mean([np.roll(X, s, axis=0) for X, s in zip(shapes, shifts)], axis=0)
    return ref, shifts


def fit_affine(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """A (2×2) ที่ทำให้ ‖X Aᵀ − Y‖² น้อยสุด: Aᵀ = (XᵀX)⁻¹ XᵀY  (X, Y จัดศูนย์กลางแล้ว จึงไม่มีพจน์เลื่อน)"""
    XtX = X.T @ X
    if abs(np.linalg.det(XtX)) < 1e-9:
        return np.eye(2)
    return (np.linalg.inv(XtX) @ X.T @ Y).T


def affine_ok(A: np.ndarray, lim: float, max_rot_deg: float) -> bool:
    """ตรวจด้วย SVD: A = U Σ Vᵀ  — σ ต้องใกล้ 1 (ไม่ยืด/บีบมาก), det > 0 (ไม่พลิก), มุมหมุนสุทธิไม่มาก"""
    if np.linalg.det(A) <= 0:
        return False
    U, S, Vt = np.linalg.svd(A)
    if S.min() < 1 - lim or S.max() > 1 + lim:
        return False
    Rm = U @ Vt
    return abs(np.degrees(np.arctan2(Rm[1, 0], Rm[0, 0]))) <= max_rot_deg


def shape_consistency(glyphs: list, lay, line_T0a: dict, p: Params) -> dict:
    gl = {g.gid: g for g in glyphs}
    lines = {L.lid: L for L in lay.lines}
    N = p.shape_points
    items, cur = [], {}
    for lid, (T0a, y_t) in line_T0a.items():
        L = lines[lid]
        for i, T in T0a.items():
            g = gl[i]
            if g.kind != "text" or g.frozen or g.merged or g.cls in _SKIP_CLS or len(g.outer) < 16:
                continue
            if g.h < 0.4 * L.xh:
                continue
            P = G.apply(T, g.outer)
            Hs = [G.apply(T, h) for h in g.holes if len(h) >= 8]
            w, h = np.ptp(P[:, 0]), np.ptp(P[:, 1])
            if w < 2 or h < 2:
                continue
            src = G.resample_n(g.outer, N)            # จุดบนตัวต้นฉบับ (ใช้ระบุตำแหน่งการบิดตอนวาด)
            Pc = G.apply(T, src)                      # จุดเดียวกันหลังแก้เอียง/ขนาด → จับคู่จุดได้ตรงกันเสมอ
            cur[i] = (Pc, src, lid, y_t, h)
            items.append((i, g.cls, len(Hs), w, h, raster_vector(P, Hs)))

    groups = [m for m in cluster_glyphs(items, p) if len(m) >= p.shape_min_members]
    adjusted, rejected = 0, 0
    I2 = np.eye(2)
    for cid, mem in enumerate(groups):
        shapes = [cur[i][0] - cur[i][0].mean(axis=0) for i in mem]
        # ปรับขนาดให้เท่ากันก่อนเฉลี่ย (ต้นแบบควรเป็นรูปทรงเฉลี่ย ไม่ถูกตัวใหญ่ครอบงำ)
        scales = [_rms(X) for X in shapes]
        s_ref = float(np.median(scales))
        normed = [X * (s_ref / max(sc, 1e-6)) for X, sc in zip(shapes, scales)]
        proto, shifts = prototype(normed)
        for i, X0, sc, sh in zip(mem, shapes, scales, shifts):
            g = gl[i]
            g.shape_cluster = cid
            Pc, src, lid, y_t, h = cur[i]
            X = np.roll(X0, sh, axis=0)               # จุดเรียงตามต้นแบบแล้ว (ขนาดจริงของตัวนี้)
            S = np.roll(src, sh, axis=0)
            Y = proto * (sc / s_ref)                  # ต้นแบบในขนาดของตัวนี้ → ไม่เปลี่ยนขนาด (ให้ C3 ดูแล)
            r0 = _rms(X - Y) / max(_rms(Y), 1e-6)
            if r0 > p.shape_max_dev:                  # ต่างจากต้นแบบมากเกิน = อาจคนละตัว / จับคู่จุดไม่ดี → ไม่แตะ
                rejected += 1
                continue
            A = fit_affine(X, Y)
            if not affine_ok(A, p.shape_max_change, p.shape_max_rot_deg):
                A = I2                                # affine แปลกเกิน → ใช้แค่การบิดเฉพาะจุด
            Ab = I2 + p.shape_strength * (A - I2)
            cx = float(Pc[:, 0].mean())
            # จุดหมุน: ตัวกลาง/ตัวสูง → บนเส้นบรรทัด (ฐานไม่ลอย), ตัวหางลง → ที่กึ่งกลางตัว
            py = y_t if g.cls in ("mid", "asc") else float(Pc[:, 1].mean())
            T0a, _ = line_T0a[lid]
            T0a[i] = G.about(Ab, cx, py) @ T0a[i]

            # ส่วนที่ affine ทำไม่ได้ (โค้ง/ป่องเฉพาะจุด): R = Y − X·Aᵀ  ทำให้เรียบด้วย moving average
            # ตามแนว contour แล้วใช้บางส่วน จำกัดขนาด → เวกเตอร์เลื่อนรายจุด (non-rigid)
            if p.shape_nonrigid > 0:
                R = Y - X @ A.T
                R = G.moving_average_closed(R, 7)
                R = G.moving_average_closed(R, 7)
                D = p.shape_nonrigid * R
                lim = p.shape_max_disp_h * h
                nrm = np.linalg.norm(D, axis=1, keepdims=True)
                D = D * np.minimum(1.0, lim / np.maximum(nrm, 1e-9))
                # เวกเตอร์เลื่อนอยู่ในกรอบหลังปรับ affine: หมุน/ยืดตาม Ab ด้วย (เล็กน้อย)
                g.disp = (S, D @ Ab.T)
            g.notes.append(f"shape:{cid}")
            adjusted += 1
    return {"clusters": len(groups), "clustered": sum(len(m) for m in groups),
            "adjusted": adjusted, "rejected": rejected}


# ------------------------------------------------------------------ (2) line spacing / indent
def _line_extent(L, gl) -> tuple:
    xs0, xs1, ys = [], [], []
    for i in L.members:
        P = G.apply(gl[i].T, gl[i].outer)
        xs0.append(P[:, 0].min())
        xs1.append(P[:, 0].max())
        if gl[i].cls in ("mid", "asc"):
            ys.append(P[:, 1].max())
    return min(xs0), max(xs1), (float(np.median(ys)) if ys else None)


def regularize_lines(glyphs: list, lay, p: Params, ruled_paper: bool) -> dict:
    gl = {g.gid: g for g in glyphs}
    nontext = [g for g in glyphs if g.kind == "nontext"]
    info = {"blocks": 0, "moved_y": 0, "moved_x": 0, "chains": 0}
    lay.blocks, lay.indent_groups = [], []

    cand = {}
    for L in lay.lines:
        if L.frozen or len(L.members) < 2 or any(gl[i].frozen for i in L.members):
            continue
        if any("short_line" in gl[i].notes for i in L.members):
            continue
        x0, x1, yb = _line_extent(L, gl)
        if yb is None:
            continue
        cand[L.lid] = {"L": L, "x0": x0, "x1": x1, "y": yb, "xh": L.xh}
    if len(cand) < 2:
        return info

    # สิ่งกีดขวางในแถว (ลูกศร กรอบ ภาพ อยู่ในแนวเดียวกับบรรทัด) → บรรทัดนี้เป็นหมุด ไม่ขยับ
    anchored = set()
    for lid, c in cand.items():
        for g in nontext:
            if g.y1 > c["y"] - 1.5 * c["xh"] and g.y0 < c["y"] + 0.3 * c["xh"] and g.x1 > c["x0"] and g.x0 < c["x1"]:
                anchored.add(lid)
                break

    def blocked(a, b):     # มี non-text คั่นระหว่างบรรทัด a (บน) กับ b (ล่าง) ในช่วง x ที่ทับกัน
        lo, hi = max(a["x0"], b["x0"]), min(a["x1"], b["x1"])
        for g in nontext:
            if g.y0 > a["y"] and g.y1 < b["y"] - b["xh"] and g.x1 > lo and g.x0 < hi:
                return True
        return False

    # ---- (a) ต่อบรรทัดเป็นสายตามแนวตั้ง
    order = sorted(cand, key=lambda k: cand[k]["y"])
    # บรรทัดถัดไป = บรรทัดใต้ที่ใกล้ที่สุดซึ่งช่วง x ทับกัน (ไม่ข้ามบรรทัด)
    # ถ้ามีหลายบรรทัดชี้มาที่บรรทัดเดียวกัน เก็บอันที่อยู่ใกล้ที่สุดไว้ อันอื่นจบสายตรงนั้น
    succ = {}
    for k in order:
        a = cand[k]
        best = None
        for j in order:
            b = cand[j]
            if b["y"] <= a["y"] + 0.8 * a["xh"]:
                continue
            ov = min(a["x1"], b["x1"]) - max(a["x0"], b["x0"])
            if ov < 0.3 * min(a["x1"] - a["x0"], b["x1"] - b["x0"]):
                continue
            if best is None or b["y"] < cand[best]["y"]:
                best = j
        if best is not None:
            succ[k] = best
    pred = {}
    for k, j in succ.items():
        if j not in pred or cand[k]["y"] > cand[pred[j]]["y"]:
            pred[j] = k
    nxt = {k: j for j, k in pred.items()}
    has_prev = set(pred)
    chains = []
    for k in order:
        if k in has_prev:
            continue
        ch = [k]
        while ch[-1] in nxt:
            ch.append(nxt[ch[-1]])
        chains.append(ch)
    info["chains"] = len(chains)

    dy = defaultdict(float)
    dx = defaultdict(float)
    for ch in chains:
        # ---- (b) ตัดเป็น block ตามช่องว่างที่ผิดปกติ / สิ่งกีดขวาง
        pitches = [cand[ch[t + 1]]["y"] - cand[ch[t]]["y"] for t in range(len(ch) - 1)]
        if pitches:
            med = float(np.median(pitches))
            blocks, curb = [], [ch[0]]
            for t, pt in enumerate(pitches):
                a, b = cand[ch[t]], cand[ch[t + 1]]
                if pt > p.line_break_ratio * med or pt < med / p.line_break_ratio or blocked(a, b) \
                        or pt > 8.0 * a["xh"]:
                    blocks.append(curb)
                    curb = []
                curb.append(ch[t + 1])
            blocks.append(curb)
        else:
            blocks = [ch]

        # ---- (c) ระยะบรรทัดเท่ากัน: least squares  y_k = c0 + c1·k
        if p.regularize_spacing and not ruled_paper:
            for blk in blocks:
                if len(blk) < 3:
                    continue
                info["blocks"] += 1
                lay.blocks.append(list(blk))
                ks = np.arange(len(blk), dtype=float)
                ys = np.array([cand[l]["y"] for l in blk])
                # หมุดที่ขยับไม่ได้ให้น้ำหนักมาก (เส้นตรงต้องผ่านใกล้หมุด)
                w = np.array([25.0 if l in anchored else 1.0 for l in blk])
                A = np.column_stack([ks, np.ones_like(ks)]) * np.sqrt(w)[:, None]
                c1, c0 = np.linalg.lstsq(A, ys * np.sqrt(w), rcond=None)[0]
                for k, l in zip(ks, blk):
                    if l in anchored:
                        continue
                    d = (c0 + c1 * k) - cand[l]["y"]
                    lim = p.line_max_shift_xh * cand[l]["xh"]
                    if abs(d) > 2 * lim:          # ห่างจากแนวมาก = ผู้เขียนตั้งใจ (เว้นที่ไว้) → ไม่แตะ
                        continue
                    dy[l] = float(np.clip(p.alpha * d, -lim, lim))

        # ---- (d) แนวหน้าบรรทัด: กลุ่มจุดเริ่มที่ใกล้กัน → มัธยฐานของกลุ่ม
        if p.regularize_indent and len(ch) >= 2:
            xh = float(np.median([cand[l]["xh"] for l in ch]))
            srt = sorted(ch, key=lambda l: cand[l]["x0"])
            groups, g0 = [], [srt[0]]
            for l in srt[1:]:
                if cand[l]["x0"] - cand[g0[0]]["x0"] <= p.indent_tol_xh * xh:   # วัดจากตัวแรกของกลุ่ม (กันกลุ่มลากยาวเป็นลูกโซ่)
                    g0.append(l)
                else:
                    groups.append(g0)
                    g0 = [l]
            groups.append(g0)
            for grp in groups:
                if len(grp) < 2:
                    continue
                lay.indent_groups.append(list(grp))
                fixed = [cand[l]["x0"] for l in grp if l in anchored]
                tgt = float(np.median(fixed)) if fixed else float(np.median([cand[l]["x0"] for l in grp]))
                for l in grp:
                    if l in anchored:
                        continue
                    d = tgt - cand[l]["x0"]
                    if abs(d) <= p.indent_tol_xh * 1.5 * xh:
                        dx[l] = p.alpha * d

    for lid in set(dx) | set(dy):
        L = cand[lid]["L"]
        if abs(dx[lid]) < 0.5 and abs(dy[lid]) < 0.5:
            continue
        M = G.T_translate(dx[lid], dy[lid])
        for i in L.members:
            gl[i].T = M @ gl[i].T
        info["moved_y"] += int(abs(dy[lid]) >= 0.5)
        info["moved_x"] += int(abs(dx[lid]) >= 0.5)
    for g in glyphs:      # สระ/วรรณยุกต์/เครื่องหมายตามเจ้าของ
        if g.role == "mark" and g.owner >= 0 and not g.frozen:
            g.T = gl[g.owner].T.copy()
    return info