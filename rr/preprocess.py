"""
A2–A4. เตรียมภาพ

A2  ตัดไฮไลต์ (สว่าง + อิ่มตัวสูง) และรวมหมึกทุกสีด้วย min-channel
A3  Gaussian blur (2D convolution) → Adaptive Threshold
A4  ลบเส้นบรรทัดกระดาษ (morphological opening แนวนอน) → แก้หน้าเอียงด้วย projection profile
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import Params
from .geometry import fit_line_lsq
from .io_loader import PageInput


@dataclass
class Prepped:
    color: np.ndarray          # ภาพสีต้นฉบับ หลังหมุนแก้เอียงแล้ว (ใช้เก็บสีหมึก / วาง non-text)
    binary: np.ndarray         # หมึก = 255, พื้น = 0 (หลังหมุน)
    angle: float               # มุมที่หมุนแก้ (องศา)
    dark: bool                 # พื้นหลังมืด (หมึกสว่าง)
    bg_color: tuple            # สีพื้นหลังของหน้า (BGR) ใช้เป็นพื้นตอนวาดผล
    h_rough: float
    steps: dict = field(default_factory=dict)   # ภาพแต่ละขั้นสำหรับ UI
    alpha: np.ndarray | None = None             # ความเข้มหมึก 0..1 ต่อพิกเซล (หลังหมุน) ใช้วาดแบบ warp
    R: np.ndarray | None = None                 # เมทริกซ์ 2×3 ที่หมุนภาพต้นฉบับ → กรอบที่แก้เอียงแล้ว


def _odd(n: int) -> int:
    n = int(n)
    return n if n % 2 else n + 1


def rough_height(binary: np.ndarray) -> float:
    """ประมาณ h คร่าว ๆ: มัธยฐานความสูงของ connected component (ตัดตัวจิ๋วและตัวยักษ์)"""
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    hs = stats[1:, cv2.CC_STAT_HEIGHT]
    areas = stats[1:, cv2.CC_STAT_AREA]
    hs = hs[(hs >= 4) & (areas >= 8)]
    if len(hs) == 0:
        return 20.0
    h0 = float(np.median(hs))
    sel = hs[(hs > 0.4 * h0) & (hs < 2.5 * h0)]
    return float(np.median(sel)) if len(sel) else h0


def _rotate(img: np.ndarray, angle: float, border, interp=cv2.INTER_LINEAR, reflect: bool = False) -> np.ndarray:
    h, w = img.shape[:2]
    R = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    if reflect:   # มุมภาพที่ว่างหลังหมุน เติมด้วยภาพสะท้อน → ลายกระดาษ/กริดต่อเนื่องไปถึงขอบ
        return cv2.warpAffine(img, R, (w, h), flags=interp, borderMode=cv2.BORDER_REFLECT_101)
    return cv2.warpAffine(img, R, (w, h), flags=interp, borderValue=border)


def skew_with_gain(img: np.ndarray, max_deg: float, step: float) -> tuple[float, float]:
    """มุมที่ทำให้ horizontal projection แยกเป็นแถวคมที่สุด + อัตราส่วน variance เทียบกับมุม 0
    (gain สูง = มั่นใจว่าภาพมีแถวชัดเจน)"""
    h, w = img.shape
    s = min(1.0, 1200 / max(h, w))
    small = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    v0 = float(np.var(small.sum(axis=1, dtype=np.float64)))
    best, best_v = 0.0, -1.0
    for a in np.arange(-max_deg, max_deg + 1e-9, step):
        v = float(np.var(_rotate(small, a, 0).sum(axis=1, dtype=np.float64)))
        if v > best_v:
            best, best_v = float(a), v
    return best, (best_v / v0 if v0 > 0 else 1.0)


def dot_grid_angle(pat: np.ndarray, max_deg: float):
    """มุมของกระดาษแบบจุด (dot grid) จากเวกเตอร์ไปหาจุดข้างเคียงที่ใกล้ที่สุด
    จุดบนกริดเรียงเป็นแถว/คอลัมน์ → เวกเตอร์ถึงเพื่อนบ้านใกล้สุดชี้ตามแนวกริด
    เอามุมของเวกเตอร์ (mod 90°) มาหามัธยฐาน = มุมเอียงของกระดาษ
    คืน (มุม, ความมั่นใจ 0..1) หรือ None ถ้าไม่ใช่กระดาษจุด"""
    n, _, stats, cents = cv2.connectedComponentsWithStats(pat, connectivity=8)
    if n < 60:
        return None
    w, h, a = stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT], stats[1:, cv2.CC_STAT_AREA]
    ok = (a >= 2) & (w <= 14) & (h <= 14) & (np.abs(w - h) <= 4)
    P = cents[1:][ok]
    if len(P) < 60:
        return None
    if len(P) > 3000:
        P = P[np.random.default_rng(0).choice(len(P), 3000, replace=False)]
    nn = np.empty_like(P)
    for s0 in range(0, len(P), 500):                     # หาเพื่อนบ้านใกล้สุดทีละก้อน (ประหยัดหน่วยความจำ)
        D = ((P[s0:s0 + 500, None, :] - P[None, :, :]) ** 2).sum(-1)
        D[np.arange(D.shape[0]), np.arange(s0, s0 + D.shape[0])] = np.inf
        nn[s0:s0 + 500] = P[np.argmin(D, axis=1)]
    V = nn - P
    ang = np.degrees(np.arctan2(V[:, 1], V[:, 0]))
    ang = (ang + 45) % 90 - 45                          # รวมทิศแถวกับคอลัมน์ไว้ด้วยกัน
    ang = ang[np.abs(ang) <= max_deg]
    if len(ang) < 40:
        return None
    med = float(np.median(ang))
    conf = float(np.mean(np.abs(ang - med) < 1.5))
    return med, conf


def paper_pattern(flat_gray: np.ndarray, binary: np.ndarray, ink_max: int) -> np.ndarray:
    """ลายของกระดาษ (จุด dot grid / เส้นกริด / เส้นบรรทัดจาง): พิกเซลที่จางกว่าหมึกแต่ไม่ใช่พื้นขาว
    และไม่อยู่ติดตัวหนังสือ (กันขอบนุ่มของเส้นหมึกปน)"""
    pat = ((flat_gray > ink_max) & (flat_gray < 245)).astype(np.uint8) * 255
    near_ink = cv2.dilate(binary, np.ones((9, 9), np.uint8)) > 0
    pat[near_ink] = 0
    return pat


def estimate_skew(binary: np.ndarray, max_deg: float, step: float) -> float:
    """หมุนหลายมุมแล้วเลือกมุมที่ variance ของ horizontal projection สูงสุด
    (บรรทัดกับช่องว่างแยกกันคมที่สุด)"""
    h, w = binary.shape
    s = min(1.0, 1200 / max(h, w))
    small = cv2.resize(binary, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    best, best_v = 0.0, -1.0
    v0 = float(np.var(small.sum(axis=1, dtype=np.float64)))
    for a in np.arange(-max_deg, max_deg + 1e-9, step):
        r = _rotate(small, a, 0)
        v = float(np.var(r.sum(axis=1, dtype=np.float64)))
        if v > best_v:
            best, best_v = float(a), v
    # หมุนแล้วดีขึ้นไม่ชัด (หน้ามีบรรทัดน้อย / มีข้อความเขียนเฉียงตั้งใจ) → ไม่หมุนทั้งหน้า
    # ปล่อยให้การหา baseline รายบรรทัดจัดการแทน
    if v0 > 0 and best_v < 1.15 * v0:
        return 0.0
    return best



def detect_ruled_lines(binary: np.ndarray, h: float, p: Params):
    """หาเส้นบรรทัดกระดาษ: เปิดภาพด้วย kernel แนวนอน → HoughLinesP → รวม segment ที่อยู่บนเส้นเดียวกัน
    เก็บเฉพาะกลุ่มที่ยาวรวมเกิน ruled_min_width_frac ของความกว้างหน้า (เส้นใต้คำสั้น ๆ ไม่โดนลบ)"""
    H, W = binary.shape
    k = max(15, int(1.0 * h))
    hor = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    segs = cv2.HoughLinesP(hor, 1, np.pi / 720, threshold=max(30, int(0.06 * W)),
                           minLineLength=int(0.10 * W), maxLineGap=int(1.2 * h))
    if segs is None:
        return [], 0.0
    max_t = np.radians(p.deskew_max_deg + 1)
    items = []
    for x1, y1, x2, y2 in np.asarray(segs).reshape(-1, 4).astype(float):
        if x2 < x1:
            x1, y1, x2, y2 = x2, y2, x1, y1
        th = np.arctan2(y2 - y1, x2 - x1)
        if abs(th) > max_t:
            continue
        yc = y1 + (W / 2 - x1) * np.tan(th)
        items.append((yc, th, x1, x2, np.hypot(x2 - x1, y2 - y1)))
    if not items:
        return [], 0.0
    items.sort()
    groups, cur = [], [items[0]]
    for it in items[1:]:
        if it[0] - cur[-1][0] <= max(3.0, 0.25 * h):
            cur.append(it)
        else:
            groups.append(cur)
            cur = [it]
    groups.append(cur)
    lines = []
    for g in groups:
        arr = np.array(g)
        # ความยาวที่ครอบคลุมจริง (union ของช่วง x) ไม่นับซ้ำ
        spans = sorted((a[2], a[3]) for a in g)
        cover, (s0, s1) = 0.0, spans[0]
        for a, b in spans[1:]:
            if a <= s1:
                s1 = max(s1, b)
            else:
                cover += s1 - s0
                s0, s1 = a, b
        cover += s1 - s0
        if cover < p.ruled_min_width_frac * W:
            continue
        wts = arr[:, 4]
        lines.append({"yc": float(np.average(arr[:, 0], weights=wts)),
                      "theta": float(np.average(arr[:, 1], weights=wts))})
    # ปรับเส้นให้แม่นด้วย least squares บนพิกเซลจริงรอบเส้น (Hough ให้แค่ค่าประมาณ)
    ys_all, xs_all = np.nonzero(hor)
    for L in lines:
        for tol in (max(5.0, 0.15 * h), max(3.0, 0.08 * h), max(3.0, 0.08 * h)):
            pred = L["yc"] + (xs_all - W / 2) * np.tan(L["theta"])
            sel = np.abs(ys_all - pred) <= tol
            if sel.sum() < 50:
                break
            a, b = fit_line_lsq(xs_all[sel].astype(float), ys_all[sel].astype(float))
            L["theta"] = float(np.arctan(a))
            L["yc"] = float(a * W / 2 + b)
    # ความหนาเส้น: นับพิกเซลแนวตั้งของ hor รอบเส้นที่พบ
    thick = []
    for L in lines:
        for x in np.linspace(0.05 * W, 0.95 * W, 40).astype(int):
            y = int(round(L["yc"] + (x - W / 2) * np.tan(L["theta"])))
            col = hor[max(0, y - int(0.3 * h)): min(H, y + int(0.3 * h) + 1), x]
            n = int(np.count_nonzero(col))
            if n:
                thick.append(n)
    return lines, float(np.median(thick)) if thick else 2.0


def remove_ruled_lines(binary: np.ndarray, lines: list, thick: float, h: float,
                       gray: np.ndarray | None = None, ink_dark: float = 0.0):
    """ลบเส้นบรรทัดกระดาษ แต่พยายามไม่กินลายมือที่ทับเส้น
    1) ในแถบรอบเส้น ลบเฉพาะพิกเซลที่เป็น "ช่วงแนวนอนยาว" (opening แนวนอน)
    2) ลบเศษชิ้นเล็กที่อยู่ในแถบทั้งชิ้น
    3) ต่อเส้นลายมือที่ขาดตรงจุดตัดด้วย closing แนวตั้ง"""
    H, W = binary.shape
    bt = int(round(thick)) + 4
    band = np.zeros_like(binary)
    for L in lines:
        y0 = L["yc"] + (0 - W / 2) * np.tan(L["theta"])
        y1 = L["yc"] + (W - W / 2) * np.tan(L["theta"])
        cv2.line(band, (0, int(round(y0))), (W - 1, int(round(y1))), 255, bt)
    hor = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, int(0.4 * h)), 1)))
    # หมึกเข้มจริง (ขอบกรอบที่วาดทับเส้นกริด, เส้นใต้) ไม่ลบ — เส้นกระดาษพิมพ์มักจางกว่าหมึกมาก
    dark = (gray < ink_dark) if (gray is not None and ink_dark > 0) else np.zeros_like(binary, bool)
    removed = binary.copy()
    removed[(band > 0) & (hor > 0) & ~dark] = 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(removed, connectivity=8)
    if n > 1:
        inband = np.bincount(lab[band > 0].ravel(), minlength=n)
        total = stats[:, cv2.CC_STAT_AREA]
        darkcnt = np.bincount(lab[dark].ravel(), minlength=n)
        kill = (inband >= 0.7 * total) & (stats[:, cv2.CC_STAT_HEIGHT] <= bt + 2) & (darkcnt < 0.5 * total)
        kill[0] = False
        removed[kill[lab]] = 0
    closed = cv2.morphologyEx(removed, cv2.MORPH_CLOSE, np.ones((bt + 2, 1), np.uint8))
    out = np.where(band > 0, closed, removed).astype(np.uint8)
    return out, band


def preprocess(page: PageInput, p: Params) -> Prepped:
    color = page.image
    steps: dict = {}

    # ---- A2 พื้นหลังสว่างหรือมืด
    # หมึกกินพื้นที่ส่วนน้อยของหน้า → มัธยฐานของทั้งหน้า ≈ สีพื้นหลัง
    small = cv2.resize(color, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    bg_color = tuple(int(v) for v in np.median(small.reshape(-1, 3), axis=0))
    page_lum = float(np.median(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)))
    dark = p.background_mode == "dark" or (p.background_mode == "auto" and page_lum < p.dark_threshold)
    # พื้นมืด: กลับสีภาพสำหรับขั้นวิเคราะห์ (หมึกสว่าง → เข้ม, พื้นดำ → ขาว)
    # ขั้นต่อจากนี้ทั้งหมดจึงใช้โค้ดเดียวกับกระดาษขาวได้  ส่วนสีจริงยังเก็บไว้ใน color
    analysis = 255 - color if dark else color

    # ---- A2 สี + ไฮไลต์ + แสงไม่สม่ำเสมอ
    # min-channel: หมึกสี (น้ำเงิน แดง) เข้มเท่าหมึกดำ
    gray = analysis.min(axis=2)
    if dark:
        steps["inverted"] = analysis
    if p.remove_highlighter and not dark:
        # (ตัวเลือกเสริม) ตัดไฮไลต์ด้วยสี: สว่าง + อิ่มตัวสูง
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        L = cv2.cvtColor(color, cv2.COLOR_BGR2LAB)[..., 0]
        hl = (hsv[..., 1] >= p.highlight_s_min) & (L >= p.highlight_l_min)
        hl = (cv2.dilate(hl.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & (L >= p.highlight_l_min - 40)
        gray = gray.copy()
        gray[hl] = 255
        steps["highlight_mask"] = (hl * 255).astype(np.uint8)
    if p.flatten_background:
        # ประมาณพื้นหลังด้วย morphological closing (ลบเส้นหมึกที่บางกว่า kernel ทิ้ง)
        # แล้วหาร → กระดาษสว่างเท่ากันทั้งหน้า, แถบไฮไลต์ (กว้างกว่า kernel) กลายเป็นพื้นขาว
        Hh, Ww = gray.shape
        k = _odd(p.flatten_kernel) if p.flatten_kernel > 0 else _odd(max(21, min(Hh, Ww) // 60))
        bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        gray = np.clip(gray.astype(np.float32) / np.maximum(bg.astype(np.float32), 1) * 255, 0, 255).astype(np.uint8)
        steps["background"] = bg

    # ---- A3 blur + threshold
    flat_gray = gray.copy()          # ก่อน blur: ใช้คำนวณความเข้มหมึกจริงตอนวาดกลับ
    use_blur = p.blur == "on" or (p.blur == "auto" and page.source_type in ("pdf_scan", "img_photo"))
    if use_blur:
        k = _odd(max(3, p.blur_ksize))
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    steps["gray"] = gray

    H, W = gray.shape
    block = _odd(p.thresh_block) if p.thresh_block > 0 else _odd(max(15, min(H, W) // 25))
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, block, p.thresh_C)
    binary[gray > 240] = 0
    faint = gray > p.ink_max            # จางกว่าหมึกจริง (จุด dot grid, เส้นบรรทัดจาง) — ตัดทิ้งหลังหาเส้นบรรทัด
    steps["binary_raw"] = binary.copy()

    h = rough_height(binary)

    # ---- A4 ลบเส้นบรรทัดกระดาษ (ทำบนภาพที่ยังไม่หมุน: เส้นที่ตรวจได้มีมุมของตัวเองอยู่แล้ว)
    ruled = []
    if p.remove_ruled_lines:
        ruled, thick = detect_ruled_lines(binary, h, p)
        if ruled:
            binary, band = remove_ruled_lines(binary, ruled, thick, h, gray, 0.85 * p.ink_max)
            steps["ruled_lines"] = band
    # ตัดส่วนที่จางทิ้ง "หลัง" หาเส้นบรรทัด เพราะเส้นบรรทัดจาง ๆ ยังใช้หามุมหน้าเอียงได้
    binary[faint] = 0

    # ---- A4 แก้หน้าเอียง: ถ้ามีเส้นบรรทัดกระดาษ ≥ 2 เส้น ใช้มุมของเส้นพวกนั้น (แม่นสุด)
    #      ไม่มีก็ใช้ projection profile ของตัวหนังสือ
    #      ลำดับความเชื่อถือ: เส้นบรรทัดกระดาษ → ลายกระดาษ (จุด/กริด) → แถวตัวหนังสือ
    #      ใช้กระดาษเป็นหลัก เพราะต้องการให้ "กระดาษตรง" (ตัวหนังสือจัดตามกระดาษทีหลังอยู่แล้ว)
    angle, source = 0.0, "none"
    if p.deskew:
        if len(ruled) >= 2:
            angle = float(np.degrees(np.median([r["theta"] for r in ruled])))
            angle = float(np.clip(angle, -p.deskew_max_deg, p.deskew_max_deg))
            source = "ruled_lines"
        else:
            pat = paper_pattern(flat_gray, binary, p.ink_max)
            steps["paper_pattern"] = pat
            dots = dot_grid_angle(pat, p.deskew_max_deg)
            if dots is not None and dots[1] >= 0.5:  # กระดาษจุดที่เรียงเป็นกริดชัดเจน → เชื่อกระดาษ
                angle, source = dots[0], "dot_grid"
                if abs(angle) < 0.3:
                    angle = 0.0                      # เอียงน้อยมาก = ตรงอยู่แล้ว ไม่ต้องหมุน (กันภาพเบลอจากการหมุน)
            else:
                angle = estimate_skew(binary, p.deskew_max_deg, p.deskew_step_deg)
                source = "text" if abs(angle) > 1e-6 else "none"
        if abs(angle) > 1e-6:
            binary = _rotate(binary, angle, 0, cv2.INTER_NEAREST)
            color = _rotate(color, angle, bg_color, reflect=True)
            flat_gray = _rotate(flat_gray, angle, 255, reflect=True)
    steps["deskew_source"] = source

    steps["binary"] = binary
    # ความเข้มหมึก: 0 = พื้น, 1 = หมึกเต็ม (เทียบกับความเข้มมัธยฐานของหมึกในหน้า)
    ink_level = float(np.median(flat_gray[binary > 0])) if (binary > 0).any() else 0.0
    alpha = np.clip((255.0 - flat_gray.astype(np.float32)) / max(255.0 - ink_level, 1.0), 0, 1)
    Hh0, Ww0 = binary.shape
    R = cv2.getRotationMatrix2D((Ww0 / 2, Hh0 / 2), angle, 1.0)
    return Prepped(color=color, binary=binary, angle=angle, dark=dark, bg_color=bg_color, alpha=alpha, R=R,
                   h_rough=h, steps=steps)