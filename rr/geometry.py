"""
เครื่องมือพีชคณิตเชิงเส้นที่ใช้ทั้ง pipeline

ทุกฟังก์ชันทำงานกับ "เมทริกซ์จุด" P ขนาด (N, 2) ที่แต่ละแถวคือจุด (x, y) หนึ่งจุด
พิกัดภาพ: x ไปทางขวา, y ลงล่าง
"""
from __future__ import annotations

import numpy as np

I3 = np.eye(3)


# ---------------------------------------------------------------- เวกเตอร์พื้นฐาน
def direction_vectors(P: np.ndarray, k: int = 1, closed: bool = True) -> np.ndarray:
    """เวกเตอร์ทิศทาง v_i = p_{i+k} − p_i (การลบเวกเตอร์)"""
    if closed:
        return np.roll(P, -k, axis=0) - P
    return P[k:] - P[:-k]


def norms(V: np.ndarray) -> np.ndarray:
    """ขนาดของเวกเตอร์แต่ละแถว ||v||"""
    return np.sqrt((V * V).sum(axis=1))


def turning_angles(P: np.ndarray, k: int = 2) -> np.ndarray:
    """มุมเลี้ยว (องศา) ที่แต่ละจุดของ contour ปิด จาก dot product
    cos θ = (u·v) / (||u|| ||v||)  โดย u = p_i − p_{i−k}, v = p_{i+k} − p_i
    θ ใกล้ 0 = เส้นตรง, θ ใหญ่ = โค้งแรงหรือมุมแหลม"""
    u = P - np.roll(P, k, axis=0)
    v = np.roll(P, -k, axis=0) - P
    dot = (u * v).sum(axis=1)
    den = norms(u) * norms(v) + 1e-9
    return np.degrees(np.arccos(np.clip(dot / den, -1.0, 1.0)))


def resample_closed(P: np.ndarray, step: float = 1.0) -> np.ndarray:
    """resample contour ปิดให้จุดห่างเท่ากันตามความยาวเส้น (arc length)"""
    if len(P) < 3:
        return P.astype(np.float64)
    Q = np.vstack([P, P[:1]]).astype(np.float64)
    seg = norms(np.diff(Q, axis=0))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = s[-1]
    if L < step * 3:
        return P.astype(np.float64)
    n = max(int(round(L / step)), 3)
    t = np.linspace(0, L, n, endpoint=False)
    return np.column_stack([np.interp(t, s, Q[:, 0]), np.interp(t, s, Q[:, 1])])


def resample_n(P: np.ndarray, n: int) -> np.ndarray:
    """resample contour ปิดให้มี n จุดพอดี"""
    Q = np.vstack([P, P[:1]]).astype(np.float64)
    s = np.concatenate([[0.0], np.cumsum(norms(np.diff(Q, axis=0)))])
    t = np.linspace(0, s[-1], n, endpoint=False)
    return np.column_stack([np.interp(t, s, Q[:, 0]), np.interp(t, s, Q[:, 1])])


# ---------------------------------------------------------------- covariance / eigen
def covariance(P: np.ndarray) -> np.ndarray:
    """Covariance matrix C = (P − μ)ᵀ(P − μ) / (N − 1)  (เมทริกซ์สมมาตร 2×2)"""
    X = P - P.mean(axis=0)
    return X.T @ X / max(len(P) - 1, 1)


def principal_axes(P: np.ndarray):
    """คืน (μ, λ เรียงมากไปน้อย, eigenvectors เป็นคอลัมน์)
    eigenvector ของเมทริกซ์สมมาตรตั้งฉากกันเสมอ → ได้แกนหลัก/แกนรองของตัวอักษร"""
    mu = P.mean(axis=0)
    lam, vec = np.linalg.eigh(covariance(P))
    order = np.argsort(lam)[::-1]
    return mu, lam[order], vec[:, order]


def oriented_bbox(P: np.ndarray, min_ratio: float = 1.5):
    """Oriented Bounding Box จาก eigenvector
    ถ้า λ1/λ2 < min_ratio (รูปเกือบกลม ทิศไม่เสถียร) ใช้กรอบตามแกนภาพแทน
    คืน (มุม 4 จุด (4,2), ratio, used_obb)"""
    if len(P) < 3:
        x0, y0 = P.min(axis=0)
        x1, y1 = P.max(axis=0)
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float), 1.0, False
    mu, lam, E = principal_axes(P)
    ratio = float(lam[0] / max(lam[1], 1e-9))
    if ratio < min_ratio:
        E = np.eye(2)
    Y = (P - mu) @ E                     # ฉายจุดลงบนแกนใหม่ (vector projection)
    lo, hi = Y.min(axis=0), Y.max(axis=0)
    corners_local = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    return corners_local @ E.T + mu, ratio, ratio >= min_ratio


# ---------------------------------------------------------------- least squares
def fit_line_lsq(x: np.ndarray, y: np.ndarray):
    """หาเส้น y = a·x + b ที่ดีที่สุดด้วย normal equation
        θ = (AᵀA)⁻¹ Aᵀ y ,  A = [x 1]
    (ระบบมีสมการมากกว่าตัวแปร → หาคำตอบที่ residual กำลังสองน้อยสุด)"""
    A = np.column_stack([x, np.ones_like(x)])
    AtA = A.T @ A
    if abs(np.linalg.det(AtA)) < 1e-9:
        return 0.0, float(np.median(y))
    a, b = np.linalg.inv(AtA) @ (A.T @ y)
    return float(a), float(b)


# ---------------------------------------------------------------- affine (พิกัด homogeneous 3×3)
def T_translate(tx: float, ty: float) -> np.ndarray:
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], float)


def about(M2: np.ndarray, cx: float, cy: float) -> np.ndarray:
    """ทำการแปลงเชิงเส้น 2×2 รอบจุด (cx, cy):  Tr(c) · L · Tr(−c)"""
    L = np.eye(3)
    L[:2, :2] = M2
    return T_translate(cx, cy) @ L @ T_translate(-cx, -cy)


def shear_matrix(k: float) -> np.ndarray:
    """Shear แนวนอน: x' = x + k·y, y' = y   (ความสูงไม่เปลี่ยน)"""
    return np.array([[1.0, k], [0.0, 1.0]])


def scale_matrix(sx: float, sy: float) -> np.ndarray:
    """Scaling = เมทริกซ์แนวทแยง diag(sx, sy)"""
    return np.diag([sx, sy])


def blend(T: np.ndarray, alpha: float) -> np.ndarray:
    """แก้บางส่วน: T_α = I + α(T − I)"""
    return I3 + alpha * (T - I3)


def apply(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    """แปลงทุกจุดพร้อมกันด้วยการคูณเมทริกซ์ครั้งเดียว: [P 1] · Tᵀ"""
    H = np.column_stack([P, np.ones(len(P))])
    return (H @ T.T)[:, :2]


def is_identity(T: np.ndarray, eps: float = 1e-9) -> bool:
    return bool(np.abs(T - I3).max() < eps)


# ---------------------------------------------------------------- moving average (circulant)
def circulant_ma_matrix(n: int, w: int) -> np.ndarray:
    """เมทริกซ์ moving average แบบ circulant (สำหรับ contour ปิด)
    แถว i มีค่า 1/w ที่คอลัมน์ i−r … i+r (วนรอบ) โดย r = w // 2
    แสดงไว้เพื่ออธิบายหลักการ — ตอนรันจริงใช้ moving_average_closed() ซึ่งให้ผลเท่ากันแต่เร็วกว่า"""
    r = w // 2
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(-r, r + 1):
            M[i, (i + j) % n] = 1.0 / w
    return M


def moving_average_closed(P: np.ndarray, w: int, keep: np.ndarray | None = None) -> np.ndarray:
    """P' = M·P  โดย M = circulant moving-average matrix
    คำนวณด้วย np.roll (ผลเท่ากับคูณเมทริกซ์ M แต่ใช้หน่วยความจำ O(N))
    keep: mask จุดที่ไม่ต้อง smooth (มุมแหลม) เพื่อรักษาเอกลักษณ์"""
    r = w // 2
    if r == 0 or len(P) < 2 * w:
        return P
    acc = np.zeros_like(P, dtype=np.float64)
    for j in range(-r, r + 1):
        acc += np.roll(P, -j, axis=0)
    out = acc / (2 * r + 1)
    if keep is not None:
        out[keep] = P[keep]
    return out


# ---------------------------------------------------------------- slant
def slant_segments(P: np.ndarray, k: int, max_dev_deg: float):
    """มุม (องศา) ของช่วงเส้นที่เกือบตั้ง จาก direction vector บน contour
    คืนมุมเทียบแนวดิ่ง: บวก = ปลายบนเอียงไปทางขวา"""
    if len(P) < 2 * k + 2:
        return np.empty(0)
    V = direction_vectors(P, k)
    n = norms(V)
    good = n > 0.6 * k                     # ตัด segment ที่สั้นผิดปกติ (หักมุม)
    V = V[good]
    flip = V[:, 1] > 0                     # ให้ทุกเวกเตอร์ชี้ขึ้น (y ลด)
    V[flip] *= -1
    ang = np.degrees(np.arctan2(V[:, 0], -V[:, 1]))
    return ang[np.abs(ang) <= max_dev_deg]


# ---------------------------------------------------------------- shape similarity
def shape_vector(P: np.ndarray, n: int) -> np.ndarray:
    Q = resample_n(P, n)
    Q = Q - Q.mean(axis=0)
    nn = np.linalg.norm(Q)
    return Q / (nn + 1e-12)


def shape_cosine(P: np.ndarray, Q: np.ndarray, n: int = 64) -> float:
    """Cosine similarity ระหว่างรูปทรงสองรูป หลัง resample เป็น n จุด จัดศูนย์กลาง และ normalize
    ลองทุกจุดเริ่มต้น (cyclic shift) แล้วเลือกค่าที่ดีที่สุด
    → ไม่ไวต่อการเลื่อนและการย่อขยายแบบสม่ำเสมอ แต่ไวต่อการบิดรูป (shear) และการ smooth"""
    a = shape_vector(P, n).ravel()
    B = shape_vector(Q, n)
    best = -1.0
    for s in range(n):
        best = max(best, float(a @ np.roll(B, s, axis=0).ravel()))
    return best