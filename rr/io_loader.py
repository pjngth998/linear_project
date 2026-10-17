"""
A1. นำเข้าไฟล์และแยกประเภท

รองรับ: .pdf, .png, .jpg/.jpeg, .webp, .bmp, .tif/.tiff
ไฟล์ .goodnotes ต้อง export เป็น PDF จากแอป GoodNotes ก่อน (รูปแบบไฟล์เป็น proprietary)

ประเภทที่แยกได้ (source_type):
    pdf_vector   PDF ที่ลายมือเป็น vector path (มักมาจาก GoodNotes / แท็บเล็ต)
    pdf_digital  PDF ดิจิทัลอื่น ๆ (ตัวพิมพ์ / เส้นน้อย) ไม่มีภาพสแกน
    pdf_scan     PDF ที่เป็นภาพสแกน
    img_digital  ภาพพื้นหลังเรียบ (screenshot / export จากแอป)
    img_photo    ภาพถ่ายหรือสแกน
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class PageInput:
    name: str
    image: np.ndarray               # BGR uint8
    source_type: str
    info: dict = field(default_factory=dict)


def parse_pages(spec: str | None, n_pages: int) -> list[int]:
    """'1,3-5' → [0, 2, 3, 4] (นับจาก 0) | 'all' → ทุกหน้า"""
    if not spec or spec.strip() == "":
        spec = "1"
    spec = spec.strip().lower()
    if spec == "all":
        return list(range(n_pages))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a) - 1, int(b)))
        elif part:
            out.append(int(part) - 1)
    return [i for i in dict.fromkeys(out) if 0 <= i < n_pages]


def _is_digital(img: np.ndarray) -> bool:
    """ภาพดิจิทัลมีพื้นหลังเป็นค่าสีเดียวเป๊ะ ๆ เป็นส่วนใหญ่ ภาพถ่าย/สแกนมี noise"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = np.bincount(g.ravel(), minlength=256)
    return hist.max() / g.size > 0.35


def _flatten_alpha(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        a = img[..., 3:4].astype(np.float32) / 255
        rgb = img[..., :3].astype(np.float32)
        return (rgb * a + 255 * (1 - a)).astype(np.uint8)   # พื้นโปร่งใส → ขาว
    return img


def load_bytes(data: bytes, filename: str, pages: str | None = "1", dpi: int = 300) -> list[PageInput]:
    ext = os.path.splitext(filename)[1].lower()
    stem = os.path.splitext(os.path.basename(filename))[0]

    if ext == ".goodnotes":
        raise ValueError(
            f"{filename}: ไฟล์ .goodnotes อ่านตรงไม่ได้ ให้ export เป็น PDF จาก GoodNotes ก่อน "
            "(Share → Export → PDF)"
        )

    if ext == ".pdf":
        import pymupdf

        doc = pymupdf.open(stream=data, filetype="pdf")
        out = []
        for i in parse_pages(pages, doc.page_count):
            page = doc[i]
            drawings = page.get_drawings()
            n_draw = len(drawings)
            n_items = sum(len(dw.get("items", [])) for dw in drawings)
            imgs = page.get_images(full=True)
            img_cover = 0.0
            for im in imgs:
                try:
                    for r in page.get_image_rects(im[0]):
                        img_cover += abs(r.width * r.height)
                except Exception:
                    pass
            img_cover /= max(abs(page.rect.width * page.rect.height), 1)
            has_text = bool(page.get_text().strip())
            if (n_draw >= 30 or n_items >= 300) and img_cover < 0.5:
                kind = "pdf_vector"
            elif img_cover < 0.5 and (has_text or n_draw > 0):
                kind = "pdf_digital"
            else:
                kind = "pdf_scan"
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            out.append(PageInput(
                name=f"{stem}_p{i + 1}",
                image=bgr,
                source_type=kind,
                info={"file": filename, "page": i + 1, "pages_total": doc.page_count,
                      "vector_paths": n_draw, "vector_items": n_items, "image_cover": round(img_cover, 3), "dpi": dpi},
            ))
        if not out:
            raise ValueError(f"{filename}: ไม่มีหน้าที่ตรงกับ '{pages}' (ทั้งหมด {doc.page_count} หน้า)")
        return out

    if ext in IMAGE_EXT:
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise ValueError(f"{filename}: เปิดภาพไม่ได้")
        if arr.dtype != np.uint8:
            arr = cv2.convertScaleAbs(arr, alpha=255.0 / max(float(arr.max()), 1.0))
        bgr = _flatten_alpha(arr)
        return [PageInput(
            name=stem,
            image=bgr,
            source_type="img_digital" if _is_digital(bgr) else "img_photo",
            info={"file": filename, "page": 1, "pages_total": 1},
        )]

    raise ValueError(f"{filename}: ไม่รองรับนามสกุล {ext}")


def load_file(path: str, pages: str | None = "1", dpi: int = 300) -> list[PageInput]:
    with open(path, "rb") as f:
        return load_bytes(f.read(), os.path.basename(path), pages, dpi)