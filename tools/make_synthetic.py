"""
สร้างภาพ "ลายมือรก" สังเคราะห์ไว้ทดสอบ pipeline

แนวคิด: เรนเดอร์ข้อความด้วยฟอนต์ลายมือไทยทีละกลุ่มอักษร (พยัญชนะ + สระบน/ล่าง/วรรณยุกต์)
แล้วสุ่มความเอียง ขนาด และตำแหน่งแนวตั้งของแต่ละกลุ่ม ให้ดูเหมือนเขียนเร็ว ๆ
จากนั้นใส่เส้นบรรทัดกระดาษ ไฮไลต์ ลูกศร/กรอบ (non-text) และแสงไม่สม่ำเสมอ

ใช้:
    python tools/make_synthetic.py --out samples
ได้ไฟล์:
    samples/synthetic_photo.png   (แบบถ่ายรูป: มีแสงเงา noise และหน้าเอียง)
    samples/synthetic_clean.png   (แบบดิจิทัล: พื้นขาวสะอาด)
    samples/synthetic_dark.png    (แบบ dark mode: หมึกสว่างบนพื้นดำ)
    samples/synthetic_scan.pdf    (แบบสแกน: ภาพ photo ฝังใน PDF)
"""
from __future__ import annotations

import argparse
import os
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

THAI_COMBINING = set(chr(c) for c in [0x0E31, *range(0x0E34, 0x0E3B), *range(0x0E47, 0x0E4F)])
SARA_AM = "ำ"

TEXT_LINES = [
    "พีชคณิตเชิงเส้น บทที่ 3 การแปลงเชิงเส้น",
    "เมทริกซ์ A ขนาด 2x2 มีค่าเฉพาะ 2 ค่า",
    "Eigenvector คือเวกเตอร์ที่ทิศทางไม่เปลี่ยน",
    "det(A) = ad - bc ถ้าไม่เท่ากับ 0 จะหาตัวผกผันได้",
    "ข้อสอบกลางภาคปี 2569 ออกเรื่องนี้ทุกปี",
    "Shear matrix ทำให้รูปเอียงโดยความสูงคงเดิม",
    "สรุป: น้ำหนักคะแนน 30 เปอร์เซ็นต์ จำให้ดี",
]


def clusters(text: str) -> list[str]:
    """แบ่งข้อความเป็นกลุ่มอักษร (ฐาน + เครื่องหมายที่ซ้อนบน/ล่าง)"""
    out: list[str] = []
    for ch in text:
        if out and (ch in THAI_COMBINING or ch == SARA_AM):
            out[-1] += ch
        else:
            out.append(ch)
    return out


def render_cluster(font: ImageFont.FreeTypeFont, s: str):
    """คืนภาพ alpha ของกลุ่มอักษร และจุด (left, baseline) ในภาพนั้น"""
    size = font.size
    W, H = int(size * 3), int(size * 3)
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    ox, oy = size, int(size * 1.8)
    d.text((ox, oy), s, font=font, fill=255, anchor="ls")
    adv = font.getlength(s)
    return np.array(img), (ox, oy), adv


def make_page(font_path: str, seed: int = 7, size: int = 64, width: int = 2400):
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    font = ImageFont.truetype(font_path, size)
    line_gap = int(size * 1.9)
    margin = int(size * 1.2)
    height = margin * 2 + line_gap * (len(TEXT_LINES) + 1)
    ink = np.zeros((height, width), np.float32)          # ความเข้มหมึก 0..1
    ink_color = np.zeros((height, width, 3), np.float32)  # สีหมึก

    writer_slant = 0.18  # ความเอียงประจำตัวผู้เขียน (tan ของมุม)
    for li, text in enumerate(TEXT_LINES):
        color = (40, 30, 20) if li % 3 else (140, 50, 20)   # BGR: ดำ / น้ำเงิน
        x = float(margin + rng.uniform(-10, 30))
        y_base = margin + line_gap * (li + 1)
        drift = rng.uniform(-0.035, 0.035)                  # บรรทัดลาดขึ้น/ลง
        wave_amp, wave_f = rng.uniform(0, 6), rng.uniform(0.002, 0.006)
        for cl in clusters(text):
            if cl == " ":
                x += size * rng.uniform(0.35, 0.65)
                continue
            alpha, (ox, oy), adv = render_cluster(font, cl)
            shear = writer_slant + rng.gauss(0, 0.13)
            sc = rng.uniform(0.82, 1.22)
            dy = rng.gauss(0, size * 0.07) + drift * (x - margin) + wave_amp * np.sin(wave_f * x)
            # affine รอบจุด (ox, oy) = มุมซ้ายของ baseline
            M = np.array([[sc, -shear * sc, 0], [0, sc, 0]], np.float32)
            M[:, 2] = np.array([x, y_base + dy]) - M[:, :2] @ np.array([ox, oy])
            warped = cv2.warpAffine(alpha, M, (width, height), flags=cv2.INTER_LINEAR).astype(np.float32) / 255
            ink = np.maximum(ink, warped)
            ink_color[warped > 0.05] = color
            x += adv * sc + rng.uniform(-0.04, 0.18) * size

    # ---- ประกอบหน้ากระดาษ (สีหมึก + พื้นขาว)
    page = np.full((height, width, 3), 255, np.float32)
    page = page * (1 - ink[..., None]) + ink_color * ink[..., None]

    # เส้นบรรทัดกระดาษสีฟ้าอ่อน
    for li in range(len(TEXT_LINES) + 1):
        yy = margin + line_gap * li + int(size * 0.35)
        cv2.line(page, (0, yy), (width, yy), (235, 205, 170), 2)

    # ไฮไลต์สีเหลืองทับคำบรรทัดที่ 2
    hl = page.copy()
    y2 = margin + line_gap * 2
    cv2.rectangle(hl, (margin + 300, y2 - int(size * 0.75)), (margin + 900, y2 + int(size * 0.2)), (120, 245, 255), -1)
    page = np.where(ink[..., None] > 0.3, page, 0.55 * page + 0.45 * hl)

    # non-text: กรอบล้อมคำ + ลูกศร
    yb = margin + line_gap * 5
    cv2.rectangle(page, (width - 760, yb - int(size * 1.0)), (width - 240, yb + int(size * 0.5)), (30, 30, 30), 4)
    yb = margin + line_gap * 7
    cv2.arrowedLine(page, (width - 900, yb - 40), (width - 300, yb - 90), (30, 30, 30), 5, tipLength=0.08)
    clean = np.clip(page, 0, 255).astype(np.uint8)

    # ---- เวอร์ชันถ่ายรูป: แสงไม่สม่ำเสมอ + noise + หมุนหน้า
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    light = 0.78 + 0.22 * (xx / width) - 0.10 * ((yy / height - 0.5) ** 2) * 4
    photo = page * light[..., None] * np.array([0.97, 0.99, 1.0])
    photo += nrng.normal(0, 6, photo.shape)
    photo = np.clip(photo, 0, 255).astype(np.uint8)
    R = cv2.getRotationMatrix2D((width / 2, height / 2), 1.6, 1.0)
    photo = cv2.warpAffine(photo, R, (width, height), borderValue=(200, 205, 210))
    return clean, photo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="samples")
    ap.add_argument("--font", default=os.path.join(os.path.dirname(__file__), "Sriracha.ttf"))
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    clean, photo = make_page(a.font, a.seed)
    cv2.imwrite(os.path.join(a.out, "synthetic_clean.png"), clean)
    cv2.imwrite(os.path.join(a.out, "synthetic_photo.png"), photo)
    # แบบพื้นดำ (dark mode): กลับสี → หมึกสว่างบนพื้นเกือบดำ
    dark = np.maximum(255 - clean, np.array([30, 28, 26], np.uint8))
    cv2.imwrite(os.path.join(a.out, "synthetic_dark.png"), dark)

    import pymupdf as fitz
    ok, buf = cv2.imencode(".jpg", photo, [cv2.IMWRITE_JPEG_QUALITY, 85])
    doc = fitz.open()
    h, w = photo.shape[:2]
    pg = doc.new_page(width=w * 72 / 300, height=h * 72 / 300)
    pg.insert_image(pg.rect, stream=buf.tobytes())
    doc.save(os.path.join(a.out, "synthetic_scan.pdf"))
    print("saved to", a.out)


if __name__ == "__main__":
    main()