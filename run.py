"""
รัน pipeline จาก command line

ตัวอย่าง
    python run.py notes1.pdf
    python run.py notes1.pdf notes2.png notes3.pdf --pages 1-2
    python run.py notes1.pdf --pages 3 --alpha 0.6 --no-spacing
    python run.py notes.png --set ma_window=7 --set slant_target=vertical

ผลลัพธ์อยู่ที่ outputs/<ชื่อไฟล์>_p<หน้า>/
    input.png      ภาพเข้า (หลังแก้หน้าเอียง ตำแหน่งตรงกับ output)
    output.png     ผลลัพธ์
    compare.png    ภาพเทียบซ้าย-ขวา
    debug.png      ภาพตรวจสอบ (baseline, กลุ่มตัวอักษร, mark → เจ้าของ ฯลฯ)
    metrics.json   ตัวชี้วัดก่อน/หลัง + พารามิเตอร์ที่ใช้
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

from rr import Params, load_file, run_page, side_by_side
from rr.pipeline import SOURCE_LABEL


def _coerce(v: str, like):
    if isinstance(like, bool):
        return v.lower() in ("1", "true", "yes", "on")
    if isinstance(like, int):
        return int(v)
    if isinstance(like, float):
        return float(v)
    return v


def _num(x):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Rewrite Reborn: จัดระเบียบลายมือ")
    ap.add_argument("files", nargs="+", help="ไฟล์ PDF / PNG / JPG (แนะนำครั้งละ 1-3 ไฟล์)")
    ap.add_argument("--pages", default="1", help="หน้าของ PDF เช่น 1 | 1,3 | 2-4 | all  (ค่าเริ่มต้น: 1)")
    ap.add_argument("--out", default="outputs", help="โฟลเดอร์ผลลัพธ์")
    ap.add_argument("--alpha", type=float, help="ความแรงการแก้ 0-1")
    ap.add_argument("--dpi", type=int)
    for stage in ("slant", "scale", "baseline", "spacing", "smoothing"):
        ap.add_argument(f"--no-{stage}", action="store_true", help=f"ปิดขั้น {stage}")
    ap.add_argument("--ocr", action="store_true", help="วัด OCR confidence ก่อน/หลัง (ต้องมี tesseract)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="ตั้งพารามิเตอร์ใดก็ได้ใน rr/config.py เช่น --set ma_window=7")
    a = ap.parse_args(argv)

    if len(a.files) > 3:
        print(f"⚠ ใส่มา {len(a.files)} ไฟล์ (แนะนำ 1-3 ไฟล์ต่อครั้งเพื่อดูผลทีละชุด) — จะรันทั้งหมด")

    p = Params()
    if a.alpha is not None:
        p.alpha = a.alpha
    if a.dpi:
        p.dpi = a.dpi
    for stage in ("slant", "scale", "baseline", "spacing", "smoothing"):
        if getattr(a, f"no_{stage}"):
            setattr(p, f"enable_{stage}", False)
    p.ocr = p.ocr or a.ocr
    for kv in a.set:
        k, _, v = kv.partition("=")
        if not hasattr(p, k):
            sys.exit(f"ไม่รู้จักพารามิเตอร์ '{k}' (ดูใน rr/config.py)")
        setattr(p, k, _coerce(v, getattr(p, k)))

    os.makedirs(a.out, exist_ok=True)
    summary = []
    for f in a.files:
        try:
            pages = load_file(f, a.pages, p.dpi)
        except Exception as e:
            print(f"✗ {f}: {e}")
            continue
        for pg in pages:
            print(f"\n▶ {pg.name}  [{SOURCE_LABEL.get(pg.source_type, pg.source_type)}]")
            r = run_page(pg, p)
            d = os.path.join(a.out, r.name)
            os.makedirs(d, exist_ok=True)
            cv2.imwrite(os.path.join(d, "input.png"), r.input_aligned)
            cv2.imwrite(os.path.join(d, "output.png"), r.output)
            cv2.imwrite(os.path.join(d, "compare.png"), side_by_side(r.input_aligned, r.output))
            cv2.imwrite(os.path.join(d, "debug.png"), r.debug)
            rec = {"name": r.name, "source_type": r.source_type, "info": r.info,
                   "page_stats": r.page_stats, "correction": r.correction_info,
                   "metrics": r.metrics, "timings": r.timings, "notes": r.notes, "params": p.to_dict()}
            with open(os.path.join(d, "metrics.json"), "w", encoding="utf-8") as fh:
                json.dump(rec, fh, ensure_ascii=False, indent=2, default=float)

            c = r.metrics.get("counts", {})
            print(f"  h={r.page_stats['h']}px  หมุนแก้ {r.page_stats['deskew_deg']}°  "
                  f"บรรทัด {c.get('lines', 0)}  ตัวฐาน {c.get('base', 0)}  สระ/วรรณยุกต์ {c.get('marks', 0)}  "
                  f"non-text {c.get('nontext', 0)}  ({r.timings['total']}s)")
            for row in r.metrics.get("rows", []):
                imp = row["improve_pct"]
                imp_s = "" if np.isnan(imp) else f"  {'▼' if imp > 0 else '▲'} {abs(imp):.0f}%"
                print(f"  {row['label']:<38s} {_num(row['before'])} → {_num(row['after'])}{imp_s}")
            if "identity_cos" in r.metrics:
                print(f"  {'ความคงเอกลักษณ์ (cosine, 1 = เหมือนเดิม)':<38s} {r.metrics['identity_cos']:.4f}"
                      f"  (ต่ำสุด {r.metrics['identity_cos_min']:.4f})")
            if p.ocr:
                ob, oa = r.metrics.get("ocr_before", {}), r.metrics.get("ocr_after", {})
                if ob.get("ok"):
                    print(f"  OCR confidence ({ob['lang']}): {ob['mean_conf']:.1f} → {oa.get('mean_conf', 0):.1f}")
                else:
                    print(f"  OCR: {ob.get('msg')}")
            for n in r.notes:
                print(f"  ⚠ {n}")
            print(f"  → {d}/")
            summary.append(r.name)
    if not summary:
        sys.exit(1)


if __name__ == "__main__":
    main()