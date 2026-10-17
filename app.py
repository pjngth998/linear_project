"""
UI สำหรับทดลอง pipeline และเทียบภาพก่อน/หลัง

รัน:
    streamlit run app.py
"""
from __future__ import annotations

import base64
import io
import json
import os
import zipfile

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from rr import Params, load_bytes, run_page, side_by_side
from rr.pipeline import SOURCE_LABEL
from rr.render import legend_text

st.set_page_config(page_title="Rewrite Reborn", page_icon="✍️", layout="wide")

MAX_FILES = 3

st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
div[data-testid="stMetricValue"] {font-size: 1.5rem;}
.rr-note {background:#fff6e5;border-left:4px solid #f0a020;padding:.5rem .8rem;border-radius:4px;margin:.3rem 0;}
.rr-badge {display:inline-block;padding:.1rem .55rem;border-radius:999px;background:#eef1f6;color:#334;font-size:.85rem;margin-left:.4rem;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
def rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img


def png_bytes(img: np.ndarray) -> bytes:
    return cv2.imencode(".png", img)[1].tobytes()


def b64jpg(img: np.ndarray, q: int = 90) -> str:
    return base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1]).decode()


def compare_slider(before: np.ndarray, after: np.ndarray, key: str, zoom: float = 1.0, height: int = 720):
    """ภาพซ้อนกัน ลากเส้นแบ่งซ้าย-ขวาเพื่อเทียบตำแหน่งเดียวกัน (ไม่ต้องติดตั้ง component เพิ่ม)"""
    h, w = after.shape[:2]
    a, b = b64jpg(before), b64jpg(after)
    html = f"""
    <div style="font-family:sans-serif;font-size:13px;color:#555;margin-bottom:6px">
      ลากแถบเลื่อนด้านล่างเพื่อเปิด/ปิดผลลัพธ์ • ซ้าย = ก่อน, ขวา = หลัง • เลื่อน (scroll) ในกรอบเพื่อดูส่วนอื่น
    </div>
    <div id="wrap-{key}" style="overflow:auto;height:{height - 70}px;border:1px solid #ddd;border-radius:6px;background:#fff">
      <div style="position:relative;width:{zoom * 100:.0f}%;aspect-ratio:{w}/{h}">
        <img src="data:image/jpeg;base64,{b}" style="position:absolute;inset:0;width:100%;height:100%">
        <img id="top-{key}" src="data:image/jpeg;base64,{a}"
             style="position:absolute;inset:0;width:100%;height:100%;clip-path:inset(0 50% 0 0)">
        <div id="bar-{key}" style="position:absolute;top:0;bottom:0;left:50%;width:2px;background:#e33;box-shadow:0 0 3px #e33"></div>
      </div>
    </div>
    <input id="rng-{key}" type="range" min="0" max="100" value="50" step="0.5" style="width:100%;margin-top:8px">
    <script>
      const r = document.getElementById("rng-{key}");
      const t = document.getElementById("top-{key}");
      const bar = document.getElementById("bar-{key}");
      r.addEventListener("input", () => {{
        t.style.clipPath = "inset(0 " + (100 - r.value) + "% 0 0)";
        bar.style.left = r.value + "%";
      }});
    </script>
    """
    if hasattr(st, "iframe"):          # Streamlit รุ่นใหม่
        st.iframe(html, height=height)
    else:                              # รุ่นเก่า
        components.html(html, height=height)


def overlay_diff(before: np.ndarray, after: np.ndarray, dark: bool = False) -> np.ndarray:
    """แดง = หมึกก่อน, น้ำเงิน = หมึกหลัง, ม่วงเข้ม = ทับกันพอดี (แสดงบนพื้นขาวเสมอ)"""
    gb = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ga = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if not dark:                 # กระดาษสว่าง: หมึก = ส่วนที่เข้ม → กลับค่าให้หมึกมีค่าสูง
        gb, ga = 255 - gb, 255 - ga
    gb = np.clip(gb - np.median(gb), 0, None)   # ตัดพื้นหลัง (ไม่ว่าจะสีอะไร) ออก
    ga = np.clip(ga - np.median(ga), 0, None)
    gb = np.clip(gb / max(gb.max(), 1) * 1.6, 0, 1)
    ga = np.clip(ga / max(ga.max(), 1) * 1.6, 0, 1)
    out = np.full(before.shape, 255, np.float32)
    out[..., 0] -= 200 * gb       # B ลด → แดง
    out[..., 1] -= 200 * np.maximum(gb, ga)
    out[..., 2] -= 200 * ga       # R ลด → น้ำเงิน
    return np.clip(out, 0, 255).astype(np.uint8)


def _code_version() -> str:
    """ลายนิ้วมือของโค้ดใน rr/ — ถ้าไฟล์ไหนถูกแก้ ค่านี้เปลี่ยน แคชเก่าจะไม่ถูกใช้ซ้ำ"""
    import hashlib
    h = hashlib.md5()
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rr")
    for f in sorted(os.listdir(d)):
        if f.endswith(".py"):
            with open(os.path.join(d, f), "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()


@st.cache_data(show_spinner=False, max_entries=24)
def process(data: bytes, filename: str, pages: str, params_json: str, code_version: str = ""):
    p = Params.from_dict(json.loads(params_json))
    out = []
    for pg in load_bytes(data, filename, pages, p.dpi):
        out.append(run_page(pg, p))
    return out


def outputs_pdf(results) -> bytes:
    import pymupdf
    doc = pymupdf.open()
    for r in results:
        img = r.output
        h, w = img.shape[:2]
        page = doc.new_page(width=w * 72 / 300, height=h * 72 / 300)
        page.insert_image(page.rect, stream=png_bytes(img))
    return doc.tobytes()


def results_zip(results) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for r in results:
            z.writestr(f"{r.name}/input.png", png_bytes(r.input_aligned))
            z.writestr(f"{r.name}/output.png", png_bytes(r.output))
            z.writestr(f"{r.name}/compare.png", png_bytes(side_by_side(r.input_aligned, r.output)))
            z.writestr(f"{r.name}/debug.png", png_bytes(r.debug))
            z.writestr(f"{r.name}/metrics.json", json.dumps(
                {"page_stats": r.page_stats, "correction": r.correction_info, "metrics": r.metrics,
                 "timings": r.timings, "notes": r.notes}, ensure_ascii=False, indent=2, default=float))
    return buf.getvalue()


def fmt(x, nd=3):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


# ------------------------------------------------------------------ sidebar
st.sidebar.title("✍️ Rewrite Reborn")
st.sidebar.caption("จัดระเบียบลายมือด้วยการแปลงเชิงเส้น โดยรักษาเอกลักษณ์เดิม")

uploads = st.sidebar.file_uploader(f"ไฟล์ลายมือ (สูงสุด {MAX_FILES} ไฟล์)", type=["pdf", "png", "jpg", "jpeg", "webp"],
                                   accept_multiple_files=True)
pages = st.sidebar.text_input("หน้าของ PDF", "1", help="เช่น 1 | 1,3 | 2-4 | all")

d = Params()
with st.sidebar.expander("ความแรงและขั้นที่ใช้", expanded=True):
    alpha = st.slider("α (ความแรงของการแก้)", 0.0, 1.0, d.alpha, 0.05,
                      help="T_α = I + α(T − I)   0 = ไม่แก้, 1 = แก้เต็มที่")
    c1, c2 = st.columns(2)
    en_slant = c1.checkbox("แก้เอียง", d.enable_slant)
    en_scale = c2.checkbox("ปรับขนาด", d.enable_scale)
    en_base = c1.checkbox("จัดเส้นบรรทัด", d.enable_baseline)
    en_space = c2.checkbox("จัดช่องไฟ", d.enable_spacing)
    en_smooth = c1.checkbox("ทำเส้นเรียบ", d.enable_smoothing)
    slant_target = st.radio("มุมเป้าหมาย", ["page_median", "vertical"], horizontal=True,
                            format_func=lambda v: "ความเอียงประจำตัว" if v == "page_median" else "ตั้งตรง")
    slant_level = st.radio("แก้เอียงระดับ", ["glyph", "word"], horizontal=True,
                           format_func=lambda v: "รายตัว (ถ่วงน้ำหนัก)" if v == "glyph" else "ทั้งคำ")

with st.sidebar.expander("เตรียมภาพ (A)"):
    dpi = st.select_slider("DPI ของ PDF", [150, 200, 250, 300, 400], d.dpi)
    flatten = st.checkbox("ปรับแสงให้เรียบ (ลบไฮไลต์/เงา)", d.flatten_background)
    blur = st.selectbox("Gaussian blur", ["auto", "on", "off"], index=0)
    bg_mode = st.radio("พื้นหลัง", ["auto", "light", "dark"], horizontal=True,
                       format_func={"auto": "ตรวจเอง", "light": "สว่าง", "dark": "มืด"}.get)
    thresh_C = st.slider("Threshold C", 2, 30, d.thresh_C, help="มาก = เก็บเฉพาะหมึกเข้ม")
    ink_max = st.slider("ความจางสูงสุดของหมึก", 100, 240, d.ink_max, 5,
                        help="พิกเซลที่จางกว่านี้ไม่นับเป็นหมึก (ตัดจุด dot grid / เส้นจาง) ปากกาสีอ่อนให้เพิ่มค่า")
    _bg_opts = ["paper", "auto", "page", "white"]
    out_bg = st.radio("พื้นของผลลัพธ์", _bg_opts, index=_bg_opts.index(d.output_bg), horizontal=True,
                      format_func={"auto": "อัตโนมัติ", "page": "สีพื้นเดิม", "white": "ขาว",
                                   "paper": "กระดาษเดิม (กริด/ไฮไลต์)"}.get)
    rm_lines = st.checkbox("ลบเส้นบรรทัดกระดาษ", d.remove_ruled_lines)
    deskew = st.checkbox("แก้หน้าเอียง", d.deskew)
    straighten = st.checkbox("หมุนทั้งหน้าให้ตรง (เมื่อกระดาษเอียง)", d.straighten_page,
                             help="ปิด = คงกรอบภาพต้นฉบับ กระดาษเอียงเท่าเดิม ตัวหนังสือขนานกับกระดาษ")
    rm_hl = st.checkbox("ตัดไฮไลต์ด้วยสี (เสริม)", d.remove_highlighter)

with st.sidebar.expander("เลย์เอาต์ (B)"):
    nontext_h = st.slider("non-text: สูงเกิน (×h)", 1.5, 6.0, d.nontext_height_h, 0.25)
    nontext_w = st.slider("non-text: กว้างเกิน (×h)", 3.0, 15.0, d.nontext_width_h, 0.5)
    mark_h = st.slider("ตัวเตี้ยกว่า (×h) = สระ/วรรณยุกต์", 0.3, 0.9, d.mark_height_h, 0.05)
    min_glyphs = st.slider("บรรทัดสั้นกว่านี้ไม่แก้", 2, 8, d.baseline_min_glyphs)
    st.markdown("**สิ่งที่ไม่ใช่ตัวอักษร**")
    freeze_reg = st.checkbox("คงข้อความในกรอบ / ภาพวาดไว้ตามเดิม", d.freeze_regions)
    prot_img = st.checkbox("ไม่แตะภาพที่แปะมา (รูป/ตาราง/ภาพหน้าจอ)", d.protect_images)
    prot_head = st.checkbox("ไม่แตะหัวเรื่องตัวใหญ่ / ตัวพิมพ์", d.protect_headings)
    neat = st.checkbox("ไม่แตะบรรทัดที่เรียบร้อยอยู่แล้ว", d.freeze_neat_lines)
    bind_frac = st.checkbox("ผูกเศษส่วน (dy/dx) เป็นหน่วยเดียว", d.bind_fractions,
                            help="เศษส่วนเลื่อนตามบรรทัดทั้งก้อน ไม่ถูกเอียง/ย่อขยาย/แยกช่องไฟ")
    freeze_math = st.checkbox("คงบรรทัดสูตรที่มีตัวซ้อนกันไว้ตามเดิม", d.freeze_math)
    bracket_h = st.slider("วงเล็บ/ปีกกา: สูงเกิน (×h)", 1.5, 4.0, d.bracket_h_h, 0.1)
    sparse_fill = st.slider("ภาพวาด: หมึกกินพื้นที่น้อยกว่า", 0.02, 0.3, d.sparse_fill, 0.01)
    freeze_frac = st.slider("freeze ทั้งบรรทัดเมื่อเกิน", 0.1, 0.8, d.freeze_line_frac, 0.05,
                            help="สัดส่วนตัวในบรรทัดที่อยู่ในกรอบ/สูตร")

with st.sidebar.expander("ปรับแก้และเส้นเรียบ (C–D)"):
    scale_rng = st.slider("ช่วงการย่อ/ขยาย", 0.5, 1.6, (d.scale_min, d.scale_max), 0.01)
    slant_clip = st.slider("แก้มุมได้ไม่เกิน (องศา)", 5.0, 40.0, d.slant_clip_deg, 1.0)
    shrink_n0 = st.slider("n₀ ของการแก้เอียงรายตัว", 5.0, 200.0, d.slant_shrink_n0, 5.0,
                          help="w = n/(n + n₀)  ยิ่งน้อย ยิ่งเชื่อมุมของแต่ละตัว (เป็นระเบียบขึ้น แต่เปลี่ยนรูปมากขึ้น)")
    render_mode = st.radio("วิธีวาดผล", ["warp", "contour"], horizontal=True,
                           format_func={"warp": "ยกพิกเซลจริง (คงเส้นเดิม)", "contour": "เติมจาก contour (เส้นเรียบ)"}.get,
                           help="warp: ความหนาและความนุ่มของเส้นตามต้นฉบับ · contour: ผ่าน moving average แต่เส้นอาจหนาขึ้น")
    ma_window = st.select_slider("หน้าต่าง moving average (โหมด contour)", [3, 5, 7, 9, 11], d.ma_window)
    corner = st.slider("มุมแหลม (องศา) ที่ไม่ smooth", 20.0, 120.0, d.corner_deg, 5.0)

with st.sidebar.expander("ประเมินผล (E)"):
    ocr = st.checkbox("วัด OCR confidence (ต้องมี tesseract)", False)
    ocr_lang = st.text_input("ภาษา tesseract", d.ocr_lang)

params = Params(
    dpi=dpi, flatten_background=flatten, blur=blur, thresh_C=thresh_C, ink_max=ink_max, background_mode=bg_mode, output_bg=out_bg, remove_ruled_lines=rm_lines,
    deskew=deskew, straighten_page=straighten, remove_highlighter=rm_hl,
    nontext_height_h=nontext_h, nontext_width_h=nontext_w, mark_height_h=mark_h, baseline_min_glyphs=min_glyphs,
    bind_fractions=bind_frac, freeze_regions=freeze_reg, protect_images=prot_img, protect_headings=prot_head, freeze_neat_lines=neat, freeze_math=freeze_math, bracket_h_h=bracket_h, sparse_fill=sparse_fill,
    freeze_line_frac=freeze_frac,
    enable_slant=en_slant, enable_scale=en_scale, enable_baseline=en_base, enable_spacing=en_space,
    enable_smoothing=en_smooth, slant_target=slant_target, slant_level=slant_level, slant_clip_deg=slant_clip, slant_shrink_n0=shrink_n0,
    scale_min=scale_rng[0], scale_max=scale_rng[1], alpha=alpha, ma_window=ma_window, corner_deg=corner,
    render_mode=render_mode,
    ocr=ocr, ocr_lang=ocr_lang,
)

run = st.sidebar.button("▶ รัน pipeline", type="primary", width="stretch")

# ------------------------------------------------------------------ run
files: list[tuple[str, bytes]] = [(u.name, u.getvalue()) for u in (uploads or [])]
if len(files) > MAX_FILES:
    st.sidebar.warning(f"เลือกมา {len(files)} ไฟล์ จะใช้แค่ {MAX_FILES} ไฟล์แรก")
    files = files[:MAX_FILES]

if run:
    if not files:
        st.sidebar.error("ยังไม่ได้เลือกไฟล์")
    else:
        results, errors = [], []
        pj = json.dumps(params.to_dict(), sort_keys=True)
        prog = st.progress(0.0, text="กำลังประมวลผล…")
        for k, (name, data) in enumerate(files):
            prog.progress(k / len(files), text=f"กำลังประมวลผล {name}")
            try:
                results.extend(process(data, name, pages, pj, _code_version()))
            except Exception as e:
                errors.append(f"{name}: {e}")
        prog.empty()
        st.session_state["results"] = results
        st.session_state["errors"] = errors
        st.session_state["params_used"] = params.to_dict()

results = st.session_state.get("results", [])
for e in st.session_state.get("errors", []):
    st.error(e)

if not results:
    st.title("Rewrite Reborn")
    st.write("เลือกไฟล์ลายมือ 1–3 ไฟล์ทางซ้าย แล้วกด **รัน pipeline** "
             "ผลลัพธ์จะแสดงภาพก่อน/หลังแบบเลื่อนเทียบ ภาพ debug และตัวชี้วัด")
    st.markdown("""
| ขั้น | ทำอะไร |
|---|---|
| **A** เตรียมภาพ | ปรับแสง → blur → adaptive threshold → ลบเส้นบรรทัดกระดาษ → แก้หน้าเอียง |
| **B** เลย์เอาต์ | contour + hierarchy → แยก non-text → จัดบรรทัด → baseline (least squares) → โซน → ผูกสระ/วรรณยุกต์ → กลุ่มหยาบ → ช่องว่าง |
| **C** ปรับแก้ | shear (มุมเอียง) · scale (ขนาด) · translate (เส้นบรรทัด/ช่องไฟ) รวมเป็น affine เดียว แล้วแก้บางส่วนด้วย α |
| **D** วาดกลับ | resample → circulant moving average (คงมุมแหลม) → เติมสีหมึกเดิม → วาง non-text ที่เดิม |
| **E** ประเมินผล | σ จาก baseline, SD มุมเอียง, CV ความสูง/ช่องไฟ, cosine similarity รูปทรง |
""")
    st.stop()

# ------------------------------------------------------------------ results
top = st.columns([3, 1, 1])
top[0].subheader(f"ผลลัพธ์ {len(results)} หน้า")
top[1].download_button("⬇ ผลลัพธ์ทั้งหมด (PDF)", outputs_pdf(results), "rewrite_reborn_output.pdf",
                       "application/pdf", width="stretch")
top[2].download_button("⬇ ทุกอย่าง (ZIP)", results_zip(results), "rewrite_reborn_results.zip",
                       "application/zip", width="stretch")

tabs = st.tabs([r.name for r in results])
for ti, (tab, r) in enumerate(zip(tabs, results)):
    with tab:
        st.markdown(f"**{r.name}** <span class='rr-badge'>{SOURCE_LABEL.get(r.source_type, r.source_type)}</span>"
                    + ("<span class='rr-badge'>พื้นมืด</span>" if r.page_stats.get("dark_background") else "")
                    + f"<span class='rr-badge'>h ≈ {r.page_stats['h']} px</span>"
                    f"<span class='rr-badge'>หมุนแก้ {r.page_stats['deskew_deg']}°</span>"
                    f"<span class='rr-badge'>{r.timings['total']} s</span>", unsafe_allow_html=True)
        for n in r.notes:
            st.markdown(f"<div class='rr-note'>⚠ {n}</div>", unsafe_allow_html=True)

        rows = r.metrics.get("rows", [])
        mc = st.columns(len(rows) + 1) if rows else []
        short = {"baseline_sigma": "เบี่ยงจากเส้นบรรทัด", "slant_std_deg": "SD มุมเอียง (°)",
                 "height_cv": "CV ความสูง", "gap_cv": "CV ช่องไฟ"}
        for c, row in zip(mc, rows):
            delta = None
            if not (np.isnan(row["before"]) or np.isnan(row["after"])):
                delta = f"{row['after'] - row['before']:+.3f} ({-row['improve_pct']:+.0f}%)"
            c.metric(short[row["metric"]], fmt(row["after"]), delta, delta_color="inverse",
                     help=f"{row['label']} — ก่อน {fmt(row['before'])} · ยิ่งต่ำยิ่งเป็นระเบียบ")
        if mc:
            mc[-1].metric("ความคงเอกลักษณ์", fmt(r.metrics.get("identity_cos"), 4),
                          help="cosine similarity รูปทรงก่อน/หลัง (1 = เหมือนเดิม) "
                               f"· ต่ำสุดรายตัว {fmt(r.metrics.get('identity_cos_min'), 4)}")

        v1, v2, v3, v4 = st.tabs(["เทียบก่อน/หลัง", "Debug", "ขั้นตอนเตรียมภาพ", "ตัวชี้วัดละเอียด"])

        with v1:
            mode = st.radio("มุมมอง", ["เลื่อนเทียบ", "ซ้าย-ขวา", "ซ้อนทับ"], horizontal=True, key=f"mode{ti}")
            if mode == "เลื่อนเทียบ":
                zoom = st.slider("ซูม", 1.0, 4.0, 1.0, 0.25, key=f"zoom{ti}")
                compare_slider(r.input_aligned, r.output, key=f"c{ti}", zoom=zoom)
            elif mode == "ซ้าย-ขวา":
                a, b = st.columns(2)
                a.image(rgb(r.input_aligned), caption="ก่อน (ต้นฉบับ)", width="stretch")
                b.image(rgb(r.output), caption="หลัง", width="stretch")
                if r.page_stats.get("deskew_deg"):
                    st.caption(f"ระบบวิเคราะห์ในกรอบที่หมุนแก้ {r.page_stats['deskew_deg']}° "
                               "แล้วหมุนผลกลับเข้ากรอบต้นฉบับ (กระดาษไม่ถูกหมุน)")
            else:
                st.image(rgb(overlay_diff(r.input_aligned, r.output, r.page_stats.get("dark_background", False))), width="stretch",
                         caption="แดง = ตำแหน่งเดิม · น้ำเงิน = ตำแหน่งใหม่ · ม่วง = ทับกัน")
            d1, d2 = st.columns(2)
            d1.download_button("⬇ output.png", png_bytes(r.output), f"{r.name}_output.png", "image/png",
                               key=f"dlo{ti}", width="stretch")
            d2.download_button("⬇ compare.png", png_bytes(side_by_side(r.input_aligned, r.output)),
                               f"{r.name}_compare.png", "image/png", key=f"dlc{ti}", width="stretch")

        with v2:
            st.caption(legend_text())
            st.image(rgb(r.debug), width="stretch")

        with v3:
            names = {"inverted": "กลับสี (พื้นมืด) (A2)", "background": "พื้นหลังที่ประมาณได้ (A2)", "highlight_mask": "ไฮไลต์ที่ตัดออก (A2)",
                     "gray": "หลังปรับแสง + blur (A3)", "binary_raw": "Adaptive threshold (A3)",
                     "ruled_lines": "แถบเส้นบรรทัดที่ลบ (A4)", "binary": "ภาพขาวดำสุดท้าย (หลังแก้เอียง)"}
            keys = [k for k in names if k in r.steps]
            for i in range(0, len(keys), 2):
                cols = st.columns(2)
                for c, k in zip(cols, keys[i:i + 2]):
                    img = r.steps[k]
                    if k in ("binary_raw", "binary", "ruled_lines", "highlight_mask"):
                        img = 255 - img
                    c.image(img, caption=names[k], width="stretch")

        with v4:
            if rows:
                st.dataframe(
                    [{"ตัวชี้วัด": x["label"], "ก่อน": fmt(x["before"]), "หลัง": fmt(x["after"]),
                      "ดีขึ้น (%)": fmt(x["improve_pct"], 1)} for x in rows],
                    hide_index=True, width="stretch")
            if r.metrics.get("ocr_before"):
                ob, oa = r.metrics["ocr_before"], r.metrics["ocr_after"]
                if ob.get("ok"):
                    st.write(f"OCR confidence ({ob['lang']}): **{ob['mean_conf']:.1f} → {oa['mean_conf']:.1f}** "
                             f"(คำที่อ่านได้ {ob['words']} → {oa['words']})")
                    if ob.get("missing"):
                        st.caption(f"ไม่มีภาษา {', '.join(ob['missing'])} ใน tesseract — ติดตั้งเพิ่มเพื่อวัดภาษาไทย")
                else:
                    st.caption(f"OCR: {ob.get('msg')}")
            c1, c2, c3 = st.columns(3)
            c1.write("**จำนวน**")
            c1.json(r.metrics.get("counts", {}))
            c2.write("**ค่าของหน้า**")
            c2.json(r.page_stats)
            c3.write("**การปรับแก้ / เวลา (s)**")
            c3.json({**r.correction_info, "timings": r.timings})
            st.download_button("⬇ metrics.json", json.dumps(
                {"page_stats": r.page_stats, "correction": r.correction_info, "metrics": r.metrics,
                 "params": st.session_state.get("params_used", {})}, ensure_ascii=False, indent=2, default=float),
                f"{r.name}_metrics.json", "application/json", key=f"dlm{ti}")