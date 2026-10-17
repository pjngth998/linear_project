"""
พารามิเตอร์ทั้งหมดของ pipeline อยู่ที่นี่ที่เดียว

ค่าที่เกี่ยวกับขนาด ส่วนใหญ่กำหนดเป็น "สัดส่วนของ h" (h = ความสูงมัธยฐานของตัวอักษรในหน้า)
เพื่อให้ใช้ได้ทุกความละเอียดภาพและทุกขนาดลายมือ
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields


@dataclass
class Params:
    # ---------- A. นำเข้าและเตรียมภาพ ----------
    dpi: int = 300                      # ความละเอียดตอน rasterize PDF
    flatten_background: bool = True     # A2 ปรับแสงให้เรียบ (หารด้วยพื้นหลังที่ประมาณจาก closing) → ไฮไลต์หายไปด้วย
    flatten_kernel: int = 0             # ขนาด kernel ประมาณพื้นหลัง (0 = อัตโนมัติ) ต้องใหญ่กว่าความหนาเส้นหลายเท่า
    remove_highlighter: bool = False    # A2 (เสริม) ตัดไฮไลต์ด้วยกฎสี  ระวัง: อาจกินเส้นบรรทัดสีฟ้า/หมึกสีอ่อน
    highlight_s_min: int = 45           # ความอิ่มตัวสีขั้นต่ำที่ถือว่าเป็นไฮไลต์ (HSV S, 0-255)
    highlight_l_min: int = 185          # ความสว่างขั้นต่ำ (Lab L, 0-255) ไฮไลต์สว่าง หมึกสีเข้มกว่า
    blur: str = "auto"                  # A3 "auto" | "on" | "off"  (auto = เบลอเฉพาะสแกน/ภาพถ่าย)
    blur_ksize: int = 5                 # ขนาด Gaussian kernel (เลขคี่)
    thresh_block: int = 0               # ขนาดบล็อก adaptive threshold (0 = อัตโนมัติ)
    thresh_C: int = 12                  # ค่าลบออกจากค่าเฉลี่ยท้องถิ่น
    ink_max: int = 160                  # พิกเซลที่สว่างกว่านี้ (หลังปรับแสง/กลับสี) ไม่นับเป็นหมึก → ตัดจุด dot grid / เส้นจาง
    background_mode: str = "auto"       # "auto" | "light" | "dark"  พื้นหลังสว่าง (กระดาษ) หรือมืด (dark mode)
    dark_threshold: int = 110           # auto: ความสว่างมัธยฐานของหน้าต่ำกว่านี้ = พื้นมืด
    remove_ruled_lines: bool = True     # A4 ลบเส้นบรรทัดกระดาษ
    ruled_min_width_frac: float = 0.75  # เส้นต้องยาวรวมอย่างน้อยกี่ส่วนของความกว้างหน้า จึงจะถือเป็นเส้นบรรทัด
                                        # (ต่ำไปจะกินขอบกรอบ/ตารางที่วาดเอง)
    deskew: bool = True                 # A4 แก้หน้าเอียง
    straighten_page: bool = True        # True = หมุนทั้งหน้าให้ตรง (กระดาษ+ตัวหนังสือ) | False = คงกรอบภาพต้นฉบับ
    deskew_max_deg: float = 10.0
    deskew_step_deg: float = 0.2


    # ---------- B. วิเคราะห์เลย์เอาต์ ----------
    nontext_height_h: float = 3.0       # สูงเกินกี่ h = non-text
    nontext_width_h: float = 8.0        # กว้างเกินกี่ h = non-text
    nontext_line_ratio: float = 25.0    # λ1/λ2 เกินเท่านี้ + ยาวเกิน 2.5h = เส้นตรง (ลูกศร/เส้นใต้)
    frame_hole_w_h: float = 1.5         # รูกว้างกว่า (×h) และ
    frame_hole_h_h: float = 1.0         # สูงกว่า (×h) และกินพื้นที่ ≥ 35% ของกรอบ = กรอบล้อม (box/วงกลมใหญ่)
    bracket_h_h: float = 2.2            # สูงเกิน (×h) แต่
    bracket_w_h: float = 0.6            # แคบกว่า (×h) = วงเล็บใหญ่ / ปีกกา / ∫ / เส้นตั้ง
    sparse_fill: float = 0.12           # ชิ้นใหญ่ (สูง > 2h หรือกว้าง > 3h) ที่หมึกกินพื้นที่กรอบน้อยกว่านี้ = ภาพวาด/กราฟ
    image_fill: float = 0.45            # ชิ้นใหญ่ที่หมึกทึบเกินสัดส่วนนี้ของกรอบ = ภาพที่แปะมา
    image_min_w_h: float = 3.0          # ภาพต้องกว้างกว่า (×h)
    image_min_h_h: float = 2.0          # และสูงกว่า (×h)
    protect_images: bool = True         # ไม่แตะทุกอย่างที่อยู่ในภาพที่แปะมา
    protect_headings: bool = True       # ไม่แตะหัวเรื่องตัวใหญ่ (ตัวพิมพ์)
    heading_min_h: float = 1.8          # หัวเรื่อง = ชิ้นสูงกว่า (×h) เรียงเป็นแถว ฐานตรงกัน
    freeze_neat_lines: bool = True      # ไม่แตะบรรทัดที่เรียบร้อยอยู่แล้ว (ตัวพิมพ์)
    neat_resid_xh: float = 0.03         # ฐานเบี่ยงไม่เกิน (σ/x-height) …
    neat_height_cv: float = 0.08        # … และความสูงสม่ำเสมอ (CV) = เรียบร้อยแล้ว
    freeze_regions: bool = True         # ข้อความในกรอบ / ในภาพวาด → คงไว้ตามต้นฉบับ
    bind_scripts: bool = True           # ตัวยก/ตัวห้อย (t², v₀) ผูกกับตัวทางซ้าย
    bind_fractions: bool = True         # เศษส่วน (dy/dx) ผูกเป็นหน่วยแข็ง: เลื่อนตามบรรทัด ไม่เอียง/ไม่ย่อขยาย
    freeze_math: bool = True            # สูตร (เศษส่วน, ตัวซ้อนแนวตั้ง) → คงไว้ตามต้นฉบับ
    math_stack_min: int = 2             # จำนวนคู่ตัวที่ซ้อนแนวตั้งในบรรทัด ขั้นต่ำที่ถือว่าเป็นสูตร
    freeze_line_frac: float = 0.3       # ตัวในบรรทัดถูก freeze เกินสัดส่วนนี้ → freeze ทั้งบรรทัด
    noise_area_h2: float = 0.004        # พื้นที่น้อยกว่า (k·h²) = noise ทิ้งได้
    mark_height_h: float = 0.6          # ตัวที่เตี้ยกว่า (k·h) = ผู้สมัครเป็นสระ/วรรณยุกต์/จุด
    line_join_dx_h: float = 4.0         # ระยะห่างแนวนอนสูงสุดที่ยังนับเป็นบรรทัดเดียวกัน
    line_join_dy_h: float = 0.6         # ความต่างแนวตั้งสูงสุดที่ยังนับเป็นบรรทัดเดียวกัน
    line_merge_tol: float = 0.5         # รวมบรรทัดที่ต่อกันถ้า baseline ต่างไม่เกิน (k·x-height) — ลายมือที่บรรทัดลอยขึ้นลงต้องใช้ค่าสูง
    baseline_min_glyphs: int = 3        # บรรทัดที่มีตัวน้อยกว่านี้ = ไม่เชื่อถือ (fallback)
    baseline_outlier_k: float = 2.0     # ตัด outlier ที่ residual > k·σ
    baseline_iters: int = 3
    gap_small_xh: float = 0.35          # (บรรทัดที่แยกสองกลุ่มไม่ชัด) ช่องแคบกว่านี้ ×x-height = ช่องในคำ
    gap_large_xh: float = 1.0           # (บรรทัดที่แยกสองกลุ่มไม่ชัด) ช่องกว้างกว่านี้ ×x-height = ช่องคำ
    gap_amb_ratio: float = 1.6          # ช่องที่ห่างจากศูนย์กลางกลุ่มเกินเท่านี้ = ก้ำกึ่ง → ไม่ปรับ
    merged_width_ratio: float = 1.5     # กว้างกว่ามัธยฐาน x เท่า = ตัวอักษรติดกัน

    # ---------- C. คำนวณการปรับแก้ ----------
    enable_slant: bool = True
    enable_scale: bool = True
    enable_baseline: bool = True
    enable_spacing: bool = True
    enable_smoothing: bool = True
    slant_target: str = "page_median"   # "page_median" (คงความเอียงประจำตัวผู้เขียน) | "vertical"
    slant_max_dev_deg: float = 30.0     # ใช้เฉพาะช่วงเส้นที่ห่างจากแนวดิ่งไม่เกินกี่องศา
    slant_min_segments: int = 12        # จำนวน segment ขั้นต่ำต่อคำ (ไม่ถึงจะใช้ค่าของบรรทัด/หน้า)
    slant_level: str = "glyph"          # "glyph" = แก้รายตัวแบบถ่วงน้ำหนักกับค่าของคำ | "word" = แก้ทั้งคำเท่ากัน
    slant_shrink_n0: float = 40.0       # น้ำหนักรายตัว w = n/(n + n0)  (n = จำนวน segment ของตัวนั้น)
    slant_clip_deg: float = 25.0        # แก้มุมได้ไม่เกินกี่องศา
    obb_min_ratio: float = 1.5          # λ1/λ2 ขั้นต่ำที่จะใช้ Oriented Bounding Box
    scale_min: float = 0.75
    scale_max: float = 1.33
    scale_skip_outliers: bool = True    # ตัวที่ต้องย่อ/ขยายเกินช่วงนี้ = ไม่ปรับเลย (แทนการตัดที่ขอบช่วง)

    # ---- C3.5 รูปทรงสม่ำเสมอ (ต้นแบบจากลายมือของผู้เขียนเอง)
    shape_consistency: bool = True
    shape_strength: float = 0.6         # β: ดึงเข้าหาต้นแบบกี่ส่วน (0 = ไม่แตะ, 1 = เท่าต้นแบบเท่าที่ affine ทำได้)
    shape_cos_min: float = 0.92         # cosine similarity ขั้นต่ำที่ถือว่าเป็นตัวเดียวกัน
    shape_min_members: int = 3          # กลุ่มต้องมีอย่างน้อยกี่ตัวถึงสร้างต้นแบบ
    shape_points: int = 64              # จำนวนจุดที่ resample contour
    shape_max_dev: float = 0.25         # ต่างจากต้นแบบเกินนี้ (RMS/ขนาด) → ไม่แตะ
    shape_max_change: float = 0.2       # singular value ของ A ต้องอยู่ใน [1−x, 1+x]
    shape_max_rot_deg: float = 10.0
    shape_nonrigid: float = 0.5         # บิดเฉพาะจุด (ส่วนที่ affine ทำไม่ได้) เข้าหาต้นแบบกี่ส่วน
    shape_max_disp_h: float = 0.06      # บิดได้ไม่เกิน (× ความสูงตัว) ต่อจุด

    # ---- C6 ระยะบรรทัดและแนวหน้าบรรทัด
    regularize_spacing: bool = True     # ระยะห่างระหว่างบรรทัดเท่ากัน (ไม่ทำบนกระดาษมีเส้นบรรทัด)
    regularize_indent: bool = True      # จุดเริ่มบรรทัดที่ระดับย่อหน้าเดียวกันตรงกัน
    line_break_ratio: float = 1.45      # ระยะบรรทัดต่างจากค่ากลางเกินกี่เท่า = ขึ้นช่วงใหม่ (ไม่ปรับข้าม)
    line_max_shift_xh: float = 0.5      # เลื่อนบรรทัดขึ้น/ลงได้ไม่เกิน (× x-height)
    indent_tol_xh: float = 0.9          # จุดเริ่มห่างกันไม่เกินนี้ = ระดับย่อหน้าเดียวกัน
    resid_max_xh: float = 1.2           # ยกตัวที่ลอย/จมจากเส้นบรรทัดกลับได้ไม่เกิน (k·x-height) เกินนี้ถือว่าวัดพลาด ไม่แก้
    word_gap_alpha: float = 0.5         # ช่องระหว่างคำปรับเข้าหาค่ากลางเบากว่าช่องในคำ (คูณกับ α)
    word_letter_ratio: float = 2.5      # ช่องคำต้องกว้างอย่างน้อยกี่เท่าของช่องในคำ (ค่ากลางของบรรทัด)
    alpha: float = 0.8                  # ความแรงของการแก้: T_α = I + α(T − I)

    # ---------- D. แปลงและวาดกลับ ----------
    render_mode: str = "warp"           # "warp" = แปลงพิกเซลหมึกจริง (คงความหนาเส้นเดิม) | "contour" = เติมจาก contour + smoothing
    ma_window: int = 5                  # หน้าต่าง moving average (เลขคี่)
    corner_deg: float = 55.0            # มุมเลี้ยวเกินนี้ = มุมแหลม ไม่ smooth
    resample_step: float = 1.0          # ระยะห่างจุดหลัง resample (px)
    output_bg: str = "paper"            # "paper" | "auto" | "page" | "white"
                                        # auto = ภาพถ่าย/สแกนพื้นสว่าง → ขาว, อื่น ๆ → สีพื้นเดิม
                                        # paper = คงกระดาษเดิม (กริด เส้นบรรทัด ไฮไลต์) แล้ววางตัวหนังสือที่จัดแล้วทับ
    supersample: int = 1                # วาดที่ขนาดกี่เท่าแล้วย่อ (1 = ใช้ anti-alias ของ fillPoly อย่างเดียว ประหยัดหน่วยความจำ)

    # ---------- E. ประเมินผล ----------
    cos_points: int = 64
    ocr: bool = False                   # ใช้ tesseract วัด confidence ก่อน/หลัง
    ocr_lang: str = "tha+eng"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Params":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})