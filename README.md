# Rewrite Reborn

จัดระเบียบลายมือภาษาไทย ภาษาอังกฤษ และตัวเลขอารบิก ด้วยการแปลงเชิงเส้น โดยรักษาเอกลักษณ์ลายมือเดิม

## ติดตั้ง

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## ใช้งาน

### UI (แนะนำ)

```bash
streamlit run app.py
```

1. เลือกไฟล์ 1–3 ไฟล์ทางซ้าย
2. ใส่หน้าของ PDF ที่ต้องการ เช่น `1`, `1,3`, `2-4`
3. ปรับพารามิเตอร์ แล้วกด **รัน pipeline**
4. ดูผลได้ 4 แท็บ

| แท็บ | ใช้ทำอะไร |
|---|---|
| เทียบก่อน/หลัง | มี 3 มุมมอง: เลื่อนเทียบ (มีซูม), ซ้าย-ขวา, ซ้อนทับ (แดง = เดิม, น้ำเงิน = ใหม่) |
| Debug | baseline, x-height, กลุ่มตัวอักษร, เส้นโยงสระ/วรรณยุกต์ไปตัวเจ้าของ, non-text |
| ขั้นตอนเตรียมภาพ | ภาพผลของแต่ละขั้นใน A |
| ตัวชี้วัดละเอียด | ตารางก่อน/หลัง, จำนวน, เวลาที่ใช้ |

ผลที่ได้เมื่อรันจะถูก cache ไว้ ถ้ากดรันซ้ำด้วยพารามิเตอร์เดิมจะไม่คำนวณใหม่

### Command line

```bash
python run.py notes.pdf
python run.py a.pdf b.png c.pdf --pages 1-2
python run.py notes.pdf --alpha 0.6 --no-spacing
python run.py notes.pdf --set ma_window=7 --set slant_target=vertical
python run.py notes.pdf --ocr                # ต้องมี tesseract (+ ภาษา tha ถ้าจะวัดภาษาไทย)
```

ผลลัพธ์ของแต่ละหน้าอยู่ใน `outputs/<ชื่อไฟล์>_p<หน้า>/`:

| ไฟล์ | เนื้อหา |
|---|---|
| `input.png` | ภาพเข้าหลังแก้หน้าเอียง |
| `output.png` | ผลลัพธ์ |
| `compare.png` | ภาพเทียบซ้าย-ขวา |
| `debug.png` | ภาพตรวจสอบ |
| `metrics.json` | ตัวชี้วัด + พารามิเตอร์ที่ใช้ |

### ไฟล์ GoodNotes

ไฟล์ `.goodnotes` อ่านตรงไม่ได้ ให้ export เป็น PDF จากแอปก่อน (Share → Export → PDF) ระบบจะตรวจเองว่าเป็น PDF แบบ vector

## โครงสร้างโค้ด ↔ ขั้นตอน pipeline

```
rr/
  config.py       พารามิเตอร์ทั้งหมด (ค่าขนาดเป็นสัดส่วนของ h)
  io_loader.py    A1  นำเข้า PDF/ภาพ, แยกประเภท (pdf_vector / pdf_digital / pdf_scan / img_digital / img_photo)
  preprocess.py   A2  ปรับแสง (หารด้วยพื้นหลังจาก closing) → ไฮไลต์/เงาหายไป
                  A3  Gaussian blur (เฉพาะสแกน/ภาพถ่าย) → Adaptive Threshold
                  A4  หาเส้นบรรทัดกระดาษ (Hough + least squares) → ลบ → แก้หน้าเอียง
  components.py   B1  contour + hierarchy (outer + hole) = glyph
                  B2  h (ความสูงมัธยฐาน), ความหนาเส้น
                  B3  แยก text / non-text / noise
  layout.py       B4  จัดบรรทัด (line tracking) + รวมบรรทัดที่ต่อกัน + ยุบบรรทัดสั้นที่เป็นสระ/วรรณยุกต์
                  B5  baseline ด้วย normal equation (AᵀA)⁻¹Aᵀy + ตัด outlier, x-height
                  B6  แบ่งโซนด้วย vector projection
                  B7  ผูกสระ/วรรณยุกต์กับตัวฐาน ส่วนเครื่องหมายยืนเดี่ยว (= - .) เป็นหน่วยของตัวเอง
                  B8  กลุ่มหยาบ mid / asc / desc / tall
                  B9  ตัวติดกัน
                  B10 ช่องว่างตัว/คำ (1D k-means)
  correction.py   C1  มุมเอียงจากเส้นเกือบตั้ง (direction vector), แก้รายตัวแบบ shrinkage เข้าหาค่าของคำ
                  C2  (Primary OBB ใช้ใน debug)
                  C3  ขนาดเป้าหมาย = มัธยฐานของกลุ่ม
                  C4  ช่องไฟเป้าหมาย = มัธยฐานของกลุ่ม small/large
                  C5  บรรทัดสั้น → ไม่แก้
                  C6  T = Tr(dx) · [I + α(Tr(dy)·K·S − I)]
  geometry.py     เครื่องมือ LA: covariance, eigen, OBB, LSQ, affine 3×3, circulant MA, cosine similarity
  render.py       D1–D4 แปลง → resample → moving average (คงมุมแหลม) → วาด AA → วาง non-text ที่เดิม
  evaluate.py     E   ตัวชี้วัดก่อน/หลัง + OCR (ไม่บังคับ)
  pipeline.py     รวมทุกขั้นสำหรับหน้าเดียว
app.py            UI (Streamlit)
run.py            CLI
```

## พารามิเตอร์ที่ควรลองจูนก่อน

| พารามิเตอร์ | ผล |
|---|---|
| `alpha` | ความแรงรวม: ยิ่งสูงยิ่งเป็นระเบียบ แต่เปลี่ยนรูปมากขึ้น |
| `slant_shrink_n0` | ยิ่งน้อยยิ่งแก้เอียงรายตัวแรง |
| `slant_target` | `page_median` คงความเอียงประจำตัวผู้เขียน, `vertical` ตั้งตรง |
| `thresh_C` | ภาพจางหรือหมึกหาย → ลด, มี noise เยอะ → เพิ่ม |
| `mark_height_h` | สระ/วรรณยุกต์ถูกนับเป็นตัวฐาน → เพิ่ม, ตัวเล็กถูกนับเป็น mark → ลด |
| `nontext_height_h` / `nontext_width_h` | แผนภาพถูกนับเป็นตัวอักษร → ลด |

## ข้อจำกัดที่รู้แล้ว

- **PDF แบบ vector:** ตอนนี้ render เป็นภาพคมชัดแล้วเข้า pipeline ปกติ ยังไม่ได้ดึงเส้นปากกาจาก path โดยตรง
- **ตัวอักษรติดกัน:** ปรับเป็นหน่วยเดียว ไม่แยกตัว
- **ระยะห่างระหว่างบรรทัด:** ยังไม่ปรับ ปรับเฉพาะภายในบรรทัด
- **เส้นตารางหรือเส้นใต้ยาว:** ถ้ายาวเกิน 40% ของความกว้างหน้า จะถูกลบเหมือนเส้นบรรทัดกระดาษ ปิดได้ด้วย `remove_ruled_lines`
- **การผูก ำ:** นิคหิตผูกกับตัวที่อยู่ใต้มัน ซึ่งโดยมากถูก แต่ถ้าเขียนเยื้องไปทาง า มากอาจผูกผิดตัว