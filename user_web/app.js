/* Rewrite Reborn — หน้าเว็บสำหรับผู้ใช้ */
(() => {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const el = (tag, attrs = {}, ...kids) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") n.className = v;
      else if (k === "text") n.textContent = v;
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
      else if (v !== false && v != null) n.setAttribute(k, v === true ? "" : v);
    }
    kids.flat().forEach((c) => c != null && n.append(c));
    return n;
  };

  const state = {
    files: [],              // {fid, name, pages, kind, error, sel:Set, last}
    settings: {
      strength: "medium", fix_baseline: true, fix_slant: true, fix_size: true, fix_spacing: true,
      fix_shape: true, fix_lines: true, slant_style: "keep", output_bg: "paper", background: "auto",
      light_ink: false, sensitivity: 50, keep_math: true,
    },
    job: null, items: [], poll: null,
  };

  // ------------------------------------------------------------ toast
  function toast(msg, err = false) {
    const t = el("div", { class: "toast" + (err ? " err" : ""), text: msg });
    $("#toasts").append(t);
    setTimeout(() => t.remove(), err ? 6500 : 3200);
  }

  // ------------------------------------------------------------ hero demo: ตัวอักษรเบี้ยว → เข้าแนว
  function setupDemo() {
    // บรรทัดไทยมีสระบน-ล่างและวรรณยุกต์ (ตามพยัญชนะไปด้วย) · บรรทัดอังกฤษบอกว่ายังเป็นลายมือเดิม
    const lines = [["#demoLine", "จดเร็วแค่ไหน ก็อ่านง่ายขึ้น"], ["#demoLine2", "same handwriting, just neater"]];
    const seg = window.Intl && Intl.Segmenter ? new Intl.Segmenter("th", { granularity: "grapheme" }) : null;
    const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    const all = [];
    for (const [sel, text] of lines) {
      const box = $(sel);
      const parts = seg ? [...seg.segment(text)].map((s) => s.segment) : [...text];
      parts.forEach((g) => {
        const s = el("span", { class: "ch" + (g.trim() ? "" : " sp"), text: g.trim() ? g : " " });
        box.append(s);
        all.push(s);
      });
    }
    const paper = $(".scribble .paper");
    const messUp = () => { paper.classList.remove("is-tidy"); all.forEach((s) => {
      const r = (Math.random() - .5) * 22, y = (Math.random() - .5) * 18, sc = .82 + Math.random() * .36;
      s.style.transition = "none";
      s.style.transform = `translateY(${y}px) rotate(${r}deg) scale(${sc})`;
    }); };
    const tidy = () => {
      all.forEach((s, i) => {
        s.style.transition = "";
        s.style.transitionDelay = reduce ? "0ms" : `${i * 18}ms`;
        s.style.transform = "none";
      });
      // ตัวอักษรเข้าที่แล้ว → เส้นบรรทัดเปลี่ยนจากเส้นประเป็นเส้นทึบ
      setTimeout(() => paper.classList.add("is-tidy"), reduce ? 0 : 700 + all.length * 18);
    };
    messUp();
    if (reduce) { tidy(); return; }
    setTimeout(tidy, 900);
    paper.addEventListener("click", () => { messUp(); setTimeout(tidy, 500); });
  }

  // ------------------------------------------------------------ steps
  function setStep(n) {
    $$(".steps li").forEach((li) => {
      const s = +li.dataset.step;
      li.classList.toggle("is-active", s === n || (n === 2 && s === 3));
      li.classList.toggle("is-done", s < n && !(n === 2 && s === 3));
    });
  }

  // ------------------------------------------------------------ upload
  async function upload(fileList) {
    const files = [...fileList];
    if (!files.length) return;
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f));
    const drop = $("#drop");
    drop.classList.add("is-over");
    $("strong", drop).textContent = `กำลังอัปโหลด ${files.length} ไฟล์…`;
    try {
      const res = await fetch("/api/files", { method: "POST", body: fd });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      let added = 0;
      for (const f of data.files) {
        if (f.error) { toast(`${f.name}: ${f.error}`, true); state.files.push({ ...f, sel: new Set() }); continue; }
        state.files.push({ ...f, sel: new Set(Array.from({ length: f.pages }, (_, i) => i + 1)), last: null });
        added += 1;
      }
      if (added) toast(`เพิ่ม ${added} ไฟล์แล้ว · เลือกไว้ทุกหน้า แตะหน้าที่ไม่ต้องการเพื่อยกเลิก`);
    } catch (e) {
      toast("อัปโหลดไม่สำเร็จ ตรวจว่าเซิร์ฟเวอร์ยังทำงานอยู่ แล้วลองอีกครั้ง", true);
    } finally {
      drop.classList.remove("is-over");
      $("strong", drop).textContent = state.files.length ? "เพิ่มไฟล์อื่น" : "ลากไฟล์โน้ตมาวางที่นี่";
      renderFiles();
    }
  }

  function renderFiles() {
    const list = $("#fileList");
    list.replaceChildren();
    const has = state.files.length > 0;
    $("#workspace").hidden = !has;
    $("#hero").hidden = has;            // มีไฟล์แล้ว เก็บส่วนแนะนำออก ให้พื้นที่กับหน้าที่เลือก
    $("#drop").classList.toggle("compact", has);
    if (has) setStep(2);
    state.files.forEach((f, fi) => {
      if (f.error) {
        list.append(el("div", { class: "file has-error" },
          el("div", { class: "file-head" }, el("span", { class: "file-name", text: f.name }),
            el("div", { class: "file-actions" }, el("button", { class: "tiny-btn danger", type: "button", text: "นำออก", onclick: () => removeFile(fi) }))),
          el("div", { class: "file-error", text: f.error })));
        return;
      }
      const meta = el("span", { class: "file-meta", "data-fi": fi });
      const thumbs = el("div", { class: "thumbs" });
      const card = el("div", { class: "file" },
        el("div", { class: "file-head" },
          el("span", { class: "file-name", text: f.name }), meta,
          el("div", { class: "file-actions" },
            el("button", { class: "tiny-btn", type: "button", text: "เลือกทั้งหมด", onclick: () => { for (let i = 1; i <= f.pages; i++) f.sel.add(i); refreshSel(); } }),
            el("button", { class: "tiny-btn", type: "button", text: "ไม่เลือก", onclick: () => { f.sel.clear(); refreshSel(); } }),
            el("button", { class: "tiny-btn danger", type: "button", text: "นำออก", onclick: () => removeFile(fi) }))),
        thumbs);
      meta.textContent = `เลือก ${f.sel.size} จาก ${f.pages} หน้า`;
      for (let n = 1; n <= f.pages; n++) {
        const on = f.sel.has(n);
        const b = el("button", {
          class: "thumb", type: "button", "aria-pressed": on ? "true" : "false", "aria-label": `หน้า ${n}`, "data-fi": fi, "data-n": n,
          onclick: (ev) => {
            if (ev.target.closest(".zoom")) return;
            if (ev.shiftKey && f.last != null) {
              const [a, z] = [Math.min(f.last, n), Math.max(f.last, n)];
              const turnOn = !f.sel.has(n);
              for (let i = a; i <= z; i++) turnOn ? f.sel.add(i) : f.sel.delete(i);
            } else {
              f.sel.has(n) ? f.sel.delete(n) : f.sel.add(n);
            }
            f.last = n;
            refreshSel();
          },
        },
          el("div", { class: "img" }, el("img", { src: `/api/files/${f.fid}/pages/${n}/thumb?w=300`, alt: "", loading: "lazy" })),
          el("span", { class: "check", text: "✓" }),
          el("span", { class: "zoom", role: "button", tabindex: "0", "aria-label": `ดูหน้า ${n} ให้ใหญ่ขึ้น`, text: "⤢",
            onclick: () => openPagePreview(fi, n), onkeydown: (e) => { if (e.key === "Enter") openPagePreview(fi, n); } }),
          el("span", { class: "num" }, el("span", { text: `หน้า ${n}` })));
        thumbs.append(b);
      }
      list.append(card);
    });
    updateRunBtn();
  }

  // อัปเดตสถานะการเลือกโดยไม่สร้างภาพย่อใหม่
  function refreshSel() {
    $$(".thumb").forEach((t) => {
      const f = state.files[+t.dataset.fi];
      t.setAttribute("aria-pressed", f.sel.has(+t.dataset.n) ? "true" : "false");
    });
    $$(".file-meta").forEach((m) => {
      const f = state.files[+m.dataset.fi];
      m.textContent = `เลือก ${f.sel.size} จาก ${f.pages} หน้า`;
    });
    updateRunBtn();
  }

  function removeFile(i) {
    const f = state.files[i];
    if (f.fid) fetch(`/api/files/${f.fid}`, { method: "DELETE" });
    state.files.splice(i, 1);
    renderFiles();
    if (!state.files.length) setStep(1);
  }

  function selectedPages() {
    return state.files.filter((f) => f.fid).flatMap((f) => [...f.sel].sort((a, b) => a - b).map((p) => ({ fid: f.fid, page: p })));
  }

  function updateRunBtn() {
    const n = selectedPages().length;
    const btn = $("#runBtn");
    btn.disabled = n === 0;
    btn.textContent = n ? `จัดลายมือ ${n} หน้า` : "เลือกหน้าก่อน";
    $("#etaHint").textContent = n ? `ใช้เวลาประมาณ ${fmtTime(n * 5)}` : "";
  }
  const fmtTime = (s) => (s < 60 ? `${Math.max(5, Math.round(s / 5) * 5)} วินาที` : `${Math.round(s / 60)} นาที`);

  // ------------------------------------------------------------ page preview
  let pp = null;
  function openPagePreview(fi, n) {
    pp = { fi, n };
    const f = state.files[fi];
    $("#ppTitle").textContent = `${f.name} · หน้า ${n}`;
    $("#ppImg").src = `/api/files/${f.fid}/pages/${n}/thumb?w=1100`;
    syncPpBtn();
    $("#pagePreview").hidden = false;
    $("#ppClose").focus();
  }
  function syncPpBtn() {
    const f = state.files[pp.fi];
    $("#ppToggle").textContent = f.sel.has(pp.n) ? "ยกเลิกการเลือกหน้านี้" : "เลือกหน้านี้";
  }
  $("#ppToggle").addEventListener("click", () => {
    const f = state.files[pp.fi];
    f.sel.has(pp.n) ? f.sel.delete(pp.n) : f.sel.add(pp.n);
    syncPpBtn(); refreshSel();
  });
  $("#ppClose").addEventListener("click", () => { $("#pagePreview").hidden = true; });

  // ------------------------------------------------------------ settings
  const STRENGTH_HINT = {
    light: "ปรับนิดเดียว เหมาะกับลายมือที่ค่อนข้างเรียบร้อยอยู่แล้ว",
    medium: "ปรับพอให้เป็นระเบียบ และยังเห็นจังหวะการเขียนเดิม",
    strong: "จัดเต็มที่ ตัวอักษรเข้าแนวมากที่สุด",
  };
  $$(".seg").forEach((g) => g.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    $$("button", g).forEach((x) => x.setAttribute("aria-checked", x === b ? "true" : "false"));
    state.settings[g.dataset.name] = b.dataset.v;
    if (g.dataset.name === "strength") $("#strengthHint").textContent = STRENGTH_HINT[b.dataset.v];
  }));
  $$(".chip").forEach((c) => c.addEventListener("click", () => {
    const on = c.getAttribute("aria-pressed") !== "true";
    c.setAttribute("aria-pressed", on ? "true" : "false");
    state.settings[c.dataset.k] = on;
    if (!$$(".chip").some((x) => x.getAttribute("aria-pressed") === "true")) toast("ไม่ได้เลือกสิ่งที่จะจัดเลย ผลลัพธ์จะเหมือนต้นฉบับ");
  }));
  $("#sens").addEventListener("input", (e) => { state.settings.sensitivity = +e.target.value; });
  $("#lightInk").addEventListener("change", (e) => { state.settings.light_ink = e.target.checked; });
  $("#keepMath").addEventListener("change", (e) => { state.settings.keep_math = e.target.checked; });

  // ------------------------------------------------------------ run job
  $("#runBtn").addEventListener("click", startJob);
  async function startJob() {
    const pages = selectedPages();
    if (!pages.length) return;
    const btn = $("#runBtn");
    btn.disabled = true;
    try {
      const res = await fetch("/api/jobs", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pages, settings: state.settings }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || "เริ่มงานไม่ได้");
      const data = await res.json();
      state.job = data.job;
      state.items = [];
      buildResultShell(pages);
      setStep(4);
      clearInterval(state.poll);
      state.poll = setInterval(pollJob, 1000);
      pollJob();
    } catch (e) {
      toast(String(e.message || e), true);
    } finally {
      updateRunBtn();
    }
  }

  function setProgress(frac) {
    const p = $("#progPath");
    const L = p.getTotalLength();
    p.style.strokeDasharray = L;
    p.style.strokeDashoffset = L * (1 - Math.max(0.02, frac));
  }

  function buildResultShell(pages) {
    const sec = $("#results");
    sec.hidden = false;
    sec.classList.remove("is-done");
    $("#resTitle").textContent = "กำลังจัดลายมือ…";
    $("#resSub").textContent = `0 จาก ${pages.length} หน้า`;
    $("#pdfBtn").hidden = $("#zipBtn").hidden = $("#againBtn").hidden = true;
    $("#cancelBtn").hidden = false;
    setProgress(0);
    const cards = $("#cards");
    cards.replaceChildren();
    pages.forEach((pg, i) => {
      const f = state.files.find((x) => x.fid === pg.fid);
      cards.append(el("article", { class: "card", id: `card${i}` },
        el("div", { class: "card-head" },
          el("span", { class: "card-title", text: f.pages > 1 ? `${f.name} · หน้า ${pg.page}` : f.name }),
          el("div", { class: "badges" }, el("span", { class: "badge", text: "รอคิว" }))),
        el("div", { class: "compare" }, el("div", { class: "pending" }, penSvg(), el("span", { text: "รอคิว" })))));
    });
    sec.scrollIntoView({ behavior: "smooth", block: "start" });
  }
  const penSvg = () => {
    const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    s.setAttribute("viewBox", "0 0 34 18"); s.setAttribute("class", "pen");
    s.innerHTML = '<path d="M2 12c4-8 7-8 9 0s5 6 8 0 5-7 13-3"/>';
    return s;
  };

  async function pollJob() {
    if (!state.job) return;
    let data;
    try {
      const res = await fetch(`/api/jobs/${state.job}`);
      if (!res.ok) throw new Error();
      data = await res.json();
    } catch { return; }
    setProgress(data.done / data.total);
    $("#resSub").textContent = `${data.done} จาก ${data.total} หน้า`;
    data.items.forEach((it, i) => {
      const prev = state.items[i];
      if (prev && prev.status === it.status) return;
      state.items[i] = it;
      renderCard(i, it);
    });
    if (data.status !== "running") {
      clearInterval(state.poll);
      const ok = data.items.filter((x) => x.status === "done").length;
      const bad = data.items.filter((x) => x.status === "error").length;
      $("#results").classList.add("is-done");
      $("#resTitle").textContent = data.status === "cancelled" ? `หยุดแล้ว · เสร็จ ${ok} หน้า` : `เสร็จแล้ว ${ok} หน้า`;
      $("#resSub").textContent = bad ? `มี ${bad} หน้าที่จัดไม่สำเร็จ ดูรายละเอียดที่การ์ดของหน้านั้น` : "แตะภาพเพื่อดูเต็มจอและซูมเทียบ";
      $("#cancelBtn").hidden = true;
      $("#againBtn").hidden = false;
      if (ok) {
        $("#pdfBtn").href = `/api/jobs/${state.job}/download.pdf`;
        $("#zipBtn").href = `/api/jobs/${state.job}/download.zip`;
        $("#pdfBtn").hidden = $("#zipBtn").hidden = false;
        toast(`จัดลายมือเสร็จ ${ok} หน้า`);
      }
    }
  }

  function renderCard(i, it) {
    const card = $(`#card${i}`);
    if (!card) return;
    const badges = $(".badges", card);
    const cmp = $(".compare", card);
    if (it.status === "running") {
      badges.replaceChildren(el("span", { class: "badge mark", text: "กำลังจัด…" }));
      cmp.replaceChildren(el("div", { class: "pending" }, penSvg(), el("span", { text: "กำลังจัดลายมือหน้านี้" })));
      return;
    }
    if (it.status === "error") {
      badges.replaceChildren(el("span", { class: "badge err", text: "จัดไม่สำเร็จ" }));
      cmp.replaceChildren(el("div", { class: "pending", text: `หน้านี้จัดไม่สำเร็จ: ${it.error || "ไม่ทราบสาเหตุ"} · ลองเลื่อน "การจับเส้นหมึก" ในตั้งค่าเพิ่มเติม แล้วจัดใหม่` }));
      return;
    }
    if (it.status !== "done") return;
    const s = it.summary || {};
    const b = [];
    if (s.regularity_pct != null) b.push(el("span", { class: "badge good", text: s.regularity_pct >= 0 ? `เป็นระเบียบขึ้น ${s.regularity_pct}%` : "แทบไม่เปลี่ยน" }));
    if (s.identity_label) b.push(el("span", { class: "badge mark", text: s.identity_label }));
    b.push(el("span", { class: "badge", text: `${it.seconds} วินาที` }));
    badges.replaceChildren(...b);
    const head = $(".card-head", card);
    $(".card-actions", head)?.remove();
    head.append(el("div", { class: "card-actions" },
      el("button", { class: "tiny-btn", type: "button", text: "ดูเต็มจอ", onclick: () => openViewer(i) }),
      el("a", { class: "tiny-btn", href: `/api/jobs/${state.job}/img/${i}/after?fmt=png`, download: `${it.name}.png`, text: "ดาวน์โหลด PNG" })));
    const fig = (which, cap) => el("figure", { onclick: () => openViewer(i) },
      el("figcaption", { text: cap }),
      el("img", { src: `/api/jobs/${state.job}/img/${i}/${which}?w=900`, alt: `${cap} ${it.name}`, loading: "lazy" }));
    cmp.replaceChildren(fig("before", "ต้นฉบับ"), fig("after", "หลังจัด"));
    if (it.notes && it.notes.length) card.append(el("div", { class: "notes", text: it.notes.join(" · ") }));
  }

  $("#cancelBtn").addEventListener("click", () => state.job && fetch(`/api/jobs/${state.job}/cancel`, { method: "POST" }));
  $("#againBtn").addEventListener("click", () => {
    $("#settings").scrollIntoView({ behavior: "smooth", block: "center" });
    setStep(2);
  });

  // ------------------------------------------------------------ viewer (ซูม/เลื่อน 2 ฝั่งพร้อมกัน)
  const V = { i: 0, s: 1, x: 0, y: 0, w: 1, h: 1 };
  function doneIdx() { return state.items.map((it, i) => (it && it.status === "done" ? i : -1)).filter((i) => i >= 0); }
  function openViewer(i) {
    V.i = i;
    const it = state.items[i];
    $("#vTitle").textContent = it.name;
    const a = $("#vBefore"), b = $("#vAfter");
    let loaded = 0;
    const onload = () => { if (++loaded === 2) { V.w = b.naturalWidth; V.h = b.naturalHeight; fit(); } };
    a.onload = b.onload = onload;
    a.src = `/api/jobs/${state.job}/img/${i}/before?w=2200`;
    b.src = `/api/jobs/${state.job}/img/${i}/after?w=2200`;
    $("#viewer").hidden = false;
    document.body.style.overflow = "hidden";
    $("#vClose").focus();
  }
  function closeViewer() { $("#viewer").hidden = true; document.body.style.overflow = ""; }
  function apply() {
    const t = `translate(${V.x}px, ${V.y}px) scale(${V.s})`;
    $("#vBefore").style.transform = t;
    $("#vAfter").style.transform = t;
    $("#vZoom").textContent = `${Math.round(V.s * 100)}%`;
  }
  function fit() {
    const pane = $(".pane");
    const r = pane.getBoundingClientRect();
    V.s = Math.min(r.width / V.w, r.height / V.h);
    V.x = (r.width - V.w * V.s) / 2;
    V.y = Math.max(0, (r.height - V.h * V.s) / 2);
    apply();
  }
  function zoomAt(f, cx, cy) {
    const ns = Math.min(8, Math.max(0.05, V.s * f));
    V.x = cx - (cx - V.x) * (ns / V.s);
    V.y = cy - (cy - V.y) * (ns / V.s);
    V.s = ns;
    apply();
  }
  $$(".pane").forEach((pane) => {
    pane.addEventListener("wheel", (e) => {
      e.preventDefault();
      const r = pane.getBoundingClientRect();
      zoomAt(e.deltaY < 0 ? 1.15 : 1 / 1.15, e.clientX - r.left, e.clientY - r.top);
    }, { passive: false });
    let drag = null;
    pane.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY, ox: V.x, oy: V.y }; pane.setPointerCapture(e.pointerId); pane.classList.add("dragging"); });
    pane.addEventListener("pointermove", (e) => { if (!drag) return; V.x = drag.ox + e.clientX - drag.x; V.y = drag.oy + e.clientY - drag.y; apply(); });
    const end = () => { drag = null; pane.classList.remove("dragging"); };
    pane.addEventListener("pointerup", end);
    pane.addEventListener("pointercancel", end);
  });
  $$(".viewer-tools [data-z]").forEach((b) => b.addEventListener("click", () => {
    const z = +b.dataset.z;
    if (z === 0) return fit();
    const r = $(".pane").getBoundingClientRect();
    zoomAt(z > 0 ? 1.3 : 1 / 1.3, r.width / 2, r.height / 2);
  }));
  const step = (d) => {
    const ids = doneIdx();
    const k = ids.indexOf(V.i);
    if (ids.length) openViewer(ids[(k + d + ids.length) % ids.length]);
  };
  $("#vPrev").addEventListener("click", () => step(-1));
  $("#vNext").addEventListener("click", () => step(1));
  $("#vClose").addEventListener("click", closeViewer);
  window.addEventListener("resize", () => { if (!$("#viewer").hidden) fit(); });

  // ------------------------------------------------------------ dialogs & keyboard
  $("#howBtn").addEventListener("click", () => { $("#how").hidden = false; $("#howClose").focus(); });
  $("#howClose").addEventListener("click", () => { $("#how").hidden = true; });
  $("#how").addEventListener("click", (e) => { if (e.target.id === "how") $("#how").hidden = true; });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { closeViewer(); $("#pagePreview").hidden = true; $("#how").hidden = true; }
    if (!$("#viewer").hidden && e.key === "ArrowRight") step(1);
    if (!$("#viewer").hidden && e.key === "ArrowLeft") step(-1);
  });

  // ------------------------------------------------------------ drag & drop
  const drop = $("#drop");
  const input = $("#fileInput");
  input.addEventListener("change", () => { upload(input.files); input.value = ""; });
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  $("#addMore").addEventListener("click", () => input.click());
  ["dragenter", "dragover"].forEach((t) => window.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("is-over"); }));
  ["dragleave", "drop"].forEach((t) => window.addEventListener(t, (e) => {
    e.preventDefault();
    if (t === "dragleave" && e.relatedTarget) return;
    drop.classList.remove("is-over");
  }));
  window.addEventListener("drop", (e) => { if (e.dataTransfer?.files?.length) upload(e.dataTransfer.files); });

  setupDemo();
})();