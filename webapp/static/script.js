(() => {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file");
  const pickBtn = document.getElementById("pick-btn");
  const uploadCard = document.getElementById("upload-card");
  const progressCard = document.getElementById("progress-card");
  const resultCard = document.getElementById("result-card");
  const errorCard = document.getElementById("error-card");
  const errorMsg = document.getElementById("error-msg");
  const filenameEl = document.getElementById("filename");
  const elapsedEl = document.getElementById("elapsed");
  const percentEl = document.getElementById("percent");
  const barFill = document.getElementById("bar-fill");
  const stepsList = document.getElementById("steps");
  const crossedText = document.getElementById("crossed-text");
  const crossedCount = document.getElementById("crossed-count");
  const borderlineText = document.getElementById("borderline-text");
  const borderlineCount = document.getElementById("borderline-count");
  const borderlineWrap = document.getElementById("borderline-wrap");
  const copyBtn = document.getElementById("copy-btn");
  const copyStatus = document.getElementById("copy-status");
  const mapImg = document.getElementById("map-img");
  const mapPending = document.getElementById("map-pending");
  const resetBtn = document.getElementById("reset-btn");
  const errorReset = document.getElementById("error-reset");

  let elapsedTimer = null;
  let startedAt = 0;
  let activeES = null;
  let stepEls = new Map();

  // ---- pick & drop ----
  pickBtn.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("click", (e) => {
    if (e.target === pickBtn) return;
    fileInput.click();
  });
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files[0]) submit(fileInput.files[0]);
  });
  ["dragenter", "dragover"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add("dragover"); })
  );
  ["dragleave", "drop"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.remove("dragover"); })
  );
  dropzone.addEventListener("drop", (e) => {
    const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) submit(f);
  });

  // ---- reset ----
  resetBtn.addEventListener("click", reset);
  errorReset.addEventListener("click", reset);

  function reset() {
    if (activeES) { activeES.close(); activeES = null; }
    if (elapsedTimer) { clearInterval(elapsedTimer); elapsedTimer = null; }
    fileInput.value = "";
    uploadCard.classList.remove("hidden");
    progressCard.classList.add("hidden");
    resultCard.classList.add("hidden");
    errorCard.classList.add("hidden");
    stepsList.innerHTML = "";
    stepEls.clear();
    barFill.style.width = "0%";
    percentEl.textContent = "0%";
    crossedText.value = "";
    crossedCount.textContent = "0";
    borderlineText.value = "";
    borderlineCount.textContent = "0";
    borderlineWrap.classList.add("hidden");
    mapImg.src = "";
    mapImg.classList.add("hidden");
    mapPending.classList.remove("hidden");
    copyStatus.textContent = "";
    copyBtn.classList.remove("copied");
    copyBtn.firstChild && (copyBtn.lastChild.textContent = " Kopiuj");
  }

  // ---- copy ----
  copyBtn.addEventListener("click", async () => {
    const txt = crossedText.value.trim();
    if (!txt) return;
    try {
      await navigator.clipboard.writeText(txt);
      copyBtn.classList.add("copied");
      copyStatus.textContent = "Skopiowano do schowka.";
      setTimeout(() => {
        copyBtn.classList.remove("copied");
        copyStatus.textContent = "";
      }, 1800);
    } catch {
      // fallback
      crossedText.select();
      document.execCommand("copy");
      copyStatus.textContent = "Skopiowano (fallback).";
    }
  });

  // ---- submit ----
  async function submit(file) {
    if (!/\.pdf$/i.test(file.name)) {
      showError("Wybierz plik PDF.");
      return;
    }
    if (file.size > 25 * 1024 * 1024) {
      showError("Plik jest za duży (limit 25 MB).");
      return;
    }
    uploadCard.classList.add("hidden");
    progressCard.classList.remove("hidden");
    filenameEl.textContent = file.name;
    startTimer();

    const fd = new FormData();
    fd.append("pdf", file);
    let res;
    try {
      res = await fetch("/analyze", { method: "POST", body: fd });
    } catch (e) {
      showError("Nie mogę połączyć się z serwerem.");
      return;
    }
    if (!res.ok) {
      let msg = "Błąd serwera";
      try { const j = await res.json(); msg = j.error || msg; } catch {}
      showError(msg);
      return;
    }
    const { job_id } = await res.json();
    streamEvents(job_id);
  }

  // ---- SSE ----
  function streamEvents(jobId) {
    const es = new EventSource(`/events/${jobId}`);
    activeES = es;
    es.addEventListener("message", (m) => {
      let ev;
      try { ev = JSON.parse(m.data); } catch { return; }
      handleEvent(ev);
    });
    es.addEventListener("error", () => {
      // SSE auto-reconnects, ale jeśli job zakończył się błędem, nie wskrzeszamy
      if (es.readyState === EventSource.CLOSED) return;
    });
  }

  function handleEvent(ev) {
    if (ev.type === "init") {
      stepsList.innerHTML = "";
      stepEls.clear();
      ev.steps.forEach((s) => {
        const li = document.createElement("li");
        li.dataset.key = s.key;
        li.innerHTML = `<span class="icon"></span><span class="label">${s.label}</span>`;
        stepsList.appendChild(li);
        stepEls.set(s.key, li);
      });
      // pierwszy krok jako aktywny
      const first = stepsList.querySelector("li");
      if (first) first.classList.add("active");
    } else if (ev.type === "step") {
      const li = stepEls.get(ev.step);
      if (!li) return;
      // odznacz wszystkie aktywne
      stepsList.querySelectorAll("li.active").forEach((x) => {
        if (x !== li) x.classList.remove("active");
      });
      if (ev.complete) {
        li.classList.remove("active");
        li.classList.add("done");
        // następny niezakończony krok zostaje aktywny
        const next = li.nextElementSibling;
        if (next && !next.classList.contains("done")) next.classList.add("active");
      } else {
        li.classList.add("active");
      }
    } else if (ev.type === "progress") {
      const p = Math.max(0, Math.min(100, ev.percent || 0));
      barFill.style.width = p + "%";
      percentEl.textContent = p + "%";
    } else if (ev.type === "done") {
      finishOk(ev);
    } else if (ev.type === "error") {
      showError(ev.message || "Nieznany błąd.");
    } else if (ev.type === "close") {
      if (activeES) { activeES.close(); activeES = null; }
      stopTimer();
    }
  }

  function finishOk(ev) {
    barFill.style.width = "100%";
    percentEl.textContent = "100%";
    // wszystkie kroki — done
    stepsList.querySelectorAll("li").forEach((li) => {
      li.classList.remove("active");
      li.classList.add("done");
    });
    stopTimer();

    resultCard.classList.remove("hidden");
    const crossed = ev.crossed || [];
    const borderline = ev.borderline || [];
    crossedText.value = crossed.join(", ");
    crossedCount.textContent = String(crossed.length);
    if (borderline.length) {
      borderlineWrap.classList.remove("hidden");
      borderlineText.value = borderline.join(", ");
      borderlineCount.textContent = String(borderline.length);
    }
    if (ev.image_url) {
      mapImg.src = ev.image_url + "?t=" + Date.now();
      mapImg.onload = () => {
        mapPending.classList.add("hidden");
        mapImg.classList.remove("hidden");
      };
    }
  }

  function showError(msg) {
    if (activeES) { activeES.close(); activeES = null; }
    stopTimer();
    progressCard.classList.add("hidden");
    resultCard.classList.add("hidden");
    uploadCard.classList.add("hidden");
    errorCard.classList.remove("hidden");
    errorMsg.textContent = msg;
  }

  function startTimer() {
    startedAt = Date.now();
    if (elapsedTimer) clearInterval(elapsedTimer);
    elapsedEl.textContent = "0 s";
    elapsedTimer = setInterval(() => {
      const s = Math.round((Date.now() - startedAt) / 1000);
      elapsedEl.textContent = s + " s";
    }, 250);
  }
  function stopTimer() {
    if (elapsedTimer) { clearInterval(elapsedTimer); elapsedTimer = null; }
  }
})();
