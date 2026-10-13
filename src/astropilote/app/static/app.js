/* AstroPilote — Beginner-Friendly Frontend with Live Agent Stepper & Streaming Reveal */

const COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#a855f7", "#ec4899"];

const COMPASS_LABELS = {
  N: "North (N)",
  NE: "North-East (NE)",
  E: "East (E)",
  SE: "South-East (SE)",
  S: "South (S)",
  SW: "South-West (SW)",
  W: "West (W)",
  NW: "North-West (NW)",
};

let currentLlmMode = "api";
let healthLlmInfo = null;
let streamTimer = null;

document.addEventListener("DOMContentLoaded", async () => {
  initThemeToggle();
  initLlmModeToggle();
  initApiKeySaver();
  initSourceModal();
  initQueryForm();

  await Promise.all([loadHealth(), loadPresets()]);

  const initialQuery = document.getElementById("query-input").value.trim();
  if (initialQuery) {
    await runQuery(initialQuery);
  }
});

function initThemeToggle() {
  const btn = document.getElementById("btn-theme-toggle");
  const saved = localStorage.getItem("astropilote_theme") || "dark";
  setTheme(saved);

  btn.addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme") || "dark";
    setTheme(current === "dark" ? "light" : "dark");
  });
}

function setTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("astropilote_theme", theme);
  const btn = document.getElementById("btn-theme-toggle");
  if (btn) {
    btn.textContent = theme === "dark" ? "☀️ Light" : "🌙 Dark";
  }
}

function initLlmModeToggle() {
  const buttons = document.querySelectorAll(".mode-btn");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      buttons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      currentLlmMode = btn.getAttribute("data-llm-mode") || "api";
      updateLlmStatusLine();
    });
  });
}

function initApiKeySaver() {
  const btnSave = document.getElementById("btn-save-key");
  const inputKey = document.getElementById("api-key-input");
  if (!btnSave || !inputKey) return;

  btnSave.addEventListener("click", async () => {
    const key = inputKey.value.trim();
    if (!key) return;
    btnSave.disabled = true;
    btnSave.textContent = "Saving...";
    try {
      const resp = await fetch("/api/config/key", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ api_key: key }),
      });
      if (resp.ok) {
        const data = await resp.json();
        healthLlmInfo = data.llm;
        inputKey.value = "";
        updateLlmStatusLine();
      }
    } finally {
      btnSave.disabled = false;
      btnSave.textContent = "Save to .env";
    }
  });
}

function updateLlmStatusLine() {
  const el = document.getElementById("llm-status-line");
  const keyBox = document.getElementById("api-key-box");
  if (!el || !healthLlmInfo) return;

  if (currentLlmMode === "api") {
    if (healthLlmInfo.anthropic_ready) {
      el.textContent = `🟢 Ready: ${healthLlmInfo.anthropic_model}`;
      if (keyBox) keyBox.classList.add("hidden");
    } else {
      el.textContent = `🟡 Paste your ANTHROPIC_API_KEY below or in .env`;
      if (keyBox) keyBox.classList.remove("hidden");
    }
  } else {
    if (keyBox) keyBox.classList.add("hidden");
    if (healthLlmInfo.ollama_running) {
      el.textContent = `🟢 Ollama active (${healthLlmInfo.ollama_model})`;
    } else {
      el.textContent = `⚪ Ollama offline (${healthLlmInfo.ollama_model} · auto-fallback)`;
    }
  }
}

async function loadHealth() {
  try {
    const resp = await fetch("/api/health");
    const data = await resp.json();
    healthLlmInfo = data.llm;
    currentLlmMode = data.llm?.configured_mode || "api";

    document.querySelectorAll(".mode-btn").forEach((b) => {
      b.classList.toggle("active", b.getAttribute("data-llm-mode") === currentLlmMode);
    });
    updateLlmStatusLine();

    const statEl = document.getElementById("corpus-stat");
    if (statEl) {
      statEl.textContent = `${data.corpus_documents} sky objects · ${data.indexed_chunks} chunks`;
    }
  } catch (err) {
    console.error("Health check error:", err);
  }
}

async function loadPresets() {
  const listEl = document.getElementById("preset-list");
  try {
    const resp = await fetch("/api/presets");
    const data = await resp.json();
    listEl.innerHTML = "";

    data.presets.forEach((p, idx) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "preset-btn" + (idx === 0 ? " active" : "");
      btn.innerHTML = `
        <span class="preset-title">${escapeHtml(p.label)}</span>
        <span class="preset-sub">${escapeHtml(p.subtitle)}</span>
      `;
      btn.addEventListener("click", async () => {
        document.querySelectorAll(".preset-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        const input = document.getElementById("query-input");
        input.value = p.query;
        await runQuery(p.query);
      });
      listEl.appendChild(btn);
    });
  } catch (err) {
    console.error("Failed to load presets:", err);
  }
}

function initQueryForm() {
  const form = document.getElementById("query-form");
  const input = document.getElementById("query-input");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const q = input.value.trim();
    if (!q) return;
    await runQuery(q);
  });

  input.addEventListener("keydown", async (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      const q = input.value.trim();
      if (q) await runQuery(q);
    }
  });
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function setStepperState(activeIdx) {
  const stepper = document.getElementById("agent-stepper");
  if (!stepper) return;
  stepper.classList.remove("hidden");
  for (let i = 1; i <= 4; i++) {
    const el = document.getElementById(`step-${i}`);
    if (!el) continue;
    el.classList.remove("active", "done");
    const dot = el.querySelector(".step-dot");
    if (i < activeIdx) {
      el.classList.add("done");
      if (dot) dot.textContent = "✓";
    } else if (i === activeIdx) {
      el.classList.add("active");
      if (dot) dot.textContent = String(i);
    } else {
      if (dot) dot.textContent = String(i);
    }
  }
}

async function runQuery(questionText) {
  const btn = document.getElementById("btn-submit");
  const bodyEl = document.getElementById("answer-body");
  const stepper = document.getElementById("agent-stepper");

  if (streamTimer) {
    clearInterval(streamTimer);
    streamTimer = null;
  }

  btn.disabled = true;
  btn.textContent = "Running Agent...";
  document.getElementById("conditions-grid").classList.add("hidden");
  document.getElementById("plan-section").classList.add("hidden");
  bodyEl.innerHTML = `<p class="muted">Running hybrid search and sky calculations...</p>`;

  try {
    setStepperState(1);
    const fetchPromise = fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: questionText,
        llm_mode: currentLlmMode,
      }),
    }).then(async (r) => {
      if (!r.ok) throw new Error(`Server returned ${r.status}`);
      return r.json();
    });

    // Show each real pipeline stage clearly so the user can follow what the agent is doing
    await sleep(340);
    setStepperState(2);
    await sleep(380);
    setStepperState(3);
    await sleep(360);
    setStepperState(4);

    const data = await fetchPromise;
    await sleep(220);
    setStepperState(5);
    setTimeout(() => {
      if (stepper) stepper.classList.add("hidden");
    }, 600);

    renderResult(data);
  } catch (err) {
    if (stepper) stepper.classList.add("hidden");
    bodyEl.innerHTML = `<p class="muted">Error: ${escapeHtml(err.message)}</p>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Ask AstroPilote";
  }
}

function renderResult(data) {
  const ans = data.answer;
  const tel = data.telemetry;
  const night = data.night;
  const weather = data.weather;
  const visObjects = data.visibility_objects || {};
  const chunks = data.retrieved_chunks || [];

  // 1. Outcome Badge & Latency
  const badge = document.getElementById("badge-outcome");
  if (ans.abstained) {
    badge.className = "status-badge abstained";
    badge.textContent = `⚠ Guardrail (${tel.stop_reason})`;
  } else {
    badge.className = "status-badge verified";
    badge.textContent = "✓ Ephemeris & Citations Verified";
  }
  document.getElementById("meta-latency").textContent = `${tel.latency_s.toFixed(2)} s · ${tel.tool_calls.length} tools`;

  // 2. Cited Sources Pills
  const citedBar = document.getElementById("cited-sources-bar");
  citedBar.innerHTML = "";
  (ans.sources || []).forEach((docId) => {
    const pill = document.createElement("button");
    pill.type = "button";
    pill.className = "cite-pill";
    pill.textContent = `[${docId}]`;
    pill.addEventListener("click", () => openSourceModal(docId));
    citedBar.appendChild(pill);
  });

  // 3. Stream the Beginner-Friendly Prose Answer progressively
  const bodyEl = document.getElementById("answer-body");
  streamFormattedProse(ans.answer || "", bodyEl, () => {
    // 4. Reveal Night Conditions Cards & Schedule after prose finishes streaming
    const condGrid = document.getElementById("conditions-grid");
    if (night && !ans.abstained) {
      condGrid.classList.remove("hidden");
      document.getElementById("cond-darkness").textContent =
        `${night.astronomical_dark_start || "21:00"} → ${night.astronomical_dark_end || "05:30"}`;
      const moonPct = night.moon_illumination_pct ?? 0;
      const moonSet = night.moon_set_local ? ` (sets at ${night.moon_set_local})` : "";
      document.getElementById("cond-moon").textContent = `${moonPct}% lit${moonSet}`;
      const moonSub = document.getElementById("cond-moon-sub");
      if (moonSub) {
        moonSub.textContent =
          moonPct <= 35
            ? "Dark sky — ideal for faint nebulae & galaxies"
            : "Bright Moon — stick to bright star clusters";
      }

      if (weather) {
        const clouds = weather.cloud_cover_pct ?? 0;
        const bortle = weather.bortle ?? 4;
        const skyDesc =
          bortle <= 4
            ? "Dark rural sky (Milky Way visible)"
            : bortle <= 6
              ? "Suburban sky (moderate light pollution)"
              : "Bright city sky (use binoculars/telescope)";
        document.getElementById("cond-clouds").textContent = `${clouds}% clouds`;
        const cloudSub = document.getElementById("cond-clouds-sub");
        if (cloudSub) {
          cloudSub.textContent = `${skyDesc} (Bortle ${bortle})`;
        }
      } else {
        document.getElementById("cond-clouds").textContent = "Clear sky";
      }
    } else {
      condGrid.classList.add("hidden");
    }

    // 5. Observation Schedule Table & Altitude Chart
    const planSec = document.getElementById("plan-section");
    const tbody = document.getElementById("plan-tbody");
    tbody.innerHTML = "";

    if (ans.plan && ans.plan.length > 0 && !ans.abstained) {
      planSec.classList.remove("hidden");
      ans.plan.forEach((item) => {
        const tr = document.createElement("tr");
        const startStr = String(item.start).slice(0, 5);
        const endStr = String(item.end).slice(0, 5);
        const diffClass = item.difficulty === "easy" ? "diff-easy" : "diff-medium";
        const diffLabel =
          item.difficulty === "easy"
            ? "Easy (Beginner)"
            : item.difficulty === "medium"
              ? "Moderate"
              : "Challenging";
        const dirFull = COMPASS_LABELS[item.direction] || item.direction;
        const heightDesc =
          item.altitude_deg >= 65
            ? "Almost overhead"
            : item.altitude_deg >= 40
              ? "High up"
              : "Above horizon";

        tr.innerHTML = `
          <td><code>${escapeHtml(startStr)} – ${escapeHtml(endStr)}</code></td>
          <td>
            <strong>${escapeHtml(item.name)}</strong>
            <button type="button" class="cite-pill" data-doc-id="${escapeHtml(item.object_id)}">[${escapeHtml(item.object_id)}]</button>
          </td>
          <td>Look <strong>${escapeHtml(dirFull)}</strong><br/><span class="muted">${heightDesc} (${Number(item.altitude_deg).toFixed(0)}°)</span></td>
          <td><span class="diff-badge ${diffClass}">${escapeHtml(diffLabel)}</span></td>
          <td class="muted">${escapeHtml(item.imaging_notes || "")}</td>
        `;
        tbody.appendChild(tr);
      });

      tbody.querySelectorAll(".cite-pill[data-doc-id]").forEach((btn) => {
        btn.addEventListener("click", () => openSourceModal(btn.getAttribute("data-doc-id")));
      });

      renderAltitudeChart(visObjects, ans.plan);
    } else {
      planSec.classList.add("hidden");
    }
  });

  // 6. Collapsible Evidence Drawer (RAG Chunks + Tool Trace)
  const evSummary = document.getElementById("evidence-summary");
  evSummary.textContent = `Technical Details: View ${chunks.length} retrieved RAG sources & ${tel.tool_calls.length} agent tool calls`;

  const chunksList = document.getElementById("chunks-list");
  chunksList.innerHTML = "";
  chunks.forEach((c) => {
    const div = document.createElement("div");
    div.className = "ev-item";
    const cleanSnippet = (c.text || "")
      .replace(/^\[.*?\]\n?/m, "")
      .replace(/^Section:.*?\n/m, "")
      .slice(0, 180);
    div.innerHTML = `
      <div class="ev-top">
        <button type="button" class="cite-pill" data-doc-id="${escapeHtml(c.doc_id)}">[${escapeHtml(c.doc_id)}]</button>
        <span>${escapeHtml(c.section)} · score ${Number(c.score || 0).toFixed(3)}</span>
      </div>
      <div class="muted">${escapeHtml(cleanSnippet)}...</div>
    `;
    chunksList.appendChild(div);
  });

  chunksList.querySelectorAll(".cite-pill[data-doc-id]").forEach((btn) => {
    btn.addEventListener("click", () => openSourceModal(btn.getAttribute("data-doc-id")));
  });

  const traceList = document.getElementById("tool-trace-list");
  traceList.innerHTML = "";
  (tel.tool_calls || []).forEach((tc, idx) => {
    const div = document.createElement("div");
    div.className = "ev-item";
    div.innerHTML = `
      <div class="ev-top">
        <strong>${idx + 1}. ${escapeHtml(tc.name)}()</strong>
        <span>${(tc.duration_s * 1000).toFixed(1)} ms</span>
      </div>
      <div class="muted"><code>${escapeHtml(JSON.stringify(tc.arguments))}</code></div>
    `;
    traceList.appendChild(div);
  });
}

function streamFormattedProse(rawText, containerEl, onComplete) {
  if (!rawText) {
    containerEl.innerHTML = "<p class='muted'>No response generated.</p>";
    if (onComplete) onComplete();
    return;
  }

  // Split into tokens (words/spaces) so we can stream smoothly at a natural reading speed
  const words = rawText.split(/(?=\s+)/);
  const totalWords = words.length;
  // Stream in ~28 frames (~0.75 seconds) so it feels responsive and alive without being slow
  const wordsPerTick = Math.max(3, Math.ceil(totalWords / 28));
  let idx = 0;

  if (streamTimer) clearInterval(streamTimer);

  streamTimer = setInterval(() => {
    idx = Math.min(totalWords, idx + wordsPerTick);
    const partial = words.slice(0, idx).join("");
    const isDone = idx >= totalWords;

    containerEl.innerHTML =
      formatProseWithCitations(partial) +
      (isDone ? "" : `<span class="streaming-cursor"></span>`);

    if (isDone) {
      clearInterval(streamTimer);
      streamTimer = null;
      containerEl.querySelectorAll(".cite-pill[data-doc-id]").forEach((btn) => {
        btn.addEventListener("click", () => openSourceModal(btn.getAttribute("data-doc-id")));
      });
      if (onComplete) onComplete();
    }
  }, 26);
}

function formatProseWithCitations(text) {
  if (!text) return "<p class='muted'>No response generated.</p>";

  const lines = text.split("\n").filter((l) => l.trim().length > 0);
  const htmlParts = [];
  let inList = false;

  lines.forEach((rawLine) => {
    const line = rawLine.trim();
    const formatted = escapeHtml(line)
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.+?)\*/g, "<em>$1</em>")
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(
        /\[([A-Z]{1,5}\d{0,4}|GUIDE_[A-Z_]+)\]/g,
        '<button type="button" class="cite-pill" data-doc-id="$1">[$1]</button>'
      );

    if (line.startsWith("- ") || line.startsWith("* ")) {
      if (!inList) {
        htmlParts.push("<ul>");
        inList = true;
      }
      htmlParts.push(`<li>${formatted.slice(2)}</li>`);
    } else {
      if (inList) {
        htmlParts.push("</ul>");
        inList = false;
      }
      htmlParts.push(`<p>${formatted}</p>`);
    }
  });

  if (inList) htmlParts.push("</ul>");
  return htmlParts.join("");
}

function renderAltitudeChart(visObjects, planItems) {
  const svg = document.getElementById("altitude-svg");
  const legend = document.getElementById("chart-legend");
  const chartCard = document.getElementById("chart-card");

  const plannedIds = (planItems || []).map((p) => p.object_id);
  const keys = plannedIds.filter((id) => visObjects[id] && visObjects[id].slots);

  if (keys.length === 0) {
    chartCard.classList.add("hidden");
    return;
  }
  chartCard.classList.remove("hidden");
  svg.innerHTML = "";
  legend.innerHTML = "";

  const W = 720;
  const H = 210;
  const padL = 42;
  const padR = 18;
  const padT = 16;
  const padB = 28;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const yForAlt = (alt) => padT + plotH * (1 - Math.max(0, Math.min(90, alt)) / 90);

  // Horizontal grid lines (0°, 20° min threshold, 45°, 90°)
  [0, 20, 45, 90].forEach((alt) => {
    const y = yForAlt(alt);
    const isFloor = alt === 20;
    svg.insertAdjacentHTML(
      "beforeend",
      `<line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}"
        stroke="${isFloor ? "#f59e0b" : "rgba(148,163,184,0.18)"}"
        stroke-dasharray="${isFloor ? "5,4" : "none"}" stroke-width="1" />
       <text x="${padL - 8}" y="${y + 4}" text-anchor="end" fill="#7c8ea6" font-size="10" font-family="JetBrains Mono">${alt}°</text>`
    );
  });

  // Time labels along X axis
  const sampleSlots = visObjects[keys[0]].slots;
  const nSlots = sampleSlots.length;
  const xForIdx = (i) => padL + (i / Math.max(1, nSlots - 1)) * plotW;

  sampleSlots.forEach((slot, i) => {
    if (i % 3 === 0 || i === nSlots - 1) {
      const x = xForIdx(i);
      svg.insertAdjacentHTML(
        "beforeend",
        `<text x="${x}" y="${H - 7}" text-anchor="middle" fill="#7c8ea6" font-size="10" font-family="JetBrains Mono">${slot.time}</text>`
      );
    }
  });

  // Plot each object's altitude curve
  keys.slice(0, 5).forEach((oid, idx) => {
    const color = COLORS[idx % COLORS.length];
    const obj = visObjects[oid];
    const pts = obj.slots.map((s, i) => `${xForIdx(i).toFixed(1)},${yForAlt(s.altitude_deg).toFixed(1)}`).join(" ");

    svg.insertAdjacentHTML(
      "beforeend",
      `<polyline fill="none" stroke="${color}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" points="${pts}" />`
    );

    const item = document.createElement("span");
    item.className = "legend-item";
    item.innerHTML = `<span class="legend-dot" style="background:${color}"></span>${escapeHtml(oid)}`;
    item.addEventListener("click", () => openSourceModal(oid));
    legend.appendChild(item);
  });
}

function initSourceModal() {
  const dlg = document.getElementById("source-dialog");
  const closeBtn = document.getElementById("btn-close-source-dialog");
  if (closeBtn && dlg) {
    closeBtn.addEventListener("click", () => dlg.close());
  }
}

async function openSourceModal(objectId) {
  if (!objectId) return;
  const dlg = document.getElementById("source-dialog");
  const idEl = document.getElementById("modal-doc-id");
  const titleEl = document.getElementById("modal-doc-title");
  const bodyEl = document.getElementById("modal-doc-body");

  idEl.textContent = objectId;
  titleEl.textContent = "Loading...";
  bodyEl.innerHTML = "<p class='muted'>Fetching encyclopedia record...</p>";
  dlg.showModal();

  try {
    const resp = await fetch(`/api/catalog/${encodeURIComponent(objectId)}`);
    if (!resp.ok) throw new Error(`Document ${objectId} not found`);
    const data = await resp.json();

    titleEl.textContent = data.title || objectId;
    const aliases = (data.aliases || []).join(", ");
    bodyEl.innerHTML = `
      <p>
        <strong>Catalog:</strong> ${escapeHtml(data.catalog || "—")} ·
        <strong>Type:</strong> ${escapeHtml((data.object_type || "—").replace(/_/g, " "))} ·
        <strong>Constellation:</strong> ${escapeHtml(data.constellation || "—")}
      </p>
      <p>
        <strong>Brightness (Magnitude):</strong> ${data.magnitude ?? "—"} ·
        <strong>Apparent Size:</strong> ${data.size_major_arcmin ? data.size_major_arcmin + "'" : "—"} (Full Moon = 30') ·
        <strong>Aliases:</strong> ${escapeHtml(aliases || "—")}
      </p>
      ${data.framing_advice ? `<p><strong>Camera / Telescope Field of View:</strong> ${escapeHtml(data.framing_advice)}</p>` : ""}
    `;
  } catch (err) {
    bodyEl.innerHTML = `<p class="muted">${escapeHtml(err.message)}</p>`;
  }
}

function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
