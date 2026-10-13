/* AstroPilote Clean UI Client */

const CURVE_COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#ec4899", "#a855f7"];

const appState = {
  llmMode: "api",
  health: null,
};

document.addEventListener("DOMContentLoaded", async () => {
  initTheme();
  initLlmModeToggle();
  initModal();

  await Promise.all([fetchHealth(), fetchPresets()]);

  const queryInput = document.getElementById("query-input");
  if (queryInput && queryInput.value.trim()) {
    await runQuery(queryInput.value.trim());
  }

  document.getElementById("query-form")?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const q = queryInput?.value.trim();
    if (q) await runQuery(q);
  });
});

function initTheme() {
  const btn = document.getElementById("btn-theme-toggle");
  const saved = localStorage.getItem("astropilote-theme") || "dark";
  document.documentElement.dataset.theme = saved;
  if (btn) {
    btn.textContent = saved === "dark" ? "☀️ Light" : "🌙 Dark";
    btn.addEventListener("click", () => {
      const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      localStorage.setItem("astropilote-theme", next);
      btn.textContent = next === "dark" ? "☀️ Light" : "🌙 Dark";
    });
  }
}

function initLlmModeToggle() {
  const buttons = document.querySelectorAll("[data-llm-mode]");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      appState.llmMode = btn.dataset.llmMode;
      buttons.forEach((b) => b.classList.toggle("active", b === btn));
      updateLlmStatusText();
    });
  });
}

function updateLlmStatusText() {
  const el = document.getElementById("llm-status-line");
  if (!el || !appState.health) return;
  const llm = appState.health.llm || {};
  if (appState.llmMode === "api") {
    el.textContent = llm.anthropic_ready
      ? `✓ Anthropic (${llm.anthropic_model}) active`
      : `Set ANTHROPIC_API_KEY in .env (using grounded synthesis)`;
  } else {
    el.textContent = llm.ollama_running
      ? `✓ Local Ollama (${llm.ollama_model}) connected`
      : `Ollama offline (using grounded synthesis)`;
  }
}

function initModal() {
  const dialog = document.getElementById("source-dialog");
  document.getElementById("btn-close-source-dialog")?.addEventListener("click", () => {
    dialog?.close();
  });

  document.body.addEventListener("click", async (e) => {
    const pill = e.target.closest("[data-doc-id]");
    if (pill) {
      await openCatalogModal(pill.dataset.docId);
    }
  });
}

async function fetchHealth() {
  try {
    const res = await fetch("/api/health");
    if (!res.ok) return;
    const data = await res.json();
    appState.health = data;
    if (data.llm?.configured_mode) {
      appState.llmMode = data.llm.configured_mode;
      document.querySelectorAll("[data-llm-mode]").forEach((b) => {
        b.classList.toggle("active", b.dataset.llmMode === appState.llmMode);
      });
    }
    updateLlmStatusText();
  } catch (_) {}
}

async function fetchPresets() {
  try {
    const res = await fetch("/api/presets");
    if (!res.ok) return;
    const data = await res.json();
    const listEl = document.getElementById("preset-list");
    if (!listEl) return;
    listEl.innerHTML = "";

    (data.presets || []).forEach((p, idx) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `preset-btn ${idx === 0 ? "active" : ""}`;
      btn.innerHTML = `
        <span class="preset-title">${escapeHtml(p.label)}</span>
        <span class="preset-sub">${escapeHtml(p.subtitle || "")}</span>
      `;
      btn.addEventListener("click", async () => {
        document.querySelectorAll(".preset-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        const input = document.getElementById("query-input");
        if (input) input.value = p.query;
        await runQuery(p.query);
      });
      listEl.appendChild(btn);
    });
  } catch (_) {}
}

async function runQuery(question) {
  const btn = document.getElementById("btn-submit");
  if (btn) {
    btn.disabled = true;
    btn.textContent = "Thinking...";
  }

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        llm_mode: appState.llmMode,
      }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const payload = await res.json();
    renderResult(payload);
  } catch (err) {
    const body = document.getElementById("answer-body");
    if (body) body.innerHTML = `<p class="muted">Error: ${escapeHtml(String(err))}</p>`;
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.textContent = "Ask AstroPilote";
    }
  }
}

function renderResult(payload) {
  const { answer, telemetry, night, weather, visibility_objects, retrieved_chunks } = payload;

  // 1. Outcome Badge & Latency
  const badge = document.getElementById("badge-outcome");
  if (badge) {
    if (answer.abstained || telemetry.stop_reason !== "answered") {
      badge.className = "status-badge abstained";
      badge.textContent =
        telemetry.stop_reason === "refused" ? "⚠ Guard Refusal" : "⚠ Abstained (Out of Corpus)";
    } else {
      badge.className = "status-badge verified";
      badge.textContent = "✓ Verified Answer";
    }
  }
  const latEl = document.getElementById("meta-latency");
  if (latEl) {
    latEl.textContent = `${Number(telemetry.latency_s).toFixed(2)} s · ${(telemetry.tool_calls || []).length} tools`;
  }

  // 2. Cited Sources Pills
  const citedBar = document.getElementById("cited-sources-bar");
  if (citedBar) {
    citedBar.innerHTML = (answer.sources || [])
      .map((s) => `<button type="button" class="cite-pill" data-doc-id="${escapeHtml(s)}">${escapeHtml(s)}</button>`)
      .join("");
  }

  // 3. Night Conditions Grid (only when night data exists)
  const condGrid = document.getElementById("conditions-grid");
  if (condGrid) {
    if (night && night.astronomical_dark_start && !answer.abstained) {
      condGrid.classList.remove("hidden");
      document.getElementById("cond-darkness").textContent = night.has_astronomical_darkness
        ? `${night.astronomical_dark_start} – ${night.astronomical_dark_end}`
        : "No true darkness (Sun > -18°)";
      const moonSet = night.moon_set_local ? ` · sets ${night.moon_set_local}` : "";
      document.getElementById("cond-moon").textContent = `${night.moon_illumination_pct ?? 0}%${moonSet}`;
      document.getElementById("cond-clouds").textContent =
        weather && weather.cloud_cover_pct != null
          ? `${weather.cloud_cover_pct}% cloud cover`
          : "Clear sky";
    } else {
      condGrid.classList.add("hidden");
    }
  }

  // 4. Observation Plan Table & Altitude Chart
  const planSec = document.getElementById("plan-section");
  const planTbody = document.getElementById("plan-tbody");
  if (planSec && planTbody) {
    if (answer.plan && answer.plan.length > 0) {
      planSec.classList.remove("hidden");
      planTbody.innerHTML = answer.plan
        .map((item) => {
          const tStart = String(item.start).slice(0, 5);
          const tEnd = String(item.end).slice(0, 5);
          const dClass = `diff-${item.difficulty || "easy"}`;
          return `
            <tr>
              <td><code>${tStart} – ${tEnd}</code></td>
              <td>
                <strong>${escapeHtml(item.name)}</strong>
                <button type="button" class="cite-pill" data-doc-id="${escapeHtml(item.object_id)}" style="margin-left:0.35rem;">${escapeHtml(item.object_id)}</button>
              </td>
              <td><code>${Number(item.altitude_deg).toFixed(0)}° ${escapeHtml(item.direction)}</code></td>
              <td><span class="diff-badge ${dClass}">${escapeHtml(item.difficulty)}</span></td>
              <td>${escapeHtml(item.imaging_notes || "")}</td>
            </tr>
          `;
        })
        .join("");
      renderAltitudeChart(visibility_objects || {});
    } else {
      planSec.classList.add("hidden");
    }
  }

  // 5. Readable Prose Answer (strip redundant Night Summary line if conditions cards are visible)
  let rawAnswer = answer.answer || "";
  if (night && night.astronomical_dark_start) {
    rawAnswer = rawAnswer.replace(/^\*\*Night Summary:\*\*.*$/m, "").trim();
  }
  const ansBody = document.getElementById("answer-body");
  if (ansBody) {
    ansBody.innerHTML = formatProse(rawAnswer);
  }

  // 6. Collapsible Evidence Drawer
  const evSummary = document.getElementById("evidence-summary");
  if (evSummary) {
    evSummary.textContent = `Inspect Retrieved RAG Passages (${(retrieved_chunks || []).length}) & Tool Calls (${(telemetry.tool_calls || []).length})`;
  }

  const chunksList = document.getElementById("chunks-list");
  if (chunksList) {
    chunksList.innerHTML =
      !retrieved_chunks || retrieved_chunks.length === 0
        ? `<p class="muted">No chunks retrieved.</p>`
        : retrieved_chunks
            .map(
              (c) => `
              <div class="ev-item">
                <div class="ev-top">
                  <button type="button" class="cite-pill" data-doc-id="${escapeHtml(c.doc_id)}">${escapeHtml(c.doc_id)}</button>
                  <span class="muted">${escapeHtml(c.section)} · score ${Number(c.score).toFixed(3)}</span>
                </div>
                <div><strong>${escapeHtml(c.title)}</strong></div>
              </div>
            `
            )
            .join("");
  }

  const traceList = document.getElementById("tool-trace-list");
  if (traceList) {
    traceList.innerHTML =
      !telemetry.tool_calls || telemetry.tool_calls.length === 0
        ? `<p class="muted">0 tools called (handled by safety guard).</p>`
        : telemetry.tool_calls
            .map(
              (tc, idx) => `
              <div class="ev-item">
                <div class="ev-top">
                  <strong>${idx + 1}. <code>${escapeHtml(tc.name)}</code></strong>
                  <span class="muted">${(Number(tc.duration_s) * 1000).toFixed(1)} ms</span>
                </div>
                <div class="muted" style="font-family:var(--font-mono); font-size:0.72rem; word-break:break-all;">
                  ${escapeHtml(JSON.stringify(tc.arguments))}
                </div>
              </div>
            `
            )
            .join("");
  }
}

function renderAltitudeChart(visObjects) {
  const svg = document.getElementById("altitude-svg");
  const legend = document.getElementById("chart-legend");
  if (!svg || !legend) return;

  const entries = Object.entries(visObjects).filter(
    ([, obj]) => obj && Array.isArray(obj.slots) && obj.slots.length > 0 && obj.observable
  );
  if (entries.length === 0) {
    svg.innerHTML = "";
    legend.innerHTML = "";
    return;
  }

  const W = 720;
  const H = 210;
  const padL = 38;
  const padR = 18;
  const padT = 16;
  const padB = 26;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const slots = entries[0][1].slots;
  const n = slots.length;
  const xAt = (i) => padL + (i / Math.max(1, n - 1)) * plotW;
  const yAt = (alt) => {
    const c = Math.max(0, Math.min(90, Number(alt)));
    return padT + ((90 - c) / 90) * plotH;
  };

  let html = "";
  [20, 45, 70, 90].forEach((alt) => {
    const y = yAt(alt).toFixed(1);
    const isFloor = alt === 20;
    const stroke = isFloor ? "rgba(245, 158, 11, 0.55)" : "rgba(148, 163, 184, 0.16)";
    const dash = isFloor ? 'stroke-dasharray="5 4"' : "";
    html += `
      <line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}" stroke="${stroke}" stroke-width="1" ${dash} />
      <text x="${padL - 8}" y="${Number(y) + 3}" text-anchor="end" fill="#7c8ea6" font-family="JetBrains Mono, monospace" font-size="10">${alt}°</text>
    `;
  });

  slots.forEach((s, i) => {
    if (i % 3 === 0 || i === n - 1) {
      html += `<text x="${xAt(i).toFixed(1)}" y="${H - 6}" text-anchor="middle" fill="#7c8ea6" font-family="JetBrains Mono, monospace" font-size="10">${escapeHtml(s.time)}</text>`;
    }
  });

  legend.innerHTML = "";
  entries.slice(0, 5).forEach(([docId, obj], idx) => {
    const color = CURVE_COLORS[idx % CURVE_COLORS.length];
    const pts = obj.slots.map((s, i) => `${xAt(i).toFixed(1)},${yAt(s.altitude_deg).toFixed(1)}`);
    html += `<polyline fill="none" stroke="${color}" stroke-width="2.4" points="${pts.join(" ")}" />`;

    const item = document.createElement("span");
    item.className = "legend-item";
    item.dataset.docId = docId;
    item.innerHTML = `<span class="legend-dot" style="background:${color}"></span><span>${escapeHtml(docId)}</span>`;
    legend.appendChild(item);
  });

  svg.innerHTML = html;
}

async function openCatalogModal(docId) {
  const dialog = document.getElementById("source-dialog");
  const idEl = document.getElementById("modal-doc-id");
  const titleEl = document.getElementById("modal-doc-title");
  const bodyEl = document.getElementById("modal-doc-body");
  if (!dialog || !bodyEl) return;

  idEl.textContent = docId;
  titleEl.textContent = docId;
  bodyEl.innerHTML = `<p class="muted">Loading catalog sheet...</p>`;
  dialog.showModal();

  try {
    const res = await fetch(`/api/catalog/${encodeURIComponent(docId)}`);
    if (!res.ok) throw new Error(`Object ${docId} not found`);
    const data = await res.json();
    titleEl.textContent = data.title || docId;
    bodyEl.innerHTML = `
      <p style="margin-top:0; color:var(--text-muted); font-family:var(--font-mono); font-size:0.8rem;">
        Catalog: ${escapeHtml(data.catalog)} · Type: ${escapeHtml(data.object_type)} · Constellation: ${escapeHtml(data.constellation)}
        ${data.magnitude != null ? ` · Magnitude: ${data.magnitude}` : ""}
        ${data.size_major_arcmin != null ? ` · Size: ${data.size_major_arcmin}'` : ""}
      </p>
      <div class="modal-pre">${escapeHtml(data.full_text || "")}</div>
    `;
  } catch (err) {
    bodyEl.innerHTML = `<p class="muted">${escapeHtml(String(err))}</p>`;
  }
}

function formatProse(text) {
  let safe = escapeHtml(text);
  safe = safe.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  safe = safe.replace(/\*(.+?)\*/g, "<em>$1</em>");
  safe = safe.replace(
    /\[((?:M\d{3}|C\d{3}|NGC\d{4}|CON_[A-Z]{3}))\]/g,
    '<button type="button" class="cite-pill" data-doc-id="$1">$1</button>'
  );

  const lines = safe.split("\n");
  let html = "";
  let inList = false;
  for (const line of lines) {
    const t = line.trim();
    if (!t) continue;
    if (t.startsWith("- ")) {
      if (!inList) {
        html += "<ul>";
        inList = true;
      }
      html += `<li>${t.slice(2)}</li>`;
    } else {
      if (inList) {
        html += "</ul>";
        inList = false;
      }
      html += `<p>${t}</p>`;
    }
  }
  if (inList) html += "</ul>";
  return html;
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
