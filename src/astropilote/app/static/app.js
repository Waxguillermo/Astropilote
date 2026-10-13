/* AstroPilote Observatory Client — Interactive SVG Sky Chart, FOV Framing, & Google ADK Telemetry */

const CURVE_COLORS = ["#4f83ff", "#10b981", "#f59e0b", "#ec4899", "#a855f7", "#22d3ee"];

const state = {
  presets: [],
  retrievalAblation: {},
  chunkingExperiment: {},
  evalSummary: {},
  lastResponse: null,
};

document.addEventListener("DOMContentLoaded", async () => {
  initThemeSwitcher();
  initModals();
  initRigInputs();

  await Promise.all([loadHealth(), loadPresets()]);
  renderFovDiagram();

  // Execute initial showcase query automatically so the dashboard opens populated
  const queryInput = document.getElementById("query-input");
  if (queryInput && queryInput.value.trim()) {
    await runCopilotQuery(queryInput.value.trim());
  }

  const form = document.getElementById("query-form");
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const q = queryInput.value.trim();
    if (q) {
      await runCopilotQuery(q);
    }
  });
});

function initThemeSwitcher() {
  const buttons = document.querySelectorAll("[data-set-theme]");
  const currentTheme = document.documentElement.dataset.theme || "dark";
  buttons.forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.setTheme === currentTheme);
    btn.addEventListener("click", () => {
      const mode = btn.dataset.setTheme;
      document.documentElement.dataset.theme = mode;
      const meta = document.querySelector('meta[name="color-scheme"]');
      const schemeVal = mode === "light" ? "light" : "dark";
      if (meta) meta.content = schemeVal;
      try {
        localStorage.setItem("astropilote-theme", mode);
        localStorage.setItem("astropilote-color-scheme", schemeVal);
      } catch (_) {}
      buttons.forEach((b) => b.classList.toggle("active", b === btn));
    });
  });
}

function initModals() {
  const sourceDialog = document.getElementById("source-dialog");
  const closeSourceBtn = document.getElementById("btn-close-source-dialog");
  if (closeSourceBtn && sourceDialog) {
    closeSourceBtn.addEventListener("click", () => sourceDialog.close());
  }

  const benchDialog = document.getElementById("benchmarks-dialog");
  const openBenchBtn = document.getElementById("btn-open-benchmarks");
  const closeBenchBtn = document.getElementById("btn-close-benchmarks-dialog");
  if (openBenchBtn && benchDialog) {
    openBenchBtn.addEventListener("click", () => {
      renderBenchmarksModal();
      benchDialog.showModal();
    });
  }
  if (closeBenchBtn && benchDialog) {
    closeBenchBtn.addEventListener("click", () => benchDialog.close());
  }

  // Delegate clicks on [DOC_ID] citation buttons anywhere in the document
  document.body.addEventListener("click", async (e) => {
    const btn = e.target.closest("[data-doc-id]");
    if (btn) {
      const docId = btn.dataset.docId;
      await openSourceInspector(docId);
    }
  });
}

function initRigInputs() {
  const locEl = document.getElementById("input-location");
  const dateEl = document.getElementById("input-date");
  const apEl = document.getElementById("input-aperture");
  const flEl = document.getElementById("input-focal");

  const onRigChange = async () => {
    renderFovDiagram();
    await updateEphemerisPreview();
  };

  [locEl, dateEl, apEl, flEl].forEach((el) => {
    if (el) el.addEventListener("change", onRigChange);
  });
}

async function loadHealth() {
  try {
    const res = await fetch("/api/health");
    if (!res.ok) return;
    const data = await res.json();
    const modelBadge = document.getElementById("header-model-badge");
    const devBadge = document.getElementById("header-device-badge");
    if (modelBadge) {
      const ollamaUp = data.ollama_status && data.ollama_status.running;
      modelBadge.textContent = `${data.llm_model} · ${ollamaUp ? "Live ADK" : "Grounded ADK"}`;
    }
    if (devBadge) {
      devBadge.textContent = `${String(data.accelerator_device).toUpperCase()} · ${data.indexed_chunks} chunks`;
    }
  } catch (_) {}
}

async function loadPresets() {
  try {
    const res = await fetch("/api/presets");
    if (!res.ok) return;
    const data = await res.json();
    state.presets = data.presets || [];
    state.retrievalAblation = data.retrieval_ablation || {};
    state.chunkingExperiment = data.chunking_experiment || {};
    state.evalSummary = data.eval_summary || {};

    const listEl = document.getElementById("preset-list");
    if (!listEl) return;
    listEl.innerHTML = "";

    state.presets.forEach((p, idx) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `preset-btn ${idx === 0 ? "active" : ""}`;
      btn.innerHTML = `
        <div class="preset-btn-top">
          <span>${escapeHtml(p.label)}</span>
          <span class="preset-badge">${escapeHtml(p.badge)}</span>
        </div>
      `;
      btn.addEventListener("click", async () => {
        document.querySelectorAll(".preset-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        const queryInput = document.getElementById("query-input");
        if (queryInput) {
          queryInput.value = p.query;
        }
        await runCopilotQuery(p.query);
      });
      listEl.appendChild(btn);
    });
  } catch (_) {}
}

function renderFovDiagram() {
  const ap = parseFloat(document.getElementById("input-aperture")?.value || "80");
  const fl = parseFloat(document.getElementById("input-focal")?.value || "480");
  const fRatio = (fl / Math.max(10, ap)).toFixed(1);
  const fTag = document.getElementById("rig-fratio-tag");
  if (fTag) fTag.textContent = `f/${fRatio}`;

  const fovW = ((57.2958 * 60.0 * 23.5) / Math.max(10, fl)).toFixed(1);
  const fovH = ((57.2958 * 60.0 * 15.6) / Math.max(10, fl)).toFixed(1);
  const readout = document.getElementById("fov-readout");
  if (readout) readout.textContent = `${fovW}' × ${fovH}'`;

  const svg = document.getElementById("fov-svg");
  if (!svg) return;

  // Reference targets to compare inside APS-C sensor box: M31 (178'), M45 (110'), M27 (8')
  const sensorW = parseFloat(fovW);
  const m31Px = Math.min(115, Math.max(14, (178.0 / sensorW) * 90));
  const m45Px = Math.min(100, Math.max(10, (110.0 / sensorW) * 90));
  const m13Px = Math.min(80, Math.max(5, (20.0 / sensorW) * 90));

  svg.innerHTML = `
    <rect x="4" y="4" width="252" height="112" rx="6" fill="rgba(7,11,20,0.75)" stroke="rgba(125,162,255,0.22)" />
    <!-- APS-C Sensor Frame -->
    <rect x="40" y="14" width="180" height="92" rx="3" fill="rgba(79,131,255,0.08)" stroke="#4f83ff" stroke-width="1.5" stroke-dasharray="4 2" />
    <!-- M31 ellipse -->
    <ellipse cx="130" cy="60" rx="${m31Px}" ry="${(m31Px * 0.38).toFixed(1)}" transform="rotate(-28 130 60)"
      fill="rgba(245,158,11,0.16)" stroke="#f59e0b" stroke-width="1.2" />
    <!-- M45 circle -->
    <circle cx="130" cy="60" r="${(m45Px * 0.5).toFixed(1)}" fill="rgba(34,211,238,0.14)" stroke="#22d3ee" stroke-width="1.2" />
    <!-- M13 core -->
    <circle cx="130" cy="60" r="${Math.max(3, m13Px * 0.5).toFixed(1)}" fill="rgba(16,185,129,0.4)" stroke="#10b981" stroke-width="1.2" />
    <text x="46" y="26" fill="#9bb0d3" font-family="JetBrains Mono, monospace" font-size="9">APS-C Frame (${fovW}'×${fovH}')</text>
    <text x="46" y="100" fill="#f59e0b" font-family="JetBrains Mono, monospace" font-size="8.5">M31 (178') · M45 (110') · M13 (20')</text>
  `;
}

async function updateEphemerisPreview() {
  const loc = document.getElementById("input-location")?.value || "Fontainebleau";
  const obsDate = document.getElementById("input-date")?.value || "2026-10-17";
  const ap = parseFloat(document.getElementById("input-aperture")?.value || "80");
  const fl = parseFloat(document.getElementById("input-focal")?.value || "480");

  try {
    const res = await fetch("/api/ephemeris", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        place_name: loc,
        obs_date: obsDate,
        focal_length_mm: fl,
        aperture_mm: ap,
      }),
    });
    if (!res.ok) return;
    const data = await res.json();
    renderSkyChart(data.visibility?.objects || {}, data.night || {});
  } catch (_) {}
}

async function runCopilotQuery(question) {
  const statusPill = document.getElementById("agent-status-pill");
  const submitBtn = document.getElementById("btn-submit");
  const fl = parseFloat(document.getElementById("input-focal")?.value || "480");
  const ap = parseFloat(document.getElementById("input-aperture")?.value || "80");
  const obsDate = document.getElementById("input-date")?.value || "2026-10-17";

  if (statusPill) {
    statusPill.textContent = "Running ADK Pipeline...";
    statusPill.classList.add("running");
  }
  if (submitBtn) submitBtn.disabled = true;

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        date_override: obsDate,
        focal_length_mm: fl,
        aperture_mm: ap,
      }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const payload = await res.json();
    state.lastResponse = payload;
    renderCopilotResponse(payload);
  } catch (err) {
    const ansBody = document.getElementById("answer-body");
    if (ansBody) {
      ansBody.innerHTML = `<p style="color: var(--accent-rose);">Error communicating with AstroPilote backend: ${escapeHtml(String(err))}</p>`;
    }
  } finally {
    if (statusPill) {
      statusPill.textContent = "Ready";
      statusPill.classList.remove("running");
    }
    if (submitBtn) submitBtn.disabled = false;
  }
}

function renderCopilotResponse(payload) {
  const { answer, telemetry, intent, night, visibility_objects, retrieved_chunks } = payload;

  // 1. Update 5-Node Pipeline Visualizer & Telemetry Strip
  const isRefused = telemetry.stop_reason === "refused";
  document.querySelectorAll(".p-node").forEach((nodeEl) => {
    const name = nodeEl.dataset.node;
    nodeEl.classList.remove("active", "refused");
    if (isRefused && name === "guard") {
      nodeEl.classList.add("refused");
    } else if (!isRefused) {
      nodeEl.classList.add("active");
    }
  });

  const toolsCountEl = document.getElementById("node-tools-count");
  if (toolsCountEl) {
    toolsCountEl.textContent = `${(telemetry.tool_calls || []).length} tool calls`;
  }
  const valStatusEl = document.getElementById("node-validate-status");
  if (valStatusEl) {
    valStatusEl.textContent = answer.verified ? "PASS (4/4)" : "RETRY";
  }

  document.getElementById("metric-outcome").textContent = String(telemetry.stop_reason).toUpperCase();
  document.getElementById("metric-verified").textContent = answer.verified ? "✓ VERIFIED" : "⚠ UNVERIFIED";
  document.getElementById("metric-latency").textContent = `${Number(telemetry.latency_s).toFixed(2)} s`;
  document.getElementById("metric-tokens").textContent = `${telemetry.tokens_in} / ${telemetry.tokens_out}`;

  // 2. Conditions Banner
  const condBanner = document.getElementById("conditions-banner");
  const condText = document.getElementById("conditions-text");
  if (answer.conditions) {
    condBanner.classList.remove("hidden");
    condText.textContent = answer.conditions;
  } else {
    condBanner.classList.add("hidden");
  }

  // 3. Scheduled Night Plan Table
  const planSec = document.getElementById("plan-section");
  const planTbody = document.getElementById("plan-tbody");
  if (answer.plan && answer.plan.length > 0) {
    planSec.classList.remove("hidden");
    planTbody.innerHTML = answer.plan
      .map((item) => {
        const startStr = String(item.start).slice(0, 5);
        const endStr = String(item.end).slice(0, 5);
        const diffClass = `diff-${item.difficulty || "easy"}`;
        const srcBtns = (item.sources || [])
          .map((s) => `<button type="button" class="cite-btn" data-doc-id="${escapeHtml(s)}">[${escapeHtml(s)}]</button>`)
          .join(" ");
        return `
          <tr>
            <td><code>${startStr} – ${endStr}</code></td>
            <td><strong>${escapeHtml(item.name)}</strong></td>
            <td><code>${Number(item.altitude_deg).toFixed(1)}° ${escapeHtml(item.direction)}</code></td>
            <td><span class="diff-pill ${diffClass}">${escapeHtml(item.difficulty)}</span></td>
            <td>${escapeHtml(item.imaging_notes || "")}</td>
            <td>${srcBtns}</td>
          </tr>
        `;
      })
      .join("");
  } else {
    planSec.classList.add("hidden");
    planTbody.innerHTML = "";
  }

  // 4. Cited Sources Pills & Natural Language Synthesis
  const citedBar = document.getElementById("cited-sources-bar");
  citedBar.innerHTML = (answer.sources || [])
    .map((s) => `<button type="button" class="cite-btn" data-doc-id="${escapeHtml(s)}">📄 ${escapeHtml(s)}</button>`)
    .join(" ");

  const answerBody = document.getElementById("answer-body");
  answerBody.innerHTML = formatMarkdownWithCitations(answer.answer || "");

  // 5. Render Interactive SVG Night Sky Altitude Chart
  renderSkyChart(visibility_objects || {}, night || {});

  // 6. Render ADK Intent & Tool Trace
  const traceTag = document.getElementById("trace-count-tag");
  if (traceTag) traceTag.textContent = `${(telemetry.tool_calls || []).length} calls`;

  const intentBox = document.getElementById("intent-summary");
  if (intent) {
    intentBox.innerHTML = `
      <strong>Intent:</strong> <code>${escapeHtml(intent.request_type)}</code> ·
      <strong>Mode:</strong> <code>${escapeHtml(intent.mode)}</code>
      ${intent.location_name ? ` · <strong>Site:</strong> ${escapeHtml(intent.location_name)}` : ""}
      ${intent.target_date ? ` · <strong>Date:</strong> ${escapeHtml(String(intent.target_date))}` : ""}
    `;
  } else {
    intentBox.innerHTML = "";
  }

  const traceList = document.getElementById("tool-trace-list");
  if (!telemetry.tool_calls || telemetry.tool_calls.length === 0) {
    traceList.innerHTML = `<p class="muted">0 tools invoked (handled directly by <code>guard → refuse</code>).</p>`;
  } else {
    traceList.innerHTML = telemetry.tool_calls
      .map(
        (tc, i) => `
        <div class="trace-item">
          <div class="trace-item-top">
            <strong>#${i + 1} <code>${escapeHtml(tc.name)}</code></strong>
            <span class="muted">${(Number(tc.duration_s) * 1000).toFixed(1)} ms</span>
          </div>
          <div class="chunk-snippet"><code>${escapeHtml(JSON.stringify(tc.arguments))}</code></div>
        </div>
      `
      )
      .join("");
  }

  // 7. Render Retrieved Hybrid RAG Chunks
  const chunksTag = document.getElementById("chunks-count-tag");
  if (chunksTag) chunksTag.textContent = `${(retrieved_chunks || []).length} passages`;
  const chunksList = document.getElementById("chunks-list");
  if (!retrieved_chunks || retrieved_chunks.length === 0) {
    chunksList.innerHTML = `<p class="muted">No chunks retrieved for this turn.</p>`;
  } else {
    chunksList.innerHTML = retrieved_chunks
      .map(
        (ch) => `
        <div class="chunk-card">
          <div class="chunk-card-top">
            <button type="button" class="cite-btn" data-doc-id="${escapeHtml(ch.doc_id)}">[${escapeHtml(ch.doc_id)}]</button>
            <span>${escapeHtml(ch.section)} · <strong>${Number(ch.score).toFixed(3)}</strong></span>
          </div>
          <div style="font-weight:600; margin-bottom:0.2rem;">${escapeHtml(ch.title)}</div>
          <div class="chunk-snippet">${escapeHtml(ch.text)}</div>
        </div>
      `
      )
      .join("");
  }
}

function renderSkyChart(visObjects, nightInfo) {
  const svg = document.getElementById("altitude-svg");
  const legend = document.getElementById("chart-legend");
  const dateTag = document.getElementById("sky-chart-date-tag");
  if (!svg || !legend) return;

  if (dateTag && nightInfo.date) {
    dateTag.textContent = nightInfo.date;
  }

  const entries = Object.entries(visObjects).filter(([, obj]) => obj && Array.isArray(obj.slots) && obj.slots.length > 0);
  if (entries.length === 0) {
    svg.innerHTML = `<text x="230" y="115" text-anchor="middle" fill="#687ea3" font-family="Inter, sans-serif" font-size="12">No sky visibility curves for this query (Abstained / Refused).</text>`;
    legend.innerHTML = "";
    return;
  }

  const W = 460;
  const H = 230;
  const padL = 36;
  const padR = 16;
  const padT = 18;
  const padB = 28;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const slots = entries[0][1].slots;
  const numSlots = slots.length;
  const xForIdx = (i) => padL + (i / Math.max(1, numSlots - 1)) * plotW;
  // Altitude range: -10 deg to +90 deg
  const yForAlt = (alt) => {
    const clamped = Math.max(-10, Math.min(90, Number(alt)));
    return padT + ((90 - clamped) / 100) * plotH;
  };

  let html = `
    <!-- Astronomical Darkness Shaded Region -->
    <rect x="${padL}" y="${padT}" width="${plotW}" height="${plotH}" rx="4" fill="rgba(79,131,255,0.04)" />
  `;

  // Horizontal altitude grid lines (0°, 20° threshold, 45°, 70°, 90°)
  [0, 20, 45, 70, 90].forEach((alt) => {
    const y = yForAlt(alt).toFixed(1);
    const isThreshold = alt === 20;
    const stroke = isThreshold ? "rgba(245,158,11,0.55)" : "rgba(125,162,255,0.14)";
    const dash = isThreshold ? 'stroke-dasharray="4 3"' : "";
    html += `
      <line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}" stroke="${stroke}" stroke-width="1" ${dash} />
      <text x="${padL - 6}" y="${Number(y) + 3}" text-anchor="end" fill="#687ea3" font-family="JetBrains Mono, monospace" font-size="9">${alt}°</text>
    `;
  });

  // Time labels along X axis
  slots.forEach((s, idx) => {
    if (idx % 3 === 0 || idx === numSlots - 1) {
      const x = xForIdx(idx).toFixed(1);
      html += `<text x="${x}" y="${H - 8}" text-anchor="middle" fill="#687ea3" font-family="JetBrains Mono, monospace" font-size="9">${escapeHtml(s.time)}</text>`;
    }
  });

  // Plot altitude curves for each object
  legend.innerHTML = "";
  entries.slice(0, 6).forEach(([docId, obj], cIdx) => {
    const color = CURVE_COLORS[cIdx % CURVE_COLORS.length];
    const pts = obj.slots.map((s, idx) => `${xForIdx(idx).toFixed(1)},${yForAlt(s.altitude_deg).toFixed(1)}`);
    html += `<polyline fill="none" stroke="${color}" stroke-width="2.2" points="${pts.join(" ")}" />`;

    obj.slots.forEach((s, idx) => {
      if (idx % 3 === 0) {
        html += `<circle cx="${xForIdx(idx).toFixed(1)}" cy="${yForAlt(s.altitude_deg).toFixed(1)}" r="2.6" fill="${color}" />`;
      }
    });

    const item = document.createElement("span");
    item.className = "legend-item";
    item.dataset.docId = docId;
    item.innerHTML = `<span class="legend-swatch" style="background:${color}"></span><span>${escapeHtml(docId)} (${Number(obj.max_altitude_deg).toFixed(0)}°)</span>`;
    legend.appendChild(item);
  });

  svg.innerHTML = html;
}

async function openSourceInspector(docId) {
  const dialog = document.getElementById("source-dialog");
  const idEl = document.getElementById("modal-doc-id");
  const titleEl = document.getElementById("modal-doc-title");
  const bodyEl = document.getElementById("modal-doc-body");
  if (!dialog || !bodyEl) return;

  const fl = parseFloat(document.getElementById("input-focal")?.value || "480");
  idEl.textContent = docId;
  titleEl.textContent = `Loading ${docId}...`;
  bodyEl.innerHTML = `<p class="muted">Fetching canonical record from 506-document corpus...</p>`;
  dialog.showModal();

  try {
    const res = await fetch(`/api/catalog/${encodeURIComponent(docId)}?focal_length_mm=${fl}`);
    if (!res.ok) throw new Error(`Object ${docId} not found`);
    const data = await res.json();
    titleEl.textContent = data.title || docId;
    bodyEl.innerHTML = `
      <div style="display:flex; flex-wrap:wrap; gap:0.5rem; margin-bottom:0.85rem;">
        <span class="badge badge-adk">Catalog: ${escapeHtml(data.catalog || "N/A")}</span>
        <span class="badge badge-model">Type: ${escapeHtml(data.object_type || "N/A")}</span>
        <span class="badge badge-mps">Constellation: ${escapeHtml(data.constellation || "N/A")}</span>
        ${data.magnitude != null ? `<span class="badge">Mag: ${data.magnitude}</span>` : ""}
        ${data.size_major_arcmin != null ? `<span class="badge">Size: ${data.size_major_arcmin}'</span>` : ""}
      </div>
      ${data.framing_advice ? `<div class="conditions-banner" style="margin-bottom:0.85rem;">🔭 ${escapeHtml(data.framing_advice)}</div>` : ""}
      <div class="dialog-pre">${escapeHtml(data.full_text || "")}</div>
    `;
  } catch (err) {
    titleEl.textContent = docId;
    bodyEl.innerHTML = `<p class="muted">${escapeHtml(String(err))}</p>`;
  }
}

function renderBenchmarksModal() {
  const body = document.getElementById("benchmarks-modal-body");
  if (!body) return;

  const chRows = Object.entries(state.chunkingExperiment)
    .map(
      ([k, v]) => `
      <tr>
        <td><code>${escapeHtml(k)}</code></td>
        <td>${v.num_chunks}</td>
        <td>${v.avg_chunk_chars} chars</td>
        <td><strong>${(v.recall_at_5 * 100).toFixed(2)}%</strong></td>
        <td>${(v.hit_rate_at_5 * 100).toFixed(2)}%</td>
        <td><strong>${(v.mrr_at_5 * 100).toFixed(2)}%</strong></td>
      </tr>
    `
    )
    .join("");

  const abRows = Object.entries(state.retrievalAblation)
    .map(
      ([k, v]) => `
      <tr>
        <td><code>${escapeHtml(k)}</code></td>
        <td><strong>${(v.recall_at_5 * 100).toFixed(2)}%</strong></td>
        <td>${(v.hit_rate_at_5 * 100).toFixed(2)}%</td>
        <td>${(v.mrr_at_5 * 100).toFixed(2)}%</td>
        <td><code>${v.avg_latency_ms} ms</code></td>
      </tr>
    `
    )
    .join("");

  body.innerHTML = `
    <h4 style="margin-top:0;">1. Three-Way Chunking &amp; Contextual Header Ablation (506 Documents)</h4>
    <p class="panel-hint">
      Comparing naive fixed-size windows (<code>512 chars / 64 overlap</code>) against contextual metadata headers and section-aware boundary splitting:
    </p>
    <div class="table-wrap" style="margin-bottom:1.25rem;">
      <table class="plan-table">
        <thead>
          <tr>
            <th>Strategy</th>
            <th>Total Chunks</th>
            <th>Avg Size</th>
            <th>Recall@5</th>
            <th>HitRate@5</th>
            <th>MRR@5</th>
          </tr>
        </thead>
        <tbody>${chRows}</tbody>
      </table>
    </div>

    <h4>2. Four-Way Hybrid Retrieval &amp; Cross-Encoder Reranking Ablation</h4>
    <p class="panel-hint">
      Comparing lexical <code>bm25s</code>, dense <code>BAAI/bge-small-en-v1.5</code> (Qdrant on Apple M4 MPS), Reciprocal Rank Fusion (<code>k=60</code>), and calibrated Cross-Encoder Reranking (<code>ms-marco-MiniLM-L-6-v2</code>):
    </p>
    <div class="table-wrap">
      <table class="plan-table">
        <thead>
          <tr>
            <th>Retrieval Pipeline</th>
            <th>Recall@5</th>
            <th>HitRate@5</th>
            <th>MRR@5</th>
            <th>Avg Latency (MPS)</th>
          </tr>
        </thead>
        <tbody>${abRows}</tbody>
      </table>
    </div>
  `;
}

function formatMarkdownWithCitations(text) {
  let safe = escapeHtml(text);
  // Bold **...**
  safe = safe.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  // Convert [DOC_ID] citations into interactive buttons
  safe = safe.replace(
    /\[((?:M\d{3}|C\d{3}|NGC\d{4}|CON_[A-Z]{3}))\]/g,
    '<button type="button" class="cite-btn" data-doc-id="$1">[$1]</button>'
  );
  // Paragraphs & bullet lists
  const lines = safe.split("\n");
  let out = "";
  let inList = false;
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    if (trimmed.startsWith("- ")) {
      if (!inList) {
        out += "<ul>";
        inList = TrueBool();
      }
      out += `<li>${trimmed.slice(2)}</li>`;
    } else {
      if (inList) {
        out += "</ul>";
        inList = false;
      }
      out += `<p>${trimmed}</p>`;
    }
  }
  if (inList) out += "</ul>";
  return out;
}

function TrueBool() {
  return true;
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
