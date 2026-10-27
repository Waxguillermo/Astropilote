/* ============================================================================
   AstroPilote · Daylight Keynote / Gemini Argon Frontend Logic
   Adapted from jbouniol/french-data-rag ("Alter-Ego") + Starfield Sky Canvas:
   - 3-Mode Spring Segmented Control: Agent | RAG | Sky Vision (Alessandro's tool)
   - Live execution with reactive sky-blue aura & bloom launch transition
   - Discreet collapsible tool trace with rich step cards on expand
   - Right-side Deep-Sky Encyclopedia & WCS Pixel Inspector panel
   ============================================================================ */

const SESSION_HISTORY_KEY = 'astropilote_session_history_v2';
const EQUIPMENT_PROFILE_KEY = 'astropilote_equipment_profile_v2';
const CHART_COLORS = ['#38BDF8', '#34D399', '#FBBF24', '#C084FC', '#F87171'];

const SENSOR_SPECS_UI = {
  'full-frame': { w: 36.0, h: 24.0, label: 'Full-Frame', crop: 1.0 },
  'APS-C': { w: 23.5, h: 15.6, label: 'APS-C', crop: 1.5 },
  'micro-4/3': { w: 17.3, h: 13.0, label: 'Micro 4/3', crop: 2.0 },
  '1-inch': { w: 13.2, h: 8.8, label: '1" Sensor', crop: 2.7 },
  smartphone: { w: 7.0, h: 5.2, label: 'Smartphone', crop: 5.1 },
};

const MOUNT_SHORT_LABELS = {
  equatorial_goto: 'Eq. GoTo',
  star_tracker: 'Star Tracker',
  altaz_goto: 'Smart Scope / Alt-Az',
  tripod: 'Fixed Tripod',
};

const CARD_BUTTON_LABELS = {
  none: 'Rig',
  smartphone_night: 'Smartphone',
  dslr_50mm_tripod: '50mm · Tripod',
  tele_135_tracker: '135mm · Tracker',
  seestar_s50: 'Smart Scope',
  apo_80ed_eq: '80ED · GoTo',
};

const DEFAULT_EQUIPMENT_PROFILE = {
  enabled: false,
  preset_id: 'none',
  rig_name: '',
  telescope_or_lens: '50mm f/1.8 Prime Lens',
  focal_length_mm: 50.0,
  aperture_mm: 28.0,
  camera_model: 'APS-C Mirrorless / DSLR',
  sensor_format: 'APS-C',
  mount_type: 'tripod',
  has_autoguiding: false,
  filters: ['broadband'],
  bortle_class: 4,
};

const COMPASS_LABELS = {
  N: 'North (N)',
  NE: 'North-East (NE)',
  E: 'East (E)',
  SE: 'South-East (SE)',
  S: 'South (S)',
  SW: 'South-West (SW)',
  W: 'West (W)',
  NW: 'North-West (NW)',
};

const SUGGESTION_ICONS = [
  `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 3l2.2 5.8L20 11l-5.8 2.2L12 19l-2.2-5.8L4 11l5.8-2.2L12 3z" stroke-linejoin="round"/></svg>`,
  `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="9"/><ellipse cx="12" cy="12" rx="9" ry="3.6" transform="rotate(-24 12 12)"/></svg>`,
  `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2" stroke-linecap="round"/></svg>`,
  `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 3l7 3v6c0 4.5-3 7.8-7 9-4-1.2-7-4.5-7-9V6l7-3z" stroke-linejoin="round"/></svg>`,
];

const VISION_ICONS = [
  `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="8" stroke-dasharray="3 2"/><circle cx="12" cy="12" r="2.5"/></svg>`,
  `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="5" width="18" height="14" rx="3"/><circle cx="12" cy="12" r="3.2"/></svg>`,
  `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 2v4M12 18v4M2 12h4M18 12h4"/><circle cx="12" cy="12" r="5"/></svg>`,
  `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="9" cy="12" r="4"/><circle cx="16" cy="11" r="2.6"/></svg>`,
];

const state = {
  mode: 'agent',             // 'agent' | 'rag' | 'vision'
  llmMode: 'api',            // 'api' | 'local'
  llmHealth: null,
  corpusDocs: 567,
  localizableDocs: 447,
  indexedChunks: 3139,
  corpusItems: [],
  corpusFilterCat: 'ALL',
  presets: [],
  visionPresets: [],
  equipmentPresets: [],
  equipmentProfile: { ...DEFAULT_EQUIPMENT_PROFILE },
  selectedVisionPresetId: 'andromeda_widefield',
  uploadedVisionFile: null,  // { name, base64 }
  activeTurns: [],
  currentTrace: null,
  timerInterval: null,
  chatHistory: [],
  activeChatId: null,
};

document.addEventListener('DOMContentLoaded', async () => {
  initSpaceCanvas();
  initDialogLightDismiss();
  loadEquipmentProfile();
  loadSessionHistory();
  updateNewChatButtonState();
  await Promise.all([fetchSystemStatus(), fetchPresets()]);

  const params = new URLSearchParams(window.location.search);
  const initMode = params.get('mode');
  if (initMode && ['agent', 'rag', 'vision'].includes(initMode)) {
    setMode(initMode);
  }
  const runVisionId = params.get('run_vision');
  if (runVisionId) {
    setMode('vision');
    await selectVisionPresetAndRun(runVisionId);
  } else {
    const runPresetId = params.get('run_preset');
    if (runPresetId) {
      await selectPresetScenario(runPresetId);
    }
  }

  document.addEventListener('click', (e) => {
    const apiAnchor = document.querySelector('.api-popover-anchor');
    const apiPop = document.getElementById('api-popover');
    if (apiPop && !apiPop.classList.contains('hidden') && apiAnchor && !apiAnchor.contains(e.target)) {
      apiPop.classList.add('hidden');
    }
    const histAnchor = document.querySelector('.history-popover-anchor');
    const histPop = document.getElementById('history-popover');
    if (histPop && !histPop.classList.contains('hidden') && histAnchor && !histAnchor.contains(e.target)) {
      histPop.classList.add('hidden');
    }
  });
});

function initDialogLightDismiss() {
  ['source-dialog', 'corpus-dialog', 'equipment-dialog'].forEach((id) => {
    const dlg = document.getElementById(id);
    if (!dlg) return;
    dlg.addEventListener('click', (e) => {
      if (e.target === dlg) {
        dlg.close();
      }
    });
  });
}

// ============================================================================
// System Status, Model Switcher & API Key Popover
// ============================================================================
async function fetchSystemStatus() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    state.corpusDocs = data.corpus_documents || 567;
    state.localizableDocs = data.localizable_documents || 447;
    state.indexedChunks = data.indexed_chunks || 3139;
    applyLlmStatus(data.llm || {});
  } catch (err) {
    const dot = document.getElementById('ollama-dot');
    if (dot) dot.classList.add('offline');
  }
}

async function fetchPresets() {
  try {
    const res = await fetch('/api/presets');
    const data = await res.json();
    state.presets = Array.isArray(data.presets) ? data.presets : [];
    state.visionPresets = Array.isArray(data.vision_presets) ? data.vision_presets : [];
    state.equipmentPresets = Array.isArray(data.equipment_presets) ? data.equipment_presets : [];
    if (state.visionPresets.length > 0) {
      applyVisionPresetToInputs(state.visionPresets[0]);
    }
    renderSuggestions();
    renderEquipmentProfileUi();
  } catch (err) {
    console.error('Failed to load presets:', err);
  }
}

// ============================================================================
// User Photography & Optical Equipment Profile (Modal + Natural Language Parser)
// ============================================================================
function loadEquipmentProfile() {
  try {
    const raw = localStorage.getItem(EQUIPMENT_PROFILE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed === 'object') {
        state.equipmentProfile = {
          ...DEFAULT_EQUIPMENT_PROFILE,
          ...parsed,
          filters: Array.isArray(parsed.filters) ? parsed.filters : DEFAULT_EQUIPMENT_PROFILE.filters,
        };
      }
    }
  } catch (e) {
    state.equipmentProfile = { ...DEFAULT_EQUIPMENT_PROFILE };
  }
  syncEquipmentModalFromState();
  renderEquipmentProfileUi();
}

function saveEquipmentProfile() {
  try {
    localStorage.setItem(EQUIPMENT_PROFILE_KEY, JSON.stringify(state.equipmentProfile));
  } catch (e) {
    // ignore storage errors
  }
  renderEquipmentProfileUi();
}

function openEquipmentModal() {
  const dlg = document.getElementById('equipment-dialog');
  if (!dlg) return;
  syncEquipmentModalFromState();
  renderEquipmentProfileUi();
  dlg.showModal();
}

function selectEquipmentCard(presetId) {
  if (presetId === 'none') {
    state.equipmentProfile = {
      ...state.equipmentProfile,
      enabled: false,
      preset_id: 'none',
      rig_name: '',
    };
    syncEquipmentModalFromState();
    saveEquipmentProfile();
    return;
  }

  const found = (state.equipmentPresets || []).find((pr) => pr.preset_id === presetId);
  const fallbackMap = {
    smartphone_night: {
      preset_id: 'smartphone_night',
      rig_name: 'Smartphone on Tripod (Night / Astro Mode)',
      telescope_or_lens: 'Smartphone Main Wide Lens 24mm f/1.8',
      focal_length_mm: 24.0,
      aperture_mm: 13.3,
      camera_model: 'Smartphone Night / RAW Sensor',
      sensor_format: 'smartphone',
      mount_type: 'tripod',
      has_autoguiding: false,
      filters: ['broadband'],
      bortle_class: state.equipmentProfile.bortle_class || 5,
    },
    dslr_50mm_tripod: {
      preset_id: 'dslr_50mm_tripod',
      rig_name: '50mm f/1.8 Lens + APS-C Camera on Fixed Tripod',
      telescope_or_lens: '50mm f/1.8 Prime Lens',
      focal_length_mm: 50.0,
      aperture_mm: 28.0,
      camera_model: 'APS-C Mirrorless / DSLR',
      sensor_format: 'APS-C',
      mount_type: 'tripod',
      has_autoguiding: false,
      filters: ['broadband'],
      bortle_class: state.equipmentProfile.bortle_class || 4,
    },
    tele_135_tracker: {
      preset_id: 'tele_135_tracker',
      rig_name: '135mm f/2 Telephoto + Full-Frame on Star Tracker',
      telescope_or_lens: '135mm f/2 Telephoto Lens',
      focal_length_mm: 135.0,
      aperture_mm: 67.5,
      camera_model: 'Full-Frame 36×24 mm Camera',
      sensor_format: 'full-frame',
      mount_type: 'star_tracker',
      has_autoguiding: false,
      filters: ['broadband'],
      bortle_class: state.equipmentProfile.bortle_class || 4,
    },
    seestar_s50: {
      preset_id: 'seestar_s50',
      rig_name: 'Smart Telescope 50/250mm (Seestar S50 / Alt-Az GoTo)',
      telescope_or_lens: 'Integrated 50/250mm f/5 APO Triplet',
      focal_length_mm: 250.0,
      aperture_mm: 50.0,
      camera_model: 'Built-in 1" CMOS Sensor',
      sensor_format: '1-inch',
      mount_type: 'altaz_goto',
      has_autoguiding: false,
      filters: ['broadband', 'dual_band'],
      bortle_class: state.equipmentProfile.bortle_class || 6,
    },
    apo_80ed_eq: {
      preset_id: 'apo_80ed_eq',
      rig_name: '80mm f/6 APO Refractor + APS-C on Equatorial GoTo Mount',
      telescope_or_lens: '80/480mm f/6 APO Refractor',
      focal_length_mm: 480.0,
      aperture_mm: 80.0,
      camera_model: 'APS-C Astro / Mirrorless Camera',
      sensor_format: 'APS-C',
      mount_type: 'equatorial_goto',
      has_autoguiding: true,
      filters: ['broadband', 'dual_band'],
      bortle_class: state.equipmentProfile.bortle_class || 4,
    },
  };

  const presetData = found || fallbackMap[presetId] || fallbackMap.apo_80ed_eq;
  state.equipmentProfile = {
    ...state.equipmentProfile,
    ...presetData,
    enabled: true,
    preset_id: presetId,
    filters: [...(presetData.filters || ['broadband'])],
  };
  syncEquipmentModalFromState();
  saveEquipmentProfile();
}

function parseNaturalEquipmentInput(rawText, currentProfile) {
  const text = (rawText || '').trim();
  const lower = text.toLowerCase();
  const next = {
    ...currentProfile,
    enabled: true,
    rig_name: text,
    telescope_or_lens: text,
  };

  let focal = Number(next.focal_length_mm || 135);
  let aperture = Number(next.aperture_mm || 45);
  let fRatio = aperture > 0 ? focal / aperture : 4.0;
  let explicitRatio = null;

  // 1. Check explicit f-stop like "f/1.8", "f/2", "f/2.8", "f/5.6", "f2.8"
  const fMatch = lower.match(/\bf\s*\/?\s*(\d{1,2}(?:[.,]\d{1,2})?)\b/);
  if (fMatch) {
    const val = parseFloat(fMatch[1].replace(',', '.'));
    if (val >= 0.95 && val <= 25) {
      explicitRatio = val;
    }
  }

  // 2. Check telescope "aperture/focal" notation like "150/750", "80/480", "200/1000", "72/420"
  const slashMatch = lower.match(/\b(\d{2,3})\s*\/\s*(\d{3,4})\s*(?:mm)?\b/);
  if (slashMatch) {
    const ap = parseFloat(slashMatch[1]);
    const fl = parseFloat(slashMatch[2]);
    if (ap >= 20 && ap <= 600 && fl >= 100 && fl <= 5000) {
      aperture = ap;
      focal = fl;
      fRatio = focal / aperture;
    }
  } else {
    // 3. Check focal length like "135mm", "50 mm", "480mm", "1280mm"
    const mmMatch = lower.match(/\b(\d{2,4})\s*mm\b/);
    if (mmMatch) {
      const fl = parseFloat(mmMatch[1]);
      if (fl >= 10 && fl <= 5000) {
        focal = fl;
        if (explicitRatio) {
          aperture = Number((focal / explicitRatio).toFixed(1));
        } else if (fl <= 85) {
          aperture = Number((focal / 2.0).toFixed(1));
        } else if (fl <= 200) {
          aperture = Number((focal / 2.8).toFixed(1));
        } else {
          aperture = Number((focal / 5.6).toFixed(1));
        }
      }
    } else if (/\b(?:80\s*ed|72\s*ed)\b/.test(lower)) {
      if (lower.includes('72')) {
        aperture = 72;
        focal = 420;
      } else {
        aperture = 80;
        focal = 480;
      }
    } else if (/\bc8\b/.test(lower)) {
      aperture = 203;
      focal = explicitRatio && explicitRatio < 7 ? 1280 : 2030;
    } else if (/\bseestar\s*s30\b/.test(lower)) {
      aperture = 30;
      focal = 150;
    } else if (/\bseestar\b/.test(lower)) {
      aperture = 50;
      focal = 250;
    } else if (/\bvespera\b/.test(lower)) {
      aperture = 50;
      focal = 200;
    } else if (/\bdwarf\b/.test(lower)) {
      aperture = 24;
      focal = 100;
    } else if (explicitRatio) {
      aperture = Number((focal / explicitRatio).toFixed(1));
    }
  }

  if (explicitRatio && !slashMatch) {
    aperture = Number((focal / explicitRatio).toFixed(1));
  }

  next.focal_length_mm = focal;
  next.aperture_mm = Math.max(5.0, aperture);

  // 4. Detect camera / sensor format
  if (/\b(iphone|samsung|galaxy\s*s|pixel|smartphone|téléphone)\b/.test(lower)) {
    next.sensor_format = 'smartphone';
  } else if (/\b(full[-\s]?frame|plein\s*format|24x36|a7|a9|eos\s*r[3568]|eos\s*6d|eos\s*5d|z[56789]|d850|d750|asi6200|asi2400)\b/.test(lower)) {
    next.sensor_format = 'full-frame';
  } else if (/\b(asi533|asi183|asi585|asi462|seestar|vespera|dwarf|1[\s-]?inch|1\s*pouce)\b/.test(lower)) {
    next.sensor_format = '1-inch';
  } else if (/\b(micro\s*4\/3|m43|mft|om-?1|olympus|lumix\s*g|gh[456]|asi294|asi1600)\b/.test(lower)) {
    next.sensor_format = 'micro-4/3';
  } else if (/\b(aps-?c|asi2600|asi071|600d|650d|700d|800d|2000d|4000d|90d|r7|r10|r50|a6\d00|xt-?\d|x-t\d|x-s\d|d5\d00|d7\d00|z50|zfc)\b/.test(lower)) {
    next.sensor_format = 'APS-C';
  }

  // 5. Detect mount / tracking
  if (/\b(trépied|trepied|sans\s+suivi|fixe|tripod)\b/.test(lower)) {
    next.mount_type = 'tripod';
    next.has_autoguiding = false;
  } else if (/\b(star\s*adventurer|star\s*tracker|skyguider|nomade|move\s*shoot|minitrack|tracker)\b/.test(lower)) {
    next.mount_type = 'star_tracker';
    next.has_autoguiding = /\b(autoguid|guidage|asiair)\b/.test(lower);
  } else if (/\b(seestar|vespera|dwarf|alt-?az|smart\s*scope|unistellar|evscope)\b/.test(lower)) {
    next.mount_type = 'altaz_goto';
    next.has_autoguiding = false;
  } else if (/\b(eq6|heq5|eq5|eqm-?35|am5|am3|cem|gem|avx|goto|équatoriale|equatoriale|newton|lunette|c8)\b/.test(lower)) {
    next.mount_type = 'equatorial_goto';
    next.has_autoguiding = true;
  }

  // 6. Detect filters
  const detectedFilters = new Set(next.filters || ['broadband']);
  if (/\b(l-?extreme|l-?enhance|l-?ultimate|dual[-\s]?band|duo[-\s]?band|alp-?t|nbz)\b/.test(lower)) {
    detectedFilters.add('dual_band');
  }
  if (/\b(uhc|cls|l-?pro)\b/.test(lower)) {
    detectedFilters.add('uhc_cls');
  }
  if (/\b(sho|h-?alpha|sii|oiii)\b/.test(lower)) {
    detectedFilters.add('narrowband_sho');
  }
  next.filters = Array.from(detectedFilters);
  return next;
}

function handleFreeTextEquipmentInput(value) {
  const trimmed = (value || '').trim();
  if (!trimmed) {
    renderEquipmentProfileUi();
    return;
  }
  state.equipmentProfile = parseNaturalEquipmentInput(trimmed, state.equipmentProfile);
  if (!state.equipmentProfile.preset_id || state.equipmentProfile.preset_id === 'none') {
    state.equipmentProfile.preset_id = 'custom';
  }
  syncAdvancedInputsFromState();
  saveEquipmentProfile();
}

function applyFreeTextExample(exampleText) {
  const input = document.getElementById('eq-free-text');
  if (input) {
    input.value = exampleText;
  }
  state.equipmentProfile.preset_id = 'custom';
  handleFreeTextEquipmentInput(exampleText);
  syncEquipmentModalFromState();
}

function selectSkyBortle(bortleVal) {
  state.equipmentProfile.bortle_class = Number(bortleVal) || 4;
  document.querySelectorAll('#eq-sky-pills .eq-sky-pill').forEach((btn) => {
    btn.classList.toggle('active', Number(btn.dataset.bortle) === state.equipmentProfile.bortle_class);
  });
  saveEquipmentProfile();
}

function toggleEquipmentFilter(filterKey) {
  const filters = new Set(state.equipmentProfile.filters || []);
  if (filters.has(filterKey)) {
    filters.delete(filterKey);
  } else {
    filters.add(filterKey);
  }
  if (filters.size === 0) filters.add('broadband');
  state.equipmentProfile.filters = Array.from(filters);
  syncAdvancedInputsFromState();
  saveEquipmentProfile();
}

function handleManualFineTune() {
  const focalVal = parseFloat(document.getElementById('eq-focal')?.value || '135');
  const fRatioVal = parseFloat(document.getElementById('eq-fratio')?.value || '2.8');
  const sensorVal = document.getElementById('eq-sensor')?.value || 'APS-C';
  const mountVal = document.getElementById('eq-mount')?.value || 'tripod';
  const autoguideVal = Boolean(document.getElementById('eq-autoguiding')?.checked);

  const safeFocal = Number.isFinite(focalVal) && focalVal > 0 ? focalVal : 135.0;
  const safeFRatio = Number.isFinite(fRatioVal) && fRatioVal >= 0.9 ? fRatioVal : 2.8;
  const computedAperture = Number((safeFocal / safeFRatio).toFixed(1));

  state.equipmentProfile = {
    ...state.equipmentProfile,
    enabled: true,
    focal_length_mm: safeFocal,
    aperture_mm: Math.max(5.0, computedAperture),
    sensor_format: sensorVal,
    mount_type: mountVal,
    has_autoguiding: autoguideVal,
  };
  saveEquipmentProfile();
}

function syncAdvancedInputsFromState() {
  const p = state.equipmentProfile || DEFAULT_EQUIPMENT_PROFILE;
  const focal = Number(p.focal_length_mm || 50);
  const aperture = Number(p.aperture_mm || 28);
  const fRatio = aperture > 0 ? (focal / aperture).toFixed(1) : '2.8';

  const setVal = (id, v) => {
    const el = document.getElementById(id);
    if (el) el.value = v ?? '';
  };
  setVal('eq-focal', Math.round(focal));
  setVal('eq-fratio', fRatio);
  setVal('eq-sensor', p.sensor_format || 'APS-C');
  setVal('eq-mount', p.mount_type || 'tripod');
  const autoEl = document.getElementById('eq-autoguiding');
  if (autoEl) autoEl.checked = Boolean(p.has_autoguiding);

  const activeFilters = new Set(Array.isArray(p.filters) ? p.filters : ['broadband']);
  document.querySelectorAll('#eq-filter-chips .eq-filter-chip').forEach((chip) => {
    chip.classList.toggle('active', activeFilters.has(chip.dataset.filter));
  });
}

function syncEquipmentModalFromState() {
  const p = state.equipmentProfile || DEFAULT_EQUIPMENT_PROFILE;
  const activePreset = p.enabled ? p.preset_id || 'custom' : 'none';

  document.querySelectorAll('#eq-cards-grid .eq-visual-card').forEach((card) => {
    card.classList.toggle('active', card.dataset.preset === activePreset);
  });

  const detailsSec = document.getElementById('eq-details-section');
  if (detailsSec) {
    detailsSec.classList.toggle('is-muted', !p.enabled);
  }

  const freeInput = document.getElementById('eq-free-text');
  if (freeInput) {
    freeInput.value = p.enabled ? p.rig_name || p.telescope_or_lens || '' : '';
  }

  const bortle = Number(p.bortle_class || 4);
  const bucket = bortle <= 3 ? 2 : bortle <= 4 ? 4 : bortle <= 6 ? 6 : 8;
  document.querySelectorAll('#eq-sky-pills .eq-sky-pill').forEach((btn) => {
    btn.classList.toggle('active', Number(btn.dataset.bortle) === bucket);
  });

  syncAdvancedInputsFromState();
}

function renderEquipmentProfileUi() {
  const p = state.equipmentProfile || DEFAULT_EQUIPMENT_PROFILE;
  const spec = SENSOR_SPECS_UI[p.sensor_format] || SENSOR_SPECS_UI['APS-C'];
  const focal = Number(p.focal_length_mm || 50);
  const aperture = Number(p.aperture_mm || 28);
  const fRatio = aperture > 0 ? (focal / aperture).toFixed(1) : '2.8';
  const fovWDeg = (((57.2958 * 60.0 * spec.w) / focal) / 60.0).toFixed(1);
  const fovHDeg = (((57.2958 * 60.0 * spec.h) / focal) / 60.0).toFixed(1);
  const mountShort = MOUNT_SHORT_LABELS[p.mount_type] || p.mount_type;

  let subHint = '120s–240s (Autoguided)';
  if (p.mount_type === 'tripod') {
    const npf = Math.max(0.5, Math.min(15.0, 500.0 / (focal * spec.crop))).toFixed(1);
    subHint = `${npf}s max sub (Fixed tripod)`;
  } else if (p.mount_type === 'altaz_goto') {
    subHint = '10s–20s (Auto Live-Stack)';
  } else if (p.mount_type === 'star_tracker') {
    subHint = focal <= 150 ? '60s–120s (Star Tracker)' : '30s–60s (Star Tracker)';
  } else if (!p.has_autoguiding) {
    subHint = '60s–120s (Unguided Eq.)';
  }

  // Update decoded badges inside the modal
  const badgesEl = document.getElementById('eq-decoded-badges');
  if (badgesEl) {
    if (!p.enabled) {
      badgesEl.innerHTML = `
        <span class="eq-decoded-pill muted">No specific camera rig active — General &amp; visual sky guidance</span>
      `;
    } else {
      const hasDual = (p.filters || []).includes('dual_band');
      const filterLabel = hasDual ? 'Dual-Band (Hα+OIII) available' : 'Broadband / Unfiltered';
      badgesEl.innerHTML = `
        <span class="eq-decoded-pill">Focal <strong>${Math.round(focal)} mm</strong> (f/${fRatio})</span>
        <span class="eq-decoded-pill">FOV <strong>${fovWDeg}° × ${fovHDeg}°</strong> (${spec.label})</span>
        <span class="eq-decoded-pill">Sub <strong>${subHint}</strong></span>
        <span class="eq-decoded-pill">${filterLabel} · ${mountShort}</span>
      `;
    }
  }

  // Update active state on the 6 visual cards
  const activePreset = p.enabled ? p.preset_id || 'custom' : 'none';
  document.querySelectorAll('#eq-cards-grid .eq-visual-card').forEach((card) => {
    card.classList.toggle('active', card.dataset.preset === activePreset);
  });
  const detailsSec = document.getElementById('eq-details-section');
  if (detailsSec) {
    detailsSec.classList.toggle('is-muted', !p.enabled);
  }

  // Update the ultra-discreet ghost button in Home and Chat prompt bars
  let btnLabel = 'Rig';
  if (p.enabled) {
    if (CARD_BUTTON_LABELS[p.preset_id]) {
      btnLabel = CARD_BUTTON_LABELS[p.preset_id];
    } else {
      btnLabel = `${Math.round(focal)}mm f/${fRatio}`;
    }
  }

  ['home', 'chat'].forEach((prefix) => {
    const btn = document.getElementById(`${prefix}-eq-btn`);
    const dot = document.getElementById(`${prefix}-eq-dot`);
    const lbl = document.getElementById(`${prefix}-eq-label`);
    if (lbl) lbl.textContent = btnLabel;
    if (dot) {
      dot.classList.toggle('hidden', !p.enabled);
      dot.classList.toggle('active', Boolean(p.enabled));
    }
    if (btn) btn.classList.toggle('has-rig', Boolean(p.enabled));
  });
}

function getProviderLogoSvg(llmMode) {
  if (llmMode === 'api') {
    return `
      <svg viewBox="0 0 24 24" width="15" height="15" fill="#D97757" aria-hidden="true">
        <path d="M17.304 3.541h-3.672l6.696 16.918H24Zm-10.608 0L0 20.459h3.744l1.37-3.553h7.005l1.369 3.553h3.744L10.536 3.541Zm-.371 10.223L8.616 7.82l2.291 5.945Z"/>
      </svg>
    `;
  }
  return `
    <svg viewBox="0 0 24 24" width="15" height="15" fill="none" aria-hidden="true">
      <defs>
        <linearGradient id="astro-pill-grad-dyn" x1="2" y1="2" x2="22" y2="22" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stop-color="#38BDF8"/>
          <stop offset="52%" stop-color="#0EA5E9"/>
          <stop offset="100%" stop-color="#818CF8"/>
        </linearGradient>
      </defs>
      <path d="M12 1.5C12 7.3 16.7 12 22.5 12C16.7 12 12 16.7 12 22.5C12 16.7 7.3 12 1.5 12C7.3 12 12 7.3 12 1.5Z" fill="url(#astro-pill-grad-dyn)"/>
    </svg>
  `;
}

function triggerMorphAnimation(el, className = 'morphing') {
  if (!el) return;
  el.classList.remove(className);
  void el.offsetWidth;
  el.classList.add(className);
  setTimeout(() => el.classList.remove(className), 520);
}

function applyLlmStatus(llm) {
  state.llmHealth = llm;
  if (!state.llmMode) {
    state.llmMode = llm.configured_mode || 'api';
  }

  const dot = document.getElementById('ollama-dot');
  const statusText = document.getElementById('ollama-status-text');
  const versionTag = document.getElementById('model-version-tag');
  const corpusPillText = document.getElementById('corpus-pill-text');
  const homeEngineLabel = document.getElementById('home-engine-label');
  const visionEngineLabel = document.getElementById('vision-engine-label');

  const isApi = state.llmMode === 'api';
  const modelName = isApi
    ? (llm.anthropic_model || 'claude-sonnet-4-20250514').replace(/-\d{8}$/, '')
    : (llm.ollama_model || 'llama3.1:8b');
  const isReady = isApi ? Boolean(llm.anthropic_ready) : Boolean(llm.ollama_running);

  if (dot) {
    dot.innerHTML = getProviderLogoSvg(state.llmMode);
    dot.classList.toggle('offline', !isReady);
  }
  if (corpusPillText) {
    corpusPillText.textContent = `${state.corpusDocs} Wikipedia Pages`;
  }
  if (homeEngineLabel && state.mode !== 'vision') {
    homeEngineLabel.textContent =
      state.mode === 'rag'
        ? `${state.corpusDocs} Wikipedia Pages · Hybrid RAG ↗`
        : `${state.corpusDocs} Sky Objects · OpenNGC ↗`;
  }
  if (visionEngineLabel) {
    visionEngineLabel.textContent = `WCS Plate Solver & Overlay · ${state.localizableDocs} Localizable DSOs ↗`;
  }
  if (statusText) {
    if (isApi) {
      statusText.textContent = isReady
        ? `${modelName} · ${state.corpusDocs} objects`
        : `${modelName} · API key optional`;
    } else {
      statusText.textContent = isReady
        ? `${modelName} · Local Ollama`
        : `${modelName} · Deterministic fallback`;
    }
  }
  if (versionTag) {
    versionTag.textContent =
      state.mode === 'vision'
        ? `WCS TAN · ${state.localizableDocs} localizable DSOs`
        : `${modelName} · ${state.corpusDocs} DSOs`;
  }

  ['home-model-select', 'chat-model-select'].forEach((id) => {
    const sel = document.getElementById(id);
    if (sel) sel.value = state.llmMode;
  });

  renderApiPopover();
}

function selectLlmMode(mode) {
  if (mode !== 'api' && mode !== 'local') return;
  state.llmMode = mode;
  applyLlmStatus(state.llmHealth || {});
  triggerMorphAnimation(document.getElementById('ollama-dot'));
  triggerMorphAnimation(document.getElementById('ollama-status-text'));
  triggerMorphAnimation(document.getElementById('model-version-tag'));
}

function renderApiPopover() {
  const llm = state.llmHealth || {};
  const isApi = state.llmMode === 'api';
  const tabApi = document.getElementById('api-tab-api');
  const tabLocal = document.getElementById('api-tab-local');
  const dotApi = document.getElementById('provider-dot-api');
  const dotLocal = document.getElementById('provider-dot-local');
  const titleEl = document.getElementById('api-popover-title');
  const badgeEl = document.getElementById('api-key-badge');
  const descEl = document.getElementById('api-popover-desc');
  const inputRow = document.getElementById('api-input-row');

  if (tabApi) tabApi.classList.toggle('active', isApi);
  if (tabLocal) tabLocal.classList.toggle('active', !isApi);
  if (dotApi) dotApi.classList.toggle('active', Boolean(llm.anthropic_ready));
  if (dotLocal) dotLocal.classList.toggle('active', Boolean(llm.ollama_running));

  if (isApi) {
    if (titleEl) titleEl.textContent = 'ANTHROPIC API · .ENV';
    if (badgeEl) {
      badgeEl.textContent = llm.anthropic_ready ? '● Active (.env)' : 'Deterministic fallback';
      badgeEl.classList.toggle('active', Boolean(llm.anthropic_ready));
    }
    if (descEl) {
      descEl.innerHTML = `Paste your <span class="mono">ANTHROPIC_API_KEY</span> into <span class="mono">.env</span> or save it here:`;
    }
    if (inputRow) inputRow.classList.remove('hidden');
  } else {
    if (titleEl) titleEl.textContent = 'LOCAL OLLAMA ENGINE';
    if (badgeEl) {
      badgeEl.textContent = llm.ollama_running ? '● Daemon online' : '○ Daemon offline';
      badgeEl.classList.toggle('active', Boolean(llm.ollama_running));
    }
    if (descEl) {
      descEl.innerHTML = `Runs locally via Ollama (<span class="mono">${escapeHtml(llm.ollama_model || 'llama3.1:8b')}</span>) with zero external API calls.`;
    }
    if (inputRow) inputRow.classList.add('hidden');
  }
}

async function toggleApiPopover() {
  const pop = document.getElementById('api-popover');
  if (!pop) return;
  const histPop = document.getElementById('history-popover');
  if (histPop) histPop.classList.add('hidden');
  const willShow = pop.classList.contains('hidden');
  if (willShow) {
    await fetchSystemStatus();
    pop.classList.remove('hidden');
  } else {
    pop.classList.add('hidden');
  }
}

async function saveApiKeyFromUi() {
  const input = document.getElementById('api-key-input');
  const key = (input ? input.value : '').trim();
  if (!key) return;
  try {
    const res = await fetch('/api/config/key', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ api_key: key }),
    });
    if (res.ok) {
      const data = await res.json();
      if (input) input.value = '';
      applyLlmStatus(data.llm || {});
      const pop = document.getElementById('api-popover');
      if (pop) pop.classList.add('hidden');
    }
  } catch (err) {
    console.error(err);
  }
}

// ============================================================================
// 3-Mode Segmented Control (Agent | RAG | Sky Vision) & Suggestions
// ============================================================================
function setMode(mode) {
  if (!['agent', 'rag', 'vision'].includes(mode)) return;
  const changed = state.mode !== mode;
  state.mode = mode;

  ['seg-agent', 'chat-seg-agent'].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.classList.toggle('active', mode === 'agent');
  });
  ['seg-rag', 'chat-seg-rag'].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.classList.toggle('active', mode === 'rag');
  });
  ['seg-vision', 'chat-seg-vision'].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.classList.toggle('active', mode === 'vision');
  });

  document.querySelectorAll('.segmented-pill').forEach((pill) => {
    pill.classList.toggle('mode-rag', mode === 'rag');
    pill.classList.toggle('mode-vision', mode === 'vision');
  });

  // Toggle Home card between Text Prompt mode and Sky Vision Studio mode
  const textBox = document.getElementById('home-text-mode-box');
  const visionBox = document.getElementById('home-vision-mode-box');
  const heroHeading = document.getElementById('hero-title-heading');

  if (mode === 'vision') {
    if (textBox) textBox.classList.add('hidden');
    if (visionBox) visionBox.classList.remove('hidden');
    if (heroHeading) {
      heroHeading.innerHTML = `Plate-solve &amp; identify <span class="hero-blue">your sky photo.</span>`;
    }
  } else {
    if (textBox) textBox.classList.remove('hidden');
    if (visionBox) visionBox.classList.add('hidden');
    if (heroHeading) {
      heroHeading.innerHTML =
        mode === 'rag'
          ? `Query the <span class="hero-blue">${state.corpusDocs} Wikipedia pages.</span>`
          : `Explore &amp; identify <span class="hero-blue">tonight's sky.</span>`;
    }
  }

  applyLlmStatus(state.llmHealth || {});
  renderSuggestions();

  if (changed) {
    document.querySelectorAll('.prompt-card').forEach((card) => {
      card.classList.add('mode-ripple');
      clearTimeout(card._rippleTimer);
      card._rippleTimer = setTimeout(() => card.classList.remove('mode-ripple'), 280);
    });
  }
}

function openVisionFromChat() {
  resetToHome();
  setMode('vision');
}

function renderSuggestions() {
  const list = document.getElementById('suggestions-list');
  if (!list) return;

  if (state.mode === 'vision') {
    list.innerHTML = state.visionPresets
      .map((vp, i) => {
        const icon = VISION_ICONS[i % VISION_ICONS.length];
        return `
          <button type="button" class="suggestion-row" onclick="selectVisionPresetAndRun('${escapeHtml(vp.id)}')">
            <span class="suggestion-icon">${icon}</span>
            <span>${escapeHtml(vp.label)}</span>
            <span class="suggestion-badge">${escapeHtml(vp.badge || 'WCS TAN')}</span>
          </button>
        `;
      })
      .join('');
    return;
  }

  const items = state.presets.slice(0, 4);
  list.innerHTML = items
    .map((sc, i) => {
      const icon = SUGGESTION_ICONS[i % SUGGESTION_ICONS.length];
      return `
        <button type="button" class="suggestion-row" onclick="selectPresetScenario('${escapeHtml(sc.id)}')">
          <span class="suggestion-icon">${icon}</span>
          <span>${escapeHtml(sc.label)} — <span style="color:var(--text-muted);font-weight:400">${escapeHtml(sc.subtitle)}</span></span>
        </button>
      `;
    })
    .join('');
}

// ============================================================================
// Vision Studio File Upload & Preset Helpers
// ============================================================================
function applyVisionPresetToInputs(vp) {
  if (!vp) return;
  state.selectedVisionPresetId = vp.id;
  state.uploadedVisionFile = null;

  const raEl = document.getElementById('vis-ra');
  const decEl = document.getElementById('vis-dec');
  const focalEl = document.getElementById('vis-focal');
  const sensorEl = document.getElementById('vis-sensor');
  const fovEl = document.getElementById('vis-fov');
  const labelEl = document.getElementById('vision-file-label');
  const subEl = document.getElementById('vision-file-sub');
  const badgeEl = document.getElementById('vision-selected-badge');

  if (raEl) raEl.value = vp.ra_hint_deg;
  if (decEl) decEl.value = vp.dec_hint_deg;
  if (focalEl) focalEl.value = vp.focal_length_mm;
  if (sensorEl) sensorEl.value = vp.sensor_width_mm;
  if (fovEl) fovEl.value = vp.field_width_deg;
  if (labelEl) {
    labelEl.innerHTML = `Preset Ready: <strong>${escapeHtml(vp.label)}</strong> (or drop your own photo)`;
  }
  if (subEl) {
    subEl.textContent = `${vp.telescope} · ${vp.camera}`;
  }
  if (badgeEl) {
    badgeEl.textContent = vp.badge || 'Preset Active';
  }
}

function handleVisionDragOver(e) {
  e.preventDefault();
  const dz = document.getElementById('vision-dropzone');
  if (dz) dz.classList.add('dragover');
}

function handleVisionDragLeave(e) {
  e.preventDefault();
  const dz = document.getElementById('vision-dropzone');
  if (dz) dz.classList.remove('dragover');
}

function handleVisionDrop(e) {
  e.preventDefault();
  const dz = document.getElementById('vision-dropzone');
  if (dz) dz.classList.remove('dragover');
  const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
  if (file) loadUploadedVisionFile(file);
}

function handleVisionFileSelect(e) {
  const file = e.target && e.target.files && e.target.files[0];
  if (file) loadUploadedVisionFile(file);
}

function loadUploadedVisionFile(file) {
  const reader = new FileReader();
  reader.onload = () => {
    state.uploadedVisionFile = {
      name: file.name,
      base64: String(reader.result || ''),
    };
    state.selectedVisionPresetId = null;

    // Clear leftover preset hints so Astrometry.net (`solve-field`) runs unconstrained blind plate-solving by default
    ['vis-ra', 'vis-dec', 'vis-focal', 'vis-sensor', 'vis-fov'].forEach((id) => {
      const el = document.getElementById(id);
      if (el) el.value = '';
    });

    const labelEl = document.getElementById('vision-file-label');
    const subEl = document.getElementById('vision-file-sub');
    const badgeEl = document.getElementById('vision-selected-badge');
    if (labelEl) {
      labelEl.innerHTML = `Uploaded file: <span class="mono">${escapeHtml(file.name)}</span>`;
    }
    if (subEl) {
      subEl.textContent = `${(file.size / 1024).toFixed(1)} KB · Blind Astrometry.net plate-solve ready (optional hints cleared)`;
    }
    if (badgeEl) {
      badgeEl.textContent = 'Custom Photo · Auto Solve';
    }
    if (file.name.toLowerCase().endsWith('.fits') || file.name.toLowerCase().endsWith('.fit')) {
      const trustEl = document.getElementById('vis-trust-wcs');
      if (trustEl) trustEl.checked = true;
    }
  };
  reader.readAsDataURL(file);
}

async function selectVisionPresetAndRun(presetId) {
  if (document.body.classList.contains('is-thinking')) return;
  const vp = state.visionPresets.find((p) => p.id === presetId);
  if (!vp) return;
  applyVisionPresetToInputs(vp);
  await submitVisionFromHome();
}

async function submitVisionFromHome() {
  if (document.body.classList.contains('is-thinking')) return;

  const raVal = parseFloat(document.getElementById('vis-ra')?.value || '');
  const decVal = parseFloat(document.getElementById('vis-dec')?.value || '');
  const focalVal = parseFloat(document.getElementById('vis-focal')?.value || '');
  const sensorVal = parseFloat(document.getElementById('vis-sensor')?.value || '');
  const fovVal = parseFloat(document.getElementById('vis-fov')?.value || '');
  const trustWcs = Boolean(document.getElementById('vis-trust-wcs')?.checked);

  const payload = {
    preset_id: state.uploadedVisionFile ? null : (state.selectedVisionPresetId || 'andromeda_widefield'),
    image_base64: state.uploadedVisionFile ? state.uploadedVisionFile.base64 : null,
    filename: state.uploadedVisionFile ? state.uploadedVisionFile.name : null,
    ra_hint_deg: Number.isFinite(raVal) ? raVal : null,
    dec_hint_deg: Number.isFinite(decVal) ? decVal : null,
    focal_length_mm: Number.isFinite(focalVal) ? focalVal : null,
    sensor_width_mm: Number.isFinite(sensorVal) ? sensorVal : null,
    field_width_deg: Number.isFinite(fovVal) ? fovVal : null,
    trust_fits_wcs: trustWcs,
  };

  const presetObj = state.visionPresets.find((p) => p.id === payload.preset_id);
  const titleText = state.uploadedVisionFile
    ? `🔭 Sky Vision · ${state.uploadedVisionFile.name} (RA ${payload.ra_hint_deg ?? '—'}°, Dec ${payload.dec_hint_deg ?? '—'}°)`
    : `🔭 Sky Vision · ${presetObj ? presetObj.label : payload.preset_id}`;

  await executeVisionTurn(titleText, payload);
}

// ============================================================================
// Controls & Session Chat History
// ============================================================================
function isCurrentlyOnEmptyChat() {
  const homeStage = document.getElementById('home-stage');
  return state.activeChatId === null && homeStage && !homeStage.classList.contains('hidden');
}

function updateNewChatButtonState() {
  const btn = document.getElementById('new-chat-btn');
  if (!btn) return;
  btn.classList.toggle('is-empty-chat', isCurrentlyOnEmptyChat());
}

function handleNewChatClick() {
  const histPop = document.getElementById('history-popover');
  if (histPop) histPop.classList.add('hidden');
  if (isCurrentlyOnEmptyChat()) {
    const homeInput = document.getElementById('home-input');
    if (homeInput && state.mode !== 'vision') homeInput.focus();
    return;
  }
  resetToHome();
}

function resetToHome() {
  state.activeChatId = null;
  state.currentTrace = null;
  state.activeTurns = [];
  if (state.timerInterval) {
    clearInterval(state.timerInterval);
    state.timerInterval = null;
  }
  const turnsList = document.getElementById('chat-turns-list');
  if (turnsList) turnsList.innerHTML = '';
  const homeStage = document.getElementById('home-stage');
  if (homeStage) {
    homeStage.classList.remove('is-launching', 'hidden');
  }
  document.body.classList.remove('is-launching', 'is-thinking');
  document.getElementById('chat-view').classList.add('hidden');
  closeSourcePanel();
  const homeInput = document.getElementById('home-input');
  if (homeInput) homeInput.value = '';
  const chatInput = document.getElementById('chat-input');
  if (chatInput) chatInput.value = '';
  updateNewChatButtonState();
  renderHistoryList();
}

function loadSessionHistory() {
  try {
    const raw = sessionStorage.getItem(SESSION_HISTORY_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) {
        state.chatHistory = parsed.filter((item) => item && item.question && item.turns && item.turns.length > 0);
      }
    }
  } catch (e) {
    state.chatHistory = [];
  }
  renderHistoryList();
}

function saveSessionHistory() {
  try {
    // Strip heavy base64 images when persisting to sessionStorage if needed so quota is never exceeded
    const compact = state.chatHistory.slice(0, 12).map((entry) => ({
      ...entry,
      turns: (entry.turns || []).map((t) => {
        if (t.turn_type === 'vision') {
          return { ...t, original_image_data_url: t.annotated_image_data_url };
        }
        return t;
      }),
    }));
    sessionStorage.setItem(SESSION_HISTORY_KEY, JSON.stringify(compact));
  } catch (e) {
    // Ignore sessionStorage quota warnings on large images
  }
  renderHistoryList();
}

function recordCompletedTurn(turnData, { isFollowUp = false } = {}) {
  const now = new Date();
  const timeStr = now.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });

  if (isFollowUp && state.activeChatId) {
    const existingIdx = state.chatHistory.findIndex((c) => c.id === state.activeChatId);
    if (existingIdx !== -1) {
      const existing = state.chatHistory[existingIdx];
      const updated = {
        ...existing,
        mode: turnData.mode || state.mode,
        time: timeStr,
        turns: [...(existing.turns || []), turnData],
      };
      state.chatHistory.splice(existingIdx, 1);
      state.chatHistory.unshift(updated);
      updateNewChatButtonState();
      saveSessionHistory();
      return;
    }
  }

  const chatId = `astro_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
  const entry = {
    id: chatId,
    question: turnData.question,
    mode: turnData.mode || state.mode,
    time: timeStr,
    turns: [turnData],
  };

  state.chatHistory.unshift(entry);
  if (state.chatHistory.length > 20) {
    state.chatHistory = state.chatHistory.slice(0, 20);
  }
  state.activeChatId = chatId;
  updateNewChatButtonState();
  saveSessionHistory();
}

function toggleHistoryPopover() {
  const pop = document.getElementById('history-popover');
  if (!pop) return;
  const apiPop = document.getElementById('api-popover');
  if (apiPop) apiPop.classList.add('hidden');
  pop.classList.toggle('hidden');
  if (!pop.classList.contains('hidden')) {
    renderHistoryList();
  }
}

function renderHistoryList() {
  const listEl = document.getElementById('history-list');
  const badgeEl = document.getElementById('history-count-badge');
  const clearBtn = document.getElementById('history-clear-btn');

  const count = state.chatHistory.length;
  if (badgeEl) {
    badgeEl.textContent = String(count);
    badgeEl.classList.toggle('hidden', count === 0);
  }
  if (clearBtn) {
    clearBtn.classList.toggle('hidden', count === 0);
  }
  if (!listEl) return;

  if (count === 0) {
    listEl.innerHTML = `<div class="history-empty">No session history yet.<br/>Ask a sky question or run Sky Vision to save it here.</div>`;
    return;
  }

  listEl.innerHTML = state.chatHistory
    .map((item) => {
      const isActive = item.id === state.activeChatId;
      const modeLabel = (item.mode || 'agent').toUpperCase();
      const turnCount = Array.isArray(item.turns) ? item.turns.length : 1;
      const turnsBadge = turnCount > 1 ? `<span>· ${turnCount} turns</span>` : '';
      return `
        <div class="history-item${isActive ? ' active' : ''}" onclick="openHistoryChat('${escapeHtml(item.id)}')">
          <div class="history-item-main">
            <div class="history-item-q">${escapeHtml(item.question)}</div>
            <div class="history-item-meta">
              <span class="history-mode-badge">${escapeHtml(modeLabel)}</span>
              ${turnsBadge}
              <span>· ${escapeHtml(item.time || '')}</span>
            </div>
          </div>
          <button type="button" class="history-delete-btn" onclick="deleteHistoryChat(event, '${escapeHtml(item.id)}')" title="Delete">×</button>
        </div>
      `;
    })
    .join('');
}

function openHistoryChat(chatId) {
  const item = state.chatHistory.find((c) => c.id === chatId);
  if (!item || !item.turns || item.turns.length === 0) return;

  state.activeChatId = item.id;
  state.activeTurns = [...item.turns];
  state.currentTrace = state.activeTurns[state.activeTurns.length - 1];

  const histPop = document.getElementById('history-popover');
  if (histPop) histPop.classList.add('hidden');

  document.getElementById('home-stage').classList.add('hidden');
  document.getElementById('chat-view').classList.remove('hidden');

  const turnsList = document.getElementById('chat-turns-list');
  if (turnsList) {
    turnsList.innerHTML = '';
    state.activeTurns.forEach((turnData, idx) => {
      appendTurnElement(idx, turnData.question || '');
      if (turnData.turn_type === 'vision') {
        renderVisionTurn(turnData, idx, { autoOpenPanel: idx === state.activeTurns.length - 1 });
      } else {
        renderConversationTurn(turnData, idx, { autoOpenPanel: idx === state.activeTurns.length - 1 });
      }
    });
  }

  updateNewChatButtonState();
  renderHistoryList();
}

function deleteHistoryChat(event, chatId) {
  if (event) event.stopPropagation();
  state.chatHistory = state.chatHistory.filter((c) => c.id !== chatId);
  if (state.activeChatId === chatId) {
    resetToHome();
  }
  saveSessionHistory();
}

function clearSessionHistory() {
  state.chatHistory = [];
  saveSessionHistory();
}

function toggleTraceSteps(turnIdx = 0) {
  const drawer = document.getElementById(`trace-accordion-${turnIdx}`);
  const hint = document.getElementById(`trace-toggle-hint-${turnIdx}`);
  if (!drawer) return;
  const isCollapsed = drawer.classList.toggle('collapsed');
  if (hint) {
    hint.textContent = isCollapsed ? 'Details' : 'Hide';
  }
}

function autoGrow(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 100) + 'px';
}

function appendTurnElement(turnIdx, questionText) {
  const turnsList = document.getElementById('chat-turns-list');
  if (!turnsList) return null;

  const existing = document.getElementById(`chat-turn-${turnIdx}`);
  if (existing) existing.remove();

  const wrapper = document.createElement('div');
  wrapper.className = 'chat-turn';
  wrapper.id = `chat-turn-${turnIdx}`;
  wrapper.dataset.turnIndex = String(turnIdx);
  wrapper.innerHTML = `
    <div class="user-row">
      <div class="user-pill" id="user-question-text-${turnIdx}">${escapeHtml(questionText)}</div>
    </div>

    <div class="assistant-card">
      <div class="trace-drawer collapsed" id="trace-accordion-${turnIdx}">
        <button type="button" class="trace-pill-trigger" onclick="toggleTraceSteps(${turnIdx})" title="Click to inspect tool calls, arguments, and telemetry">
          <div class="trace-trigger-left">
            <span class="trace-dot" id="trace-dot-${turnIdx}"></span>
            <span class="trace-label" id="trace-summary-text-${turnIdx}">Reasoning &amp; tools</span>
            <div class="trace-chips" id="trace-chips-${turnIdx}"></div>
          </div>
          <div class="trace-trigger-right">
            <span class="trace-hint" id="trace-toggle-hint-${turnIdx}">Details</span>
            <svg class="trace-chevron" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M6 9l6 6 6-6" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
          </div>
        </button>

        <div class="trace-details-panel" id="trace-steps-body-${turnIdx}"></div>
      </div>

      <div class="answer-body" id="answer-body-${turnIdx}"></div>
      <div id="turn-extra-widget-${turnIdx}"></div>

      <div class="answer-footer" id="answer-footer-${turnIdx}">
        <div class="source-pills-row" id="source-pills-row-${turnIdx}"></div>
        <div class="meta-line" id="meta-line-${turnIdx}"></div>
      </div>
    </div>
  `;
  turnsList.appendChild(wrapper);
  return wrapper;
}

// ============================================================================
// Live Execution: Agent & RAG Queries (`/api/chat`)
// ============================================================================
async function selectPresetScenario(scenarioId) {
  if (document.body.classList.contains('is-thinking')) return;
  const sc = state.presets.find((s) => s.id === scenarioId);
  if (!sc) return;
  if (sc.mode && sc.mode !== state.mode) {
    setMode(sc.mode);
  }
  await executeQuery({ question: sc.query, mode: state.mode, isFollowUp: false });
}

async function submitFromHome() {
  if (document.body.classList.contains('is-thinking')) return;
  if (state.mode === 'vision') {
    await submitVisionFromHome();
    return;
  }
  const input = document.getElementById('home-input');
  const q = (input ? input.value : '').trim();
  if (!q) return;
  input.value = '';
  await executeQuery({ question: q, mode: state.mode, isFollowUp: false });
}

async function submitFromChat() {
  if (document.body.classList.contains('is-thinking')) return;
  const input = document.getElementById('chat-input');
  const q = (input ? input.value : '').trim();
  if (!q) return;
  input.value = '';
  input.style.height = 'auto';
  const runMode = state.mode === 'vision' ? 'agent' : state.mode;
  await executeQuery({ question: q, mode: runMode, isFollowUp: true });
}

async function launchFromHomeIfNeeded(isFollowUp) {
  const homeStage = document.getElementById('home-stage');
  const chatView = document.getElementById('chat-view');
  const turnsList = document.getElementById('chat-turns-list');

  if (!isFollowUp) {
    state.activeTurns = [];
    state.activeChatId = null;
    if (turnsList) turnsList.innerHTML = '';
    closeSourcePanel();
  }

  if (!isFollowUp && homeStage && !homeStage.classList.contains('hidden')) {
    homeStage.classList.add('is-launching');
    document.body.classList.add('is-launching');
    setTimeout(() => document.body.classList.remove('is-launching'), 1150);
    await new Promise((resolve) => setTimeout(resolve, 680));
    homeStage.classList.add('hidden');
    homeStage.classList.remove('is-launching');
  }
  if (chatView) chatView.classList.remove('hidden');
}

async function executeQuery(payload) {
  const qText = payload.question || '';
  const isFollowUp = Boolean(payload.isFollowUp && state.activeTurns.length > 0);
  await launchFromHomeIfNeeded(isFollowUp);

  const turnIdx = state.activeTurns.length;
  const turnWrapper = appendTurnElement(turnIdx, qText);
  if (isFollowUp && turnWrapper) {
    setTimeout(() => turnWrapper.scrollIntoView({ behavior: 'smooth', block: 'end' }), 40);
  }

  const dot = document.getElementById(`trace-dot-${turnIdx}`);
  const summaryText = document.getElementById(`trace-summary-text-${turnIdx}`);
  const chipsEl = document.getElementById(`trace-chips-${turnIdx}`);
  const answerEl = document.getElementById(`answer-body-${turnIdx}`);

  dot.classList.add('running');
  document.body.classList.add('is-thinking');
  const startTs = performance.now();
  const timerChipId = `live-timer-chip-${turnIdx}`;

  const stages = [
    { title: 'Parsing astronomical intent & coordinates', detail: 'Extracting target IDs, location, date & optical parameters…' },
    { title: 'Hybrid RAG retrieval (BM25 + Qdrant + CrossEncoder)', detail: `Searching ${state.corpusDocs} sky objects & ${state.indexedChunks.toLocaleString('en-US')} encyclopedia chunks…` },
    { title: 'Computing night ephemeris & target altitudes', detail: 'Evaluating twilight window, Moon phase & horizon constraints…' },
    { title: 'Synthesizing & verifying citations', detail: 'Checking grounded [doc_id] references against retrieved set…' },
  ];

  let activeStageIdx = 0;
  function renderLiveStages() {
    const elapsed = ((performance.now() - startTs) / 1000).toFixed(1);
    summaryText.textContent = `${stages[Math.min(activeStageIdx, stages.length - 1)].title}…`;
    chipsEl.innerHTML = `<span class="trace-chip active-tool">⚙ step ${Math.min(activeStageIdx + 1, 4)}/4</span><span class="trace-chip" id="${timerChipId}">${elapsed}s</span>`;
    answerEl.innerHTML = `
      <div class="live-activity-box">
        ${stages
          .slice(0, activeStageIdx + 1)
          .map((st, idx) => {
            const isRunning = idx === activeStageIdx;
            return `
              <div class="live-activity-row ${isRunning ? 'running' : 'done'}">
                ${isRunning ? '<span class="live-step-spinner"></span>' : '<span class="live-step-check">✓</span>'}
                <div class="live-activity-text">
                  <div class="live-activity-title"><span>${escapeHtml(st.title)}</span></div>
                  <div class="live-activity-detail">${escapeHtml(st.detail)}</div>
                </div>
              </div>
            `;
          })
          .join('')}
      </div>
    `;
  }

  renderLiveStages();
  const stageTimer = setInterval(() => {
    if (activeStageIdx < stages.length - 1) {
      activeStageIdx += 1;
      renderLiveStages();
    } else {
      const elapsed = ((performance.now() - startTs) / 1000).toFixed(1);
      const chip = document.getElementById(timerChipId);
      if (chip) chip.textContent = `${elapsed}s`;
    }
  }, 280);

  try {
    const prof = state.equipmentProfile && state.equipmentProfile.enabled ? state.equipmentProfile : null;
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: qText,
        mode: payload.mode || state.mode || 'agent',
        llm_mode: state.llmMode,
        focal_length_mm: prof ? prof.focal_length_mm : 480.0,
        aperture_mm: prof ? prof.aperture_mm : 80.0,
        equipment_profile: prof,
      }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    const turnData = {
      turn_type: 'chat',
      question: qText,
      mode: data.mode || payload.mode || 'agent',
      ...data,
    };

    state.activeTurns[turnIdx] = turnData;
    state.currentTrace = turnData;
    recordCompletedTurn(turnData, { isFollowUp });
    renderConversationTurn(turnData, turnIdx, { autoOpenPanel: true });
  } catch (err) {
    answerEl.innerHTML = `<p style="color:var(--red)">Request failed: ${escapeHtml(err.message)}</p>`;
  } finally {
    clearInterval(stageTimer);
    dot.classList.remove('running');
    document.body.classList.remove('is-thinking');
  }
}

// ============================================================================
// Live Execution: Sky Vision Turn (`/api/vision/identify` — Alessandro's Tool)
// ============================================================================
async function executeVisionTurn(titleText, visionPayload) {
  const isFollowUp = Boolean(state.activeTurns.length > 0 && !isCurrentlyOnEmptyChat());
  await launchFromHomeIfNeeded(isFollowUp);

  const turnIdx = state.activeTurns.length;
  const turnWrapper = appendTurnElement(turnIdx, titleText);
  if (isFollowUp && turnWrapper) {
    setTimeout(() => turnWrapper.scrollIntoView({ behavior: 'smooth', block: 'end' }), 40);
  }

  const dot = document.getElementById(`trace-dot-${turnIdx}`);
  const summaryText = document.getElementById(`trace-summary-text-${turnIdx}`);
  const chipsEl = document.getElementById(`trace-chips-${turnIdx}`);
  const answerEl = document.getElementById(`answer-body-${turnIdx}`);

  dot.classList.add('running');
  document.body.classList.add('is-thinking');
  summaryText.textContent = 'Running WCS plate projection & catalog overlay…';
  chipsEl.innerHTML = `<span class="trace-chip active-tool">⚙ image_catalog.py</span><span class="trace-chip active-tool">⚙ visualize.py</span>`;
  answerEl.innerHTML = `
    <div class="live-activity-box">
      <div class="live-activity-row running">
        <span class="live-step-spinner"></span>
        <div class="live-activity-text">
          <div class="live-activity-title"><span>Projecting ${state.localizableDocs} J2000 catalog centres onto image WCS</span></div>
          <div class="live-activity-detail">Computing spherical gnomonic TAN coordinates &amp; rendering cyan overlay markers…</div>
        </div>
      </div>
    </div>
  `;

  try {
    const res = await fetch('/api/vision/identify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(visionPayload),
    });
    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody.detail || `Server returned ${res.status}`);
    }
    const data = await res.json();

    const turnData = {
      turn_type: 'vision',
      question: titleText,
      mode: 'vision',
      ...data,
    };

    state.activeTurns[turnIdx] = turnData;
    state.currentTrace = turnData;
    recordCompletedTurn(turnData, { isFollowUp });
    renderVisionTurn(turnData, turnIdx, { autoOpenPanel: true });
  } catch (err) {
    answerEl.innerHTML = `<p style="color:var(--red)">Vision error: ${escapeHtml(err.message)}</p>`;
  } finally {
    dot.classList.remove('running');
    document.body.classList.remove('is-thinking');
  }
}

// ============================================================================
// Render Conversation Turn (Agent & RAG)
// ============================================================================
function renderConversationTurn(data, turnIdx = 0, { autoOpenPanel = true } = {}) {
  const ans = data.answer || {};
  const tel = data.telemetry || {};
  const toolObs = Array.isArray(data.tool_observations) ? data.tool_observations : [];
  const sources = Array.isArray(data.sources_full) ? data.sources_full : [];

  // 1. Discreet Summary Pill
  const summaryEl = document.getElementById(`trace-summary-text-${turnIdx}`);
  if (summaryEl) {
    if (ans.abstained) {
      summaryEl.textContent = `Scope & prudence guardrail (${tel.stop_reason || 'abstain'})`;
    } else if (data.mode === 'rag') {
      summaryEl.textContent = 'Hybrid RAG retrieval';
    } else {
      summaryEl.textContent = `${toolObs.length} tool${toolObs.length === 1 ? '' : 's'} called`;
    }
  }

  const chipsEl = document.getElementById(`trace-chips-${turnIdx}`);
  if (chipsEl) {
    const uniqueTools = [...new Set(toolObs.map((t) => t.name))];
    const toolChips = uniqueTools.map((t) => `<span class="trace-chip">${escapeHtml(t)}</span>`).join('');
    const verChip = ans.abstained
      ? `<span class="trace-chip abstained">guardrail ✓</span>`
      : `<span class="trace-chip verified">verified ✓</span>`;
    chipsEl.innerHTML = toolChips + verChip;
  }

  // 2. Detailed Tool Step Cards inside Expandable Drawer
  const stepsEl = document.getElementById(`trace-steps-body-${turnIdx}`);
  if (stepsEl) {
    stepsEl.innerHTML = toolObs.map((tc, i) => renderToolStepCard(tc, i + 1)).join('');
  }

  // 3. Staggered Answer Prose with Clickable [M031] Citation Buttons
  const answerEl = document.getElementById(`answer-body-${turnIdx}`);
  if (answerEl) {
    answerEl.innerHTML = formatAnswerWithCitations(ans.answer || '', turnIdx);
  }

  // 4. Optional Observation Session Plan & Altitude Curve Widget
  const extraEl = document.getElementById(`turn-extra-widget-${turnIdx}`);
  if (extraEl) {
    extraEl.innerHTML = '';
    if (!ans.abstained && Array.isArray(ans.plan) && ans.plan.length > 0) {
      extraEl.innerHTML = buildSessionPlanHtml(data, turnIdx);
      renderTurnAltitudeChart(data.visibility_objects || {}, ans.plan, turnIdx);
    }
  }

  // 5. Footer Source Pills & Telemetry Line
  const cited = sources.filter((s) => s.is_cited);
  const displaySources = cited.length > 0 ? cited : sources.slice(0, 6);
  const pillsRow = document.getElementById(`source-pills-row-${turnIdx}`);
  if (pillsRow) {
    pillsRow.innerHTML = displaySources
      .map(
        (s) => `
        <button type="button" class="src-pill" data-doc-id="${escapeHtml(s.doc_id)}" onclick="openSourcePanel('${escapeHtml(s.doc_id)}', ${turnIdx})">
          <span class="src-dot"></span>
          <span>${escapeHtml(s.doc_id)} · ${escapeHtml(shortTitle(s.title))}</span>
        </button>
      `
      )
      .join('');
  }

  const metaEl = document.getElementById(`meta-line-${turnIdx}`);
  if (metaEl) {
    const secs = Number(tel.latency_s || 0).toFixed(2);
    const cCount = (ans.sources || []).length;
    metaEl.textContent = ans.abstained
      ? `stop='${tel.stop_reason || 'abstain'}' · ${secs}s`
      : `${cCount}/${cCount} citations verified · ${secs}s · ${toolObs.length} tools`;
  }

  if (autoOpenPanel) {
    if (displaySources.length > 0 && !ans.abstained) {
      openSourcePanel(displaySources[0].doc_id, turnIdx);
    } else if (turnIdx === 0) {
      closeSourcePanel();
    }
  }
}

// ============================================================================
// Render Sky Vision Turn (Alessandro's Image Catalog + Visualize Overlay)
// ============================================================================
function renderVisionTurn(data, turnIdx = 0, { autoOpenPanel = true } = {}) {
  const vis = data.vision || {};
  const objects = Array.isArray(vis.objects) ? vis.objects : [];
  const traceSteps = Array.isArray(data.trace_steps) ? data.trace_steps : [];
  const sources = Array.isArray(data.sources_full) ? data.sources_full : [];

  // 1. Discreet Trace Header
  const summaryEl = document.getElementById(`trace-summary-text-${turnIdx}`);
  if (summaryEl) {
    summaryEl.textContent = `WCS & catalog projection (${objects.length} object${objects.length === 1 ? '' : 's'} matched)`;
  }

  const chipsEl = document.getElementById(`trace-chips-${turnIdx}`);
  if (chipsEl) {
    chipsEl.innerHTML = `
      <span class="trace-chip">image_catalog.py</span>
      <span class="trace-chip">visualize.py</span>
      <span class="trace-chip verified">schema v${escapeHtml(vis.schema_version || '1.0')} ✓</span>
    `;
  }

  const stepsEl = document.getElementById(`trace-steps-body-${turnIdx}`);
  if (stepsEl) {
    stepsEl.innerHTML = traceSteps.map((tc, i) => renderToolStepCard(tc, i + 1)).join('');
  }

  // 2. Concise Summary Paragraph above the Annotated Sky Image
  const centerRa = Number(vis.astrometry?.centre_icrs_deg?.ra || 0).toFixed(4);
  const centerDec = Number(vis.astrometry?.centre_icrs_deg?.dec || 0).toFixed(4);
  const imgW = vis.image?.width_px || 960;
  const imgH = vis.image?.height_px || 640;
  const cdelt = Math.abs(Number(vis.astrometry?.wcs_header?.CDELT1 || 0.005));
  const fovW = (cdelt * imgW).toFixed(2);
  const fovH = (cdelt * imgH).toFixed(2);
  const scaleArcsec = (cdelt * 3600).toFixed(2);
  const foundBadges = objects
    .map(
      (o) =>
        `<button type="button" class="cite-btn" onclick="openSourcePanel('${escapeHtml(o.doc_id)}', ${turnIdx})" onmouseenter="highlightSourceById('${escapeHtml(o.doc_id)}', true)" onmouseleave="highlightSourceById('${escapeHtml(o.doc_id)}', false)">${escapeHtml(o.doc_id)}</button>`
    )
    .join(' ');

  const answerEl = document.getElementById(`answer-body-${turnIdx}`);
  if (answerEl) {
    answerEl.innerHTML = `
      <p>
        Projected <strong>${vis.catalogue?.eligible_documents || state.localizableDocs}</strong> J2000 catalog objects onto the
        <span class="mono">${imgW}×${imgH} px</span> celestial WCS footprint centered at
        <span class="mono">RA ${centerRa}°, Dec ${centerDec}°</span>
        (field of view <span class="mono">${fovW}° × ${fovH}°</span>, pixel scale <span class="mono">${scaleArcsec}"/px</span>).
        Identified <strong>${objects.length}</strong> catalogued deep-sky object centre${objects.length === 1 ? '' : 's'} inside the frame:
        ${foundBadges || '<span class="mono">none</span>'}.
      </p>
    `;
  }

  // 3. Annotated Sky Photo Showcase + Detected Objects Table
  const extraEl = document.getElementById(`turn-extra-widget-${turnIdx}`);
  if (extraEl) {
    const rowsHtml = objects
      .map((obj) => {
        const meta = obj.metadata || {};
        const px = obj.pixel || {};
        const otype = String(meta.object_type || 'object').replace(/_/g, ' ');
        return `
          <tr onclick="openSourcePanel('${escapeHtml(obj.doc_id)}', ${turnIdx})" style="cursor:pointer">
            <td>
              <button type="button" class="cite-btn" onclick="event.stopPropagation();openSourcePanel('${escapeHtml(obj.doc_id)}', ${turnIdx})">${escapeHtml(obj.doc_id)}</button>
            </td>
            <td>
              <strong>${escapeHtml(obj.title)}</strong><br/>
              <span style="font-size:0.72rem;color:var(--text-muted)">${escapeHtml(otype)} · ${escapeHtml(meta.constellation || '')}</span>
            </td>
            <td class="mono">(${Number(px.x).toFixed(1)}, ${Number(px.y).toFixed(1)})</td>
            <td class="mono">${Number(px.distance_to_edge_px).toFixed(1)} px</td>
            <td class="mono">${meta.magnitude ?? '—'}</td>
            <td class="mono">${Number(meta.ra_deg).toFixed(3)}°, ${Number(meta.dec_deg).toFixed(3)}°</td>
          </tr>
        `;
      })
      .join('');

    extraEl.innerHTML = `
      <div class="vision-showcase-card">
        <div class="vision-toolbar">
          <div class="vision-view-toggle">
            <button type="button" class="vision-toggle-btn active" id="vis-btn-ann-${turnIdx}" onclick="switchVisionImage(${turnIdx}, 'annotated')">
              Annotated Overlay (visualize.py)
            </button>
            <button type="button" class="vision-toggle-btn" id="vis-btn-orig-${turnIdx}" onclick="switchVisionImage(${turnIdx}, 'original')">
              Original Sky Photo
            </button>
          </div>

          <div class="vision-action-group">
            <a class="vision-act-btn" href="${data.annotated_image_data_url}" download="astropilote_${turnIdx}_annotated.png">
              ⬇ Annotated PNG
            </a>
            <button type="button" class="vision-act-btn" onclick="downloadVisionCsv(${turnIdx})">
              ⬇ Objects CSV
            </button>
            ${
              objects.length > 0
                ? `<button type="button" class="vision-act-btn primary" onclick="planSessionFromVision(${turnIdx})">
                    ✨ Plan Tonight's Session for These Objects
                  </button>`
                : ''
            }
          </div>
        </div>

        <div class="vision-canvas-frame">
          <img
            id="vision-img-${turnIdx}"
            src="${data.annotated_image_data_url}"
            alt="Annotated sky photo with cyan catalog markers"
            onclick="openVisionLightbox(${turnIdx})"
          />
          <div class="vision-hud-strip">
            <span><span class="vision-hud-pill">● WCS ${escapeHtml(vis.astrometry?.method || 'TAN')}</span> · Center (${centerRa}°, ${centerDec}°)</span>
            <span>FOV ${fovW}° × ${fovH}° · ${scaleArcsec}"/px · Click image to zoom</span>
          </div>
        </div>

        ${
          objects.length > 0
            ? `
            <div class="plan-table-wrap">
              <table class="plan-table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Catalog Object &amp; Type</th>
                    <th>Pixel (x, y)</th>
                    <th>Edge Margin</th>
                    <th>Mag</th>
                    <th>J2000 (RA, Dec)</th>
                  </tr>
                </thead>
                <tbody>${rowsHtml}</tbody>
              </table>
            </div>
          `
            : ''
        }
      </div>
    `;
  }

  // 4. Footer Source Pills & Telemetry Line
  const pillsRow = document.getElementById(`source-pills-row-${turnIdx}`);
  if (pillsRow) {
    pillsRow.innerHTML = sources
      .map(
        (s) => `
        <button type="button" class="src-pill" data-doc-id="${escapeHtml(s.doc_id)}" onclick="openSourcePanel('${escapeHtml(s.doc_id)}', ${turnIdx})">
          <span class="src-dot" style="background:var(--blue)"></span>
          <span>${escapeHtml(s.doc_id)} · ${escapeHtml(shortTitle(s.title))}</span>
        </button>
      `
      )
      .join('');
  }

  const metaEl = document.getElementById(`meta-line-${turnIdx}`);
  if (metaEl) {
    metaEl.textContent = `${objects.length} catalog centre${objects.length === 1 ? '' : 's'} · ${imgW}×${imgH} px · ${Number(vis.elapsed_seconds || 0).toFixed(3)}s`;
  }

  if (autoOpenPanel && sources.length > 0) {
    openSourcePanel(sources[0].doc_id, turnIdx);
  }
}

function switchVisionImage(turnIdx, which) {
  const turnData = state.activeTurns[turnIdx];
  if (!turnData) return;
  const imgEl = document.getElementById(`vision-img-${turnIdx}`);
  const btnAnn = document.getElementById(`vis-btn-ann-${turnIdx}`);
  const btnOrig = document.getElementById(`vis-btn-orig-${turnIdx}`);
  if (!imgEl) return;

  const isAnn = which === 'annotated';
  imgEl.src = isAnn ? turnData.annotated_image_data_url : turnData.original_image_data_url;
  if (btnAnn) btnAnn.classList.toggle('active', isAnn);
  if (btnOrig) btnOrig.classList.toggle('active', !isAnn);
}

function downloadVisionCsv(turnIdx) {
  const turnData = state.activeTurns[turnIdx];
  if (!turnData || !turnData.csv_content) return;
  const blob = new Blob([turnData.csv_content], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `astropilote_vision_objects_${turnIdx}.csv`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function openVisionLightbox(turnIdx) {
  const turnData = state.activeTurns[turnIdx];
  if (!turnData) return;
  const dlg = document.getElementById('source-dialog');
  const badge = document.getElementById('modal-doc-id');
  const title = document.getElementById('modal-doc-title');
  const body = document.getElementById('modal-doc-body');
  if (!dlg || !body) return;

  if (badge) badge.textContent = 'WCS OVERLAY · VISUALIZE.PY';
  if (title) title.textContent = turnData.question || 'Annotated Sky Image';
  body.innerHTML = `<img src="${turnData.annotated_image_data_url}" alt="Full resolution annotated sky photo" />`;
  dlg.showModal();
}

async function planSessionFromVision(turnIdx) {
  const turnData = state.activeTurns[turnIdx];
  if (!turnData || !turnData.vision) return;
  const ids = (turnData.vision.objects || []).map((o) => o.doc_id);
  if (ids.length === 0) return;
  const focal = turnData.vision.input_metadata?.focal_length_mm || 480;
  const followUpQ = `I just identified ${ids.join(', ')} in my sky photo (${focal}mm focal length). Plan an observation session for these objects from Fontainebleau on October 17, 2026 and tell me how to observe each of them.`;
  setMode('agent');
  await executeQuery({ question: followUpQ, mode: 'agent', isFollowUp: true });
}

// ============================================================================
// Tool Step Cards, Answer Citations & Session Plan Builder
// ============================================================================
function renderToolStepCard(tc, index) {
  const fnName = tc.name || 'tool_call';
  const stepLabel = `STEP ${tc.step || index}`;
  const latencyBadge =
    tc.latency_ms != null ? `<span class="trace-step-latency">${tc.latency_ms} ms</span>` : '';

  let argsHtml = '';
  if (tc.arguments && typeof tc.arguments === 'object') {
    const entries = Object.entries(tc.arguments);
    if (entries.length > 0) {
      argsHtml = `<div class="trace-args-row">${entries
        .map(
          ([k, v]) =>
            `<span class="trace-arg-pill"><span class="trace-arg-key">${escapeHtml(k)}:</span> ${escapeHtml(
              JSON.stringify(v)
            )}</span>`
        )
        .join('')}</div>`;
    }
  }

  const obsText = String(tc.result || '').trim();
  const obsHtml = obsText ? `<div class="trace-step-obs">${escapeHtml(obsText)}</div>` : '';

  return `
    <div class="trace-step-card">
      <div class="trace-step-top">
        <div class="trace-step-title">
          <span class="trace-step-num">${escapeHtml(stepLabel)}</span>
          <span class="trace-step-fn">${escapeHtml(fnName)}()</span>
        </div>
        ${latencyBadge}
      </div>
      ${argsHtml}
      ${obsHtml}
    </div>
  `;
}

function formatInlineMarkdown(rawStr, turnIdx = 0, delay = 0) {
  const codeSpans = [];
  const withCodePlaceholders = String(rawStr || '').replace(/`([^`\n]+)`/g, (_, code) => {
    const idx = codeSpans.length;
    codeSpans.push(`<code class="md-inline-code mono">${escapeHtml(code)}</code>`);
    return `\u0000CODE_${idx}\u0000`;
  });

  let html = escapeHtml(withCodePlaceholders);

  // Markdown links [label](https://...)
  html = html.replace(
    /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener noreferrer" class="md-link">$1 ↗</a>'
  );

  // Bold + italic, Bold, Italic
  html = html
    .replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/__(.+?)__/g, '<strong>$1</strong>')
    .replace(/(^|[\s(])\*([^*\n]+?)\*(?=[\s.,;:!?)—-]|$)/g, '$1<em>$2</em>');

  // Bracketed corpus citations: [M031], [CON_AND], [GUIDE_BORTLE], [NGC 7000], [M031, CON_AND]
  const idPat = '(?:M|C|NGC\\s*|IC\\s*)\\d{1,4}|CON_[A-Z]{3}|GUIDE_[A-Z0-9_]+';
  const multiCiteRegex = new RegExp(`\\[((?:${idPat})(?:\\s*[,;/]\\s*(?:${idPat}))*)\\]`, 'g');
  html = html.replace(multiCiteRegex, (_, group) => {
    return group
      .split(/\s*[,;/]\s*/)
      .map((rawId) => {
        const docId = rawId.trim();
        const safeId = escapeHtml(docId);
        return `<button type="button" class="cite-btn" style="animation-delay:${delay + 60}ms" onclick="openSourcePanel('${safeId}', ${turnIdx})" onmouseenter="highlightSourceById('${safeId}', true)" onmouseleave="highlightSourceById('${safeId}', false)">${safeId}</button>`;
      })
      .join(' ');
  });

  // Restore inline code spans
  html = html.replace(/\u0000CODE_(\d+)\u0000/g, (_, idx) => codeSpans[Number(idx)] || '');
  return html;
}

function formatAnswerWithCitations(rawText, turnIdx = 0) {
  const cleaned = String(rawText || '').replace(/\r\n/g, '\n').trim();
  if (!cleaned) return '<p>No response generated.</p>';

  const lines = cleaned.split('\n');
  const htmlParts = [];
  let listType = null; // 'ul' | 'ol' | null
  let blockCounter = 0;

  const closeList = () => {
    if (listType) {
      htmlParts.push(`</${listType}>`);
      listType = null;
    }
  };

  const nextDelay = () => {
    const d = Math.min(blockCounter * 40, 360);
    blockCounter += 1;
    return d;
  };

  let i = 0;
  while (i < lines.length) {
    const rawLine = lines[i];
    const line = rawLine.trim();

    // 1. Empty line
    if (!line) {
      closeList();
      i += 1;
      continue;
    }

    // 2. Fenced code block ```...```
    if (line.startsWith('```')) {
      closeList();
      const codeLines = [];
      i += 1;
      while (i < lines.length && !lines[i].trim().startsWith('```')) {
        codeLines.push(lines[i]);
        i += 1;
      }
      if (i < lines.length) i += 1;
      const d = nextDelay();
      htmlParts.push(
        `<pre class="md-code-block md-block" style="animation-delay:${d}ms"><code class="mono">${escapeHtml(
          codeLines.join('\n')
        )}</code></pre>`
      );
      continue;
    }

    // 3. Horizontal rule (---, ***, ___)
    if (/^(?:---+|\*\*\*+|___+)\s*$/.test(line)) {
      closeList();
      htmlParts.push('<hr class="md-hr" />');
      i += 1;
      continue;
    }

    // 4. Headings (#, ##, ###, ####)
    const headingMatch = line.match(/^(#{1,4})\s+(.+?)\s*#*$/);
    if (headingMatch) {
      closeList();
      const level = headingMatch[1].length;
      const tag = level === 1 ? 'h2' : level === 2 ? 'h3' : 'h4';
      const cls = level === 1 ? 'md-h2' : level === 2 ? 'md-h3' : 'md-h4';
      const d = nextDelay();
      htmlParts.push(
        `<${tag} class="${cls} md-block" style="animation-delay:${d}ms">${formatInlineMarkdown(
          headingMatch[2],
          turnIdx,
          d
        )}</${tag}>`
      );
      i += 1;
      continue;
    }

    // 5. Markdown Table (| ... | followed by | --- |)
    if (
      line.startsWith('|') &&
      line.endsWith('|') &&
      i + 1 < lines.length &&
      /^\s*\|[\s:|-]+\|\s*$/.test(lines[i + 1])
    ) {
      closeList();
      const parseCells = (rowStr) =>
        rowStr
          .trim()
          .replace(/^\||\|$/g, '')
          .split('|')
          .map((c) => c.trim());

      const headers = parseCells(line);
      i += 2; // Skip header and separator rows
      const bodyRows = [];
      while (i < lines.length && lines[i].trim().startsWith('|') && lines[i].trim().endsWith('|')) {
        bodyRows.push(parseCells(lines[i]));
        i += 1;
      }
      const d = nextDelay();
      const theadHtml = headers.map((h) => `<th>${formatInlineMarkdown(h, turnIdx, d)}</th>`).join('');
      const tbodyHtml = bodyRows
        .map(
          (row) =>
            `<tr>${row.map((cell) => `<td>${formatInlineMarkdown(cell, turnIdx, d)}</td>`).join('')}</tr>`
        )
        .join('');
      htmlParts.push(
        `<div class="plan-table-wrap md-table-wrap md-block" style="animation-delay:${d}ms"><table class="plan-table"><thead><tr>${theadHtml}</tr></thead><tbody>${tbodyHtml}</tbody></table></div>`
      );
      continue;
    }

    // 6. Blockquote (> ...)
    if (line.startsWith('>')) {
      closeList();
      const quoteLines = [];
      while (i < lines.length && lines[i].trim().startsWith('>')) {
        quoteLines.push(lines[i].trim().replace(/^>\s?/, ''));
        i += 1;
      }
      const d = nextDelay();
      htmlParts.push(
        `<blockquote class="md-quote md-block" style="animation-delay:${d}ms">${formatInlineMarkdown(
          quoteLines.join(' '),
          turnIdx,
          d
        )}</blockquote>`
      );
      continue;
    }

    // 7. Unordered list (- ..., * ..., • ...)
    const ulMatch = line.match(/^[-*•]\s+(.+)$/);
    if (ulMatch) {
      if (listType !== 'ul') {
        closeList();
        htmlParts.push('<ul class="md-list">');
        listType = 'ul';
      }
      const d = nextDelay();
      htmlParts.push(
        `<li class="md-block" style="animation-delay:${d}ms">${formatInlineMarkdown(
          ulMatch[1],
          turnIdx,
          d
        )}</li>`
      );
      i += 1;
      continue;
    }

    // 8. Ordered list (1. ..., 2) ...)
    const olMatch = line.match(/^(\d+)[.)]\s+(.+)$/);
    if (olMatch) {
      if (listType !== 'ol') {
        closeList();
        htmlParts.push('<ol class="md-list md-ol">');
        listType = 'ol';
      }
      const d = nextDelay();
      htmlParts.push(
        `<li value="${escapeHtml(olMatch[1])}" class="md-block" style="animation-delay:${d}ms">${formatInlineMarkdown(
          olMatch[2],
          turnIdx,
          d
        )}</li>`
      );
      i += 1;
      continue;
    }

    // 9. Standard paragraph (join consecutive plain text lines)
    closeList();
    const paraLines = [line];
    i += 1;
    while (i < lines.length) {
      const nextL = lines[i].trim();
      if (
        !nextL ||
        nextL.startsWith('```') ||
        nextL.startsWith('#') ||
        nextL.startsWith('>') ||
        nextL.startsWith('|') ||
        /^(?:---+|\*\*\*+|___+)\s*$/.test(nextL) ||
        /^[-*•]\s+/.test(nextL) ||
        /^\d+[.)]\s+/.test(nextL)
      ) {
        break;
      }
      paraLines.push(nextL);
      i += 1;
    }
    const d = nextDelay();
    htmlParts.push(
      `<p class="md-p md-block" style="animation-delay:${d}ms">${formatInlineMarkdown(
        paraLines.join(' '),
        turnIdx,
        d
      )}</p>`
    );
  }

  closeList();
  return htmlParts.join('');
}

function buildSessionPlanHtml(data, turnIdx) {
  const ans = data.answer || {};
  const night = data.night || {};
  const weather = data.weather || {};
  const plan = ans.plan || [];

  const darkWindow = `${night.astronomical_dark_start || '21:00'} → ${night.astronomical_dark_end || '05:30'}`;
  const moonPct = night.moon_illumination_pct ?? 0;
  const moonSub =
    moonPct <= 35 ? 'Dark sky — ideal for faint galaxies & nebulae' : 'Bright Moon — prioritize clusters';
  const clouds = weather.cloud_cover_pct ?? 0;
  const bortle = weather.bortle ?? 4;

  const rowsHtml = plan
    .map((item) => {
      const startStr = String(item.start || '').slice(0, 5);
      const endStr = String(item.end || '').slice(0, 5);
      const dirFull = COMPASS_LABELS[item.direction] || item.direction;
      return `
        <tr onclick="openSourcePanel('${escapeHtml(item.object_id)}', ${turnIdx})" style="cursor:pointer">
          <td class="mono">${escapeHtml(startStr)} – ${escapeHtml(endStr)}</td>
          <td>
            <strong>${escapeHtml(item.name)}</strong>
            <button type="button" class="cite-btn" onclick="event.stopPropagation();openSourcePanel('${escapeHtml(item.object_id)}', ${turnIdx})">${escapeHtml(item.object_id)}</button>
          </td>
          <td>${escapeHtml(dirFull)} · <span class="mono">${Number(item.altitude_deg).toFixed(0)}°</span></td>
          <td style="color:var(--text-secondary)">${escapeHtml(item.imaging_notes || '')}</td>
        </tr>
      `;
    })
    .join('');

  return `
    <div class="session-plan-card">
      <div class="night-telemetry-grid">
        <div class="night-metric-box">
          <span class="night-metric-label">🌙 Astronomical Darkness</span>
          <span class="night-metric-val">${escapeHtml(darkWindow)}</span>
          <span class="night-metric-sub">Sun below -18° altitude</span>
        </div>
        <div class="night-metric-box">
          <span class="night-metric-label">🌕 Moon Illumination</span>
          <span class="night-metric-val">${escapeHtml(String(moonPct))}% lit</span>
          <span class="night-metric-sub">${escapeHtml(moonSub)}</span>
        </div>
        <div class="night-metric-box">
          <span class="night-metric-label">☁️ Cloud &amp; Bortle Forecast</span>
          <span class="night-metric-val">${escapeHtml(String(clouds))}% clouds · Bortle ${escapeHtml(String(bortle))}</span>
          <span class="night-metric-sub">Clear observation window</span>
        </div>
      </div>

      <div class="plan-table-wrap">
        <table class="plan-table">
          <thead>
            <tr>
              <th>Window</th>
              <th>Target Object</th>
              <th>Sky Direction &amp; Peak Alt</th>
              <th>Equipment &amp; Framing Notes</th>
            </tr>
          </thead>
          <tbody>${rowsHtml}</tbody>
        </table>
      </div>

      <div class="altitude-chart-box" id="altitude-box-${turnIdx}">
        <div class="altitude-chart-header">
          <span>Target Altitude Tonight (0° Horizon · 90° Zenith · Dashed = 20° threshold)</span>
          <div class="altitude-legend" id="altitude-legend-${turnIdx}"></div>
        </div>
        <svg id="altitude-svg-${turnIdx}" viewBox="0 0 680 185" style="width:100%;height:auto;display:block"></svg>
      </div>
    </div>
  `;
}

function renderTurnAltitudeChart(visObjects, planItems, turnIdx) {
  const svg = document.getElementById(`altitude-svg-${turnIdx}`);
  const legend = document.getElementById(`altitude-legend-${turnIdx}`);
  const box = document.getElementById(`altitude-box-${turnIdx}`);
  if (!svg || !legend || !box) return;

  const keys = (planItems || [])
    .map((p) => p.object_id)
    .filter((id) => visObjects[id] && Array.isArray(visObjects[id].slots));

  if (keys.length === 0) {
    box.classList.add('hidden');
    return;
  }

  const W = 680;
  const H = 185;
  const padL = 38;
  const padR = 16;
  const padT = 14;
  const padB = 24;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const yForAlt = (alt) => padT + plotH * (1 - Math.max(0, Math.min(90, alt)) / 90);

  [0, 20, 45, 90].forEach((alt) => {
    const y = yForAlt(alt);
    const isFloor = alt === 20;
    svg.insertAdjacentHTML(
      'beforeend',
      `<line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}"
        stroke="${isFloor ? '#FBBF24' : 'rgba(148, 163, 184, 0.18)'}"
        stroke-dasharray="${isFloor ? '5,4' : 'none'}" stroke-width="1" />
       <text x="${padL - 6}" y="${y + 3}" text-anchor="end" fill="#94A3B8" font-size="9.5" font-family="JetBrains Mono">${alt}°</text>`
    );
  });

  const sampleSlots = visObjects[keys[0]].slots;
  const nSlots = sampleSlots.length;
  const xForIdx = (i) => padL + (i / Math.max(1, nSlots - 1)) * plotW;

  sampleSlots.forEach((slot, i) => {
    if (i % 3 === 0 || i === nSlots - 1) {
      const x = xForIdx(i);
      svg.insertAdjacentHTML(
        'beforeend',
        `<text x="${x}" y="${H - 6}" text-anchor="middle" fill="#94A3B8" font-size="9.5" font-family="JetBrains Mono">${slot.time}</text>`
      );
    }
  });

  keys.slice(0, 5).forEach((oid, idx) => {
    const color = CHART_COLORS[idx % CHART_COLORS.length];
    const obj = visObjects[oid];
    const pts = obj.slots
      .map((s, i) => `${xForIdx(i).toFixed(1)},${yForAlt(s.altitude_deg).toFixed(1)}`)
      .join(' ');
    svg.insertAdjacentHTML(
      'beforeend',
      `<polyline fill="none" stroke="${color}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" points="${pts}" />`
    );

    const chip = document.createElement('span');
    chip.className = 'legend-chip';
    chip.innerHTML = `<span class="legend-dot" style="background:${color}"></span>${escapeHtml(oid)}`;
    chip.addEventListener('click', () => openSourcePanel(oid, turnIdx));
    legend.appendChild(chip);
  });
}

// ============================================================================
// Right-Side Floating Deep-Sky Encyclopedia, Wikipedia & WCS Inspector Panel
// ============================================================================
function _resolveSourcesForTurn(turnIdx) {
  const turnTrace =
    turnIdx != null && state.activeTurns[turnIdx] ? state.activeTurns[turnIdx] : state.currentTrace;
  const turnSources = (turnTrace && Array.isArray(turnTrace.sources_full) && turnTrace.sources_full) || [];
  const seen = new Set(turnSources.map((s) => s.doc_id));
  const extra = [];
  for (const t of state.activeTurns) {
    for (const s of (t && t.sources_full) || []) {
      if (s && s.doc_id && !seen.has(s.doc_id)) {
        seen.add(s.doc_id);
        extra.push(s);
      }
    }
  }
  return { turnTrace: turnTrace || state.currentTrace, sources: [...turnSources, ...extra] };
}

function highlightSourceById(docId, active) {
  document.querySelectorAll('.src-pill').forEach((p) => {
    if (p.dataset.docId === docId) {
      p.classList.toggle('pulse-highlight', active);
    }
  });
  const panel = document.getElementById('source-panel');
  if (panel && !panel.classList.contains('hidden')) {
    panel.classList.toggle('pulse-highlight', active);
  }
}

async function openSourcePanel(docId, turnIdx = null) {
  if (!docId) return;
  const { turnTrace, sources } = _resolveSourcesForTurn(turnIdx);
  if (turnTrace) state.currentTrace = turnTrace;

  let target = sources.find((s) => s.doc_id.toUpperCase() === docId.toUpperCase());
  const prof = state.equipmentProfile || DEFAULT_EQUIPMENT_PROFILE;
  if (!target || !target.wikipedia_url || !target.shooting_guide) {
    try {
      const qs = new URLSearchParams({
        focal_length_mm: String(prof.focal_length_mm || 480),
        aperture_mm: String(prof.aperture_mm || 80),
        sensor_format: String(prof.sensor_format || 'APS-C'),
        mount_type: String(prof.mount_type || 'equatorial_goto'),
        has_autoguiding: String(Boolean(prof.has_autoguiding)),
      });
      const resp = await fetch(`/api/catalog/${encodeURIComponent(docId)}?${qs.toString()}`);
      if (resp.ok) {
        const fetched = await resp.json();
        target = target
          ? {
              ...fetched,
              ...target,
              wikipedia_url: fetched.wikipedia_url,
              wikipedia_page: fetched.wikipedia_page,
              chunks: fetched.chunks,
              shooting_guide: fetched.shooting_guide || target.shooting_guide,
              framing_advice: fetched.framing_advice || target.framing_advice,
            }
          : fetched;
      }
    } catch (e) {
      // ignore
    }
  }
  if (!target) return;

  document.querySelectorAll('.src-pill').forEach((p) => {
    p.classList.toggle('active', p.dataset.docId === target.doc_id);
  });

  const panel = document.getElementById('source-panel');
  panel.classList.remove('hidden');

  const eyebrowEl = document.getElementById('panel-eyebrow');
  if (eyebrowEl) {
    eyebrowEl.textContent = target.vision_pixel
      ? 'ASTROPILOTE · WCS VISION & WIKIPEDIA'
      : 'ASTROPILOTE · DEEP-SKY CATALOG';
  }
  document.getElementById('panel-title').textContent = target.title || target.doc_id;

  const cited = sources.filter((s) => s.is_cited);
  const tabList = cited.length > 0 ? cited : sources.slice(0, 6);
  const turnArg = turnIdx != null ? `, ${turnIdx}` : '';
  const tabsEl = document.getElementById('panel-tabs');
  tabsEl.innerHTML =
    tabList.length > 1
      ? tabList
          .map(
            (s) => `
          <button type="button" class="panel-tab ${s.doc_id === target.doc_id ? 'active' : ''}" onclick="openSourcePanel('${escapeHtml(s.doc_id)}'${turnArg})">
            ${escapeHtml(s.doc_id)}
          </button>
        `
          )
          .join('')
      : '';

  const widgetEl = document.getElementById('panel-widget');
  widgetEl.classList.remove('hidden');
  const magStr = target.magnitude != null ? `${target.magnitude}` : '—';
  const sizeStr = target.size_major_arcmin != null ? `${target.size_major_arcmin}'` : '—';
  const raStr = target.ra_deg != null ? `${Number(target.ra_deg).toFixed(2)}°` : '—';
  const decStr = target.dec_deg != null ? `${Number(target.dec_deg).toFixed(2)}°` : '—';
  const constStr = target.constellation || '—';

  const visionBanner = target.vision_pixel
    ? `
      <div class="mono" style="font-size:0.68rem;font-weight:700;color:var(--blue);background:var(--blue-pill);padding:5px 8px;border-radius:8px;margin-bottom:8px;">
        ◎ WCS PIXEL (${Number(target.vision_pixel.x).toFixed(1)}, ${Number(target.vision_pixel.y).toFixed(1)}) · Edge ${Number(target.vision_pixel.distance_to_edge_px).toFixed(1)} px
      </div>
    `
    : `
      <div class="mono" style="font-size:0.67rem;font-weight:600;color:var(--blue);margin-bottom:6px;">
        ASTROMETRY · J2000 (RA ${escapeHtml(raStr)}, Dec ${escapeHtml(decStr)})
      </div>
    `;

  widgetEl.innerHTML = `
    ${visionBanner}
    <table class="astro-mini-grid">
      <thead>
        <tr>
          <th>Visual Mag</th>
          <th>Angular Size</th>
          <th>Constellation</th>
          <th>Chunks</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td class="active-cell">${escapeHtml(magStr)}</td>
          <td>${escapeHtml(sizeStr)}</td>
          <td>${escapeHtml(constStr)}</td>
          <td>${escapeHtml(String(target.num_chunks || 4))}</td>
        </tr>
      </tbody>
    </table>
  `;

  const badgeClass = target.vision_pixel ? 'vision' : 'vigueur';
  const badgeLabel = target.vision_pixel ? '● WCS IN-FRAME' : '● WIKIPEDIA CORPUS';
  const hierarchyText =
    target.hierarchy ||
    `${target.catalog || 'Catalog'} › ${target.constellation || 'Sky'} › ${(target.object_type || 'object').replace(/_/g, ' ')}`;

  const wikiPage = target.wikipedia_page || target.title || target.doc_id;
  const wikiUrl =
    target.wikipedia_url ||
    `https://en.wikipedia.org/wiki/${encodeURIComponent(String(wikiPage).replace(/ /g, '_'))}`;

  const wikiBannerHtml = `
    <div class="wiki-source-card">
      <div class="wiki-source-info">
        <span class="wiki-source-kicker">WIKIPEDIA ARTICLE SOURCE</span>
        <span class="wiki-source-page" title="${escapeHtml(wikiPage)}">${escapeHtml(wikiPage)}</span>
      </div>
      <a href="${escapeHtml(wikiUrl)}" target="_blank" rel="noopener noreferrer" class="wiki-open-btn">
        <span>Wikipedia</span>
        <span>↗</span>
      </a>
    </div>
  `;

  const clausesHtml = formatCatalogDocumentBody(target);
  const sg = target.shooting_guide || {};
  const sgOri = sg.orientation || {};
  const sgApp = sg.appearance || {};
  const sgCam = sg.camera_settings || {};
  const isDso = !String(target.doc_id || '').startsWith('CON_') && !String(target.doc_id || '').startsWith('GUIDE_');

  const shootingGuideHtml =
    isDso && (sgOri.how_to_find || target.framing_advice)
      ? `
      <div class="panel-nota-card shooting-guide-card">
        <div class="panel-nota-title">Shooting &amp; Star-Hopping Guide (${escapeHtml(sgCam.focal_length_mm ? `${Math.round(sgCam.focal_length_mm)}mm ${sgCam.f_ratio || ''} · ${sgCam.sensor_format || 'APS-C'}` : '480mm f/6 APS-C')})</div>
        <div class="sg-section">
          <strong class="sg-kicker">How to Orient &amp; Locate</strong>
          <div>Constellation <strong>${escapeHtml(sgOri.constellation_name || target.constellation || '—')}</strong> (${escapeHtml(sgOri.ra_hms || '')}, ${escapeHtml(sgOri.dec_dms || '')}) · Size: ${escapeHtml(sgOri.size_vs_full_moon || '—')}.</div>
          <div style="margin-top:3px;color:var(--text-secondary)">${escapeHtml(sgOri.how_to_find || '')}</div>
        </div>
        <div class="sg-section">
          <strong class="sg-kicker">Visual &amp; Photographic Appearance</strong>
          <div style="color:var(--text-secondary)">${escapeHtml(sgApp.visual_and_photo_look || '')}</div>
        </div>
        <div class="sg-section">
          <strong class="sg-kicker">Recommended Camera Settings</strong>
          <div>• <strong>Framing:</strong> ${escapeHtml(target.framing_advice || sgCam.fov_arcmin || '—')}</div>
          <div>• <strong>Sub-exposure:</strong> ${escapeHtml(sgCam.recommended_sub_exposure || '—')}</div>
          <div>• <strong>ISO / Gain:</strong> ${escapeHtml(sgCam.recommended_iso_or_gain || '—')}</div>
          <div>• <strong>Filter:</strong> ${escapeHtml(sgCam.filter_recommendation || '—')}</div>
        </div>
      </div>
    `
      : target.framing_advice
      ? `
      <div class="panel-nota-card">
        <div class="panel-nota-title">Optical FOV Framing</div>
        <div>${escapeHtml(target.framing_advice)}</div>
      </div>
    `
      : '';

  document.getElementById('panel-content').innerHTML = `
    ${wikiBannerHtml}
    <div class="panel-meta-badges">
      <span class="etat-badge ${badgeClass}">${escapeHtml(badgeLabel)}</span>
      <span class="docid-chip">DOC_ID · ${escapeHtml(target.doc_id)}</span>
    </div>
    <div class="panel-hierarchy">${escapeHtml(hierarchyText)}</div>
    ${shootingGuideHtml}
    <div class="legal-section-label">
      <span>Indexed Wikipedia Chunks</span>
      <span>${escapeHtml(target.doc_id)}</span>
    </div>
    <div class="legal-clauses-list">
      ${clausesHtml}
    </div>
  `;
  panel.scrollTop = 0;
}

function closeSourcePanel() {
  const panel = document.getElementById('source-panel');
  if (panel) panel.classList.add('hidden');
  document.querySelectorAll('.src-pill').forEach((p) => p.classList.remove('active'));
}

function renderChunkCard(heading, badge, bodyHtml) {
  return `
    <div class="legal-group-card">
      <div class="legal-group-header">
        <strong class="legal-group-title">${escapeHtml(heading)}</strong>
        <span class="clause-num-badge" title="${escapeHtml(badge)}">${escapeHtml(badge)}</span>
      </div>
      <div class="legal-group-body">${bodyHtml}</div>
    </div>
  `;
}

function stripChunkHeader(text) {
  // Drop the retrieval-only "[DOC_ID | title | Type … | Aliases …]\nSection: X" prefix of every indexed chunk
  return String(text || '').replace(/^\[[^\]]*\]\s*(?:Section:[^\n]*\n?)?/, '').trim();
}

function formatCatalogDocumentBody(target) {
  if (Array.isArray(target.chunks) && target.chunks.length > 0) {
    return target.chunks
      .map((ch, idx) =>
        renderChunkCard(
          ch.section || `Section ${idx + 1}`,
          ch.chunk_id || `${idx + 1}`,
          decorateAstroInline(stripChunkHeader(ch.text))
        )
      )
      .join('');
  }

  const raw = String(target.full_text || '').trim();
  if (!raw) {
    return `<div class="legal-group-card"><div class="legal-group-body">${escapeHtml(target.title || target.doc_id)}</div></div>`;
  }

  const sections = raw
    .split(/^###?\s+/m)
    .map((s) => s.trim())
    .filter(Boolean);

  return sections
    .slice(0, 4)
    .map((sec, idx) => {
      const lines = sec.split('\n').map((l) => l.trim()).filter(Boolean);
      const heading = (lines[0] || `Section ${idx + 1}`).replace(/^#\s*/, '');
      const bodyText = lines
        .slice(1)
        .filter((l) => !l.startsWith('# '))
        .join(' ')
        .slice(0, 460);
      return renderChunkCard(heading, `§${idx + 1}`, decorateAstroInline(bodyText || heading));
    })
    .join('');
}

// ============================================================================
// Corpus & Wikipedia Pages Explorer Modal (567 Wikipedia Pages + Direct Links)
// ============================================================================
async function openCorpusExplorer() {
  const dlg = document.getElementById('corpus-dialog');
  if (!dlg) return;
  dlg.showModal();

  if (!state.corpusItems || state.corpusItems.length === 0) {
    const grid = document.getElementById('corpus-grid');
    if (grid) {
      grid.innerHTML = `<div class="mono" style="color:var(--text-secondary);padding:20px;">Loading ${state.corpusDocs} Wikipedia pages &amp; OpenNGC records…</div>`;
    }
    try {
      const res = await fetch('/api/corpus');
      if (res.ok) {
        const data = await res.json();
        state.corpusItems = Array.isArray(data.documents) ? data.documents : [];
      }
    } catch (err) {
      console.error('Failed to load corpus:', err);
    }
  }
  renderCorpusList();
}

function setCorpusCatalogFilter(cat) {
  state.corpusFilterCat = cat || 'ALL';
  document.querySelectorAll('#corpus-filter-tabs .corpus-tab').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.cat === state.corpusFilterCat);
  });
  renderCorpusList();
}

function renderCorpusList() {
  const grid = document.getElementById('corpus-grid');
  if (!grid) return;
  const q = (document.getElementById('corpus-search-input')?.value || '').trim().toLowerCase();
  const cat = state.corpusFilterCat || 'ALL';

  const filtered = (state.corpusItems || []).filter((item) => {
    if (cat !== 'ALL' && item.catalog !== cat) return false;
    if (!q) return true;
    return (
      String(item.doc_id || '').toLowerCase().includes(q) ||
      String(item.title || '').toLowerCase().includes(q) ||
      String(item.wikipedia_page || '').toLowerCase().includes(q) ||
      String(item.constellation || '').toLowerCase().includes(q) ||
      String(item.object_type || '').toLowerCase().includes(q)
    );
  });

  if (filtered.length === 0) {
    grid.innerHTML = `<div class="mono" style="color:var(--text-secondary);padding:18px;">No matching Wikipedia pages for "${escapeHtml(q)}".</div>`;
    return;
  }

  // Render up to 120 cards at once for instant DOM responsiveness
  const slice = filtered.slice(0, 120);
  grid.innerHTML = slice
    .map((item) => {
      const magBadge = item.magnitude != null ? `Mag ${item.magnitude}` : item.catalog;
      const constBadge = item.constellation ? ` · ${item.constellation}` : '';
      const otype = String(item.object_type || '').replace(/_/g, ' ');
      return `
        <div class="corpus-card">
          <div>
            <div class="corpus-card-top">
              <span class="corpus-doc-id">${escapeHtml(item.doc_id)} · ${escapeHtml(item.catalog)}</span>
              <span class="corpus-meta-line">${escapeHtml(magBadge)}${escapeHtml(constBadge)} · ${escapeHtml(String(item.num_chunks || 4))} chunks</span>
            </div>
            <div class="corpus-card-title">${escapeHtml(item.title)}</div>
            <div class="corpus-meta-line" style="margin:3px 0 6px;">${escapeHtml(otype)}</div>
            <p class="corpus-card-excerpt">${escapeHtml(item.excerpt || '')}</p>
          </div>
          <div class="corpus-card-actions">
            <a
              href="${escapeHtml(item.wikipedia_url)}"
              target="_blank"
              rel="noopener noreferrer"
              class="corpus-wiki-link"
              title="Open Wikipedia page: ${escapeHtml(item.wikipedia_page)}"
            >
              <span>W · ${escapeHtml(item.wikipedia_page)}</span>
              <span>↗</span>
            </a>
            <button
              type="button"
              class="corpus-inspect-btn"
              onclick="inspectFromCorpus('${escapeHtml(item.doc_id)}')"
            >
              Inspect Chunks
            </button>
          </div>
        </div>
      `;
    })
    .join('');
}

function inspectFromCorpus(docId) {
  const dlg = document.getElementById('corpus-dialog');
  if (dlg && dlg.open) dlg.close();
  openSourcePanel(docId, null);
}

// ============================================================================
// Interactive 60fps Deep-Space Starfield, Nebula & Constellation Canvas
// ============================================================================
function initSpaceCanvas() {
  const canvas = document.getElementById('space-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let W = (canvas.width = window.innerWidth);
  let H = (canvas.height = window.innerHeight);

  window.addEventListener('resize', () => {
    W = canvas.width = window.innerWidth;
    H = canvas.height = window.innerHeight;
  });

  // Realistic stellar spectral palette (O/B blue-white, A white, G yellow, K/M amber)
  const SPECTRAL_COLORS = ['#93c5fd', '#e0f2fe', '#f8fafc', '#fef08a', '#fed7aa'];
  const NUM_STARS = 340;
  const stars = Array.from({ length: NUM_STARS }, () => {
    const layer = Math.random() < 0.15 ? 3 : Math.random() < 0.45 ? 2 : 1;
    return {
      x: Math.random() * W,
      y: Math.random() * H,
      r: layer === 3 ? 1.5 + Math.random() * 1.1 : layer === 2 ? 0.9 + Math.random() * 0.6 : 0.45 + Math.random() * 0.4,
      color: SPECTRAL_COLORS[Math.floor(Math.random() * SPECTRAL_COLORS.length)],
      alpha: 0.35 + Math.random() * 0.65,
      twinkleSpeed: 0.008 + Math.random() * 0.025,
      twinklePhase: Math.random() * Math.PI * 2,
      layer,
      vx: (Math.random() - 0.5) * 0.06 * layer,
      vy: -0.04 * layer,
    };
  });

  let mouseX = W * 0.5;
  let mouseY = H * 0.5;
  let targetMouseX = mouseX;
  let targetMouseY = mouseY;

  window.addEventListener('mousemove', (e) => {
    targetMouseX = e.clientX;
    targetMouseY = e.clientY;
  });

  let shootingStar = null;
  function maybeSpawnShootingStar() {
    if (!shootingStar && Math.random() < 0.004) {
      shootingStar = {
        x: Math.random() * W * 0.8 + W * 0.1,
        y: Math.random() * H * 0.35,
        vx: 7 + Math.random() * 5,
        vy: 3.5 + Math.random() * 3,
        len: 75 + Math.random() * 45,
        life: 1.0,
      };
    }
  }

  function frame() {
    ctx.clearRect(0, 0, W, H);
    const isWarping = document.body.classList.contains('is-thinking');
    const speedMult = isWarping ? 12.0 : 1.0;

    mouseX += (targetMouseX - mouseX) * 0.04;
    mouseY += (targetMouseY - mouseY) * 0.04;
    const parallaxX = (mouseX - W * 0.5) * 0.018;
    const parallaxY = (mouseY - H * 0.5) * 0.018;

    const brightStars = [];

    for (let i = 0; i < stars.length; i++) {
      const s = stars[i];
      s.twinklePhase += s.twinkleSpeed;
      s.x += s.vx * speedMult;
      s.y += s.vy * speedMult;

      if (s.x < 0) s.x += W;
      if (s.x > W) s.x -= W;
      if (s.y < 0) s.y += H;
      if (s.y > H) s.y -= H;

      const px = (s.x + parallaxX * s.layer + W) % W;
      const py = (s.y + parallaxY * s.layer + H) % H;
      const twinkle = 0.65 + 0.35 * Math.sin(s.twinklePhase);
      const a = Math.min(1, s.alpha * twinkle);

      if (s.layer === 3) {
        brightStars.push({ px, py });
      }

      ctx.save();
      ctx.globalAlpha = a;
      ctx.fillStyle = s.color;
      ctx.beginPath();
      if (isWarping && s.layer >= 2) {
        // Telescope slew light-streak effect
        ctx.strokeStyle = s.color;
        ctx.lineWidth = s.r * 0.95;
        ctx.moveTo(px, py);
        ctx.lineTo(px - s.vx * 18, py - s.vy * 28);
        ctx.stroke();
      } else {
        ctx.arc(px, py, s.r, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    }

    // Draw subtle astronomical constellation lines between nearby bright stars
    ctx.save();
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.11)';
    ctx.lineWidth = 0.75;
    for (let i = 0; i < brightStars.length; i++) {
      for (let j = i + 1; j < brightStars.length; j++) {
        const dx = brightStars[i].px - brightStars[j].px;
        const dy = brightStars[i].py - brightStars[j].py;
        const dist = Math.hypot(dx, dy);
        if (dist < 135) {
          ctx.globalAlpha = (1 - dist / 135) * 0.22;
          ctx.beginPath();
          ctx.moveTo(brightStars[i].px, brightStars[i].py);
          ctx.lineTo(brightStars[j].px, brightStars[j].py);
          ctx.stroke();
        }
      }
    }
    ctx.restore();

    // Periodic shooting star / meteor
    maybeSpawnShootingStar();
    if (shootingStar) {
      ctx.save();
      const grad = ctx.createLinearGradient(
        shootingStar.x,
        shootingStar.y,
        shootingStar.x - shootingStar.vx * 8,
        shootingStar.y - shootingStar.vy * 8
      );
      grad.addColorStop(0, `rgba(224, 242, 254, ${shootingStar.life})`);
      grad.addColorStop(1, 'rgba(56, 189, 248, 0)');
      ctx.strokeStyle = grad;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.moveTo(shootingStar.x, shootingStar.y);
      ctx.lineTo(shootingStar.x - shootingStar.vx * 8, shootingStar.y - shootingStar.vy * 8);
      ctx.stroke();
      ctx.restore();

      shootingStar.x += shootingStar.vx;
      shootingStar.y += shootingStar.vy;
      shootingStar.life -= 0.028;
      if (shootingStar.life <= 0 || shootingStar.x > W || shootingStar.y > H) {
        shootingStar = null;
      }
    }

    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

function decorateAstroInline(text) {
  let html = escapeHtml(text);
  // Highlight catalog IDs (M31, NGC 224, IC 434, C14)
  html = html.replace(
    /\b(M\s?\d{1,3}|NGC\s?\d{1,4}|IC\s?\d{1,4}|C\s?\d{1,3}|Caldwell\s+\d{1,3}|Messier\s+\d{1,3})\b/g,
    '<span class="legal-highlight-ref">$1</span>'
  );
  // Highlight magnitudes, arcminutes, degrees, and focal lengths
  html = html.replace(
    /\b(\d+(?:\.\d+)?\s*(?:arcmin|'|°|mm|mag|light-years))\b/gi,
    '<span class="legal-highlight-num">$1</span>'
  );
  return html;
}

function shortTitle(title) {
  const s = String(title || '');
  return s.replace(/\s*\(.*?\)\s*/g, '').trim() || s;
}

function escapeHtml(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
