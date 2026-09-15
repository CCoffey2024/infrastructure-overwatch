"use strict";

const API = "/api";

const state = {
  jobs: [],
  selectedJobId: null,
  selectedJob: null,
  alerts: [],
  events: [],
  frameIds: [],
  fusedEvents: [],
  contributors: [],
  fuseSelection: new Set(),
  activeTab: "detections",
  player: {
    frameIndex: 0,
    playing: false,
    speed: 1,
    loop: false,
    fps: 10,
    confThreshold: 0.0,
    hiddenClasses: new Set(),
    highlightedTrackId: null,
    playToken: 0, // bumped to cancel an in-flight play loop when the job/state changes
  },
};

const CLASS_COLORS = {
  car: "#4dabf7",
  truck: "#ffa94d",
  bus: "#845ef7",
  dismount: "#ff6b6b",
  drone: "#51cf66",
  launch_flash: "#ffd43b",
  vehicle_of_interest: "#4dabf7",
};
function classColor(label) {
  return CLASS_COLORS[label] || "#e6e9ef";
}

async function api(path, opts) {
  const res = await fetch(API + path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      /* no JSON body */
    }
    throw new Error(detail);
  }
  if (res.status === 204) return null;
  const contentType = res.headers.get("content-type") || "";
  return contentType.includes("application/json") ? res.json() : res.text();
}

// --- Job queue --------------------------------------------------------

async function refreshJobs() {
  state.jobs = await api("/jobs");
  renderJobQueue();
  renderFusionRunList();
  if (state.selectedJobId) {
    const stillSelected = state.jobs.find((j) => j.job_id === state.selectedJobId);
    if (stillSelected) state.selectedJob = stillSelected;
  }
}

function statusBadge(status) {
  return `<span class="status-badge status-${status}">${status}</span>`;
}

function summaryLine(job) {
  if (job.kind === "fusion") {
    const s = job.summary || {};
    return s.fused_events !== undefined
      ? `${s.fused_events} fused events · ${s.distinct_sensors} sensors`
      : "";
  }
  const s = job.summary || {};
  return s.detections !== undefined
    ? `${s.detections} det · ${s.events || 0} events · ${s.alerts || 0} alerts`
    : "";
}

function renderJobQueue() {
  const body = document.getElementById("job-table-body");
  body.innerHTML = "";
  for (const job of state.jobs) {
    const tr = document.createElement("tr");
    if (job.job_id === state.selectedJobId) tr.classList.add("selected");
    const sourceOrFusion = job.kind === "fusion" ? job.display_name : job.source;
    tr.innerHTML = `
      <td>${statusBadge(job.status)}</td>
      <td>${job.kind}</td>
      <td title="${sourceOrFusion}">${sourceOrFusion}</td>
      <td>${job.sensor_id || ""}</td>
      <td>${new Date(job.created_at).toLocaleString()}</td>
      <td>${summaryLine(job)}${job.status === "failed" ? `<br><span class="error">${job.error || ""}</span>` : ""}</td>
    `;
    tr.addEventListener("click", () => selectJob(job.job_id));
    body.appendChild(tr);
  }
}

// --- Ingest form --------------------------------------------------------

function initIngestForm() {
  const form = document.getElementById("ingest-form");
  const kindSelect = document.getElementById("source-kind");
  const splitRow = document.getElementById("split-row");

  kindSelect.addEventListener("change", () => {
    splitRow.classList.toggle("hidden", kindSelect.value !== "visdrone-vid");
  });

  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const errorEl = document.getElementById("ingest-error");
    errorEl.classList.add("hidden");
    const data = new FormData(form);
    const kind = data.get("kind");
    const path = data.get("path").trim();
    const body = {
      source: `${kind}:${path}`,
      sensor_id: data.get("sensor_id"),
      class_scheme: data.get("class_scheme"),
      stride: Number(data.get("stride")),
      cap: Number(data.get("cap")),
      fps: Number(data.get("fps")),
      conf_thresh: Number(data.get("conf_thresh")),
      track_min_hits: Number(data.get("track_min_hits")),
      calibrate: data.get("calibrate") === "on",
    };
    if (kind === "visdrone-vid") body.split = data.get("split");
    const anomalyRef = data.get("anomaly_ref").trim();
    if (anomalyRef) body.anomaly_ref = anomalyRef;

    try {
      await api("/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      await refreshJobs();
    } catch (err) {
      errorEl.textContent = err.message;
      errorEl.classList.remove("hidden");
    }
  });
}

// --- Fusion panel --------------------------------------------------------

function renderFusionRunList() {
  const container = document.getElementById("fusion-run-list");
  const completedRuns = state.jobs.filter((j) => j.kind === "run" && j.status === "completed");
  if (completedRuns.length === 0) {
    container.innerHTML = `<div class="empty">No completed runs yet.</div>`;
  } else {
    container.innerHTML = "";
    for (const job of completedRuns) {
      const label = document.createElement("label");
      const checked = state.fuseSelection.has(job.job_id) ? "checked" : "";
      label.innerHTML = `<input type="checkbox" value="${job.job_id}" ${checked}/> ${job.sensor_id} · ${job.display_name}`;
      label.querySelector("input").addEventListener("change", (ev) => {
        if (ev.target.checked) state.fuseSelection.add(job.job_id);
        else state.fuseSelection.delete(job.job_id);
        updateFuseButtonState();
      });
      container.appendChild(label);
    }
  }
  updateFuseButtonState();
}

function updateFuseButtonState() {
  // keep the selection in sync with jobs that may have disappeared (deleted) since
  const validIds = new Set(state.jobs.map((j) => j.job_id));
  for (const id of Array.from(state.fuseSelection)) if (!validIds.has(id)) state.fuseSelection.delete(id);
  document.getElementById("fuse-button").disabled = state.fuseSelection.size < 2;
}

function initFusionPanel() {
  document.getElementById("fuse-button").addEventListener("click", async () => {
    const errorEl = document.getElementById("fusion-error");
    errorEl.classList.add("hidden");
    const body = {
      source_job_ids: Array.from(state.fuseSelection),
      max_time_delta_s: Number(document.getElementById("fusion-max-delta").value),
      require_matching_label: document.getElementById("fusion-require-label").checked,
    };
    try {
      await api("/jobs/fuse", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      state.fuseSelection.clear();
      await refreshJobs();
    } catch (err) {
      errorEl.textContent = err.message;
      errorEl.classList.remove("hidden");
    }
  });
}

// --- Evidence panel / job selection --------------------------------------------------------

async function selectJob(jobId) {
  stopPlayback();
  state.selectedJobId = jobId;
  state.selectedJob = await api(`/jobs/${jobId}`);
  state.alerts = [];
  state.events = [];
  state.frameIds = [];
  state.fusedEvents = [];
  state.contributors = [];
  state.player.frameIndex = 0;
  state.player.highlightedTrackId = null;

  renderJobQueue();

  const job = state.selectedJob;
  document.getElementById("evidence-empty").classList.toggle("hidden", true);
  document.getElementById("evidence-content").classList.remove("hidden");

  if (job.status === "completed" && job.kind === "run") {
    [state.alerts, state.events, state.frameIds] = await Promise.all([
      api(`/jobs/${jobId}/alerts?limit=1000`),
      api(`/jobs/${jobId}/events?limit=1000`),
      api(`/jobs/${jobId}/frame_ids`),
    ]);
  } else if (job.status === "completed" && job.kind === "fusion") {
    [state.fusedEvents, state.contributors] = await Promise.all([
      api(`/jobs/${jobId}/fusion_events?limit=1000`),
      api(`/jobs/${jobId}/fusion_contributors?limit=1000`),
    ]);
  }

  renderEvidencePanel();
}

function renderSummaryTiles() {
  const job = state.selectedJob;
  const s = job.summary || {};
  const tiles =
    job.kind === "fusion"
      ? [
          ["Source runs", s.source_runs],
          ["Distinct sensors", s.distinct_sensors],
          ["Fused events", s.fused_events],
          ["Contributors", s.contributors],
        ]
      : [
          ["Frames", s.frames],
          ["Detections", s.detections],
          ["Tracks", s.tracks],
          ["Events", s.events],
          ["Alerts", s.alerts],
          ["Anomalies", s.anomalies],
        ];
  document.getElementById("summary-tiles").innerHTML = tiles
    .filter(([, v]) => v !== undefined)
    .map(([label, value]) => `<div class="tile"><div class="label">${label}</div><div class="value">${value}</div></div>`)
    .join("");
}

function renderClassToggles() {
  const container = document.getElementById("class-toggles");
  const classes = Array.from(new Set(state.alerts.map((a) => a.label))).sort();
  container.innerHTML = classes
    .map(
      (label) =>
        `<label><input type="checkbox" data-class="${label}" checked/> <span style="color:${classColor(label)}">${label}</span></label>`
    )
    .join("");
  for (const input of container.querySelectorAll("input")) {
    input.addEventListener("change", (ev) => {
      const label = ev.target.dataset.class;
      if (ev.target.checked) state.player.hiddenClasses.delete(label);
      else state.player.hiddenClasses.add(label);
      drawFrame(state.player.frameIndex);
    });
  }
}

function renderEvidencePanel() {
  const job = state.selectedJob;
  renderSummaryTiles();

  const isRun = job.kind === "run";
  document.getElementById("player-block").classList.toggle("hidden", !(isRun && job.status === "completed"));
  document.getElementById("evidence-tabs").classList.toggle("hidden", job.status !== "completed");

  // Fusion jobs don't have a Detections/Alerts/Anomalies/Metrics view -- only Events (fused) and Fusion evidence.
  for (const btn of document.querySelectorAll("#evidence-tabs button")) {
    const tab = btn.dataset.tab;
    const runOnly = ["detections", "alerts", "anomalies", "metrics"].includes(tab);
    btn.classList.toggle("hidden", !isRun && runOnly);
  }
  if (!isRun && ["detections", "alerts", "anomalies", "metrics"].includes(state.activeTab)) {
    state.activeTab = "fusion";
  }
  if (isRun && state.activeTab === "fusion") state.activeTab = "detections";

  if (isRun && job.status === "completed") {
    const slider = document.getElementById("frame-slider");
    slider.max = String(Math.max(0, state.frameIds.length - 1));
    slider.value = "0";
    document.getElementById("frame-counter").textContent = `0 / ${state.frameIds.length}`;
    renderClassToggles();
    drawFrame(0);
  }

  renderTabs();
  renderTabContent();
}

// --- Interactive player --------------------------------------------------------

function alertsForFrameIndex(n) {
  const fid = state.frameIds[n];
  return state.alerts.filter((a) => a.frame_id === fid);
}

function visibleDetectionsForFrame(n) {
  return alertsForFrameIndex(n).filter(
    (d) => d.confidence >= state.player.confThreshold && !state.player.hiddenClasses.has(d.label)
  );
}

let lastDrawnDetections = [];

function drawFrame(n, onDone) {
  if (!state.selectedJobId || n < 0 || n >= state.frameIds.length) return;
  const token = state.player.playToken;
  const img = new Image();
  img.onload = () => {
    if (token !== state.player.playToken) return; // a newer job/selection superseded this draw
    const canvas = document.getElementById("player-canvas");
    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0);

    const dets = visibleDetectionsForFrame(n);
    lastDrawnDetections = dets;
    for (const d of dets) {
      const highlighted = state.player.highlightedTrackId !== null && d.track_id === state.player.highlightedTrackId;
      ctx.strokeStyle = highlighted ? "#ffe600" : classColor(d.label);
      ctx.lineWidth = highlighted ? 3 : 1.5;
      ctx.strokeRect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);
      const text = `${d.label} ${d.confidence.toFixed(2)}${d.track_id !== null ? " #" + d.track_id : ""}`;
      ctx.font = "11px monospace";
      const textWidth = ctx.measureText(text).width;
      ctx.fillStyle = highlighted ? "#ffe600" : classColor(d.label);
      ctx.fillRect(d.x1, Math.max(0, d.y1 - 13), textWidth + 4, 13);
      ctx.fillStyle = "#0b0e12";
      ctx.fillText(text, d.x1 + 2, Math.max(10, d.y1 - 3));
    }
    document.getElementById("frame-slider").value = String(n);
    document.getElementById("frame-counter").textContent = `${n + 1} / ${state.frameIds.length}`;
    if (onDone) onDone();
  };
  img.src = `${API}/jobs/${state.selectedJobId}/frame/${n}`;
}

function stopPlayback() {
  state.player.playing = false;
  state.player.playToken += 1;
  const btn = document.getElementById("play-button");
  if (btn) btn.textContent = "Play";
}

function advancePlayback() {
  if (!state.player.playing) return;
  let next = state.player.frameIndex + 1;
  if (next >= state.frameIds.length) {
    if (state.player.loop) {
      next = 0;
    } else {
      stopPlayback();
      return;
    }
  }
  state.player.frameIndex = next;
  drawFrame(next, () => {
    if (!state.player.playing) return;
    setTimeout(advancePlayback, 1000 / (state.player.fps * state.player.speed));
  });
}

function initPlayerControls() {
  document.getElementById("play-button").addEventListener("click", () => {
    if (state.player.playing) {
      stopPlayback();
    } else {
      state.player.playing = true;
      document.getElementById("play-button").textContent = "Pause";
      advancePlayback();
    }
  });
  document.getElementById("prev-frame").addEventListener("click", () => {
    stopPlayback();
    state.player.frameIndex = Math.max(0, state.player.frameIndex - 1);
    drawFrame(state.player.frameIndex);
  });
  document.getElementById("next-frame").addEventListener("click", () => {
    stopPlayback();
    state.player.frameIndex = Math.min(state.frameIds.length - 1, state.player.frameIndex + 1);
    drawFrame(state.player.frameIndex);
  });
  document.getElementById("frame-slider").addEventListener("input", (ev) => {
    stopPlayback();
    state.player.frameIndex = Number(ev.target.value);
    drawFrame(state.player.frameIndex);
  });
  document.getElementById("speed-select").addEventListener("change", (ev) => {
    state.player.speed = Number(ev.target.value);
  });
  document.getElementById("loop-checkbox").addEventListener("change", (ev) => {
    state.player.loop = ev.target.checked;
  });
  document.getElementById("conf-threshold").addEventListener("input", (ev) => {
    state.player.confThreshold = Number(ev.target.value);
    document.getElementById("conf-threshold-value").textContent = state.player.confThreshold.toFixed(2);
    drawFrame(state.player.frameIndex);
  });
  document.getElementById("clear-highlight").addEventListener("click", () => {
    state.player.highlightedTrackId = null;
    document.getElementById("clear-highlight").classList.add("hidden");
    drawFrame(state.player.frameIndex);
  });

  const canvas = document.getElementById("player-canvas");
  canvas.addEventListener("click", (ev) => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (ev.clientX - rect.left) * scaleX;
    const y = (ev.clientY - rect.top) * scaleY;
    // topmost (last-drawn) box first
    for (let i = lastDrawnDetections.length - 1; i >= 0; i--) {
      const d = lastDrawnDetections[i];
      if (x >= d.x1 && x <= d.x2 && y >= d.y1 && y <= d.y2) {
        state.player.highlightedTrackId = d.track_id;
        document.getElementById("clear-highlight").classList.toggle("hidden", d.track_id === null);
        drawFrame(state.player.frameIndex);
        return;
      }
    }
  });
}

// --- Tabs --------------------------------------------------------

function renderTabs() {
  for (const btn of document.querySelectorAll("#evidence-tabs button")) {
    btn.classList.toggle("tab-active", btn.dataset.tab === state.activeTab);
  }
}

function initTabs() {
  for (const btn of document.querySelectorAll("#evidence-tabs button")) {
    btn.addEventListener("click", () => {
      state.activeTab = btn.dataset.tab;
      renderTabs();
      renderTabContent();
    });
  }
}

function renderDataTable(rows, columns) {
  if (rows.length === 0) return `<p class="hint">No rows.</p>`;
  const cols = columns || Object.keys(rows[0]);
  const head = cols.map((c) => `<th>${c}</th>`).join("");
  const body = rows
    .map(
      (r) =>
        `<tr>${cols
          .map((c) => `<td>${typeof r[c] === "number" ? Number(r[c]).toFixed(3).replace(/\.?0+$/, "") : r[c] ?? ""}</td>`)
          .join("")}</tr>`
    )
    .join("");
  return `<div class="table-scroll"><table class="data-table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

async function renderTabContent() {
  const el = document.getElementById("tab-content");
  const job = state.selectedJob;
  if (!job || job.status !== "completed") {
    el.innerHTML = "";
    return;
  }

  if (state.activeTab === "detections") {
    el.innerHTML = renderDataTable(state.alerts.slice(0, 500), [
      "frame_id", "label", "confidence", "calibrated_confidence", "triage_band", "track_id",
    ]);
  } else if (state.activeTab === "events") {
    el.innerHTML = renderDataTable(state.events, ["frame_id", "timestamp_s", "event_type", "severity", "track_id", "description"]);
  } else if (state.activeTab === "alerts") {
    const alertRows = state.events.filter((e) => e.severity === "medium" || e.severity === "high");
    el.innerHTML = renderDataTable(alertRows, ["frame_id", "timestamp_s", "event_type", "severity", "track_id", "description"]);
  } else if (state.activeTab === "anomalies") {
    const anomalyRows = state.events.filter((e) => e.event_type === "VISUAL_ANOMALY");
    el.innerHTML = renderDataTable(anomalyRows, ["frame_id", "timestamp_s", "track_id", "score", "description"]);
  } else if (state.activeTab === "metrics") {
    el.innerHTML = `<p class="hint">Loading metrics…</p>`;
    try {
      const metrics = await api(`/jobs/${job.job_id}/metrics`);
      const grid = `<div class="metrics-grid">
        <div class="tile"><div class="label">mAP@0.5</div><div class="value">${metrics.map_50.toFixed(3)}</div></div>
        <div class="tile"><div class="label">mAP@0.5:0.95</div><div class="value">${metrics.map_50_95.toFixed(3)}</div></div>
        <div class="tile"><div class="label">AP@IoU 0.3</div><div class="value">${metrics.f1_at_iou_30.toFixed(3)}</div></div>
      </div>`;
      const perClassRows = Object.entries(metrics.per_class_ap_50).map(([label, ap]) => ({ label, ap_50: ap }));
      el.innerHTML = `${grid}<h3>Per-class AP@0.5</h3>${renderDataTable(perClassRows, ["label", "ap_50"])}
        <h3>Calibration reliability</h3>${renderDataTable(metrics.calibration_reliability, ["bin_low", "bin_high", "mean_confidence", "observed_accuracy", "n"])}`;
    } catch (err) {
      el.innerHTML = `<p class="hint">${err.message}</p>`;
    }
  } else if (state.activeTab === "fusion") {
    if (job.kind === "fusion") {
      el.innerHTML = `<h3>Fused events</h3>${renderDataTable(state.fusedEvents)}<h3>Contributors</h3>${renderDataTable(state.contributors)}`;
    } else {
      el.innerHTML = `<p class="hint">Not a fusion job.</p>`;
    }
  }
}

// --- Boot --------------------------------------------------------

function init() {
  initIngestForm();
  initFusionPanel();
  initPlayerControls();
  initTabs();
  document.getElementById("refresh-button").addEventListener("click", refreshJobs);
  refreshJobs();
  setInterval(refreshJobs, 2000);
}

document.addEventListener("DOMContentLoaded", init);
