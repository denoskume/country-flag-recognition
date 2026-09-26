const $ = (id) => document.getElementById(id);

const state = {
  file: null,
  result: null,
  health: null,
  deploymentThreshold: 0.8805,
  checkpointThreshold: 0.5927,
};

function toast(message) {
  const el = $("toast");
  el.textContent = message;
  el.classList.remove("hidden");
  setTimeout(() => el.classList.add("hidden"), 2200);
}

function currentThreshold() {
  const policy = $("policySelect").value;
  if (policy === "strict") return Math.min(0.99, state.deploymentThreshold + 0.05);
  if (policy === "custom") return Number($("thresholdRange").value) / 100;
  return state.deploymentThreshold;
}

function refreshPolicyUI() {
  const policy = $("policySelect").value;
  const threshold = currentThreshold();
  $("thresholdRange").disabled = policy !== "custom";
  $("thresholdValue").textContent = (threshold * 100).toFixed(1) + "%";
  $("modePill").textContent =
    policy.charAt(0).toUpperCase() + policy.slice(1) + " · " +
    (threshold * 100).toFixed(1) + "%";
}

async function loadHealth() {
  const res = await fetch("/api/health");
  const data = await res.json();
  state.health = data;
  state.deploymentThreshold = data.deployment_threshold;
  state.checkpointThreshold = data.checkpoint_threshold;
  $("thresholdRange").value = Math.round(data.deployment_threshold * 100);
  $("modelMeta").textContent =
    data.model + " · " + data.classes + " classes · " + data.device;
  refreshPolicyUI();
  renderSystem(data);
}

function renderSystem(data) {
  $("systemMetrics").innerHTML = [
    ["Status", data.status],
    ["Model", data.model],
    ["Classes", data.classes],
    ["Production threshold", (data.deployment_threshold * 100).toFixed(1) + "%"],
    ["Checkpoint threshold", (data.checkpoint_threshold * 100).toFixed(1) + "%"],
  ].map(([k,v]) =>
    '<div class="metric-card"><span>' + k + '</span><strong>' + v + '</strong></div>'
  ).join("");
}

function setFile(file) {
  state.file = file;
  $("analyzeBtn").disabled = !file;
  $("clearBtn").disabled = !file;

  if (!file) {
    $("previewWrap").classList.add("hidden");
    $("fileInput").value = "";
    return;
  }

  $("fileName").textContent = file.name;
  const url = URL.createObjectURL(file);
  $("preview").src = url;
  $("previewWrap").classList.remove("hidden");
  $("preview").onload = () => {
    $("imageMeta").textContent =
      $("preview").naturalWidth + " × " + $("preview").naturalHeight + " px";
  };
}

async function analyze() {
  if (!state.file) return;

  $("analyzeBtn").disabled = true;
  $("analyzeBtn").textContent = "Analyzing…";

  const form = new FormData();
  form.append("file", state.file);

  const threshold = currentThreshold();
  const topK = Number($("topKSelect").value);

  try {
    const res = await fetch(
      "/api/analyze?threshold=" + encodeURIComponent(threshold) +
      "&top_k=" + encodeURIComponent(topK),
      { method: "POST", body: form }
    );

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Analysis failed.");

    state.result = data;
    renderResult(data);
    saveHistory(data);
    $("resultSection").classList.remove("hidden");
    $("resultSection").scrollIntoView({ behavior: "smooth", block: "start" });

    if (data.accepted && $("enrichmentToggle").checked) {
      loadCountry(data.top_candidate.code);
    } else {
      $("countryContent").innerHTML = "No accepted country yet.";
      $("countrySub").textContent = "Available after an accepted decision";
    }
  } catch (err) {
    toast(err.message);
  } finally {
    $("analyzeBtn").disabled = false;
    $("analyzeBtn").textContent = "Analyze image";
  }
}

function renderResult(data) {
  $("decisionMetric").textContent = data.decision;
  $("candidateMetric").textContent = data.top_candidate.country;
  $("confidenceMetric").textContent = (data.confidence * 100).toFixed(1) + "%";
  $("latencyMetric").textContent = data.latency_ms.toFixed(1) + " ms";

  const banner = $("decisionBanner");
  banner.className = "decision-banner " + (data.accepted ? "accepted" : "rejected");
  banner.textContent = data.accepted
    ? "Accepted · confidence meets the active decision policy."
    : "Rejected · top candidate shown for inspection only.";

  $("candidateList").innerHTML = data.candidates.map(c => {
    const pct = (c.confidence * 100).toFixed(1);
    return '<div class="candidate-row">' +
      '<div class="rank">#' + c.rank + '</div>' +
      '<div><div class="candidate-name">' + c.country + '</div>' +
      '<div class="candidate-code">' + c.code.toUpperCase() + '</div></div>' +
      '<div class="confidence">' + pct + '%</div>' +
      '<div class="confidence-bar"><div class="confidence-fill" style="width:' + pct + '%"></div></div>' +
      '</div>';
  }).join("");
}

async function loadCountry(code) {
  $("countryContent").textContent = "Loading country intelligence…";
  $("countrySub").textContent = "Live reference data";

  try {
    const res = await fetch("/api/country/" + encodeURIComponent(code));
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Country data unavailable.");

    const population = data.population && data.population.value
      ? new Intl.NumberFormat().format(data.population.value)
      : "Not available";

    $("countryContent").innerHTML =
      '<div class="country-grid">' +
      stat("Capital", data.capital) +
      stat("Population", population) +
      stat("Continent", data.continent) +
      stat("Currency", data.currency) +
      stat("Area", data.area_km2 ? new Intl.NumberFormat().format(Math.round(data.area_km2)) + " km²" : "Not available") +
      stat("Language(s)", data.official_languages) +
      '</div>' +
      '<div class="country-overview">' + escapeHtml(data.overview || "Not available") + '</div>';
  } catch (err) {
    $("countryContent").textContent = err.message;
  }
}

function stat(label, value) {
  return '<div class="country-stat"><span>' + label + '</span><strong>' +
    escapeHtml(String(value || "Not available")) + '</strong></div>';
}

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, ch => ({
    "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"
  })[ch]);
}

function saveHistory(data) {
  const history = JSON.parse(localStorage.getItem("flagHistory") || "[]");
  history.unshift({
    at: new Date().toISOString(),
    decision: data.decision,
    candidate: data.top_candidate.country,
    confidence: data.confidence,
    accepted: data.accepted,
  });
  localStorage.setItem("flagHistory", JSON.stringify(history.slice(0, 25)));
  renderHistory();
}

function renderHistory() {
  const history = JSON.parse(localStorage.getItem("flagHistory") || "[]");
  $("historyList").innerHTML = history.length
    ? history.map(item =>
      '<div class="history-item"><div>' +
      '<div class="history-title">' + item.decision + '</div>' +
      '<div class="history-meta">Top candidate: ' + item.candidate +
      ' · ' + new Date(item.at).toLocaleString() + '</div></div>' +
      '<div class="history-score">' + (item.confidence * 100).toFixed(1) + '%</div></div>'
    ).join("")
    : '<div class="card">No analyses yet.</div>';
}

function downloadResult() {
  if (!state.result) return;
  const blob = new Blob([JSON.stringify(state.result, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "flag-analysis.json";
  a.click();
  URL.revokeObjectURL(url);
}

document.querySelectorAll(".nav-item").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach(x => x.classList.remove("active"));
    document.querySelectorAll(".view").forEach(x => x.classList.remove("active"));
    btn.classList.add("active");
    const view = btn.dataset.view;
    $(view + "View").classList.add("active");
    $("viewName").textContent = btn.textContent;
  });
});

$("fileInput").addEventListener("change", e => setFile(e.target.files[0] || null));
$("clearBtn").addEventListener("click", () => setFile(null));
$("analyzeBtn").addEventListener("click", analyze);
$("downloadBtn").addEventListener("click", downloadResult);
$("policySelect").addEventListener("change", refreshPolicyUI);
$("thresholdRange").addEventListener("input", refreshPolicyUI);
$("clearHistoryBtn").addEventListener("click", () => {
  localStorage.removeItem("flagHistory");
  renderHistory();
});

["dragenter","dragover"].forEach(evt =>
  $("dropzone").addEventListener(evt, e => {
    e.preventDefault();
    $("dropzone").classList.add("dragging");
  })
);
["dragleave","drop"].forEach(evt =>
  $("dropzone").addEventListener(evt, e => {
    e.preventDefault();
    $("dropzone").classList.remove("dragging");
  })
);
$("dropzone").addEventListener("drop", e => setFile(e.dataTransfer.files[0] || null));

$("settingsBtn").addEventListener("click", () => toast("Settings are available in Session controls."));

loadHealth().catch(() => toast("System status unavailable."));
renderHistory();
