const API_URL = "http://localhost:5000/predict";

const sliderConfig = [
  ["heart-rate", "HR_mean"],
  ["hrv", "HRV"],
  ["eeg-alpha", "EEG_alpha"],
  ["eeg-beta", "EEG_beta"],
  ["eeg-gamma", "EEG_gamma"],
  ["eeg-delta", "EEG_delta"],
  ["eeg-theta", "EEG_theta"],
  ["temperature", "Temperature"],
  ["tremor-index", "Tremor_index"],
  ["respiratory-rate", "Respiratory_rate"],
  ["acc-energy", "ACC_energy"]
];

const formatValue = (value) => Number(value).toFixed(2).replace(/\.00$/, "");

function syncSliderOutput(slider) {
  const output = document.getElementById(`${slider.id}-output`);
  if (output) output.value = formatValue(slider.value);
}

function readFeatures() {
  return Object.fromEntries(
    sliderConfig.map(([sliderId, featureName]) => [
      featureName,
      Number(document.getElementById(sliderId).value)
    ])
  );
}

function appendHistoryRow(score, label) {
  const body = document.getElementById("signal-history-body");
  const emptyRow = body.querySelector(".empty-row");
  if (emptyRow) emptyRow.remove();

  const row = document.createElement("tr");
  const riskClass = label === "High" ? "risk-high" : "risk-low";
  row.innerHTML = `
    <td>${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</td>
    <td>${score.toFixed(1)}%</td>
    <td class="${riskClass}">${label.toUpperCase()}</td>
  `;
  body.prepend(row);

  document.getElementById("run-count").textContent = `${body.children.length} RUN${body.children.length === 1 ? "" : "S"}`;
}

async function runScoring() {
  const button = document.getElementById("run-scoring");
  const riskCircle = document.getElementById("current-risk-signal");
  const riskHeading = document.getElementById("risk-heading");
  const riskValue = document.getElementById("risk-value");
  const riskNote = document.getElementById("risk-note");
  const apiStatus = document.getElementById("api-status");

  button.disabled = true;
  button.querySelector("span:first-child").textContent = "Scoring...";
  apiStatus.className = "api-status";
  apiStatus.innerHTML = "API target <span>localhost:5000</span>";

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ patient_id: "sub-002", features: readFeatures() })
    });

    if (!response.ok) throw new Error(`API returned ${response.status}`);

    const data = await response.json();
    const score = Number(data.risk_score);
    const isHigh = data.label === "High";

    riskCircle.innerHTML = `${score.toFixed(1)}<span>%</span>`;
    riskCircle.style.setProperty("--risk-score", `${Math.min(Math.max(score, 0), 100)}%`);
    riskCircle.style.setProperty("--ring-color", isHigh ? "#f06b5d" : "#b9f34a");
    riskHeading.textContent = `Seizure Risk: ${isHigh ? "HIGH" : "LOW"}`;
    riskHeading.style.color = isHigh ? "#f06b5d" : "#b9f34a";
    riskValue.textContent = `${score.toFixed(1)} / 100`;
    riskNote.textContent = `Scored at ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })} for sub-002.`;
    appendHistoryRow(score, data.label);

    apiStatus.className = "api-status success";
    apiStatus.textContent = "API response received";
  } catch (error) {
    riskHeading.textContent = "API needs attention";
    riskHeading.style.color = "#f06b5d";
    riskNote.textContent = "Could not reach the local NeuroCare scoring service.";
    apiStatus.className = "api-status error";
    apiStatus.textContent = "API needs attention";
    console.error("NeuroCare scoring failed:", error);
  } finally {
    button.disabled = false;
    button.querySelector("span:first-child").textContent = "Run Scoring";
  }
}

sliderConfig.forEach(([sliderId]) => {
  const slider = document.getElementById(sliderId);
  slider.addEventListener("input", () => syncSliderOutput(slider));
  syncSliderOutput(slider);
});

document.getElementById("run-scoring").addEventListener("click", runScoring);
