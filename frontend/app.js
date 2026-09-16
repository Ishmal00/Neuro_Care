const API_ROOT = "http://localhost:5000";
const API_URL = `${API_ROOT}/predict`;
const patientSelect = document.getElementById("patient-select");
const windowSelect = document.getElementById("window-select");
const selectionSummary = document.getElementById("selection-summary");
const activeSubject = document.getElementById("active-subject");

function appendHistoryRow(score, label, patientId, runId, windowStartTime) {
  const body = document.getElementById("signal-history-body");
  const emptyRow = body.querySelector(".empty-row");
  if (emptyRow) emptyRow.remove();

  const row = document.createElement("tr");
  const riskClass = label === "High" ? "risk-high" : "risk-low";
  row.innerHTML = `
    <td>${patientId} / ${runId}:${windowStartTime}s</td>
    <td>${score.toFixed(1)}%</td>
    <td class="${riskClass}">${label.toUpperCase()}</td>
  `;
  body.prepend(row);

  document.getElementById("run-count").textContent = `${body.children.length} RUN${body.children.length === 1 ? "" : "S"}`;
}

function updateSelectionSummary() {
  const option = windowSelect.selectedOptions[0];
  activeSubject.textContent = patientSelect.value || "Loading...";
  if (option) selectionSummary.textContent = `${patientSelect.value} / run ${option.dataset.run} / ${option.dataset.window}s`;
}

async function loadWindows() {
  const response = await fetch(`${API_ROOT}/demo-options?patient_id=${encodeURIComponent(patientSelect.value)}`);
  if (!response.ok) throw new Error(`Could not load windows (${response.status})`);
  const data = await response.json();
  windowSelect.replaceChildren(...data.windows.map((window) => {
    const option = document.createElement("option");
    option.dataset.run = window.run_id;
    option.dataset.window = window.window_start_time;
    option.textContent = `Run ${window.run_id} / ${window.window_start_time}s`;
    return option;
  }));
  updateSelectionSummary();
}

async function loadPatients() {
  const response = await fetch(`${API_ROOT}/demo-options`);
  if (!response.ok) throw new Error(`Could not load patients (${response.status})`);
  const data = await response.json();
  patientSelect.replaceChildren(...data.patients.map((patientId) => {
    const option = document.createElement("option");
    option.value = patientId;
    option.textContent = patientId;
    return option;
  }));
  await loadWindows();
}

async function runScoring() {
  const button = document.getElementById("run-scoring");
  const riskCircle = document.getElementById("current-risk-signal");
  const riskHeading = document.getElementById("risk-heading");
  const riskValue = document.getElementById("risk-value");
  const riskNote = document.getElementById("risk-note");
  const apiStatus = document.getElementById("api-status");
  const windowOption = windowSelect.selectedOptions[0];

  button.disabled = true;
  button.querySelector("span:first-child").textContent = "Scoring...";
  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        patient_id: patientSelect.value,
        run_id: Number(windowOption.dataset.run),
        window_start_time: Number(windowOption.dataset.window)
      })
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error || `API returned ${response.status}`);
    }

    const data = await response.json();
    const score = Number(data.risk_score);
    const isHigh = data.prediction === 1;

    riskCircle.innerHTML = `${score.toFixed(1)}<span>%</span>`;
    riskCircle.style.setProperty("--risk-score", `${Math.min(Math.max(score, 0), 100)}%`);
    riskCircle.style.setProperty("--ring-color", isHigh ? "#f06b5d" : "#b9f34a");
    riskHeading.textContent = `Model prediction: ${isHigh ? "SEIZURE" : "NO SEIZURE"}`;
    riskHeading.style.color = isHigh ? "#f06b5d" : "#b9f34a";
    riskValue.textContent = `${score.toFixed(1)} / 100`;
    riskNote.textContent = `Class ${data.prediction} from ${data.patient_id}, run ${data.run_id}, ${data.window_start_time}s.`;
    appendHistoryRow(score, data.label, data.patient_id, data.run_id, data.window_start_time);

    apiStatus.className = "api-status success";
    apiStatus.textContent = `Actual model prediction: class ${data.prediction}`;
  } catch (error) {
    riskHeading.textContent = "API needs attention";
    riskHeading.style.color = "#f06b5d";
    riskNote.textContent = error.message;
    apiStatus.className = "api-status error";
    apiStatus.textContent = "Prediction request failed";
    console.error("NeuroCare scoring failed:", error);
  } finally {
    button.disabled = false;
    button.querySelector("span:first-child").textContent = "Run Scoring";
  }
}

patientSelect.addEventListener("change", () => loadWindows().catch(console.error));
windowSelect.addEventListener("change", updateSelectionSummary);
document.getElementById("run-scoring").addEventListener("click", runScoring);
loadPatients().catch((error) => {
  selectionSummary.textContent = "Dataset API unavailable";
  document.getElementById("api-status").textContent = error.message;
});
