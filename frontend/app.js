const API_ROOT = "http://localhost:5000";
const API_URL = `${API_ROOT}/predict`;
const patientSelect = document.getElementById("patient-select");
const windowSelect = document.getElementById("window-select");
const selectionSummary = document.getElementById("selection-summary");
const activeSubject = document.getElementById("active-subject");
const activeSubjectId = document.getElementById("active-subject-id");

const DEMO_PATIENT_NAMES = {
  "SUB-001": "Ahmed Khan",
  "SUB-002": "Sara Ahmed",
  "SUB-003": "Muhammad Hamza",
  "SUB-004": "Ayesha Malik",
  "SUB-005": "Usman Ali",
  "SUB-006": "Hira Noor",
  "SUB-007": "Hamza Tariq",
  "SUB-008": "Fatima Zahra",
  "SUB-009": "Bilal Ahmed",
  "SUB-010": "Zoya Hassan"
};

function getPatientName(patientId) {
  return DEMO_PATIENT_NAMES[String(patientId).toUpperCase()] || "Demo Patient";
}

function formatPatient(patientId) {
  return `${getPatientName(patientId)} — ${String(patientId).toUpperCase()}`;
}

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
  const patientId = patientSelect.value;
  activeSubject.textContent = patientId ? getPatientName(patientId) : "Loading...";
  activeSubjectId.textContent = patientId ? String(patientId).toUpperCase() : "--";
  if (option) selectionSummary.textContent = `${formatPatient(patientId)} / Run ${option.dataset.run} / ${option.dataset.window}s`;
}

function getRiskLevel(score) {
  if (score < 35) return "Low";
  if (score < 70) return "Moderate";
  return "High";
}

function updateAssessmentComparison(prediction, recordedLabel) {
  const comparison = document.getElementById("comparison-result");
  const icon = comparison.querySelector(".comparison-icon");
  const heading = comparison.querySelector("strong");
  const note = comparison.querySelector("p");
  const assessmentDetected = prediction === 1;
  const eventDetected = recordedLabel === 1;

  if (assessmentDetected && eventDetected) {
    comparison.className = "comparison-result match-positive";
    icon.textContent = "✓";
    heading.textContent = "Correctly identified recorded event";
    note.textContent = "The AI assessment and recorded event both indicate a seizure event.";
  } else if (!assessmentDetected && !eventDetected) {
    comparison.className = "comparison-result match-negative";
    icon.textContent = "✓";
    heading.textContent = "Correctly identified non-seizure window";
    note.textContent = "The AI assessment and recorded event both indicate no seizure event.";
  } else if (assessmentDetected) {
    comparison.className = "comparison-result false-alarm";
    icon.textContent = "!";
    heading.textContent = "False alarm";
    note.textContent = "The AI assessment indicated an event that was not recorded for this window.";
  } else {
    comparison.className = "comparison-result missed-event";
    icon.textContent = "!";
    heading.textContent = "Missed recorded event";
    note.textContent = "A seizure event was recorded, but the AI assessment did not identify it.";
  }
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
    option.textContent = formatPatient(patientId);
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
  const riskLevel = document.getElementById("risk-level");
  const assessmentStatus = document.getElementById("assessment-status");
  const eventHeading = document.getElementById("event-heading");
  const eventMark = document.getElementById("event-mark");
  const eventIcon = document.getElementById("event-icon");
  const eventStatus = document.getElementById("event-status");
  const eventNote = document.getElementById("event-note");
  const apiStatus = document.getElementById("api-status");
  const windowOption = windowSelect.selectedOptions[0];

  button.disabled = true;
  button.querySelector("span:first-child").textContent = "Analyzing...";
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
    const riskLevelValue = getRiskLevel(score);
    const eventDetected = data.recorded_label === 1;
    const levelColor = riskLevelValue === "High" ? "#f06b5d" : riskLevelValue === "Moderate" ? "#e8b84b" : "#b9f34a";

    riskCircle.innerHTML = `${score.toFixed(1)}<span>%</span>`;
    riskCircle.style.setProperty("--risk-score", `${Math.min(Math.max(score, 0), 100)}%`);
    riskCircle.style.setProperty("--ring-color", levelColor);
    riskHeading.textContent = `Seizure Risk: ${riskLevelValue}`;
    riskHeading.style.color = riskLevelValue === "High" ? "#f06b5d" : riskLevelValue === "Moderate" ? "#e8b84b" : "#b9f34a";
    riskValue.textContent = `${score.toFixed(1)} / 100`;
    riskNote.textContent = `Assessment for ${formatPatient(data.patient_id)}, Run ${data.run_id}, ${data.window_start_time}s.`;
    riskLevel.textContent = riskLevelValue;
    assessmentStatus.textContent = isHigh ? "Event indicated" : "No event indicated";

    eventHeading.textContent = `${formatPatient(data.patient_id)} / Run ${data.run_id}`;
    eventMark.textContent = eventDetected ? "EVENT" : "CLEAR";
    eventIcon.textContent = eventDetected ? "!" : "✓";
    eventStatus.textContent = eventDetected ? "Seizure Event Detected" : "No Seizure Event";
    eventNote.textContent = `Recorded at ${data.window_start_time}s in this monitoring window.`;
    updateAssessmentComparison(data.prediction, data.recorded_label);

    apiStatus.className = "api-status success";
    apiStatus.textContent = "Assessment complete";
  } catch (error) {
    riskHeading.textContent = "API needs attention";
    riskHeading.style.color = "#f06b5d";
    riskNote.textContent = error.message;
    apiStatus.className = "api-status error";
    apiStatus.textContent = "Assessment unavailable";
    console.error("NeuroCare scoring failed:", error);
  } finally {
    button.disabled = false;
    button.querySelector("span:first-child").textContent = "Analyze This Window";
  }
}

patientSelect.addEventListener("change", () => loadWindows().catch(console.error));
windowSelect.addEventListener("change", updateSelectionSummary);
document.getElementById("run-scoring").addEventListener("click", runScoring);
loadPatients().catch((error) => {
  selectionSummary.textContent = "Dataset API unavailable";
  document.getElementById("api-status").textContent = error.message;
});
