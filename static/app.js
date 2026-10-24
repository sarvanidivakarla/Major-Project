const result = document.querySelector("#result");
const values = document.querySelector("#values");
const toggle = document.querySelector("#toggle-values");
let latestValues = {};
let showingAll = false;

function formatValue(value) {
  return Number.isInteger(value) ? value.toLocaleString() : Number(value).toLocaleString(undefined, { maximumFractionDigits: 4 });
}

function renderValues() {
  values.innerHTML = window.FEATURES.map((feature, index) => `
    <div class="value ${!showingAll && index >= 12 ? "hidden-value" : ""}">
      <small title="${feature}">${feature}</small>
      <strong>${formatValue(latestValues[feature])}</strong>
    </div>
  `).join("");
  toggle.textContent = showingAll ? "Show fewer values" : "Show all values";
}

function renderPrediction(data) {
  const attack = data.prediction === "attack";
  result.className = `result card ${data.prediction}`;
  result.innerHTML = `
    <div class="decision">
      <div>
        <div class="badge ${data.prediction}">${attack ? "Attack detected" : "Benign traffic"}</div>
        <h2>${attack ? "The flow matches attack-like patterns." : "The flow matches benign-like patterns."}</h2>
      </div>
      <div class="probability"><strong>${(data.attack_probability * 100).toFixed(2)}%</strong>attack probability</div>
    </div>
    <div class="bar"><span class="${attack ? "" : "benign"}" style="width:${data.attack_probability * 100}%"></span></div>
    <p class="muted">Why the model decided this</p>
    <ul class="reasons">${data.reasons.map(reason => `<li>${reason}</li>`).join("")}</ul>
  `;
  latestValues = data.values;
  showingAll = false;
  renderValues();
}

async function generate(profile) {
  const buttons = document.querySelectorAll("button[data-profile]");
  buttons.forEach(button => { button.disabled = true; });
  try {
    const response = await fetch(`/api/random?profile=${profile}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Prediction failed");
    renderPrediction(data);
  } catch (error) {
    result.className = "result card";
    result.innerHTML = `<p class="error">${error.message}</p>`;
  } finally {
    buttons.forEach(button => { button.disabled = false; });
  }
}

document.querySelectorAll("button[data-profile]").forEach(button => {
  button.addEventListener("click", () => generate(button.dataset.profile));
});
toggle.addEventListener("click", () => { showingAll = !showingAll; renderValues(); });
