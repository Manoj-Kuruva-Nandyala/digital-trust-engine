const state = { type: "sms" };

const tabs = document.querySelectorAll(".tab");
const content = document.getElementById("content");
const url = document.getElementById("url");
const image = document.getElementById("image");
const label = document.getElementById("input-label");
const helper = document.getElementById("helper");
const counter = document.getElementById("counter");
const result = document.getElementById("analysis-result");
const emptyResult = document.getElementById("empty-result");
const status = document.getElementById("status");
const fileTitle = document.getElementById("file-title");
const fileHint = document.getElementById("file-hint");
const fileStatus = document.getElementById("file-status");
const dropzone = document.getElementById("dropzone");

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    tabs.forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    state.type = tab.dataset.type;

    document.getElementById("text-input-area").classList.toggle("hidden", !["sms", "email"].includes(state.type));
    document.getElementById("url-input-area").classList.toggle("hidden", state.type !== "url");
    document.getElementById("image-input-area").classList.toggle("hidden", state.type !== "screenshot");

    label.textContent = state.type === "email" ? "Email content" : "Message";
    content.placeholder = state.type === "email"
      ? "Paste the email subject and message here..."
      : "Paste the message you want to check here...";

    helper.innerHTML = '<span>i</span> ' + (state.type === "email"
      ? "For your safety, never enter passwords, OTPs, or confidential information."
      : "For your safety, never enter an OTP, password, PIN, or other secret information.");

    emptyResult.classList.remove("hidden");
    result.classList.add("hidden");
    status.textContent = "";
  });
});

content.addEventListener("input", () => {
  counter.textContent = content.value.length.toLocaleString() + " characters";
});

image.addEventListener("change", () => {
  const file = image.files && image.files[0];
  if (!file) {
    fileTitle.textContent = "Choose a screenshot";
    fileHint.textContent = "Click here or drop a PNG / JPG image";
    fileStatus.textContent = "";
    return;
  }
  fileTitle.textContent = file.name;
  fileHint.textContent = "Ready to analyze";
  fileStatus.textContent = "✓ Screenshot selected";
  status.textContent = "";
});

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragging");
  });
});
["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragging");
  });
});
dropzone.addEventListener("drop", (event) => {
  const files = event.dataTransfer.files;
  if (!files || !files.length) return;
  const file = files[0];
  if (!file.type.startsWith("image/")) {
    status.textContent = "Please choose a PNG or JPG screenshot.";
    return;
  }
  const dt = new DataTransfer();
  dt.items.add(file);
  image.files = dt.files;
  image.dispatchEvent(new Event("change"));
});

document.getElementById("clear-btn").addEventListener("click", () => {
  content.value = "";
  url.value = "";
  image.value = "";
  counter.textContent = "0 characters";
  fileTitle.textContent = "Choose a screenshot";
  fileHint.textContent = "Click here or drop a PNG / JPG image";
  fileStatus.textContent = "";
  emptyResult.classList.remove("hidden");
  result.classList.add("hidden");
  status.textContent = "";
});

document.getElementById("check-btn").addEventListener("click", async () => {
  const button = document.getElementById("check-btn");
  button.disabled = true;
  status.textContent = "Checking...";

  try {
    let response;

    if (state.type === "screenshot") {
      const file = image.files && image.files[0];
      if (!file) throw new Error("Please select a screenshot first.");
      if (!file.type.startsWith("image/")) throw new Error("Please select a PNG or JPG image.");
      if (file.size > 10 * 1024 * 1024) throw new Error("Screenshot is too large. Please use an image below 10 MB.");

      // Use browser-side OCR so the screenshot flow does not depend on
      // Tesseract being installed on the server/PC.
      if (typeof Tesseract === "undefined") {
        throw new Error("Screenshot OCR is not available. Please refresh the page and try again.");
      }

      status.textContent = "Reading screenshot...";
      const ocr = await Tesseract.recognize(file, "eng", {
        logger: (message) => {
          if (message && message.status === "recognizing text" && typeof message.progress === "number") {
            status.textContent = "Reading screenshot... " + Math.round(message.progress * 100) + "%";
          }
        }
      });

      const extractedText = (ocr.data.text || "").trim();
      if (!extractedText) {
        throw new Error("No readable text was found in the screenshot. Please use a clearer image.");
      }

      status.textContent = "Checking screenshot text...";

      response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          type: "screenshot",
          text: extractedText,
          url: ""
        })
      });
    } else {
      const input = { type: state.type, text: content.value, url: url.value };

      if (state.type === "url" && !url.value.trim()) throw new Error("Please enter a link.");
      if (["sms", "email"].includes(state.type) && !content.value.trim()) {
        throw new Error(state.type === "email" ? "Please enter an email." : "Please enter a message.");
      }

      response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(input),
      });
    }

    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Analysis failed.");
    render(data);
  } catch (error) {
    status.textContent = error.message || "The analysis could not be completed.";
  } finally {
    button.disabled = false;
  }
});

function render(data) {
  status.textContent = "";
  emptyResult.classList.add("hidden");
  result.classList.remove("hidden");

  document.getElementById("result-title").textContent = data.title;
  document.getElementById("score").textContent = Number(data.fraud_score).toFixed(2);
  document.getElementById("score-summary").textContent = data.summary;
  document.getElementById("result-subtitle").textContent = "Analysis completed";
  document.getElementById("result-note").textContent = data.note || "";

  const badge = document.getElementById("risk-badge");
  const risk = String(data.risk_level || "HIGH RISK");
  badge.textContent = risk;
  badge.className = "risk-badge " + risk.toLowerCase().replace(" ", "-");

  document.getElementById("signals").innerHTML = (data.signals || []).map((x) => "<li>" + escapeHtml(x) + "</li>").join("");
  document.getElementById("actions").innerHTML = (data.actions || []).map((x) => "<li>" + escapeHtml(x) + "</li>").join("");
  document.getElementById("technical-content").textContent = JSON.stringify(data.technical || {}, null, 2);

  const score = Math.max(0, Math.min(100, Number(data.fraud_score) || 0));
  document.getElementById("score-bar").style.width = score + "%";

  const high = risk.toLowerCase().includes("high");
  const medium = risk.toLowerCase().includes("medium");
  const icon = document.getElementById("result-icon");
  icon.textContent = high ? "⚠" : medium ? "!" : "✓";
  icon.style.color = high ? "var(--danger)" : medium ? "var(--warning)" : "var(--safe)";
  icon.style.background = high ? "rgba(255,85,112,.15)" : medium ? "rgba(255,184,77,.14)" : "rgba(63,224,160,.13)";
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
  }[character]));
}
