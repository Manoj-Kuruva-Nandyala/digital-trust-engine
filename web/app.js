const state = { type: "sms" };

const tabs = document.querySelectorAll(".tab");
const content = document.getElementById("content");
const url = document.getElementById("url");
const image = document.getElementById("image");
const label = document.getElementById("input-label");
const helper = document.getElementById("helper");
const counter = document.getElementById("counter");
const result = document.getElementById("result");
const status = document.getElementById("status");

tabs.forEach(tab => tab.addEventListener("click", () => {
  tabs.forEach(t => t.classList.remove("active"));
  tab.classList.add("active");
  state.type = tab.dataset.type;
  document.getElementById("text-input-area").classList.toggle("hidden", !["sms","email"].includes(state.type));
  document.getElementById("url-input-area").classList.toggle("hidden", state.type !== "url");
  document.getElementById("image-input-area").classList.toggle("hidden", state.type !== "screenshot");
  label.textContent = state.type === "email" ? "Email content" : "Message";
  content.placeholder = state.type === "email"
    ? "Paste the email subject and message here..."
    : "Paste the message you want to check here...";
  helper.textContent = state.type === "email"
    ? "Never enter passwords, OTPs, or confidential information."
    : "Never enter an OTP, password, PIN, or other secret information.";
  result.classList.add("hidden");
  status.textContent = "";
}));

content.addEventListener("input", () => {
  counter.textContent = content.value.length.toLocaleString() + " characters";
});

document.getElementById("check-btn").addEventListener("click", async () => {
  const button = document.getElementById("check-btn");
  button.disabled = true;
  status.textContent = "Checking...";
  try {
    const body = { type: state.type, text: content.value, url: url.value };
    if (state.type === "screenshot" && image.files[0]) {
      const form = new FormData();
      form.append("image", image.files[0]);
      const response = await fetch("/api/analyze/screenshot", { method:"POST", body:form });
      render(await response.json());
    } else {
      const response = await fetch("/api/analyze", {
        method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)
      });
      render(await response.json());
    }
  } catch (e) {
    status.textContent = "The analysis service is not available yet.";
  } finally {
    button.disabled = false;
  }
});

function render(data) {
  status.textContent = "";
  result.classList.remove("hidden");
  document.getElementById("result-title").textContent = data.title;
  document.getElementById("score").textContent = Number(data.fraud_score).toFixed(2);
  document.getElementById("score-summary").textContent = data.summary;
  document.getElementById("result-note").textContent = data.note || "";
  const badge = document.getElementById("risk-badge");
  badge.textContent = data.risk_level;
  badge.className = "risk-badge " + data.risk_level.toLowerCase().replace(" ","-");

  document.getElementById("signals").innerHTML = (data.signals || []).map(x => "<li>"+escapeHtml(x)+"</li>").join("");
  document.getElementById("actions").innerHTML = (data.actions || []).map(x => "<li>"+escapeHtml(x)+"</li>").join("");
  document.getElementById("technical-content").textContent = JSON.stringify(data.technical || {}, null, 2);

  const score = Math.max(0, Math.min(100, Number(data.fraud_score)));
  document.getElementById("score-ring").style.background =
    `conic-gradient(var(--danger) 0 ${score}%, #edf0f5 ${score}% 100%)`;
  result.scrollIntoView({behavior:"smooth", block:"start"});
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, c => ({ "&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#039;" }[c]));
}
