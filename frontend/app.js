const API_BASE = "";

document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
  });
});

function setStatus(el, text, show = true) {
  el.textContent = text;
  el.classList.toggle("hidden", !show);
}

function renderTrace(container, steps) {
  container.innerHTML = "";
  steps.forEach((s) => {
    const span = document.createElement("span");
    span.className = "trace-step";
    span.textContent = s;
    container.appendChild(span);
  });
}

// ---- Query tab ----

const queryForm = document.getElementById("query-form");
const queryInput = document.getElementById("query-input");
const queryStatus = document.getElementById("query-status");
const queryResult = document.getElementById("query-result");
const streamToggle = document.getElementById("stream-toggle");
const answerText = document.getElementById("answer-text");
const citationsList = document.getElementById("citations-list");
const findingsList = document.getElementById("findings-list");
const qaText = document.getElementById("qa-text");
const qaCard = document.getElementById("qa-card");
const pipelineTrace = document.getElementById("pipeline-trace");

queryForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = queryInput.value.trim();
  if (!query) return;

  document.getElementById("query-submit").disabled = true;
  queryResult.classList.add("hidden");
  setStatus(queryStatus, "Running planner → retriever → reasoner → writer → QA…");

  try {
    if (streamToggle.checked) {
      await runStreamingQuery(query);
    } else {
      await runQuery(query);
    }
  } catch (err) {
    setStatus(queryStatus, `Error: ${err.message}`);
  } finally {
    document.getElementById("query-submit").disabled = false;
  }
});

async function runQuery(query) {
  const res = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();

  setStatus(queryStatus, "", false);
  queryResult.classList.remove("hidden");

  renderTrace(pipelineTrace, [
    "planner",
    `retriever (${data.retrieved_chunks.length} chunks)`,
    "reasoner",
    "writer",
    "qa",
  ]);

  answerText.textContent = data.answer;

  citationsList.innerHTML = "";
  (data.citations || []).forEach((c) => {
    const li = document.createElement("li");
    li.innerHTML = `<b>${c.title || "Untitled"}</b> — ${c.source || ""} <span class="muted">(${c.chunk_id?.slice(0, 8) || ""})</span>`;
    citationsList.appendChild(li);
  });
  if (!data.citations || data.citations.length === 0) {
    citationsList.innerHTML = '<li class="muted">No citations returned.</li>';
  }

  findingsList.innerHTML = "";
  (data.key_findings || []).forEach((f) => {
    const li = document.createElement("li");
    li.textContent = f;
    findingsList.appendChild(li);
  });

  const qa = data.qa || {};
  qaCard.classList.remove("grounded", "revised");
  if (qa.is_grounded) {
    qaCard.classList.add("grounded");
    qaText.textContent = `Grounded. ${qa.faithfulness_notes || ""}`;
  } else {
    qaCard.classList.add("revised");
    qaText.textContent = `Answer was revised for grounding. ${qa.faithfulness_notes || ""}`;
  }
}

async function runStreamingQuery(query) {
  const res = await fetch(`${API_BASE}/query/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

  queryResult.classList.remove("hidden");
  answerText.textContent = "";
  citationsList.innerHTML = '<li class="muted">Not available in streaming preview mode.</li>';
  findingsList.innerHTML = "";
  qaCard.classList.add("hidden");
  renderTrace(pipelineTrace, ["planner…"]);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const events = buffer.split("\n\n");
    buffer = events.pop();

    for (const evt of events) {
      const lines = evt.split("\n");
      const eventLine = lines.find((l) => l.startsWith("event:"));
      const dataLine = lines.find((l) => l.startsWith("data:"));
      if (!eventLine || !dataLine) continue;
      const eventName = eventLine.replace("event:", "").trim();
      const data = JSON.parse(dataLine.replace("data:", "").trim());

      if (eventName === "plan") renderTrace(pipelineTrace, ["planner ✓", "retriever…"]);
      if (eventName === "retrieval") renderTrace(pipelineTrace, ["planner ✓", `retriever ✓ (${data.length})`, "reasoner…"]);
      if (eventName === "reasoning") renderTrace(pipelineTrace, ["planner ✓", "retriever ✓", "reasoner ✓", "writer…"]);
      if (eventName === "token") answerText.textContent += data;
      if (eventName === "done") renderTrace(pipelineTrace, ["planner ✓", "retriever ✓", "reasoner ✓", "writer ✓"]);
    }
  }
  setStatus(queryStatus, "", false);
  qaCard.classList.remove("hidden");
  qaCard.classList.add("grounded");
  qaText.textContent = "Streaming preview mode skips the QA verification pass.";
}

// ---- Coding tab ----

const codingForm = document.getElementById("coding-form");
const codingInput = document.getElementById("coding-input");
const codingStatus = document.getElementById("coding-status");
const codingResult = document.getElementById("coding-result");
const codesTbody = document.getElementById("codes-tbody");
const codingRationale = document.getElementById("coding-rationale");

codingForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = codingInput.value.trim();
  if (!query) return;

  document.getElementById("coding-submit").disabled = true;
  codingResult.classList.add("hidden");
  setStatus(codingStatus, "Running CrewAI coding crew (Coding Agent → Verification Agent)…");

  try {
    const res = await fetch(`${API_BASE}/coding`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    setStatus(codingStatus, "", false);
    codingResult.classList.remove("hidden");

    codesTbody.innerHTML = "";
    (data.suggested_codes || []).forEach((c) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${c.code}</td><td>${c.description}</td><td>${(c.confidence * 100).toFixed(0)}%</td>`;
      codesTbody.appendChild(tr);
    });
    if (!data.suggested_codes || data.suggested_codes.length === 0) {
      codesTbody.innerHTML = '<tr><td colspan="3" class="muted">No codes suggested.</td></tr>';
    }
    codingRationale.textContent = data.rationale || "";
  } catch (err) {
    setStatus(codingStatus, `Error: ${err.message}`);
  } finally {
    document.getElementById("coding-submit").disabled = false;
  }
});

// ---- Ingest tab ----

document.getElementById("ingest-btn").addEventListener("click", async () => {
  const btn = document.getElementById("ingest-btn");
  const statusEl = document.getElementById("ingest-status");
  const reset = document.getElementById("reset-toggle").checked;

  btn.disabled = true;
  setStatus(statusEl, "Ingesting sample documents…");
  try {
    const res = await fetch(`${API_BASE}/ingest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ directory: "data/sample_docs", reset }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    setStatus(statusEl, `Ingested ${data.chunks_ingested} chunks.`);
  } catch (err) {
    setStatus(statusEl, `Error: ${err.message}`);
  } finally {
    btn.disabled = false;
  }
});
