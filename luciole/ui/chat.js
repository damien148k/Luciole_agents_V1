// Chat de Luciole : interroge /api/agent/query (boucle) ou /api/rag/query
// (RAG seul) et affiche la réponse, ses sources et le chemin suivi.
// Aucune bibliothèque : la page doit marcher hors ligne. Tout texte venant de
// l'API passe par textContent, jamais par innerHTML.
"use strict";

const $ = (id) => document.getElementById(id);
const form = $("ask");
const select = $("use-case");
const question = $("question");
const send = $("send");
const waiting = $("waiting");
let useCases = [];

async function getJson(url, options) {
  const r = await fetch(url, options);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(errorText(r.status, body));
  return body;
}

function errorText(status, body) {
  const d = body && body.detail;
  if (typeof d === "string") return d;
  if (Array.isArray(d)) return d.map((e) => e.msg || JSON.stringify(e)).join(" ; ");
  return `erreur ${status}`;
}

function mode() {
  return form.querySelector("input[name=mode]:checked").value;
}

function showUseCase() {
  const uc = useCases.find((u) => u.name === select.value);
  $("use-case-description").textContent = uc ? uc.description : "";
  const agent = form.querySelector("input[value=agent]");
  if (uc && uc.loop) {
    agent.disabled = false;
    $("loop-status").textContent = `boucle « ${uc.loop} »`;
  } else {
    agent.disabled = true;
    form.querySelector("input[value=rag]").checked = true;
    $("loop-status").textContent = uc ? `boucle invalide : ${uc.loop_error}` : "";
  }
  try { localStorage.setItem("luciole.useCase", select.value); } catch (e) { /* stockage refusé */ }
}

async function loadUseCases() {
  try {
    useCases = await getJson("/api/use-cases");
  } catch (e) {
    $("health").textContent = `API injoignable : ${e.message}`;
    return;
  }
  select.replaceChildren(
    ...useCases.map((u) => new Option(u.title ? `${u.title} (${u.name})` : u.name, u.name))
  );
  let saved = null;
  try { saved = localStorage.getItem("luciole.useCase"); } catch (e) { /* idem */ }
  if (saved && useCases.some((u) => u.name === saved)) select.value = saved;
  showUseCase();
}

async function loadHealth() {
  try {
    const h = await getJson("/api/health");
    const loaded = Object.entries(h.loaded || {})
      .map(([name, m]) => `${name} : ${m.llm} sur ${m.hardware}`)
      .join(" · ");
    $("health").textContent =
      `version ${h.versions.app}` + (loaded ? ` · ${loaded}` : " · modèles chargés à la première question");
  } catch (e) {
    $("health").textContent = `API injoignable : ${e.message}`;
  }
}

function item(text, className) {
  const li = document.createElement("li");
  li.textContent = text;
  if (className) li.className = className;
  return li;
}

function render(entry, q, chosenMode, result, seconds) {
  entry.querySelector(".question").textContent = q;
  entry.querySelector(".mode").textContent =
    chosenMode === "agent" ? "boucle" : "RAG seul";
  entry.querySelector(".duration").textContent = `${seconds.toFixed(1)} s`;
  const verdict = entry.querySelector(".verdict");
  const trace = entry.querySelector(".trace");
  if (result.error) {
    verdict.textContent = "ERREUR";
    verdict.className = "verdict ERREUR";
    entry.querySelector(".answer").textContent = result.error;
    trace.hidden = true;
    entry.querySelector(".sources").hidden = true;
    return;
  }
  verdict.textContent = result.verdict;
  verdict.className = `verdict ${result.verdict}`;
  verdict.title = `score brut du reranker : ${result.confidence.toFixed(3)} (non calibré)`;
  entry.querySelector(".answer").textContent = result.answer;
  trace.href = `/api/traces/${encodeURIComponent(result.trace_id)}`;

  // Plusieurs passages d'une même page donnent la même citation : une ligne
  // par citation, citée si l'un de ses passages l'est.
  const byText = new Map();
  for (const c of result.citations) byText.set(c.citation, byText.get(c.citation) || c.cited);
  const unique = [...byText];
  const citations = entry.querySelector(".citations");
  citations.replaceChildren(
    ...unique.filter(([, cited]) => cited).map(([text]) => item(text)),
    ...unique.filter(([, cited]) => !cited).map(([text]) => item(`${text} (fourni, non cité)`, "seen"))
  );
  if (!result.citations.length) citations.replaceChildren(item("aucune source", "seen"));

  if (result.path) {
    const loop = entry.querySelector(".loop");
    loop.hidden = false;
    entry.querySelector(".path").replaceChildren(...result.path.map((n) => item(n)));
    entry.querySelector(".stopped").textContent =
      `Arrêt au nœud « ${result.stopped_reason} » après ${result.steps_used} étape(s).`;
    entry.querySelector(".rules").replaceChildren(
      ...result.matched_rules.map((r) =>
        item(`${r.node} : si ${r.when} → ${r.then}${r.reason ? ` (${r.reason})` : ""}`)
      )
    );
  }
}

async function ask(event) {
  event.preventDefault();
  const q = question.value.trim();
  if (!q) return;
  const chosenMode = mode();
  const url = chosenMode === "agent" ? "/api/agent/query" : "/api/rag/query";
  const entry = $("exchange").content.firstElementChild.cloneNode(true);
  const started = performance.now();
  send.disabled = true;
  waiting.hidden = false;
  const timer = setInterval(() => {
    waiting.textContent = `réponse en cours… ${Math.round((performance.now() - started) / 1000)} s`;
  }, 500);
  let result;
  try {
    result = await getJson(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q, use_case: select.value }),
    });
  } catch (e) {
    result = { error: e.message };
  } finally {
    clearInterval(timer);
    send.disabled = false;
    waiting.hidden = true;
  }
  render(entry, q, chosenMode, result, (performance.now() - started) / 1000);
  $("history").prepend(entry);
  loadHealth();
}

form.addEventListener("submit", ask);
select.addEventListener("change", showUseCase);
question.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) form.requestSubmit();
});
loadUseCases();
loadHealth();
