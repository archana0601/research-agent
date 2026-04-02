import os
import json
import re
import time
from flask import Flask, request, jsonify
from groq import Groq

app = Flask(__name__)

def get_client():
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY environment variable is not set.")
    return Groq(api_key=key)

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")


# ─── Search ───────────────────────────────────────────────────────────────────

def _search_tavily(query):
    import requests as req
    resp = req.post(
        "https://api.tavily.com/search",
        json={"api_key": TAVILY_API_KEY, "query": query, "max_results": 8},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", [])
    if not results:
        return "No results found.", []
    out = ""
    sources = []
    for r in results:
        out += f"Title: {r['title']}\nSummary: {r.get('content', '')}\nURL: {r['url']}\n\n"
        sources.append({"title": r["title"], "url": r["url"]})
    return out.strip(), sources



def web_search(query, region="wt-wt"):
    if not TAVILY_API_KEY:
        return "", []
    try:
        return _search_tavily(query)
    except Exception as e:
        print(f"[Tavily failed] {e}")
        return "", []


# ─── Research pipeline ────────────────────────────────────────────────────────

def generate_queries(question):
    res = get_client().chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": f"""Generate 5 highly targeted search queries to comprehensively research this question.
Cover: overview, statistics/data, expert opinions, recent developments, practical implications.
Return ONLY a JSON array of 5 strings, nothing else.
Question: {question}"""}],
        temperature=0.1,
    )
    return json.loads(res.choices[0].message.content.strip())


def extract_facts(raw_results, question):
    """Pass 1: pull out specific facts, numbers, names from raw search results."""
    res = get_client().chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": f"""You are a data extraction specialist. Read the search results carefully and extract every specific, concrete piece of information.

Extract and list:
- All specific numbers, percentages, prices, dates, measurements
- All brand names, product names, company names, person names
- All pros and cons mentioned
- All comparisons between options
- All expert recommendations or warnings
- Any notable quotes or claims

Be exhaustive. Format as a structured list. Do not paraphrase — use the exact figures and names from the text.

Search Results:
{raw_results}

Topic: {question}"""}],
        temperature=0.1,
    )
    return res.choices[0].message.content.strip()


def synthesize_report(question, web_results, knowledge):
    """Pass 2: build structured report, web results take priority over model knowledge."""
    res = get_client().chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": f"""You are a senior research analyst writing a structured report.

PRIORITY RULES — follow strictly:
1. WEB SEARCH RESULTS contain the freshest, most specific data. Always prefer these facts, figures, names, and prices over anything else.
2. BACKGROUND KNOWLEDGE fills gaps only where web results are silent or vague.
3. Never contradict a specific figure from web results with a generic estimate from background knowledge.
4. If web results have prices, dates, rankings, or product names — use them exactly.

Return ONLY a valid JSON object with exactly these fields:
{{
  "summary": "3 sentence executive summary — lead with the most specific findings and real numbers from web results",
  "key_stats": [
    {{"value": "exact number/$/%/year from web results", "label": "what it represents"}}
  ],
  "sections": [
    {{"title": "Section Title", "content": "Rich markdown. Use bullet points (-), bold (**text**), specific names/numbers from web results. Min 5 bullets per section."}}
  ],
  "comparison_table": {{
    "headers": ["Aspect", "Option A", "Option B", "Option C"],
    "rows": [
      ["Feature", "value", "value", "value"]
    ]
  }},
  "takeaways": [
    "Actionable insight starting with a verb and containing a specific detail from the research"
  ]
}}

Rules:
- key_stats: exactly 4, prefer real numbers from web results
- sections: exactly 4, each grounded in specific findings — no filler
- comparison_table: compare 2-4 real options from the research (products, providers, approaches)
- takeaways: exactly 5, each starting with Choose/Avoid/Look for/Check/Compare/Use/Pick/Consider

WEB SEARCH RESULTS (primary source — use these facts first):
{web_results}

BACKGROUND KNOWLEDGE (use only to fill gaps):
{knowledge}

Question: {question}"""}],
        temperature=0.2,
    )
    content = res.choices[0].message.content.strip()
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        content = match.group()
    return json.loads(content)


def research_angles(question):
    """Break question into 4 focused sub-questions to answer from knowledge."""
    res = get_client().chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": f"""Break this research question into 4 focused sub-questions that together give a complete picture.
Return ONLY a JSON array of 4 strings.
Question: {question}"""}],
        temperature=0.1,
    )
    return json.loads(res.choices[0].message.content.strip())


def answer_from_knowledge(question, angles):
    """Use model knowledge to answer each research angle in depth."""
    angles_text = "\n".join(f"{i+1}. {a}" for i, a in enumerate(angles))
    res = get_client().chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": f"""You are a senior research analyst with deep expertise. Answer each of these research sub-questions thoroughly and specifically about: {question}

Sub-questions:
{angles_text}

For each answer:
- Give specific names, figures, companies, trends
- Include real statistics or well-known estimates
- Be concrete and detailed — no vague generalities

Format each answer clearly under its sub-question number."""}],
        temperature=0.3,
    )
    return res.choices[0].message.content.strip()


def run_research(question):
    # Generate research angles
    angles = research_angles(question)

    # Detect region
    q_lower = question.lower()
    region = "wt-wt"
    if any(w in q_lower for w in ["singapore", " sg"]):
        region = "sg-en"
    elif "india" in q_lower:
        region = "in-en"
    elif any(w in q_lower for w in ["uk", "britain", "london"]):
        region = "uk-en"
    elif any(w in q_lower for w in ["australia", "sydney"]):
        region = "au-en"

    # Search the main question + all angles (web results are primary)
    search_queries = [question] + angles
    web_context = ""
    all_sources = []
    for q in search_queries:
        results, sources = web_search(q, region=region)
        if results:
            web_context += f"--- '{q}' ---\n{results}\n\n"
            all_sources.extend(sources)

    # Background knowledge fills gaps only
    knowledge = answer_from_knowledge(question, angles)

    # Synthesize: web results primary, knowledge secondary
    report = synthesize_report(question, web_context or "No web results found.", knowledge)

    seen = set()
    unique_sources = []
    for s in all_sources:
        if s["url"] not in seen:
            seen.add(s["url"])
            unique_sources.append(s)

    report["queries"] = angles
    report["sources"] = unique_sources[:8]
    return report


# ─── HTML ─────────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Research Agent</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0b0b0f; color: #e8e8f0; min-height: 100vh; display: flex; flex-direction: column; }

header { padding: 22px 40px; border-bottom: 1px solid #1e1e2e; display: flex; align-items: center; gap: 14px; flex-shrink: 0; }
.logo { font-size: 18px; font-weight: 800; background: linear-gradient(135deg, #7c6fff, #3dd8cc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
header small { font-size: 12px; color: #444; -webkit-text-fill-color: #444; }

.layout { display: flex; flex: 1; }

/* Sidebar */
.sidebar { width: 260px; flex-shrink: 0; border-right: 1px solid #1e1e2e; padding: 24px 16px; display: flex; flex-direction: column; gap: 8px; min-height: calc(100vh - 65px); }
.sidebar-title { font-size: 10px; text-transform: uppercase; letter-spacing: 2px; color: #444; font-weight: 700; padding: 0 8px; margin-bottom: 4px; display: flex; justify-content: space-between; align-items: center; }
.clear-all { font-size: 10px; color: #333; cursor: pointer; text-transform: none; letter-spacing: 0; transition: color 0.2s; }
.clear-all:hover { color: #ff6b6b; }
.history-item { padding: 10px 12px; border-radius: 10px; cursor: pointer; border: 1px solid transparent; transition: all 0.2s; }
.history-item:hover { background: #141420; border-color: #2a2a3e; }
.history-item.active { background: #1a1a2e; border-color: #7c6fff44; }
.hi-question { font-size: 13px; color: #ccc; line-height: 1.4; margin-bottom: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.hi-meta { display: flex; justify-content: space-between; align-items: center; }
.hi-date { font-size: 11px; color: #444; }
.hi-del { font-size: 11px; color: #333; cursor: pointer; transition: color 0.2s; padding: 2px 4px; }
.hi-del:hover { color: #ff6b6b; }
.no-history { font-size: 12px; color: #333; padding: 8px; text-align: center; margin-top: 16px; }

/* Main */
.main { flex: 1; overflow-x: hidden; }
.hero { padding: 56px 40px 36px; max-width: 860px; margin: 0 auto; }
.hero h1 { font-size: 36px; font-weight: 900; margin-bottom: 10px; }
.hero h1 span { background: linear-gradient(135deg, #7c6fff, #3dd8cc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hero p { color: #555; font-size: 15px; margin-bottom: 28px; }

.search-row { display: flex; gap: 10px; }
input { flex: 1; padding: 17px 22px; font-size: 15px; border: 1.5px solid #1e1e2e; border-radius: 12px; background: #141420; color: #e8e8f0; outline: none; transition: border 0.2s; }
input::placeholder { color: #444; }
input:focus { border-color: #7c6fff; }
.btn { padding: 17px 30px; background: linear-gradient(135deg, #7c6fff, #3dd8cc); color: #0b0b0f; border: none; border-radius: 12px; font-size: 15px; font-weight: 800; cursor: pointer; white-space: nowrap; transition: opacity 0.2s; }
.btn:hover { opacity: 0.88; }
.btn:disabled { opacity: 0.35; cursor: not-allowed; }

.container { max-width: 860px; margin: 0 auto; padding: 0 40px 80px; }

.loader { display: none; padding: 40px 0; }
.progress-steps { display: flex; flex-direction: column; gap: 16px; }
.ps { display: flex; align-items: center; gap: 14px; font-size: 14px; color: #444; transition: all 0.3s; }
.ps.active { color: #e8e8f0; }
.ps.done { color: #3dd8cc; }
.ps-dot { width: 8px; height: 8px; border-radius: 50%; background: #222; flex-shrink: 0; transition: background 0.3s; }
.ps.active .ps-dot { background: #7c6fff; animation: blink 1s infinite; }
.ps.done .ps-dot { background: #3dd8cc; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

.results { display: none; }
.label { font-size: 10px; text-transform: uppercase; letter-spacing: 2.5px; font-weight: 700; margin-bottom: 14px; }

.q-banner { background: linear-gradient(135deg, #13132a, #1a1a35); border: 1px solid #2a2a55; border-radius: 14px; padding: 22px 26px; margin-bottom: 24px; }
.q-banner .label { color: #7c6fff; margin-bottom: 8px; }
.q-banner h2 { font-size: 19px; font-weight: 700; line-height: 1.4; }

.summary { background: linear-gradient(135deg, #0d1f2d, #112233); border: 1px solid #1a3a55; border-radius: 14px; padding: 26px; margin-bottom: 24px; }
.summary .label { color: #3dd8cc; }
.summary p { font-size: 15px; line-height: 1.85; color: #b0c4cc; }

.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; align-items: start; }

.stats-card { background: #111118; border: 1px solid #1e1e2e; border-radius: 14px; padding: 24px; }
.stats-card .label { color: #7c6fff; }
.stat-item { display: flex; align-items: center; gap: 14px; padding: 12px 0; border-bottom: 1px solid #1a1a28; }
.stat-item:last-child { border-bottom: none; }
.stat-val { font-size: 22px; font-weight: 800; background: linear-gradient(135deg, #7c6fff, #3dd8cc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; min-width: 70px; }
.stat-lbl { font-size: 13px; color: #666; line-height: 1.4; }

.chart-card { background: #111118; border: 1px solid #1e1e2e; border-radius: 14px; padding: 24px; }
.chart-card .label { color: #7c6fff; }

.sections-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-bottom: 24px; }
.sec { background: #111118; border: 1px solid #1e1e2e; border-radius: 14px; padding: 22px; }
.sec h4 { font-size: 13px; font-weight: 700; color: #7c6fff; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 14px; }
.sec .body { font-size: 13px; color: #999; line-height: 1.85; }
.sec .body ul { padding-left: 16px; }
.sec .body li { margin-bottom: 7px; }
.sec .body strong { color: #ddd; }
.sec .body p { margin-bottom: 8px; }

.table-card { background: #111118; border: 1px solid #1e1e2e; border-radius: 14px; padding: 24px; margin-bottom: 24px; overflow-x: auto; }
.table-card .label { color: #7c6fff; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { background: #1a1a2e; color: #7c6fff; text-align: left; padding: 12px 16px; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }
td { padding: 11px 16px; border-bottom: 1px solid #1a1a28; color: #aaa; }
tr:last-child td { border-bottom: none; }
tr:hover td { background: #141422; color: #ddd; }

.takeaways { background: linear-gradient(135deg, #0a1f14, #0d1f16); border: 1px solid #1a4030; border-radius: 14px; padding: 26px; margin-bottom: 24px; }
.takeaways .label { color: #3dd8cc; }
.tw { display: flex; gap: 14px; margin-bottom: 14px; align-items: flex-start; }
.tw:last-child { margin-bottom: 0; }
.tw-n { background: linear-gradient(135deg, #7c6fff, #3dd8cc); color: #0b0b0f; width: 22px; height: 22px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 900; flex-shrink: 0; margin-top: 2px; }
.tw p { font-size: 14px; color: #b0c8b8; line-height: 1.7; }

.queries-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 24px; }
.qtag { background: #111118; border: 1px solid #222; color: #666; padding: 5px 12px; border-radius: 20px; font-size: 12px; }

.sources { border-top: 1px solid #1a1a28; padding-top: 22px; }
.sources .label { color: #444; }
.src { display: flex; align-items: center; gap: 10px; padding: 8px 0; border-bottom: 1px solid #111; }
.src:last-child { border-bottom: none; }
.src-dot { width: 5px; height: 5px; border-radius: 50%; background: #333; flex-shrink: 0; }
.src a { font-size: 13px; color: #555; text-decoration: none; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; display: block; transition: color 0.2s; }
.src a:hover { color: #7c6fff; }

.export-bar { display: flex; justify-content: flex-end; gap: 10px; margin-bottom: 20px; }
.export-btn { padding: 10px 22px; background: #1a1a28; border: 1px solid #333; color: #aaa; border-radius: 10px; font-size: 13px; font-weight: 600; cursor: pointer; transition: all 0.2s; }
.export-btn:hover { border-color: #7c6fff; color: #7c6fff; }

.followup-box { background: #111118; border: 1px solid #1e1e2e; border-radius: 14px; padding: 24px; margin-top: 24px; }
.followup-box .label { color: #7c6fff; margin-bottom: 14px; }
.followup-row { display: flex; gap: 10px; }
.followup-row input { flex: 1; padding: 13px 18px; font-size: 14px; border: 1.5px solid #1e1e2e; border-radius: 10px; background: #0b0b0f; color: #e8e8f0; outline: none; }
.followup-row input:focus { border-color: #7c6fff; }
.followup-row button { padding: 13px 22px; background: #1a1a2e; border: 1px solid #7c6fff44; color: #7c6fff; border-radius: 10px; font-size: 13px; font-weight: 700; cursor: pointer; transition: all 0.2s; white-space: nowrap; }
.followup-row button:hover { background: #7c6fff22; }
.followup-row button:disabled { opacity: 0.4; cursor: not-allowed; }
.followup-answer { margin-top: 18px; padding-top: 18px; border-top: 1px solid #1e1e2e; font-size: 14px; color: #aaa; line-height: 1.85; display: none; }
.followup-answer ul { padding-left: 18px; }
.followup-answer li { margin-bottom: 6px; }
.followup-answer strong { color: #ddd; }

@media print {
  body { background: white; color: #111; }
  header, .hero, .loader, .export-bar, .queries-row, .search-row { display: none !important; }
  .results { display: block !important; }
  .q-banner { background: #f0f0ff; border: 1px solid #ccc; -webkit-print-color-adjust: exact; }
  .q-banner h2 { color: #111; }
  .q-banner .label { color: #7c6fff; }
  .summary { background: #f0f8ff; border: 1px solid #ccc; -webkit-print-color-adjust: exact; }
  .summary p { color: #333; }
  .two-col, .sections-grid { grid-template-columns: 1fr 1fr; }
  .stat-val { color: #7c6fff !important; -webkit-text-fill-color: #7c6fff !important; }
  .stats-card, .sec, .table-card, .takeaways, .chart-card { background: #fafafa; border: 1px solid #ddd; -webkit-print-color-adjust: exact; }
  .tw-n { background: #7c6fff !important; color: white !important; -webkit-print-color-adjust: exact; }
  .tw p, .sec .body, .stat-lbl { color: #333; }
  th { background: #ebebff !important; color: #333 !important; -webkit-print-color-adjust: exact; }
  td { color: #333; }
  canvas { max-width: 100%; }
}
</style>
</head>
<body>
<header>
  <div class="logo">Research Agent</div>
  <small>Groq + DuckDuckGo — Free &amp; Open</small>
</header>

<div class="layout">
  <!-- Sidebar -->
  <div class="sidebar" id="sidebar">
    <div class="sidebar-title">
      History
      <span class="clear-all" onclick="clearHistory()">Clear all</span>
    </div>
    <div id="history-list"><div class="no-history">No searches yet</div></div>
  </div>

  <!-- Main -->
  <div class="main">
    <div class="hero">
      <h1>Deep research, <span>beautifully structured.</span></h1>
      <p>5 targeted searches. 2-pass AI analysis. Structured report with charts, comparisons &amp; actionable takeaways.</p>
      <div class="search-row">
        <input id="q" type="text" placeholder="What do you want to research?" onkeydown="if(event.key==='Enter')go()" autofocus />
        <button class="btn" id="btn" onclick="go()">Research</button>
      </div>
    </div>

    <div class="container">
      <div class="loader" id="loader">
        <div class="progress-steps">
          <div class="ps" id="p1"><div class="ps-dot"></div> Planning research angles...</div>
          <div class="ps" id="p2"><div class="ps-dot"></div> Searching the web...</div>
          <div class="ps" id="p3"><div class="ps-dot"></div> Extracting facts &amp; data points...</div>
          <div class="ps" id="p4"><div class="ps-dot"></div> Synthesizing your report...</div>
        </div>
      </div>
      <div class="results" id="results"></div>
    </div>
  </div>
</div>

<script>
var chartInstance = null;
var currentData = null;
var currentQuestion = "";

// ── History ──────────────────────────────────────────────────────────────────

function loadHistory() {
  try { return JSON.parse(localStorage.getItem("research_history") || "[]"); }
  catch(e) { return []; }
}

function saveToHistory(question, data) {
  var history = loadHistory();
  var item = {
    id: Date.now().toString(),
    question: question,
    date: new Date().toLocaleDateString("en-GB", {day:"numeric", month:"short", hour:"2-digit", minute:"2-digit"}),
    data: data
  };
  history.unshift(item);
  if (history.length > 30) history = history.slice(0, 30);
  localStorage.setItem("research_history", JSON.stringify(history));
  renderSidebar();
}

function renderSidebar() {
  var history = loadHistory();
  var list = document.getElementById("history-list");
  if (!history.length) {
    list.innerHTML = "<div class='no-history'>No searches yet</div>";
    return;
  }
  list.innerHTML = history.map(function(item) {
    return "<div class='history-item' data-id='" + item.id + "'>" +
      "<div class='hi-question'>" + item.question + "</div>" +
      "<div class='hi-meta'>" +
        "<span class='hi-date'>" + item.date + "</span>" +
        "<span class='hi-del' data-del='" + item.id + "'>✕</span>" +
      "</div></div>";
  }).join("");

  list.querySelectorAll(".history-item").forEach(function(el) {
    el.addEventListener("click", function() {
      loadReport(el.dataset.id);
    });
  });

  list.querySelectorAll(".hi-del").forEach(function(el) {
    el.addEventListener("click", function(e) {
      e.stopPropagation();
      deleteItem(el.dataset.del);
    });
  });
}

function loadReport(id) {
  var history = loadHistory();
  var item = history.find(function(h){ return h.id === id; });
  if (!item) return;
  document.getElementById("q").value = item.question;
  document.getElementById("loader").style.display = "none";
  document.querySelectorAll(".history-item").forEach(function(el){ el.classList.remove("active"); });
  var el = document.querySelector("[data-id='" + id + "']");
  if (el) el.classList.add("active");
  renderReport(item.question, item.data);
}

function deleteItem(id) {
  var history = loadHistory().filter(function(h){ return h.id !== id; });
  localStorage.setItem("research_history", JSON.stringify(history));
  renderSidebar();
}

function clearHistory() {
  localStorage.removeItem("research_history");
  renderSidebar();
}

window.addEventListener("DOMContentLoaded", function() { renderSidebar(); });

// ── Search ───────────────────────────────────────────────────────────────────

async function go() {
  var q = document.getElementById("q").value.trim();
  if (!q) return;
  document.getElementById("btn").disabled = true;
  document.getElementById("results").style.display = "none";
  document.getElementById("results").innerHTML = "";
  if (chartInstance) { chartInstance.destroy(); chartInstance = null; }

  var loader = document.getElementById("loader");
  loader.style.display = "block";

  var pids = ["p1","p2","p3","p4"];
  pids.forEach(function(id){ document.getElementById(id).className = "ps"; });
  var si = 0;
  function nextStep() {
    if (si > 0) document.getElementById(pids[si-1]).className = "ps done";
    if (si < pids.length) { document.getElementById(pids[si]).className = "ps active"; si++; }
  }
  nextStep();
  var t1 = setTimeout(function(){ nextStep(); }, 3000);
  var t2 = setTimeout(function(){ nextStep(); }, 10000);
  var t3 = setTimeout(function(){ nextStep(); }, 16000);

  try {
    var res = await fetch("/research", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({question: q})
    });
    var text = await res.text();
    var data;
    try {
      data = JSON.parse(text);
    } catch(parseErr) {
      throw new Error("Server returned unexpected response. Please try again.");
    }
    clearTimeout(t1); clearTimeout(t2); clearTimeout(t3);
    pids.forEach(function(id){ document.getElementById(id).className = "ps done"; });
    loader.style.display = "none";

    if (data.error) {
      document.getElementById("results").style.display = "block";
      document.getElementById("results").innerHTML = "<p style='color:#ff6b6b;padding:20px 0'>Error: " + data.error + "</p>";
    } else {
      saveToHistory(q, data);
      renderReport(q, data);
    }
  } catch(e) {
    clearTimeout(t1); clearTimeout(t2); clearTimeout(t3);
    loader.style.display = "none";
    document.getElementById("results").style.display = "block";
    document.getElementById("results").innerHTML = "<p style='color:#ff6b6b;padding:20px 0'>" + e.message + "</p>";
  }
  document.getElementById("btn").disabled = false;
}

function renderReport(question, d) {
  var r = document.getElementById("results");
  r.style.display = "block";

  // Stats list
  var statsHtml = "";
  (d.key_stats || []).forEach(function(s) {
    statsHtml += "<div class='stat-item'><div class='stat-val'>" + s.value + "</div><div class='stat-lbl'>" + s.label + "</div></div>";
  });

  // Sections
  var secsHtml = "";
  (d.sections || []).forEach(function(s) {
    secsHtml += "<div class='sec'><h4>" + s.title + "</h4><div class='body'>" + marked.parse(s.content) + "</div></div>";
  });

  // Comparison table
  var tableHtml = "";
  var ct = d.comparison_table;
  if (ct && ct.headers && ct.rows && ct.rows.length) {
    var thead = ct.headers.map(function(h){ return "<th>" + h + "</th>"; }).join("");
    var tbody = ct.rows.map(function(row){
      return "<tr>" + row.map(function(cell){ return "<td>" + cell + "</td>"; }).join("") + "</tr>";
    }).join("");
    tableHtml = "<div class='table-card'><div class='label'>Comparison</div><table><thead><tr>" + thead + "</tr></thead><tbody>" + tbody + "</tbody></table></div>";
  }

  // Takeaways
  var twHtml = "";
  (d.takeaways || []).forEach(function(t, i) {
    twHtml += "<div class='tw'><div class='tw-n'>" + (i+1) + "</div><p>" + t + "</p></div>";
  });

  // Queries
  var qHtml = (d.queries || []).map(function(q){ return "<span class='qtag'>" + q + "</span>"; }).join("");

  // Sources
  var srcHtml = (d.sources || []).slice(0,6).map(function(s){
    return "<div class='src'><div class='src-dot'></div><a href='" + s.url + "' target='_blank'>" + s.title + "</a></div>";
  }).join("");

  r.innerHTML =
    "<div class='export-bar'>" +
      "<button class='export-btn' id='pdfBtn' onclick='downloadPDF()'>Download PDF</button>" +
      "<button class='export-btn' onclick='window.print()'>Print</button>" +
    "</div>" +
    "<div id='report-content'>" +
      "<div class='q-banner'><div class='label'>Research Question</div><h2>" + question + "</h2></div>" +
      "<div class='summary'><div class='label'>Executive Summary</div><p>" + (d.summary || "") + "</p></div>" +
      "<div class='two-col'>" +
        "<div class='stats-card'><div class='label'>Key Statistics</div>" + statsHtml + "</div>" +
        "<div class='chart-card'><div class='label'>Stats at a Glance</div><canvas id='statsChart'></canvas></div>" +
      "</div>" +
      "<div class='sections-grid'>" + secsHtml + "</div>" +
      tableHtml +
      "<div class='takeaways'><div class='label'>Actionable Takeaways</div>" + twHtml + "</div>" +
      "<div class='queries-row'>" + qHtml + "</div>" +
      "<div class='sources'><div class='label'>Sources</div>" + srcHtml + "</div>" +
    "</div>" +
    "<div class='followup-box'>" +
      "<div class='label'>Ask a Follow-up Question</div>" +
      "<div class='followup-row'>" +
        "<input id='fuInput' type='text' placeholder='e.g. Which option is best for beginners?' />" +
        "<button id='fuBtn' onclick='askFollowup()'>Ask</button>" +
      "</div>" +
      "<div class='followup-answer' id='fuAnswer'></div>" +
    "</div>";

  currentData = d;
  currentQuestion = question;

  // Wire up follow-up Enter key
  var fuInput = document.getElementById("fuInput");
  if (fuInput) {
    fuInput.addEventListener("keydown", function(e) {
      if (e.key === "Enter") askFollowup();
    });
  }

  // Draw chart after DOM is updated
  setTimeout(function() {
    var ctx = document.getElementById("statsChart");
    if (!ctx) return;
    var labels = (d.key_stats || []).map(function(s){ return s.label; });
    var values = (d.key_stats || []).map(function(s){
      var n = parseFloat(s.value.replace(/[^0-9.]/g, ""));
      return isNaN(n) ? 0 : n;
    });
    if (chartInstance) chartInstance.destroy();
    chartInstance = new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [{
          data: values,
          backgroundColor: ["rgba(124,111,255,0.7)","rgba(61,216,204,0.7)","rgba(124,111,255,0.5)","rgba(61,216,204,0.5)"],
          borderColor: ["#7c6fff","#3dd8cc","#7c6fff","#3dd8cc"],
          borderWidth: 1.5,
          borderRadius: 6,
        }]
      },
      options: {
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: "#555", font: { size: 10 } }, grid: { color: "#1a1a28" } },
          y: { ticks: { color: "#555", font: { size: 10 } }, grid: { color: "#1a1a28" } }
        }
      }
    });
  }, 100);
}

function downloadPDF() {
  var btn = document.getElementById("pdfBtn");
  btn.disabled = true;
  btn.textContent = "Generating...";
  var el = document.getElementById("report-content");
  var opt = {
    margin: [10, 10],
    filename: (currentQuestion || "research-report").slice(0, 60) + ".pdf",
    image: { type: "jpeg", quality: 0.95 },
    html2canvas: { scale: 2, useCORS: true, backgroundColor: "#ffffff" },
    jsPDF: { unit: "mm", format: "a4", orientation: "portrait" }
  };
  // Temporarily apply light background for PDF
  el.style.background = "#fff";
  el.style.color = "#111";
  html2pdf().set(opt).from(el).save().then(function() {
    el.style.background = "";
    el.style.color = "";
    btn.disabled = false;
    btn.textContent = "Download PDF";
  });
}

async function askFollowup() {
  var input = document.getElementById("fuInput");
  var btn = document.getElementById("fuBtn");
  var answerDiv = document.getElementById("fuAnswer");
  var q = input.value.trim();
  if (!q || !currentData) return;
  btn.disabled = true;
  btn.textContent = "Thinking...";
  answerDiv.style.display = "block";
  answerDiv.innerHTML = "<span style='color:#555'>Researching your follow-up...</span>";
  try {
    var res = await fetch("/followup", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({question: currentQuestion, followup: q, report: currentData})
    });
    var text = await res.text();
    var data = JSON.parse(text);
    if (data.error) {
      answerDiv.innerHTML = "<span style='color:#ff6b6b'>" + data.error + "</span>";
    } else {
      answerDiv.innerHTML = marked.parse(data.answer);
    }
  } catch(e) {
    answerDiv.innerHTML = "<span style='color:#ff6b6b'>Something went wrong. Please try again.</span>";
  }
  btn.disabled = false;
  btn.textContent = "Ask";
}
</script>
</body>
</html>"""


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return HTML, 200, {"Content-Type": "text/html"}


@app.route("/research", methods=["POST"])
def research():
    question = request.json.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided."})
    try:
        report = run_research(question)
        return jsonify(report)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/followup", methods=["POST"])
def followup():
    data = request.json
    question = data.get("question", "").strip()
    followup_q = data.get("followup", "").strip()
    report = data.get("report", {})
    if not followup_q:
        return jsonify({"error": "No follow-up question provided."})
    try:
        context = f"Original research: {question}\n\nSummary: {report.get('summary', '')}\n\n"
        for s in report.get("sections", []):
            context += f"### {s['title']}\n{s['content']}\n\n"
        res = get_client().chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": f"""You are a research assistant. Using the research report below, answer the follow-up question with specific facts, numbers, and names. Use bullet points.

Report:
{context}

Follow-up: {followup_q}"""}],
            temperature=0.3,
        )
        return jsonify({"answer": res.choices[0].message.content.strip()})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    print("Open your browser and go to: http://127.0.0.1:5001")
    app.run(debug=True, port=5001)
