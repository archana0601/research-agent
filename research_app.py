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
        json={"api_key": TAVILY_API_KEY, "query": query, "max_results": 5},
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
        content = r.get("content", "")[:300]  # cap each result at 300 chars
        out += f"Title: {r['title']}\nSummary: {content}\nURL: {r['url']}\n\n"
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

    # Search top 3 angles only (keeps token count manageable)
    web_context = ""
    all_sources = []
    for q in angles[:3]:
        results, sources = web_search(q, region=region)
        if results:
            web_context += f"--- '{q}' ---\n{results}\n\n"
            all_sources.extend(sources)

    # Cap total web context to avoid token limit
    web_context = web_context[:6000]

    # Background knowledge fills gaps only
    knowledge = answer_from_knowledge(question, angles)
    knowledge = knowledge[:2000]

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
<title>ResearchAI — Deep Research</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Inter', -apple-system, sans-serif; background: #07070d; color: #e8e8f0; min-height: 100vh; display: flex; flex-direction: column; }

header { padding: 16px 32px; border-bottom: 1px solid rgba(255,255,255,0.05); display: flex; align-items: center; justify-content: space-between; flex-shrink: 0; background: rgba(7,7,13,0.9); backdrop-filter: blur(20px); position: sticky; top: 0; z-index: 50; }
.logo { font-size: 16px; font-weight: 800; background: linear-gradient(135deg, #a78bfa, #34d399); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -0.3px; }
.header-right { display: flex; align-items: center; gap: 16px; }
.header-back { font-size: 12px; color: #444; text-decoration: none; transition: color 0.2s; }
.header-back:hover { color: #888; }

.layout { display: flex; flex: 1; }

/* Sidebar */
.sidebar { width: 240px; flex-shrink: 0; border-right: 1px solid rgba(255,255,255,0.04); padding: 20px 12px; display: flex; flex-direction: column; gap: 4px; min-height: calc(100vh - 57px); background: rgba(255,255,255,0.01); }
.sidebar-title { font-size: 10px; text-transform: uppercase; letter-spacing: 2px; color: #333; font-weight: 700; padding: 0 10px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; }
.clear-all { font-size: 10px; color: #2a2a3a; cursor: pointer; text-transform: none; letter-spacing: 0; transition: color 0.2s; font-weight: 500; }
.clear-all:hover { color: #ff6b6b; }
.history-item { padding: 10px 10px; border-radius: 10px; cursor: pointer; border: 1px solid transparent; transition: all 0.2s; }
.history-item:hover { background: rgba(255,255,255,0.03); border-color: rgba(255,255,255,0.06); }
.history-item.active { background: rgba(124,111,255,0.08); border-color: rgba(124,111,255,0.2); }
.hi-question { font-size: 12px; color: #888; line-height: 1.4; margin-bottom: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-weight: 500; }
.history-item.active .hi-question { color: #ccc; }
.hi-meta { display: flex; justify-content: space-between; align-items: center; }
.hi-date { font-size: 10px; color: #2a2a3a; }
.hi-del { font-size: 10px; color: #222; cursor: pointer; transition: color 0.2s; padding: 2px 4px; }
.hi-del:hover { color: #ff6b6b; }
.no-history { font-size: 12px; color: #2a2a3a; padding: 8px; text-align: center; margin-top: 20px; }

/* Main */
.main { flex: 1; overflow-x: hidden; }
.hero { padding: 52px 48px 32px; max-width: 880px; margin: 0 auto; }
.hero h1 { font-size: 32px; font-weight: 900; letter-spacing: -1px; margin-bottom: 8px; }
.hero h1 span { background: linear-gradient(135deg, #a78bfa, #34d399); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hero p { color: #333; font-size: 14px; margin-bottom: 24px; font-weight: 400; }

.search-wrap { position: relative; display: flex; gap: 10px; }
.search-wrap input { flex: 1; padding: 16px 22px; font-size: 15px; border: 1px solid rgba(255,255,255,0.07); border-radius: 12px; background: rgba(255,255,255,0.04); color: #e8e8f0; outline: none; transition: all 0.2s; font-family: inherit; }
.search-wrap input::placeholder { color: #333; }
.search-wrap input:focus { border-color: rgba(124,111,255,0.4); background: rgba(124,111,255,0.05); box-shadow: 0 0 0 3px rgba(124,111,255,0.08); }
.btn { padding: 16px 28px; background: linear-gradient(135deg, #7c6fff, #3dd8cc); color: #07070d; border: none; border-radius: 12px; font-size: 14px; font-weight: 800; cursor: pointer; white-space: nowrap; transition: all 0.2s; font-family: inherit; box-shadow: 0 0 20px rgba(124,111,255,0.25); }
.btn:hover { transform: translateY(-1px); box-shadow: 0 0 30px rgba(124,111,255,0.4); }
.btn:disabled { opacity: 0.35; cursor: not-allowed; transform: none; box-shadow: none; }

.container { max-width: 880px; margin: 0 auto; padding: 0 48px 80px; }

.loader { display: none; padding: 48px 0; }
.progress-steps { display: flex; flex-direction: column; gap: 14px; }
.ps { display: flex; align-items: center; gap: 14px; font-size: 13px; color: #2a2a3a; transition: all 0.3s; font-weight: 500; }
.ps.active { color: #e8e8f0; }
.ps.done { color: #34d399; }
.ps-dot { width: 7px; height: 7px; border-radius: 50%; background: #1a1a28; flex-shrink: 0; transition: background 0.3s; }
.ps.active .ps-dot { background: #7c6fff; animation: blink 1s infinite; }
.ps.done .ps-dot { background: #34d399; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

.results { display: none; }
.label { font-size: 10px; text-transform: uppercase; letter-spacing: 2px; font-weight: 700; margin-bottom: 12px; }

.q-banner { background: linear-gradient(135deg, rgba(124,111,255,0.08), rgba(61,216,204,0.04)); border: 1px solid rgba(124,111,255,0.15); border-radius: 16px; padding: 22px 26px; margin-bottom: 20px; }
.q-banner .label { color: rgba(167,139,250,0.7); margin-bottom: 8px; }
.q-banner h2 { font-size: 18px; font-weight: 700; line-height: 1.4; letter-spacing: -0.3px; }

.summary { background: linear-gradient(135deg, rgba(61,216,204,0.06), rgba(61,216,204,0.02)); border: 1px solid rgba(61,216,204,0.12); border-radius: 16px; padding: 24px; margin-bottom: 20px; }
.summary .label { color: rgba(52,211,153,0.7); }
.summary p { font-size: 14px; line-height: 1.85; color: #888; }

.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; align-items: start; }

.stats-card { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); border-radius: 16px; padding: 22px; }
.stats-card .label { color: rgba(167,139,250,0.7); }
.stat-item { display: flex; align-items: center; gap: 14px; padding: 11px 0; border-bottom: 1px solid rgba(255,255,255,0.04); }
.stat-item:last-child { border-bottom: none; }
.stat-val { font-size: 20px; font-weight: 900; background: linear-gradient(135deg, #a78bfa, #34d399); -webkit-background-clip: text; -webkit-text-fill-color: transparent; min-width: 65px; letter-spacing: -0.5px; }
.stat-lbl { font-size: 12px; color: #444; line-height: 1.4; }

.chart-card { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); border-radius: 16px; padding: 22px; }
.chart-card .label { color: rgba(167,139,250,0.7); }

.sections-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }
.sec { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); border-radius: 16px; padding: 22px; transition: border-color 0.2s; }
.sec:hover { border-color: rgba(124,111,255,0.15); }
.sec h4 { font-size: 11px; font-weight: 700; color: rgba(167,139,250,0.8); text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 14px; }
.sec .body { font-size: 13px; color: #666; line-height: 1.85; }
.sec .body ul { padding-left: 16px; }
.sec .body li { margin-bottom: 7px; }
.sec .body strong { color: #ccc; }
.sec .body p { margin-bottom: 8px; }

.table-card { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); border-radius: 16px; padding: 22px; margin-bottom: 20px; overflow-x: auto; }
.table-card .label { color: rgba(167,139,250,0.7); }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { background: rgba(124,111,255,0.08); color: rgba(167,139,250,0.8); text-align: left; padding: 11px 14px; font-size: 10px; text-transform: uppercase; letter-spacing: 1px; font-weight: 700; }
td { padding: 10px 14px; border-bottom: 1px solid rgba(255,255,255,0.04); color: #666; }
tr:last-child td { border-bottom: none; }
tr:hover td { background: rgba(124,111,255,0.04); color: #aaa; }

.takeaways { background: linear-gradient(135deg, rgba(52,211,153,0.05), rgba(52,211,153,0.02)); border: 1px solid rgba(52,211,153,0.1); border-radius: 16px; padding: 24px; margin-bottom: 20px; }
.takeaways .label { color: rgba(52,211,153,0.7); }
.tw { display: flex; gap: 14px; margin-bottom: 14px; align-items: flex-start; }
.tw:last-child { margin-bottom: 0; }
.tw-n { background: linear-gradient(135deg, #7c6fff, #3dd8cc); color: #07070d; width: 22px; height: 22px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: 900; flex-shrink: 0; margin-top: 2px; }
.tw p { font-size: 13px; color: #666; line-height: 1.75; }

.queries-row { display: flex; flex-wrap: wrap; gap: 7px; margin-bottom: 20px; }
.qtag { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); color: #333; padding: 4px 12px; border-radius: 100px; font-size: 11px; font-weight: 500; }

.sources { border-top: 1px solid rgba(255,255,255,0.04); padding-top: 20px; }
.sources .label { color: #2a2a3a; }
.src { display: flex; align-items: center; gap: 10px; padding: 7px 0; border-bottom: 1px solid rgba(255,255,255,0.03); }
.src:last-child { border-bottom: none; }
.src-dot { width: 4px; height: 4px; border-radius: 50%; background: #2a2a3a; flex-shrink: 0; }
.src a { font-size: 12px; color: #333; text-decoration: none; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; display: block; transition: color 0.2s; }
.src a:hover { color: #a78bfa; }

.export-bar { display: flex; justify-content: flex-end; gap: 8px; margin-bottom: 18px; }
.export-btn { padding: 9px 18px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); color: #555; border-radius: 9px; font-size: 12px; font-weight: 600; cursor: pointer; transition: all 0.2s; font-family: inherit; }
.export-btn:hover { border-color: rgba(167,139,250,0.4); color: #a78bfa; }

.followup-box { background: rgba(124,111,255,0.04); border: 1px solid rgba(124,111,255,0.12); border-radius: 16px; padding: 22px; margin-top: 20px; }
.followup-box .label { color: rgba(167,139,250,0.7); margin-bottom: 14px; }
.followup-row { display: flex; gap: 10px; }
.followup-row input { flex: 1; padding: 12px 16px; font-size: 13px; border: 1px solid rgba(255,255,255,0.07); border-radius: 10px; background: rgba(255,255,255,0.03); color: #e8e8f0; outline: none; font-family: inherit; }
.followup-row input:focus { border-color: rgba(124,111,255,0.4); }
.followup-row button { padding: 12px 20px; background: rgba(124,111,255,0.1); border: 1px solid rgba(124,111,255,0.25); color: #a78bfa; border-radius: 10px; font-size: 12px; font-weight: 700; cursor: pointer; transition: all 0.2s; white-space: nowrap; font-family: inherit; }
.followup-row button:hover { background: rgba(124,111,255,0.18); }
.followup-row button:disabled { opacity: 0.4; cursor: not-allowed; }
.followup-answer { margin-top: 16px; padding-top: 16px; border-top: 1px solid rgba(255,255,255,0.05); font-size: 13px; color: #666; line-height: 1.85; display: none; }
.followup-answer ul { padding-left: 18px; }
.followup-answer li { margin-bottom: 6px; }
.followup-answer strong { color: #ccc; }

@media print {
  body { background: white; color: #111; }
  header, .hero, .loader, .export-bar, .queries-row, .search-wrap { display: none !important; }
  .sidebar { display: none !important; }
  .results { display: block !important; }
  .q-banner, .summary, .stats-card, .sec, .table-card, .takeaways, .chart-card { background: #fafafa !important; border: 1px solid #ddd !important; -webkit-print-color-adjust: exact; }
  .q-banner h2, .sec h4 { color: #111; }
  .summary p, .tw p, .sec .body, .stat-lbl, td { color: #333; }
  .stat-val { color: #7c6fff !important; -webkit-text-fill-color: #7c6fff !important; }
  .tw-n { background: #7c6fff !important; color: white !important; -webkit-print-color-adjust: exact; }
  th { background: #ebebff !important; color: #333 !important; -webkit-print-color-adjust: exact; }
  canvas { max-width: 100%; }
}
</style>
</head>
<body>
<header>
  <div class="logo">ResearchAI</div>
  <div class="header-right">
    <a href="/" class="header-back">&larr; Home</a>
  </div>
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


# ─── Landing Page ─────────────────────────────────────────────────────────────

LANDING = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ResearchAI — Deep Research, Beautifully Structured</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
* { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }
body { font-family: 'Inter', -apple-system, sans-serif; background: #07070d; color: #e8e8f0; overflow-x: hidden; }

/* Nav */
nav { display: flex; justify-content: space-between; align-items: center; padding: 20px 60px; position: fixed; top: 0; left: 0; right: 0; z-index: 100; background: rgba(7,7,13,0.85); backdrop-filter: blur(20px); border-bottom: 1px solid rgba(255,255,255,0.04); }
.nav-logo { font-size: 17px; font-weight: 800; background: linear-gradient(135deg, #a78bfa, #34d399); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -0.3px; }
.nav-links { display: flex; gap: 32px; align-items: center; }
.nav-links a { font-size: 13px; color: #666; text-decoration: none; font-weight: 500; transition: color 0.2s; }
.nav-links a:hover { color: #ccc; }
.nav-cta { padding: 9px 22px; background: linear-gradient(135deg, #7c6fff, #3dd8cc); color: #07070d; border: none; border-radius: 8px; font-size: 13px; font-weight: 700; cursor: pointer; text-decoration: none; transition: opacity 0.2s; }
.nav-cta:hover { opacity: 0.88; }

/* Hero */
.hero-wrap { position: relative; padding: 160px 24px 100px; text-align: center; overflow: hidden; }
.hero-glow { position: absolute; top: 0; left: 50%; transform: translateX(-50%); width: 800px; height: 500px; background: radial-gradient(ellipse at center top, rgba(124,111,255,0.18) 0%, rgba(61,216,204,0.08) 50%, transparent 70%); pointer-events: none; }
.hero-inner { max-width: 820px; margin: 0 auto; position: relative; }
.hero-badge { display: inline-flex; align-items: center; gap: 8px; background: rgba(124,111,255,0.1); border: 1px solid rgba(124,111,255,0.25); color: #a78bfa; padding: 6px 16px; border-radius: 100px; font-size: 12px; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 32px; }
.badge-dot { width: 6px; height: 6px; background: #7c6fff; border-radius: 50%; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.5;transform:scale(1.3)} }
.hero h1 { font-size: 68px; font-weight: 900; line-height: 1.05; letter-spacing: -2px; margin-bottom: 24px; }
.hero h1 .grad { background: linear-gradient(135deg, #a78bfa 0%, #34d399 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hero p { font-size: 18px; color: #555; line-height: 1.75; margin-bottom: 44px; max-width: 540px; margin-left: auto; margin-right: auto; font-weight: 400; }
.hero-btns { display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; margin-bottom: 64px; }
.btn-primary { padding: 15px 32px; background: linear-gradient(135deg, #7c6fff, #3dd8cc); color: #07070d; border: none; border-radius: 10px; font-size: 15px; font-weight: 700; cursor: pointer; text-decoration: none; transition: transform 0.2s, box-shadow 0.2s; box-shadow: 0 0 30px rgba(124,111,255,0.3); }
.btn-primary:hover { transform: translateY(-1px); box-shadow: 0 0 40px rgba(124,111,255,0.45); }
.btn-ghost { padding: 15px 32px; background: transparent; color: #888; border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; font-size: 15px; font-weight: 600; cursor: pointer; text-decoration: none; transition: all 0.2s; }
.btn-ghost:hover { border-color: rgba(124,111,255,0.4); color: #ccc; }

/* Stats bar */
.stats-bar { display: flex; justify-content: center; gap: 0; border: 1px solid rgba(255,255,255,0.06); border-radius: 14px; overflow: hidden; max-width: 600px; margin: 0 auto; background: rgba(255,255,255,0.02); }
.stat-item { flex: 1; padding: 20px 24px; text-align: center; border-right: 1px solid rgba(255,255,255,0.06); }
.stat-item:last-child { border-right: none; }
.stat-val { font-size: 26px; font-weight: 900; background: linear-gradient(135deg, #a78bfa, #34d399); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.stat-label { font-size: 11px; color: #444; margin-top: 4px; font-weight: 500; }

/* Section divider */
.divider { border: none; border-top: 1px solid rgba(255,255,255,0.04); margin: 0; }

/* Features */
.features-wrap { padding: 100px 60px; max-width: 1140px; margin: 0 auto; }
.section-label { text-align: center; font-size: 11px; text-transform: uppercase; letter-spacing: 3px; color: rgba(124,111,255,0.7); font-weight: 700; margin-bottom: 14px; }
.section-title { text-align: center; font-size: 40px; font-weight: 900; letter-spacing: -1px; margin-bottom: 16px; }
.section-title .grad { background: linear-gradient(135deg, #a78bfa, #34d399); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.section-sub { text-align: center; font-size: 15px; color: #444; margin-bottom: 60px; max-width: 480px; margin-left: auto; margin-right: auto; }
.features-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }
.feat { background: linear-gradient(135deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)); border: 1px solid rgba(255,255,255,0.06); border-radius: 18px; padding: 32px; transition: all 0.3s; position: relative; overflow: hidden; }
.feat::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px; background: linear-gradient(90deg, transparent, rgba(124,111,255,0.4), transparent); opacity: 0; transition: opacity 0.3s; }
.feat:hover { border-color: rgba(124,111,255,0.2); transform: translateY(-2px); }
.feat:hover::before { opacity: 1; }
.feat-icon { width: 44px; height: 44px; background: linear-gradient(135deg, rgba(124,111,255,0.15), rgba(61,216,204,0.1)); border: 1px solid rgba(124,111,255,0.2); border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 20px; margin-bottom: 18px; }
.feat h3 { font-size: 15px; font-weight: 700; margin-bottom: 10px; color: #e8e8f0; }
.feat p { font-size: 13px; color: #444; line-height: 1.75; }

/* How it works */
.how-wrap { padding: 100px 60px; max-width: 860px; margin: 0 auto; }
.steps { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.step { background: linear-gradient(135deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)); border: 1px solid rgba(255,255,255,0.06); border-radius: 18px; padding: 32px; }
.step-num { width: 36px; height: 36px; background: linear-gradient(135deg, #7c6fff, #3dd8cc); border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 14px; font-weight: 900; color: #07070d; margin-bottom: 16px; }
.step h3 { font-size: 15px; font-weight: 700; margin-bottom: 8px; }
.step p { font-size: 13px; color: #444; line-height: 1.7; }

/* CTA */
.cta-wrap { padding: 100px 24px; text-align: center; position: relative; overflow: hidden; }
.cta-glow { position: absolute; bottom: 0; left: 50%; transform: translateX(-50%); width: 600px; height: 300px; background: radial-gradient(ellipse at center bottom, rgba(61,216,204,0.12) 0%, transparent 70%); pointer-events: none; }
.cta-wrap h2 { font-size: 52px; font-weight: 900; letter-spacing: -1.5px; margin-bottom: 16px; }
.cta-wrap h2 .grad { background: linear-gradient(135deg, #a78bfa, #34d399); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.cta-wrap p { color: #444; font-size: 16px; margin-bottom: 36px; }
.cta-note { margin-top: 16px; font-size: 12px; color: #333; }

/* Footer */
footer { border-top: 1px solid rgba(255,255,255,0.04); padding: 28px 60px; display: flex; justify-content: space-between; align-items: center; }
.footer-logo { font-size: 15px; font-weight: 800; background: linear-gradient(135deg, #a78bfa, #34d399); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
footer p { font-size: 12px; color: #2a2a3a; }

@media (max-width: 768px) {
  nav { padding: 16px 20px; }
  .nav-links { display: none; }
  .hero-wrap { padding: 120px 20px 80px; }
  .hero h1 { font-size: 38px; letter-spacing: -1px; }
  .hero p { font-size: 15px; }
  .features-wrap, .how-wrap { padding: 60px 20px; }
  .features-grid { grid-template-columns: 1fr; }
  .steps { grid-template-columns: 1fr; }
  .stats-bar { flex-direction: column; }
  .stat-item { border-right: none; border-bottom: 1px solid rgba(255,255,255,0.06); }
  .stat-item:last-child { border-bottom: none; }
  footer { flex-direction: column; gap: 8px; padding: 24px 20px; }
}
</style>
</head>
<body>

<nav>
  <div class="nav-logo">ResearchAI</div>
  <div class="nav-links">
    <a href="#features">Features</a>
    <a href="#how">How it works</a>
    <a href="/app" class="nav-cta">Launch App &rarr;</a>
  </div>
</nav>

<section class="hero-wrap">
  <div class="hero-glow"></div>
  <div class="hero-inner">
    <div class="hero-badge"><span class="badge-dot"></span> Powered by Groq &amp; Tavily</div>
    <h1>Research anything.<br><span class="grad">Know everything.</span></h1>
    <p>Ask any question and get a full structured research report — with charts, comparisons, key stats, and actionable takeaways — in under 30 seconds.</p>
    <div class="hero-btns">
      <a href="/app" class="btn-primary">Start Researching Free &rarr;</a>
      <a href="#features" class="btn-ghost">See how it works</a>
    </div>
    <div class="stats-bar">
      <div class="stat-item"><div class="stat-val">&lt;30s</div><div class="stat-label">Per report</div></div>
      <div class="stat-item"><div class="stat-val">5x</div><div class="stat-label">Searches per query</div></div>
      <div class="stat-item"><div class="stat-val">100%</div><div class="stat-label">Free to use</div></div>
    </div>
  </div>
</section>

<hr class="divider">

<section class="features-wrap" id="features">
  <div class="section-label">Features</div>
  <h2 class="section-title">Everything a <span class="grad">great report</span> needs</h2>
  <p class="section-sub">Built for people who need real answers, not a wall of links.</p>
  <div class="features-grid">
    <div class="feat">
      <div class="feat-icon">&#128269;</div>
      <h3>Live Web Search</h3>
      <p>Automatically generates targeted search queries and fetches fresh results via Tavily — the search engine built for AI.</p>
    </div>
    <div class="feat">
      <div class="feat-icon">&#129504;</div>
      <h3>2-Pass AI Analysis</h3>
      <p>First extracts raw facts and figures, then synthesizes them into a coherent report. No hallucinations, no filler.</p>
    </div>
    <div class="feat">
      <div class="feat-icon">&#128202;</div>
      <h3>Charts &amp; Key Stats</h3>
      <p>Every report includes a visual bar chart and 4 highlighted key statistics pulled directly from the research.</p>
    </div>
    <div class="feat">
      <div class="feat-icon">&#128203;</div>
      <h3>Comparison Tables</h3>
      <p>Side-by-side breakdowns of products, approaches, or providers — structured so you can decide fast.</p>
    </div>
    <div class="feat">
      <div class="feat-icon">&#128172;</div>
      <h3>Follow-up Questions</h3>
      <p>Ask anything about a report and get instant, context-aware answers — like having a research assistant on call.</p>
    </div>
    <div class="feat">
      <div class="feat-icon">&#128196;</div>
      <h3>PDF Export</h3>
      <p>Download any report as a clean, formatted PDF — ready to share, present, or archive immediately.</p>
    </div>
  </div>
</section>

<hr class="divider">

<section class="how-wrap" id="how">
  <div class="section-label">How It Works</div>
  <h2 class="section-title">From question to report <span class="grad">in 4 steps</span></h2>
  <p class="section-sub">No prompting skills needed. Just ask.</p>
  <div class="steps">
    <div class="step">
      <div class="step-num">1</div>
      <h3>Type your question</h3>
      <p>Ask anything — product research, market analysis, scientific topics, travel, business strategy, or current events.</p>
    </div>
    <div class="step">
      <div class="step-num">2</div>
      <h3>AI plans the research</h3>
      <p>Groq breaks your question into 4 focused sub-questions, then runs live searches via Tavily for each one.</p>
    </div>
    <div class="step">
      <div class="step-num">3</div>
      <h3>Facts are extracted</h3>
      <p>The AI reads all results and pulls out specific numbers, names, prices, comparisons, and expert opinions.</p>
    </div>
    <div class="step">
      <div class="step-num">4</div>
      <h3>Your report is ready</h3>
      <p>A full structured report: executive summary, key stats, 4 deep sections, comparison table, and 5 takeaways.</p>
    </div>
  </div>
</section>

<hr class="divider">

<section class="cta-wrap">
  <div class="cta-glow"></div>
  <h2>Start researching<br><span class="grad">smarter today.</span></h2>
  <p>Free to use. No sign-up. No limits.</p>
  <a href="/app" class="btn-primary">Launch ResearchAI &rarr;</a>
  <div class="cta-note">Powered by Groq (LLaMA 3.3 70B) &amp; Tavily Search</div>
</section>

<footer>
  <div class="footer-logo">ResearchAI</div>
  <p>&copy; 2026 ResearchAI &mdash; Built with Groq &amp; Tavily</p>
</footer>

</body>
</html>"""


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return LANDING, 200, {"Content-Type": "text/html"}


@app.route("/app")
def app_page():
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
