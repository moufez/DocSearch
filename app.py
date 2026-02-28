"""
DocSearch -- single-file semantic search app.
Run:  python app.py
Open: http://localhost:8000
No static/ folder needed -- the UI is embedded directly in this file.
"""

import hashlib, io, json, logging, re, time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

BASE_DIR  = Path(__file__).parent
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

MODEL_NAME = "all-MiniLM-L6-v2"
_model = None

def get_model():
    global _model
    if _model is None:
        log.info("Loading model '%s' ...", MODEL_NAME)
        _model = SentenceTransformer(MODEL_NAME)
        log.info("Model ready. Dim: %d", _model.get_sentence_embedding_dimension())
    return _model

state = {"fragments": [], "vectors": None, "index": None}

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>DocSearch — Semantic Document Explorer</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=Instrument+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap" rel="stylesheet" />
<style>
/* ─── Reset & Tokens ────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --ink:      #0e0e0e;
  --paper:    #f5f2eb;
  --cream:    #ece8df;
  --muted:    #8a8478;
  --accent:   #c84b31;
  --accent2:  #2a5c8a;
  --gold:     #b89a4e;
  --success:  #2d6a4f;
  --surface:  #ffffff;
  --border:   #d6d0c4;
  --shadow:   0 2px 12px rgba(14,14,14,.08), 0 1px 3px rgba(14,14,14,.06);
  --shadow-lg:0 8px 40px rgba(14,14,14,.12), 0 2px 8px rgba(14,14,14,.06);
  --r:        4px;
  --transition: 200ms cubic-bezier(.4,0,.2,1);
}

html { scroll-behavior: smooth; }

body {
  font-family: 'Instrument Sans', sans-serif;
  background: var(--paper);
  color: var(--ink);
  min-height: 100vh;
  line-height: 1.6;
  overflow-x: hidden;
}

/* ─── Grain overlay ─────────────────────────────────────────────── */
body::before {
  content: '';
  position: fixed;
  inset: 0;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.04'/%3E%3C/svg%3E");
  pointer-events: none;
  z-index: 9999;
  opacity: .5;
}

/* ─── Header ────────────────────────────────────────────────────── */
header {
  background: var(--ink);
  color: var(--paper);
  padding: 0 2rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 64px;
  position: sticky;
  top: 0;
  z-index: 100;
  border-bottom: 2px solid var(--accent);
}

.logo {
  display: flex;
  align-items: baseline;
  gap: .5rem;
}

.logo-word {
  font-family: 'DM Serif Display', serif;
  font-size: 1.5rem;
  letter-spacing: -.02em;
  color: var(--paper);
}

.logo-dot { color: var(--accent); }

.logo-tag {
  font-family: 'DM Mono', monospace;
  font-size: .65rem;
  color: var(--muted);
  letter-spacing: .1em;
  text-transform: uppercase;
  margin-left: .25rem;
}

.header-status {
  display: flex;
  align-items: center;
  gap: .5rem;
  font-size: .8rem;
  color: var(--muted);
  font-family: 'DM Mono', monospace;
}

.status-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #555;
  transition: background var(--transition);
}
.status-dot.ready { background: var(--success); box-shadow: 0 0 6px rgba(45,106,79,.6); }

/* ─── Layout ────────────────────────────────────────────────────── */
main {
  max-width: 900px;
  margin: 0 auto;
  padding: 3rem 1.5rem 6rem;
}

/* ─── Hero ──────────────────────────────────────────────────────── */
.hero {
  text-align: center;
  padding: 3rem 0 2.5rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 2.5rem;
}

.hero h1 {
  font-family: 'DM Serif Display', serif;
  font-size: clamp(2.2rem, 6vw, 3.4rem);
  line-height: 1.1;
  letter-spacing: -.03em;
  margin-bottom: .75rem;
}

.hero h1 em {
  font-style: italic;
  color: var(--accent);
}

.hero p {
  color: var(--muted);
  font-size: 1rem;
  max-width: 480px;
  margin: 0 auto;
}

/* ─── Upload zone ───────────────────────────────────────────────── */
.upload-zone {
  border: 2px dashed var(--border);
  border-radius: 8px;
  padding: 2.5rem 2rem;
  text-align: center;
  cursor: pointer;
  background: var(--surface);
  transition: border-color var(--transition), background var(--transition), transform var(--transition);
  position: relative;
  overflow: hidden;
  margin-bottom: 1.5rem;
}

.upload-zone::before {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, transparent 60%, rgba(200,75,49,.04));
  pointer-events: none;
}

.upload-zone:hover, .upload-zone.drag-over {
  border-color: var(--accent);
  background: #fdf7f5;
  transform: translateY(-1px);
}

.upload-zone.drag-over { border-style: solid; }

.upload-icon {
  font-size: 2.5rem;
  margin-bottom: .75rem;
  display: block;
  transition: transform var(--transition);
}
.upload-zone:hover .upload-icon { transform: translateY(-4px); }

.upload-zone h3 {
  font-family: 'DM Serif Display', serif;
  font-size: 1.25rem;
  margin-bottom: .4rem;
}

.upload-zone p {
  color: var(--muted);
  font-size: .875rem;
}

.upload-zone input[type="file"] {
  position: absolute;
  inset: 0;
  opacity: 0;
  cursor: pointer;
  width: 100%;
  height: 100%;
}

/* ─── File pills ────────────────────────────────────────────────── */
.file-list {
  display: flex;
  flex-wrap: wrap;
  gap: .5rem;
  margin-bottom: 1.5rem;
  min-height: 0;
}

.file-pill {
  display: inline-flex;
  align-items: center;
  gap: .4rem;
  background: var(--cream);
  border: 1px solid var(--border);
  border-radius: 100px;
  padding: .3rem .8rem .3rem .6rem;
  font-size: .78rem;
  font-family: 'DM Mono', monospace;
  color: var(--ink);
  animation: pill-in .25s ease;
}

.file-pill.loading { color: var(--muted); }
.file-pill.done { border-color: var(--success); background: #f0f9f4; }
.file-pill.error { border-color: var(--accent); background: #fdf5f4; color: var(--accent); }

.pill-icon { font-size: .9rem; }

@keyframes pill-in {
  from { opacity: 0; transform: scale(.8) translateY(4px); }
  to   { opacity: 1; transform: scale(1) translateY(0); }
}

/* ─── Search bar ────────────────────────────────────────────────── */
.search-section {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1.5rem;
  box-shadow: var(--shadow);
  margin-bottom: 2rem;
}

.search-label {
  display: block;
  font-size: .7rem;
  font-family: 'DM Mono', monospace;
  letter-spacing: .12em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: .6rem;
}

.search-row {
  display: flex;
  gap: .75rem;
}

.search-input {
  flex: 1;
  border: 1.5px solid var(--border);
  border-radius: var(--r);
  padding: .75rem 1rem;
  font-family: 'Instrument Sans', sans-serif;
  font-size: 1rem;
  color: var(--ink);
  background: var(--paper);
  outline: none;
  transition: border-color var(--transition), box-shadow var(--transition);
}

.search-input::placeholder { color: var(--muted); }
.search-input:focus {
  border-color: var(--accent2);
  box-shadow: 0 0 0 3px rgba(42,92,138,.1);
}

.btn {
  display: inline-flex;
  align-items: center;
  gap: .4rem;
  padding: .75rem 1.4rem;
  border: none;
  border-radius: var(--r);
  cursor: pointer;
  font-family: 'Instrument Sans', sans-serif;
  font-size: .9rem;
  font-weight: 600;
  transition: all var(--transition);
  white-space: nowrap;
}

.btn-primary {
  background: var(--ink);
  color: var(--paper);
}
.btn-primary:hover:not(:disabled) {
  background: #2a2a2a;
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(14,14,14,.25);
}
.btn-primary:disabled {
  opacity: .45;
  cursor: not-allowed;
}

.btn-ghost {
  background: transparent;
  color: var(--muted);
  border: 1px solid var(--border);
  padding: .4rem .8rem;
  font-size: .8rem;
}
.btn-ghost:hover { background: var(--cream); color: var(--ink); }

/* ─── Progress bar ──────────────────────────────────────────────── */
.progress-bar {
  height: 2px;
  background: var(--border);
  border-radius: 2px;
  margin-top: .75rem;
  overflow: hidden;
  opacity: 0;
  transition: opacity .2s;
}
.progress-bar.visible { opacity: 1; }
.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--accent), var(--gold));
  border-radius: 2px;
  transition: width .4s ease;
  width: 0%;
}

/* ─── Results ───────────────────────────────────────────────────── */
.results-header {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  margin-bottom: 1rem;
}

.results-title {
  font-family: 'DM Serif Display', serif;
  font-size: 1.3rem;
}

.results-meta {
  font-size: .75rem;
  font-family: 'DM Mono', monospace;
  color: var(--muted);
}

.result-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1.4rem 1.5rem;
  margin-bottom: .875rem;
  box-shadow: var(--shadow);
  position: relative;
  overflow: hidden;
  animation: card-in .35s cubic-bezier(.2,.8,.4,1) both;
  transition: box-shadow var(--transition), transform var(--transition);
}
.result-card:hover { box-shadow: var(--shadow-lg); transform: translateY(-1px); }

.result-card:nth-child(1) { animation-delay: 0ms; }
.result-card:nth-child(2) { animation-delay: 60ms; }
.result-card:nth-child(3) { animation-delay: 120ms; }

@keyframes card-in {
  from { opacity: 0; transform: translateY(16px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* Rank stripe */
.result-card::before {
  content: '';
  position: absolute;
  left: 0; top: 0; bottom: 0;
  width: 4px;
  border-radius: 8px 0 0 8px;
}
.result-card:nth-child(1)::before { background: var(--accent); }
.result-card:nth-child(2)::before { background: var(--gold); }
.result-card:nth-child(3)::before { background: var(--accent2); }

.result-meta-row {
  display: flex;
  align-items: center;
  gap: .75rem;
  margin-bottom: .75rem;
  flex-wrap: wrap;
}

.rank-badge {
  font-family: 'DM Mono', monospace;
  font-size: .65rem;
  font-weight: 500;
  letter-spacing: .1em;
  text-transform: uppercase;
  padding: .2rem .5rem;
  border-radius: 3px;
  background: var(--cream);
  color: var(--muted);
}

.score-badge {
  display: flex;
  align-items: center;
  gap: .3rem;
  font-family: 'DM Mono', monospace;
  font-size: .8rem;
  font-weight: 500;
}

.score-bar-wrap {
  width: 60px;
  height: 4px;
  background: var(--cream);
  border-radius: 2px;
  overflow: hidden;
}
.score-bar-fill {
  height: 100%;
  border-radius: 2px;
  background: linear-gradient(90deg, var(--accent2), var(--accent));
}

.source-tag {
  font-family: 'DM Mono', monospace;
  font-size: .7rem;
  color: var(--muted);
  margin-left: auto;
  max-width: 200px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.result-text {
  font-size: .925rem;
  line-height: 1.75;
  color: #2a2a2a;
}

/* ─── Empty / Error states ──────────────────────────────────────── */
.state-box {
  text-align: center;
  padding: 3rem 2rem;
  color: var(--muted);
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
}

.state-box .state-icon { font-size: 2.5rem; margin-bottom: .75rem; display: block; }
.state-box h3 { font-family: 'DM Serif Display', serif; font-size: 1.1rem; color: var(--ink); margin-bottom: .35rem; }
.state-box p  { font-size: .875rem; }

/* ─── Stats strip ───────────────────────────────────────────────── */
.stats-strip {
  display: flex;
  gap: 1.5rem;
  padding: .875rem 1.25rem;
  background: var(--cream);
  border: 1px solid var(--border);
  border-radius: var(--r);
  margin-bottom: 1.5rem;
  font-family: 'DM Mono', monospace;
  font-size: .75rem;
  color: var(--muted);
  flex-wrap: wrap;
}

.stat { display: flex; align-items: center; gap: .35rem; }
.stat strong { color: var(--ink); font-weight: 500; }

/* ─── Toast ─────────────────────────────────────────────────────── */
#toast {
  position: fixed;
  bottom: 1.5rem;
  right: 1.5rem;
  z-index: 1000;
  display: flex;
  flex-direction: column;
  gap: .5rem;
  pointer-events: none;
}

.toast-item {
  display: flex;
  align-items: center;
  gap: .5rem;
  background: var(--ink);
  color: var(--paper);
  padding: .65rem 1rem;
  border-radius: var(--r);
  font-size: .82rem;
  font-family: 'DM Mono', monospace;
  box-shadow: var(--shadow-lg);
  animation: toast-in .3s ease;
  pointer-events: auto;
}
.toast-item.success { background: var(--success); }
.toast-item.error   { background: var(--accent); }

@keyframes toast-in {
  from { opacity: 0; transform: translateX(20px) scale(.95); }
  to   { opacity: 1; transform: translateX(0) scale(1); }
}

/* ─── Spinner ───────────────────────────────────────────────────── */
.spinner {
  width: 16px;
  height: 16px;
  border: 2px solid currentColor;
  border-top-color: transparent;
  border-radius: 50%;
  animation: spin .7s linear infinite;
  display: none;
}
.spinner.visible { display: block; }
@keyframes spin { to { transform: rotate(360deg); } }

/* ─── Responsive ────────────────────────────────────────────────── */
@media (max-width: 600px) {
  .search-row { flex-direction: column; }
  .stats-strip { gap: .75rem; }
}
</style>
</head>
<body>

<header>
  <div class="logo">
    <span class="logo-word">Doc<span class="logo-dot">.</span>Search</span>
    <span class="logo-tag">Semantic Explorer</span>
  </div>
  <div class="header-status">
    <div class="status-dot" id="statusDot"></div>
    <span id="statusLabel">Connecting…</span>
  </div>
</header>

<main>

  <div class="hero">
    <h1>Ask anything about your <em>documents</em></h1>
    <p>Upload PDF, DOCX, TXT, or JSON files — then search with plain English. Powered by semantic embeddings.</p>
  </div>

  <!-- Upload -->
  <div class="upload-zone" id="dropZone">
    <input type="file" id="fileInput" accept=".pdf,.docx,.doc,.txt,.json" multiple />
    <span class="upload-icon">📄</span>
    <h3>Drop files here or click to upload</h3>
    <p>PDF, DOCX, TXT, JSON — up to 50 MB each</p>
  </div>

  <div class="file-list" id="fileList"></div>

  <!-- Stats strip -->
  <div class="stats-strip" id="statsStrip" style="display:none">
    <div class="stat">🗂 <strong id="statFragments">0</strong> fragments indexed</div>
    <div class="stat">📁 <strong id="statSources">0</strong> documents</div>
    <div class="stat">🤖 model: <strong>all-MiniLM-L6-v2</strong></div>
    <div style="margin-left:auto">
      <button class="btn btn-ghost" onclick="resetIndex()">✕ Clear all</button>
    </div>
  </div>

  <!-- Search -->
  <div class="search-section">
    <span class="search-label">Natural Language Query</span>
    <div class="search-row">
      <input
        class="search-input"
        id="queryInput"
        type="text"
        placeholder="e.g. What are the main safety procedures?"
        autocomplete="off"
        spellcheck="false"
      />
      <button class="btn btn-primary" id="searchBtn" onclick="doSearch()" disabled>
        <div class="spinner" id="searchSpinner"></div>
        <span id="searchBtnText">Search</span>
      </button>
    </div>
    <div class="progress-bar" id="progressBar">
      <div class="progress-fill" id="progressFill"></div>
    </div>
  </div>

  <!-- Results -->
  <div id="resultsArea"></div>

</main>

<!-- Toast container -->
<div id="toast"></div>

<script>
// ── Config ──────────────────────────────────────────────────────────────────
// Auto-detect: if opened via file://, point explicitly to localhost:8000
// If opened via the server (http://localhost:8000), use same-origin empty string
const API = window.location.protocol === 'file:'
  ? 'http://localhost:8000'
  : '';

// ── State ───────────────────────────────────────────────────────────────────
let totalFragments = 0;
let totalSources   = 0;

// ── Boot ────────────────────────────────────────────────────────────────────
async function init() {
  try {
    const s = await fetchJSON('/status');
    updateStats(s.total_fragments, s.sources.length);
    setStatus('ready', `${s.total_fragments} fragments`);
    if (s.total_fragments > 0) enableSearch();
  } catch (err) {
    setStatus('offline', 'Server offline');
    document.getElementById('resultsArea').innerHTML = `
      <div class="state-box">
        <span class="state-icon">🔌</span>
        <h3>Cannot connect to the backend</h3>
        <p>Make sure the server is running:</p>
        <pre style="margin:.75rem auto;background:var(--cream);border:1px solid var(--border);border-radius:4px;padding:.75rem 1rem;text-align:left;font-family:'DM Mono',monospace;font-size:.8rem;display:inline-block">python app.py</pre>
        <p style="margin-top:.5rem">Then open <strong>http://localhost:8000</strong> in your browser.<br/>
        Do <em>not</em> open index.html directly as a file.</p>
      </div>`;
    toast('Cannot reach backend — run: python app.py', 'error');
  }
}

// ── Upload ───────────────────────────────────────────────────────────────────
const dropZone  = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const fileList  = document.getElementById('fileList');

dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  handleFiles([...e.dataTransfer.files]);
});
fileInput.addEventListener('change', () => {
  handleFiles([...fileInput.files]);
  fileInput.value = '';
});

async function handleFiles(files) {
  for (const file of files) {
    await uploadFile(file);
  }
}

function extIcon(name) {
  const ext = name.split('.').pop().toLowerCase();
  return { pdf:'📕', docx:'📘', doc:'📘', txt:'📄', json:'📋' }[ext] || '📄';
}

async function uploadFile(file) {
  const pill = document.createElement('div');
  pill.className = 'file-pill loading';
  pill.innerHTML = `<span class="pill-icon">⏳</span> <span>${file.name}</span>`;
  fileList.appendChild(pill);

  const form = new FormData();
  form.append('file', file);

  try {
    const res = await fetch(`${API}/upload`, { method: 'POST', body: form });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Upload failed');
    }
    const data = await res.json();
    pill.className = 'file-pill done';
    pill.innerHTML = `<span class="pill-icon">${extIcon(file.name)}</span> <span>${file.name}</span> <span style="color:var(--muted)">+${data.chunks_added} chunks</span>`;
    updateStats(data.total_fragments, null);
    refreshStats();
    enableSearch();
    toast(`✓ ${file.name} indexed (${data.chunks_added} fragments)`, 'success');
  } catch (err) {
    pill.className = 'file-pill error';
    pill.innerHTML = `<span class="pill-icon">✗</span> <span>${file.name}</span> <span>${err.message}</span>`;
    toast(`Error: ${err.message}`, 'error');
  }
}

// ── Search ───────────────────────────────────────────────────────────────────
const queryInput = document.getElementById('queryInput');
const searchBtn  = document.getElementById('searchBtn');

queryInput.addEventListener('keydown', e => { if (e.key === 'Enter') doSearch(); });

function enableSearch() {
  searchBtn.disabled = false;
}

function setSearchLoading(loading) {
  const spinner = document.getElementById('searchSpinner');
  const label   = document.getElementById('searchBtnText');
  const bar     = document.getElementById('progressBar');
  const fill    = document.getElementById('progressFill');

  searchBtn.disabled = loading;
  spinner.classList.toggle('visible', loading);
  label.textContent = loading ? 'Searching…' : 'Search';
  bar.classList.toggle('visible', loading);

  if (loading) {
    fill.style.width = '0%';
    setTimeout(() => { fill.style.width = '60%'; }, 50);
    setTimeout(() => { fill.style.width = '85%'; }, 400);
  } else {
    fill.style.width = '100%';
    setTimeout(() => { bar.classList.remove('visible'); fill.style.width = '0%'; }, 500);
  }
}

async function doSearch() {
  const query = queryInput.value.trim();
  if (!query) { queryInput.focus(); return; }
  if (totalFragments === 0) { toast('Upload a document first', 'error'); return; }

  setSearchLoading(true);
  const area = document.getElementById('resultsArea');

  try {
    const data = await fetchJSON('/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, top_k: 3 }),
    });
    renderResults(data, query);
  } catch (err) {
    area.innerHTML = `
      <div class="state-box">
        <span class="state-icon">⚠️</span>
        <h3>Search failed</h3>
        <p>${err.message}</p>
      </div>`;
  } finally {
    setSearchLoading(false);
    searchBtn.disabled = false;
  }
}

function renderResults(data, query) {
  const area = document.getElementById('resultsArea');

  if (!data.results || data.results.length === 0) {
    area.innerHTML = `
      <div class="state-box">
        <span class="state-icon">🔍</span>
        <h3>No results found</h3>
        <p>Try rephrasing your query or uploading more documents.</p>
      </div>`;
    return;
  }

  const headerHTML = `
    <div class="results-header">
      <span class="results-title">Top results</span>
      <span class="results-meta">${data.results.length} of ${data.total_fragments} fragments · ${data.latency_ms}ms</span>
    </div>`;

  const cardsHTML = data.results.map((r, i) => {
    const scoreWidth = Math.round(r.score * 100);
    const rankLabels = ['1st match', '2nd match', '3rd match'];
    const highlighted = highlightQuery(escapeHtml(r.text), query);
    return `
      <div class="result-card">
        <div class="result-meta-row">
          <span class="rank-badge">${rankLabels[i] || `#${i+1}`}</span>
          <div class="score-badge">
            <div class="score-bar-wrap">
              <div class="score-bar-fill" style="width:${scoreWidth}%"></div>
            </div>
            <span>${r.score.toFixed(4)}</span>
          </div>
          <span class="source-tag" title="${escapeHtml(r.source)}">📁 ${escapeHtml(r.source)}</span>
        </div>
        <p class="result-text">${highlighted}</p>
      </div>`;
  }).join('');

  area.innerHTML = headerHTML + cardsHTML;
}

function highlightQuery(text, query) {
  const words = query.split(/\\s+/).filter(w => w.length > 3);
  let result = text;
  for (const word of words) {
    const re = new RegExp(`(${escapeRe(word)})`, 'gi');
    result = result.replace(re, '<mark style="background:rgba(200,75,49,.15);border-radius:2px;padding:0 1px">$1</mark>');
  }
  return result;
}

function escapeHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function escapeRe(s) {
  return s.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&');
}

// ── Stats ────────────────────────────────────────────────────────────────────
function updateStats(fragments, sources) {
  totalFragments = fragments;
  if (sources !== null) totalSources = sources;
  document.getElementById('statFragments').textContent = fragments;
  document.getElementById('statSources').textContent   = totalSources;
  document.getElementById('statsStrip').style.display  = fragments > 0 ? 'flex' : 'none';
}

async function refreshStats() {
  try {
    const s = await fetchJSON('/status');
    updateStats(s.total_fragments, s.sources.length);
    setStatus('ready', `${s.total_fragments} fragments`);
  } catch {}
}

async function resetIndex() {
  if (!confirm('Clear all indexed documents?')) return;
  try {
    await fetch(`${API}/reset`, { method: 'DELETE' });
    updateStats(0, 0);
    fileList.innerHTML = '';
    document.getElementById('resultsArea').innerHTML = '';
    searchBtn.disabled = true;
    setStatus('ready', '0 fragments');
    toast('Index cleared', 'success');
  } catch (err) {
    toast('Reset failed: ' + err.message, 'error');
  }
}

// ── Status ───────────────────────────────────────────────────────────────────
function setStatus(state, label) {
  const dot   = document.getElementById('statusDot');
  const lbl   = document.getElementById('statusLabel');
  dot.className = 'status-dot' + (state === 'ready' ? ' ready' : '');
  lbl.textContent = label;
}

// ── Helpers ──────────────────────────────────────────────────────────────────
async function fetchJSON(path, opts = {}) {
  let res;
  try {
    res = await fetch(`${API}${path}`, opts);
  } catch (networkErr) {
    throw new Error('Network error — is the server running? (python app.py)');
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

function toast(msg, type = '') {
  const t   = document.getElementById('toast');
  const el  = document.createElement('div');
  el.className = `toast-item${type ? ' ' + type : ''}`;
  el.textContent = msg;
  t.appendChild(el);
  setTimeout(() => {
    el.style.opacity = '0';
    el.style.transform = 'translateX(20px)';
    el.style.transition = 'all .3s';
    setTimeout(() => el.remove(), 300);
  }, 4000);
}

// ── Init ─────────────────────────────────────────────────────────────────────
init();
</script>
</body>
</html>

"""

# ── Text extraction ────────────────────────────────────────────────────────

def extract_text_from_pdf(data):
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            return "\n\n".join(p.extract_text() or "" for p in pdf.pages)
    except Exception as e:
        log.warning("pdfplumber failed (%s) -- trying PyPDF2", e)
        try:
            from PyPDF2 import PdfReader
            return "\n\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)
        except Exception as e2:
            raise ValueError(f"Could not extract PDF: {e2}") from e2

def extract_text_from_docx(data):
    from docx import Document as D
    return "\n\n".join(p.text for p in D(io.BytesIO(data)).paragraphs if p.text.strip())

def extract_text_from_json(data):
    def _r(obj):
        if isinstance(obj, str):  return [obj]
        if isinstance(obj, dict): return [s for k, v in obj.items() for s in [str(k)] + _r(v)]
        if isinstance(obj, list): return [s for item in obj for s in _r(item)]
        return [str(obj)]
    return "\n".join(_r(json.loads(data.decode("utf-8"))))

def extract_text(filename, data):
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":            return extract_text_from_pdf(data)
    if ext in (".docx", ".doc"): return extract_text_from_docx(data)
    if ext == ".json":           return extract_text_from_json(data)
    for enc in ("utf-8", "latin-1", "cp1252"):
        try: return data.decode(enc)
        except UnicodeDecodeError: pass
    return data.decode("utf-8", errors="replace")

# ── Chunking ───────────────────────────────────────────────────────────────

def split_into_chunks(text, target_words=150, overlap_words=20):
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    chunks, current = [], []
    for sent in sentences:
        current.extend(sent.split())
        if len(current) >= target_words:
            chunk = " ".join(current)
            if len(chunk) > 20: chunks.append(chunk)
            current = current[-overlap_words:]
    if current:
        chunk = " ".join(current)
        if len(chunk) > 20: chunks.append(chunk)
    return chunks or [text[:2000]]

# ── Embedding + FAISS ──────────────────────────────────────────────────────

def embed_texts(texts):
    vecs = get_model().encode(texts, batch_size=64, show_progress_bar=False,
                               convert_to_numpy=True, normalize_embeddings=True)
    return vecs.astype(np.float32)

def rebuild_index(vectors):
    idx = faiss.IndexFlatIP(vectors.shape[1])
    idx.add(vectors)
    return idx

def content_hash(data):
    return hashlib.sha256(data).hexdigest()[:16]

def load_cache(h):
    p = CACHE_DIR / f"{h}.npy"
    if p.exists(): log.info("Cache hit %s", h); return np.load(str(p))
    return None

def save_cache(h, v):
    np.save(str(CACHE_DIR / f"{h}.npy"), v)

# ── FastAPI ────────────────────────────────────────────────────────────────

app = FastAPI(title="DocSearch", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class SearchRequest(BaseModel):
    query: str
    top_k: int = 3

class FragmentResult(BaseModel):
    text: str; score: float; source: str; chunk_id: int

class SearchResponse(BaseModel):
    results: list[FragmentResult]; total_fragments: int; latency_ms: float

class StatusResponse(BaseModel):
    total_fragments: int; sources: list[str]; model: str

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return HTMLResponse(content=HTML)

@app.get("/status", response_model=StatusResponse)
def get_status():
    sources = list({f["source"] for f in state["fragments"]})
    return StatusResponse(total_fragments=len(state["fragments"]), sources=sources, model=MODEL_NAME)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    t0 = time.perf_counter()
    allowed = {".pdf", ".docx", ".doc", ".txt", ".json"}
    ext = Path(file.filename or "x.txt").suffix.lower()
    if ext not in allowed:
        raise HTTPException(415, f"File type '{ext}' not supported. Use: {', '.join(sorted(allowed))}")
    data = await file.read()
    if len(data) > 50 * 1024 * 1024:
        raise HTTPException(413, "File too large (max 50 MB).")
    fname = file.filename or "document.txt"
    chash = content_hash(data)
    try:
        text = extract_text(fname, data)
    except Exception as e:
        raise HTTPException(422, f"Text extraction failed: {e}") from e
    if not text.strip():
        raise HTTPException(422, "No text could be extracted from this file.")
    chunks = split_into_chunks(text)
    log.info("'%s' -> %d chunks", fname, len(chunks))
    cached = load_cache(chash)
    new_vecs = cached if (cached is not None and cached.shape[0] == len(chunks)) else embed_texts(chunks)
    if cached is None: save_cache(chash, new_vecs)
    base_id = len(state["fragments"])
    state["fragments"].extend({"text": c, "source": fname, "chunk_id": base_id + i} for i, c in enumerate(chunks))
    state["vectors"] = new_vecs if state["vectors"] is None else np.vstack([state["vectors"], new_vecs])
    state["index"] = rebuild_index(state["vectors"])
    return {"filename": fname, "chunks_added": len(chunks),
            "total_fragments": len(state["fragments"]),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if state["index"] is None:
        raise HTTPException(400, "No documents indexed yet -- upload a file first.")
    if not req.query.strip():
        raise HTTPException(422, "Query must not be empty.")
    t0 = time.perf_counter()
    q = embed_texts([req.query])
    k = min(req.top_k, len(state["fragments"]))
    scores, idxs = state["index"].search(q, k)
    results = [FragmentResult(text=state["fragments"][i]["text"], score=round(float(s), 4),
                               source=state["fragments"][i]["source"], chunk_id=state["fragments"][i]["chunk_id"])
               for s, i in zip(scores[0], idxs[0]) if i >= 0]
    return SearchResponse(results=results, total_fragments=len(state["fragments"]),
                          latency_ms=round((time.perf_counter() - t0) * 1000, 1))

@app.delete("/reset")
def reset_index():
    state["fragments"].clear(); state["vectors"] = None; state["index"] = None
    return {"status": "reset", "total_fragments": 0}

if __name__ == "__main__":
    print("\n" + "=" * 52)
    print("  DocSearch is starting...")
    print("  Open http://localhost:8000 in your browser")
    print("  Press Ctrl+C to stop")
    print("=" * 52 + "\n")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
