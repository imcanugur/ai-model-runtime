package http

import (
	"encoding/json"
	"net/http"
)

// SwaggerUIHTML provides the comprehensive AI Model Runtime Developer API Playground & Interactive Console.
const SwaggerUIHTML = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AI Model Runtime • Developer API Playground</title>
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5.11.0/swagger-ui.css" />
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-page: #090a0f;
      --bg-surface: #11131a;
      --bg-surface-raised: #181b24;
      --bg-input: #151821;
      --border: #232736;
      --border-subtle: #2d3345;
      --border-focus: #3b82f6;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      --accent: #3b82f6;
      --accent-hover: #60a5fa;
      --accent-subtle: rgba(59, 130, 246, 0.14);
      --success: #10b981;
      --success-subtle: rgba(16, 185, 129, 0.14);
      --warning: #f59e0b;
      --warning-subtle: rgba(245, 158, 11, 0.14);
      --error: #ef4444;
      --error-subtle: rgba(239, 68, 68, 0.14);
      --font-ui: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
      --radius: 8px;
    }

    [data-theme="light"] {
      --bg-page: #f8fafc;
      --bg-surface: #ffffff;
      --bg-surface-raised: #f1f5f9;
      --bg-input: #ffffff;
      --border: #e2e8f0;
      --border-subtle: #cbd5e1;
      --border-focus: #2563eb;
      --text-main: #0f172a;
      --text-muted: #64748b;
      --text-dim: #94a3b8;
      --accent: #2563eb;
      --accent-hover: #1d4ed8;
      --accent-subtle: #eff6ff;
      --success: #10b981;
      --success-subtle: #ecfdf5;
      --warning: #f59e0b;
      --warning-subtle: #fffbeb;
      --error: #ef4444;
      --error-subtle: #fef2f2;
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      padding: 0;
      font-family: var(--font-ui);
      background-color: var(--bg-page);
      color: var(--text-main);
      -webkit-font-smoothing: antialiased;
      line-height: 1.5;
      min-height: 100vh;
    }

    /* Top Navbar */
    .console-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      height: 56px;
      padding: 0 24px;
      border-bottom: 1px solid var(--border);
      background: var(--bg-surface);
    }
    .brand-box {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .brand-icon {
      width: 28px;
      height: 28px;
      background: var(--accent);
      color: #ffffff;
      border-radius: 6px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 14px;
      font-weight: 800;
    }
    .brand-name {
      font-size: 15px;
      font-weight: 700;
      letter-spacing: -0.2px;
      color: var(--text-main);
    }
    .brand-badge {
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      background: var(--accent-subtle);
      color: var(--accent);
      padding: 2px 7px;
      border-radius: 4px;
      margin-left: 4px;
      letter-spacing: 0.4px;
    }

    .header-right {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .runtime-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      font-weight: 600;
      color: var(--success);
      background: var(--success-subtle);
      padding: 4px 10px;
      border-radius: 20px;
    }
    .runtime-dot {
      width: 7px;
      height: 7px;
      background: var(--success);
      border-radius: 50%;
    }

    .view-switch {
      display: flex;
      background: var(--bg-surface-raised);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 2px;
    }
    .view-btn {
      background: transparent;
      border: none;
      padding: 5px 12px;
      font-size: 12px;
      font-weight: 600;
      color: var(--text-muted);
      border-radius: 4px;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .view-btn.active {
      background: var(--bg-surface);
      color: var(--text-main);
      box-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }

    .theme-toggle {
      background: var(--bg-surface-raised);
      border: 1px solid var(--border);
      color: var(--text-main);
      width: 32px;
      height: 32px;
      border-radius: 6px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 13px;
    }

    /* Subheader banner */
    .console-intro {
      max-width: 1560px;
      margin: 0 auto;
      padding: 16px 24px 8px;
    }
    .intro-title {
      font-size: 18px;
      font-weight: 700;
      margin: 0 0 4px;
      letter-spacing: -0.3px;
    }
    .intro-desc {
      font-size: 13px;
      color: var(--text-muted);
      margin: 0;
    }

    /* Workspace: 50% / 50% Split View */
    .workspace-grid {
      max-width: 1560px;
      margin: 0 auto;
      padding: 12px 24px 40px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      align-items: stretch;
      min-height: calc(100vh - 160px);
    }
    @media (max-width: 1024px) {
      .workspace-grid {
        grid-template-columns: 1fr;
      }
    }

    /* Cohesive Console Panels */
    .panel {
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    .panel-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 10px 16px;
      background: var(--bg-surface-raised);
      border-bottom: 1px solid var(--border);
      min-height: 46px;
    }
    .panel-title {
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      color: var(--text-muted);
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .panel-body {
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      flex: 1;
    }

    /* Input & Select Controls */
    .ctrl-group {
      display: flex;
      flex-direction: column;
      gap: 5px;
    }
    .ctrl-label {
      font-size: 11px;
      font-weight: 700;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .ctrl-select {
      background: var(--bg-input);
      color: var(--text-main);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 8px 12px;
      font-size: 13px;
      font-family: var(--font-mono);
      outline: none;
      width: 100%;
      cursor: pointer;
      transition: border-color 0.15s ease;
    }
    .ctrl-select:focus { border-color: var(--border-focus); }

    .model-badge-row {
      display: flex;
      align-items: center;
      gap: 6px;
      flex-wrap: nowrap;
      overflow-x: auto;
      margin-top: 2px;
    }
    .pill {
      font-family: var(--font-mono);
      font-size: 11px;
      padding: 2px 6px;
      border-radius: 4px;
      background: var(--bg-surface-raised);
      border: 1px solid var(--border);
      color: var(--text-muted);
      white-space: nowrap;
    }
    .pill-task {
      color: var(--accent);
      background: var(--accent-subtle);
      border-color: transparent;
      font-weight: 600;
    }

    /* Preset selector container */
    .preset-row {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .preset-row .ctrl-select {
      flex: 1;
      min-width: 0;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .star-btn {
      flex-shrink: 0;
      width: 34px;
      height: 34px;
      background: var(--bg-input);
      border: 1px solid var(--border);
      color: var(--text-dim);
      border-radius: 6px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 14px;
      transition: all 0.15s ease;
    }
    .star-btn:hover, .star-btn.active {
      color: #eab308;
      border-color: #eab308;
      background: rgba(234, 179, 8, 0.1);
    }

    /* Editor Tabs & Textarea */
    .editor-mode-switch {
      display: flex;
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 1px;
    }
    .mode-btn {
      background: transparent;
      border: none;
      padding: 3px 10px;
      font-size: 11px;
      font-weight: 600;
      color: var(--text-muted);
      border-radius: 3px;
      cursor: pointer;
    }
    .mode-btn.active {
      background: var(--accent);
      color: #ffffff;
    }

    .editor-textarea {
      background: var(--bg-input);
      color: var(--text-main);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 12px;
      font-size: 13px;
      font-family: var(--font-mono);
      line-height: 1.6;
      outline: none;
      width: 100%;
      height: 180px;
      min-height: 180px;
      resize: vertical;
      box-sizing: border-box;
      transition: border-color 0.15s ease;
    }
    .editor-textarea:focus { border-color: var(--border-focus); }

    /* Parameters Drawer */
    .params-details {
      border: 1px solid var(--border);
      border-radius: 6px;
      background: var(--bg-surface-raised);
      overflow: hidden;
    }
    .params-summary {
      padding: 8px 12px;
      font-size: 11px;
      font-weight: 600;
      color: var(--text-muted);
      cursor: pointer;
      user-select: none;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .params-grid {
      padding: 12px;
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
      background: var(--bg-surface);
      border-top: 1px solid var(--border);
    }
    @media (max-width: 640px) {
      .params-grid { grid-template-columns: repeat(2, 1fr); }
    }
    .param-item {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .param-label {
      font-size: 10px;
      font-family: var(--font-mono);
      font-weight: 600;
      color: var(--text-muted);
    }
    .param-input {
      background: var(--bg-input);
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 5px 8px;
      font-size: 12px;
      font-family: var(--font-mono);
      color: var(--text-main);
      outline: none;
      width: 100%;
    }
    .param-input:focus { border-color: var(--border-focus); }

    /* Action Bar */
    .action-bar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-top: auto;
      padding-top: 12px;
      border-top: 1px solid var(--border);
    }
    .btn-run {
      background: var(--accent);
      color: #ffffff;
      border: none;
      border-radius: 6px;
      padding: 9px 18px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      transition: background 0.15s ease;
    }
    .btn-run:hover { background: var(--accent-hover); }
    .kbd-tag {
      font-size: 10px;
      font-family: var(--font-mono);
      opacity: 0.8;
      background: rgba(0,0,0,0.25);
      padding: 2px 6px;
      border-radius: 4px;
    }
    .btn-clear {
      background: transparent;
      border: none;
      color: var(--text-dim);
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      padding: 6px 12px;
    }
    .btn-clear:hover { color: var(--text-main); }

    /* Right Panel: Tabs Header */
    .inspector-tabs {
      display: flex;
      align-items: center;
      gap: 2px;
    }
    .tab-item {
      background: transparent;
      border: none;
      padding: 6px 12px;
      font-size: 12px;
      font-weight: 600;
      color: var(--text-muted);
      border-radius: 4px;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .tab-item.active {
      background: var(--bg-surface);
      color: var(--text-main);
    }
    .tab-item:hover:not(.active) { color: var(--text-main); }

    /* Status Indicator */
    .status-tag {
      font-family: var(--font-mono);
      font-size: 11px;
      font-weight: 700;
      padding: 3px 8px;
      border-radius: 4px;
      background: var(--bg-surface);
      border: 1px solid var(--border);
      color: var(--text-dim);
    }
    .status-tag.running {
      color: var(--warning);
      border-color: var(--warning);
      background: var(--warning-subtle);
    }
    .status-tag.success {
      color: var(--success);
      border-color: var(--success);
      background: var(--success-subtle);
    }
    .status-tag.error {
      color: var(--error);
      border-color: var(--error);
      background: var(--error-subtle);
    }

    /* Metadata Bar */
    .meta-bar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      background: var(--bg-surface-raised);
      border-bottom: 1px solid var(--border);
      padding: 8px 16px;
      font-size: 11px;
      font-family: var(--font-mono);
      color: var(--text-dim);
      flex-wrap: wrap;
    }
    .meta-item b { color: var(--text-main); }

    /* Inspector Views */
    .inspector-view {
      display: none;
      flex: 1;
      padding: 16px;
      overflow-y: auto;
    }
    .inspector-view.active {
      display: flex;
      flex-direction: column;
    }

    /* Result View Formats */
    .idle-guide {
      border: 1px dashed var(--border);
      border-radius: 6px;
      padding: 24px;
      text-align: center;
      color: var(--text-muted);
      margin: auto 0;
    }
    .guide-icon {
      font-size: 24px;
      margin-bottom: 8px;
    }

    .output-box {
      font-family: var(--font-ui);
      font-size: 14px;
      line-height: 1.7;
      color: var(--text-main);
      white-space: pre-wrap;
      background: var(--bg-input);
      padding: 16px;
      border-radius: 6px;
      border: 1px solid var(--border);
      overflow-x: auto;
    }

    /* Code Generator Area */
    .code-toolbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 10px;
    }
    .lang-tabs {
      display: flex;
      background: var(--bg-surface-raised);
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 2px;
      gap: 2px;
    }
    .lang-btn {
      background: transparent;
      border: none;
      padding: 4px 10px;
      font-size: 11px;
      font-weight: 600;
      color: var(--text-muted);
      border-radius: 3px;
      cursor: pointer;
    }
    .lang-btn.active {
      background: var(--accent);
      color: #ffffff;
    }
    .btn-copy-code {
      background: var(--bg-surface-raised);
      border: 1px solid var(--border);
      color: var(--text-main);
      padding: 4px 10px;
      font-size: 11px;
      font-weight: 600;
      border-radius: 4px;
      cursor: pointer;
    }
    .btn-copy-code:hover { border-color: var(--border-focus); }
    .code-view-box {
      margin: 0;
      font-family: var(--font-mono);
      font-size: 12px;
      line-height: 1.6;
      background: var(--bg-input);
      color: #38bdf8;
      padding: 14px;
      border-radius: 6px;
      border: 1px solid var(--border);
      overflow-x: auto;
      flex: 1;
    }

    /* History List */
    .history-container {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .history-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 8px 12px;
      background: var(--bg-surface-raised);
      border: 1px solid var(--border);
      border-radius: 6px;
      cursor: pointer;
      font-size: 12px;
      transition: border-color 0.15s ease;
    }
    .history-row:hover {
      border-color: var(--accent);
    }

    /* API Reference Workspace */
    .reference-workspace {
      display: none;
      max-width: 1560px;
      margin: 0 auto;
      padding: 20px 24px;
      flex-direction: column;
      gap: 20px;
    }
  </style>
</head>
<body data-theme="dark">

  <!-- Navbar -->
  <header class="console-header">
    <div class="brand-box">
      <div class="brand-icon">⚡</div>
      <div class="brand-text">
        <span class="brand-name">AI Model Runtime</span>
        <span class="brand-badge">Playground</span>
      </div>
    </div>

    <div class="header-right">
      <div class="runtime-pill">
        <span class="runtime-dot"></span>
        <span id="model-count-text">Connecting...</span>
      </div>
      <div class="view-switch">
        <button class="view-btn active" id="tab-playground-btn">⚡ Playground</button>
        <button class="view-btn" id="tab-reference-btn">📖 API Reference</button>
      </div>
      <button class="theme-toggle" id="theme-toggle-btn" title="Toggle theme">☀️</button>
    </div>
  </header>

  <!-- Title Banner -->
  <div class="console-intro" id="intro-banner">
    <h1 class="intro-title">Developer API Playground</h1>
    <p class="intro-desc">Explore available models and run inference directly from your browser.</p>
  </div>

  <!-- Playground Workspace (50% / 50% Balanced Grid) -->
  <div class="workspace-grid" id="playground-view">
    <!-- Left Panel: Request Builder -->
    <div class="panel">
      <div class="panel-header">
        <div class="panel-title">
          <span>REQUEST PAYLOAD</span>
        </div>
        <div class="editor-mode-switch">
          <button class="mode-btn active" id="mode-form-btn">Form</button>
          <button class="mode-btn" id="mode-json-btn">JSON</button>
        </div>
      </div>

      <div class="panel-body">
        <!-- Model Selection -->
        <div class="ctrl-group">
          <div class="ctrl-label">
            <span>Model</span>
            <span style="font-size:10px; color:var(--text-dim);">GET /v1/models</span>
          </div>
          <select class="ctrl-select" id="model-select">
            <option value="" disabled selected>Loading runtime models...</option>
          </select>
          <div class="model-badge-row" id="model-badge-row" style="display:none;">
            <span class="pill pill-task" id="badge-task">text-generation</span>
            <span class="pill" id="badge-version">v1.0.0</span>
            <span class="pill" id="badge-lang">tr</span>
            <span class="pill" id="badge-status" style="color:var(--success);">READY</span>
          </div>
        </div>

        <!-- Example Preset Selection -->
        <div class="ctrl-group">
          <div class="ctrl-label">
            <span>Example Preset</span>
            <span style="font-size:10px; color:var(--text-dim);">Capability-tailored</span>
          </div>
          <div class="preset-row">
            <select class="ctrl-select" id="example-select">
              <option value="">-- Select an example preset --</option>
            </select>
            <button class="star-btn" id="star-btn" title="Bookmark Preset">★</button>
          </div>
        </div>

        <!-- Input Editor (Form mode & JSON mode) -->
        <div class="ctrl-group" style="flex:1; display:flex; flex-direction:column;">
          <div class="ctrl-label">
            <span>Input Text</span>
            <span style="font-size:10px; color:var(--text-dim);">canonical input.text</span>
          </div>
          <div id="form-mode-container" style="flex:1; display:flex; flex-direction:column;">
            <textarea class="editor-textarea" id="form-input-text" placeholder="Enter prompt or input text for inference..."></textarea>
          </div>
          <div id="json-mode-container" style="display:none; flex:1; flex-direction:column; gap:6px;">
            <textarea class="editor-textarea" id="json-input-editor" style="color:#38bdf8;"></textarea>
            <div style="display:flex; justify-content:flex-end; gap:6px;">
              <button class="btn-clear" id="json-format-btn" style="padding:2px 8px; font-size:11px;">Format JSON</button>
              <button class="btn-clear" id="json-reset-btn" style="padding:2px 8px; font-size:11px;">Reset</button>
            </div>
          </div>
        </div>

        <!-- Advanced Options Accordion -->
        <details class="params-details" id="params-drawer">
          <summary class="params-summary">
            <span>Advanced Options</span>
            <span style="font-size:10px; color:var(--text-dim);">temperature, max_new_tokens, top_p</span>
          </summary>
          <div class="params-grid">
            <div class="param-item">
              <span class="param-label">temperature</span>
              <input type="number" step="0.1" min="0" max="2" class="param-input" id="opt-temp" placeholder="0.7" />
            </div>
            <div class="param-item">
              <span class="param-label">max_new_tokens</span>
              <input type="number" step="1" min="1" max="512" class="param-input" id="opt-tokens" placeholder="64" />
            </div>
            <div class="param-item">
              <span class="param-label">top_p</span>
              <input type="number" step="0.05" min="0" max="1" class="param-input" id="opt-topp" placeholder="0.9" />
            </div>
            <div class="param-item">
              <span class="param-label">top_k</span>
              <input type="number" step="1" min="0" max="100" class="param-input" id="opt-topk" placeholder="50" />
            </div>
          </div>
        </details>

        <!-- Footer Actions -->
        <div class="action-bar">
          <button class="btn-run" id="btn-run">
            <span>Run inference</span>
            <span class="kbd-tag">Ctrl + Enter</span>
          </button>
          <button class="btn-clear" id="btn-clear">Clear Input</button>
        </div>
      </div>
    </div>

    <!-- Right Panel: Inspector (Result, JSON, Raw, Code, History) -->
    <div class="panel">
      <div class="panel-header">
        <div class="inspector-tabs">
          <button class="tab-item active" data-tab="result" id="tab-btn-result">Result</button>
          <button class="tab-item" data-tab="json" id="tab-btn-json">JSON</button>
          <button class="tab-item" data-tab="raw" id="tab-btn-raw">Raw</button>
          <button class="tab-item" data-tab="code" id="tab-btn-code">Code</button>
          <button class="tab-item" data-tab="history" id="tab-btn-history">History</button>
        </div>
        <span class="status-tag" id="status-badge">Idle</span>
      </div>

      <!-- Metadata Strip -->
      <div class="meta-bar">
        <div>Status: <span id="meta-status">—</span></div>
        <div>Latency: <span id="meta-duration">—</span></div>
        <div>Model: <span id="meta-model">—</span></div>
        <div>Request ID: <span id="meta-reqid">—</span></div>
      </div>

      <!-- View: Result -->
      <div class="inspector-view active" id="view-result">
        <div class="idle-guide" id="result-idle-state">
          <div class="guide-icon">⚡</div>
          <div style="font-weight:600; margin-bottom:4px;">Inference Output Ready</div>
          <div style="font-size:12px; color:var(--text-dim);">Pick an example preset and press <b>Ctrl + Enter</b> to run.</div>
        </div>
        <div id="result-active-content" style="display:none; flex:1;"></div>
      </div>

      <!-- View: JSON -->
      <div class="inspector-view" id="view-json">
        <pre class="output-box" id="json-view-box" style="flex:1; margin:0;"></pre>
      </div>

      <!-- View: Raw Wire -->
      <div class="inspector-view" id="view-raw">
        <pre class="output-box" id="raw-view-box" style="flex:1; margin:0;"></pre>
      </div>

      <!-- View: Code Generator -->
      <div class="inspector-view" id="view-code">
        <div class="code-toolbar">
          <div class="lang-tabs">
            <button class="lang-btn active" data-lang="curl">cURL</button>
            <button class="lang-btn" data-lang="python">Python</button>
            <button class="lang-btn" data-lang="js">JavaScript</button>
            <button class="lang-btn" data-lang="go">Go</button>
          </div>
          <button class="btn-copy-code" id="btn-copy-snippet">Copy Code</button>
        </div>
        <pre class="code-view-box" id="code-output-box"></pre>
      </div>

      <!-- View: History -->
      <div class="inspector-view" id="view-history">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
          <span style="font-size:11px; font-weight:700; color:var(--text-muted); text-transform:uppercase;">Recent Executions</span>
          <button class="btn-clear" id="btn-clear-history" style="padding:2px 6px; font-size:11px;">Clear All</button>
        </div>
        <div class="history-container" id="history-container">
          <div style="font-size:12px; color:var(--text-dim); text-align:center; padding:24px;">No executions logged in this session yet.</div>
        </div>
      </div>
    </div>
  </div>

  <!-- API Reference Workspace -->
  <div class="reference-workspace" id="api-reference-view">
    <div style="background:var(--bg-surface); border:1px solid var(--border); border-radius:var(--radius); padding:20px;">
      <h2 style="font-size:16px; margin:0 0 6px;">OpenAPI 3.0.3 Specification</h2>
      <p style="font-size:13px; color:var(--text-muted); margin:0;">Standard model-agnostic inference endpoints.</p>
    </div>
    <div id="swagger-ui-container" style="background:#ffffff; border-radius:var(--radius); overflow:hidden;"></div>
  </div>

<script src="https://unpkg.com/swagger-ui-dist@5.11.0/swagger-ui-bundle.js" crossorigin></script>
<script src="https://unpkg.com/swagger-ui-dist@5.11.0/swagger-ui-standalone-preset.js" crossorigin></script>
<script>
  // 1. Core Preset Registry (All standardized to canonical input.text)
  const BUILTIN_EXAMPLES = [
    // --- Text Generation ---
    {
      id: "gen-basic",
      category: "Text Generation",
      task: "text-generation",
      name: "📝 Basic generation",
      input: { text: "Atatürk hakkında kısa bir cümle yaz." },
      options: { max_new_tokens: 32, temperature: 0.7 }
    },
    {
      id: "gen-question",
      category: "Text Generation",
      task: "text-generation",
      name: "🇹🇷 Turkish question",
      input: { text: "Türkiye'nin başkenti neresidir? Kısa cevap ver." },
      options: { max_new_tokens: 32, temperature: 0.5 }
    },
    {
      id: "gen-creative",
      category: "Text Generation",
      task: "text-generation",
      name: "✍️ Creative writing",
      input: { text: "İstanbul'da yağmurlu bir akşamı anlatan kısa bir paragraf yaz." },
      options: { max_new_tokens: 64, temperature: 0.8 }
    },
    {
      id: "gen-translation",
      category: "Text Generation",
      task: "text-generation",
      name: "🌍 Translation",
      input: { text: "Bugün hava çok güzel. Bu cümleyi İngilizceye çevir." },
      options: { max_new_tokens: 32, temperature: 0.3 }
    },
    {
      id: "gen-explanation",
      category: "Text Generation",
      task: "text-generation",
      name: "💡 Explanation",
      input: { text: "Yapay zekayı teknik olmayan bir kişiye iki cümleyle açıkla." },
      options: { max_new_tokens: 64, temperature: 0.7 }
    },
    {
      id: "gen-summarization",
      category: "Text Generation",
      task: "text-generation",
      name: "📄 Summarization",
      input: { text: "Aşağıdaki metni iki cümlede özetle:\n\nYapay zeka son yıllarda büyük bir ivme kazanarak sağlık, finans ve endüstri alanlarında devrim yaratmaktadır." },
      options: { max_new_tokens: 64, temperature: 0.5 }
    },
    {
      id: "gen-instructions",
      category: "Text Generation",
      task: "text-generation",
      name: "🔤 Instruction following",
      input: { text: "Aşağıdaki listeyi alfabetik sıraya koy:\nElma\nArmut\nMuz\nÇilek" },
      options: { max_new_tokens: 48, temperature: 0.2 }
    },
    {
      id: "gen-short-answer",
      category: "Text Generation",
      task: "text-generation",
      name: "⚡ Short answer",
      input: { text: "Python nedir? Tek cümleyle cevap ver." },
      options: { max_new_tokens: 32, temperature: 0.4 }
    },
    {
      id: "gen-long",
      category: "Text Generation",
      task: "text-generation",
      name: "📚 Longer generation",
      input: { text: "Türkiye'de yapay zekanın kullanım alanları ve geleceği hakkında kısa bir yazı yaz." },
      options: { max_new_tokens: 128, temperature: 0.7 }
    },

    // --- Embeddings ---
    {
      id: "emb-simple",
      category: "Embeddings",
      task: "embedding",
      name: "☀️ Simple sentence",
      input: { text: "Bugün hava çok güzel ve güneşli." },
      options: {}
    },
    {
      id: "emb-semantic",
      category: "Embeddings",
      task: "embedding",
      name: "🧠 Semantic sentence",
      input: { text: "Yapay zeka destekli metin analizi yapıyorum." },
      options: {}
    },
    {
      id: "emb-similarity",
      category: "Embeddings",
      task: "embedding",
      name: "🔄 Semantic comparison",
      input: { text: "Bugün hava oldukça güzel ve açık." },
      options: {}
    },
    {
      id: "emb-long",
      category: "Embeddings",
      task: "embedding",
      name: "📑 Longer text",
      input: { text: "Yapay zeka, doğal dil işleme ve makine öğrenmesi algoritmaları metinleri yüksek boyutlu vektör uzaylarında temsil eder." },
      options: {}
    },

    // --- Classification ---
    {
      id: "cls-berturk-academic",
      category: "Classification",
      task: "text-classification",
      name: "🛡️ BERTurk AI Text Detector",
      input: { text: "Bu çalışmada derin öğrenme tabanlı doğal dil işleme modellerinin Türkçe akademik metinler üzerindeki performansı ve sınıflandırma doğruluğu deneysel olarak incelenmiştir." },
      options: {}
    },
    {
      id: "cls-positive",
      category: "Classification",
      task: "text-classification",
      name: "😊 Positive sentiment",
      input: { text: "Bu ürün beklediğimden çok daha kaliteli çıktı, kesinlikle tavsiye ederim." },
      options: {}
    },
    {
      id: "cls-negative",
      category: "Classification",
      task: "text-classification",
      name: "🙁 Negative sentiment",
      input: { text: "Hizmet çok kötüydü ve tekrar kullanmayacağım, paketleme özensizdi." },
      options: {}
    },

    // --- Question Answering ---
    {
      id: "qa-general",
      category: "Question Answering",
      task: "question-answering",
      name: "❓ Question answering",
      input: {
        text: "Türkiye Cumhuriyeti hangi yılda ilan edilmiştir?"
      },
      options: {}
    }
  ];

  // State
  let modelsList = [];
  let currentModel = null;
  let currentExample = null;
  let editorMode = 'form';
  let activeInspectorTab = 'result';
  let activeCodeLang = 'curl';
  let lastResponse = null;
  let lastRawText = '';
  let favorites = JSON.parse(localStorage.getItem('runtime_favorites') || '[]');
  let recentRequests = JSON.parse(localStorage.getItem('runtime_recent_requests') || '[]');

  // Theme Management (Dark first default)
  const themeToggleBtn = document.getElementById('theme-toggle-btn');
  const initialTheme = localStorage.getItem('runtime_theme') || 'dark';
  document.body.setAttribute('data-theme', initialTheme);
  themeToggleBtn.textContent = initialTheme === 'dark' ? '☀️' : '🌙';

  themeToggleBtn.addEventListener('click', () => {
    const active = document.body.getAttribute('data-theme');
    const target = active === 'dark' ? 'light' : 'dark';
    document.body.setAttribute('data-theme', target);
    localStorage.setItem('runtime_theme', target);
    themeToggleBtn.textContent = target === 'dark' ? '☀️' : '🌙';
  });

  // Top Nav View Switcher
  const tabPlaygroundBtn = document.getElementById('tab-playground-btn');
  const tabReferenceBtn = document.getElementById('tab-reference-btn');
  const playgroundView = document.getElementById('playground-view');
  const referenceView = document.getElementById('api-reference-view');
  const introBanner = document.getElementById('intro-banner');

  tabPlaygroundBtn.addEventListener('click', () => {
    tabPlaygroundBtn.classList.add('active');
    tabReferenceBtn.classList.remove('active');
    playgroundView.style.display = 'grid';
    introBanner.style.display = 'block';
    referenceView.style.display = 'none';
  });

  tabReferenceBtn.addEventListener('click', () => {
    tabReferenceBtn.classList.add('active');
    tabPlaygroundBtn.classList.remove('active');
    playgroundView.style.display = 'none';
    introBanner.style.display = 'none';
    referenceView.style.display = 'flex';
  });

  // Fetch Models from GET /v1/models
  async function loadModels() {
    const pill = document.getElementById('model-count-text');
    const select = document.getElementById('model-select');
    pill.textContent = 'Fetching models...';

    try {
      const resp = await fetch('/v1/models');
      if (!resp.ok) throw new Error('HTTP ' + resp.status);
      const data = await resp.json();
      modelsList = data.models || [];

      select.innerHTML = '';
      if (modelsList.length === 0) {
        select.innerHTML = '<option value="" disabled selected>No models discovered</option>';
        pill.textContent = '0 Models Ready';
        return;
      }

      pill.textContent = modelsList.length + ' Model' + (modelsList.length > 1 ? 's' : '') + ' Ready';

      modelsList.forEach((m, i) => {
        const opt = document.createElement('option');
        opt.value = m.id;
        const ver = m.version ? (' v' + m.version) : '';
        opt.textContent = m.id + ver;
        if (i === 0) opt.selected = true;
        select.appendChild(opt);
      });

      onModelSelected(modelsList[0].id, true);
    } catch (err) {
      console.error('Failed to load models:', err);
      select.innerHTML = '<option value="" disabled selected>Error loading models</option>';
      pill.textContent = 'Offline';
    }
  }

  // Model Selected Handler
  function onModelSelected(modelId, isInitialLoad = false) {
    currentModel = modelsList.find(m => m.id === modelId) || null;
    if (!currentModel) return;

    // Badges
    const badgeRow = document.getElementById('model-badge-row');
    badgeRow.style.display = 'flex';
    document.getElementById('badge-task').textContent = currentModel.task || 'general';
    document.getElementById('badge-version').textContent = 'v' + (currentModel.version || '1.0.0');
    const langs = (currentModel.languages && currentModel.languages.length) ? currentModel.languages.join(',') : 'all';
    document.getElementById('badge-lang').textContent = langs;
    const state = currentModel.state || currentModel.status || 'READY';
    const stBadge = document.getElementById('badge-status');
    stBadge.textContent = state;
    stBadge.style.color = (state === 'READY') ? 'var(--success)' : 'var(--warning)';

    // Update examples dropdown without overwriting current prompt if user changed model
    populateExamplesForModel(currentModel.task, isInitialLoad);
    updateCodeSnippet();
  }

  document.getElementById('model-select').addEventListener('change', (e) => {
    onModelSelected(e.target.value, false);
  });

  // Populate Examples Dropdown
  function populateExamplesForModel(task, autoLoad = false) {
    const select = document.getElementById('example-select');
    select.innerHTML = '<option value="">-- Select an example preset --</option>';

    let filtered = BUILTIN_EXAMPLES.filter(ex => !task || ex.task === task);
    if (filtered.length === 0) {
      filtered = BUILTIN_EXAMPLES;
    }

    const categories = {};
    filtered.forEach(ex => {
      const cat = ex.category || 'General';
      if (!categories[cat]) categories[cat] = [];
      categories[cat].push(ex);
    });

    Object.keys(categories).forEach(cat => {
      const group = document.createElement('optgroup');
      group.label = cat;
      categories[cat].forEach(ex => {
        const isFav = favorites.includes(ex.id);
        const opt = document.createElement('option');
        opt.value = ex.id;
        opt.textContent = (isFav ? '★ ' : '') + ex.name;
        group.appendChild(opt);
      });
      select.appendChild(group);
    });

    if (autoLoad && filtered.length > 0) {
      select.value = filtered[0].id;
      loadExampleIntoEditor(filtered[0].id);
    }
  }

  // Load Example into Editor
  function loadExampleIntoEditor(exampleId) {
    currentExample = BUILTIN_EXAMPLES.find(ex => ex.id === exampleId) || null;
    if (!currentExample) return;

    // Favorite button
    const starBtn = document.getElementById('star-btn');
    if (favorites.includes(exampleId)) {
      starBtn.classList.add('active');
    } else {
      starBtn.classList.remove('active');
    }

    // Input text
    document.getElementById('form-input-text').value = (currentExample.input && currentExample.input.text) ? currentExample.input.text : '';

    // Options
    document.getElementById('opt-temp').value = (currentExample.options && currentExample.options.temperature !== undefined) ? currentExample.options.temperature : '';
    document.getElementById('opt-tokens').value = (currentExample.options && currentExample.options.max_new_tokens !== undefined) ? currentExample.options.max_new_tokens : '';
    document.getElementById('opt-topp').value = (currentExample.options && currentExample.options.top_p !== undefined) ? currentExample.options.top_p : '';
    document.getElementById('opt-topk').value = (currentExample.options && currentExample.options.top_k !== undefined) ? currentExample.options.top_k : '';

    syncFormToJson();
    updateCodeSnippet();
  }

  document.getElementById('example-select').addEventListener('change', (e) => {
    loadExampleIntoEditor(e.target.value);
  });

  // Bookmark / Favorite Toggle
  document.getElementById('star-btn').addEventListener('click', () => {
    if (!currentExample) return;
    const id = currentExample.id;
    const idx = favorites.indexOf(id);
    if (idx >= 0) {
      favorites.splice(idx, 1);
    } else {
      favorites.push(id);
    }
    localStorage.setItem('runtime_favorites', JSON.stringify(favorites));
    if (currentModel) populateExamplesForModel(currentModel.task);
  });

  // Form Mode vs JSON Mode
  const modeFormBtn = document.getElementById('mode-form-btn');
  const modeJsonBtn = document.getElementById('mode-json-btn');
  const formContainer = document.getElementById('form-mode-container');
  const jsonContainer = document.getElementById('json-mode-container');

  modeFormBtn.addEventListener('click', () => {
    modeFormBtn.classList.add('active');
    modeJsonBtn.classList.remove('active');
    formContainer.style.display = 'flex';
    jsonContainer.style.display = 'none';
    editorMode = 'form';
    syncJsonToForm();
  });

  modeJsonBtn.addEventListener('click', () => {
    modeJsonBtn.classList.add('active');
    modeFormBtn.classList.remove('active');
    formContainer.style.display = 'none';
    jsonContainer.style.display = 'flex';
    editorMode = 'json';
    syncFormToJson();
  });

  // Two-way synchronization
  function syncFormToJson() {
    const textVal = document.getElementById('form-input-text').value;

    const optionsObj = {};
    const temp = document.getElementById('opt-temp').value.trim();
    if (temp !== '') {
      const v = parseFloat(temp);
      if (!isNaN(v)) optionsObj.temperature = v;
    }
    const tokens = document.getElementById('opt-tokens').value.trim();
    if (tokens !== '') {
      const v = parseInt(tokens, 10);
      if (!isNaN(v)) optionsObj.max_new_tokens = v;
    }
    const topp = document.getElementById('opt-topp').value.trim();
    if (topp !== '') {
      const v = parseFloat(topp);
      if (!isNaN(v)) optionsObj.top_p = v;
    }
    const topk = document.getElementById('opt-topk').value.trim();
    if (topk !== '') {
      const v = parseInt(topk, 10);
      if (!isNaN(v)) optionsObj.top_k = v;
    }

    const payload = {
      input: { text: textVal },
      options: optionsObj
    };
    document.getElementById('json-input-editor').value = JSON.stringify(payload, null, 2);
  }

  function syncJsonToForm() {
    try {
      const payload = JSON.parse(document.getElementById('json-input-editor').value);
      if (payload.input && payload.input.text !== undefined) {
        document.getElementById('form-input-text').value = payload.input.text;
      }
      if (payload.options) {
        document.getElementById('opt-temp').value = payload.options.temperature !== undefined ? payload.options.temperature : '';
        document.getElementById('opt-tokens').value = payload.options.max_new_tokens !== undefined ? payload.options.max_new_tokens : '';
        document.getElementById('opt-topp').value = payload.options.top_p !== undefined ? payload.options.top_p : '';
        document.getElementById('opt-topk').value = payload.options.top_k !== undefined ? payload.options.top_k : '';
      }
    } catch (e) {
      // Keep existing values on syntax error
    }
  }

  // Live input sync
  ['form-input-text', 'opt-temp', 'opt-tokens', 'opt-topp', 'opt-topk'].forEach(id => {
    const el = document.getElementById(id);
    if (el) {
      el.addEventListener('input', () => {
        syncFormToJson();
        updateCodeSnippet();
      });
    }
  });

  document.getElementById('json-input-editor').addEventListener('input', () => {
    updateCodeSnippet();
  });

  document.getElementById('json-format-btn').addEventListener('click', () => {
    try {
      const parsed = JSON.parse(document.getElementById('json-input-editor').value);
      document.getElementById('json-input-editor').value = JSON.stringify(parsed, null, 2);
    } catch (e) {
      alert('Invalid JSON: ' + e.message);
    }
  });

  document.getElementById('json-reset-btn').addEventListener('click', () => {
    if (currentExample) loadExampleIntoEditor(currentExample.id);
  });

  document.getElementById('btn-clear').addEventListener('click', () => {
    document.getElementById('form-input-text').value = '';
    syncFormToJson();
    updateCodeSnippet();
  });

  function getPayloadObject() {
    if (editorMode === 'json') {
      try {
        const parsed = JSON.parse(document.getElementById('json-input-editor').value);
        if (!parsed.input || typeof parsed.input !== 'object') parsed.input = { text: '' };
        if (!parsed.options || typeof parsed.options !== 'object') parsed.options = {};
        return parsed;
      } catch (e) {
        syncFormToJson();
      }
    }
    syncFormToJson();
    return JSON.parse(document.getElementById('json-input-editor').value);
  }

  // Run Inference
  async function runInference() {
    if (!currentModel) {
      alert('Please select an inference model first.');
      return;
    }

    const runBtn = document.getElementById('btn-run');
    const statusBadge = document.getElementById('status-badge');
    const metaStatus = document.getElementById('meta-status');
    const metaDuration = document.getElementById('meta-duration');
    const metaModel = document.getElementById('meta-model');
    const metaReqId = document.getElementById('meta-reqid');

    const payload = getPayloadObject();
    const endpoint = '/v1/models/' + encodeURIComponent(currentModel.id) + '/predict';

    runBtn.disabled = true;
    runBtn.innerHTML = '<span>Running inference...</span>';
    statusBadge.className = 'status-tag running';
    statusBadge.textContent = 'Running...';

    const startTime = performance.now();

    try {
      const resp = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const durationMs = Math.round(performance.now() - startTime);
      const rawText = await resp.text();
      let parsedData = null;
      try {
        parsedData = JSON.parse(rawText);
      } catch (e) {
        parsedData = { raw: rawText };
      }

      let headerLines = '';
      if (resp.headers && resp.headers.forEach) {
        resp.headers.forEach((v, k) => {
          headerLines += k + ': ' + v + '\n';
        });
      }

      lastRawText = 'HTTP/1.1 ' + resp.status + ' ' + (resp.statusText || (resp.ok ? 'OK' : 'Error')) + '\n' +
        headerLines + '\n' + rawText;

      lastResponse = {
        status: resp.status,
        statusText: resp.statusText || (resp.ok ? 'OK' : 'Error'),
        ok: resp.ok,
        durationMs: durationMs,
        data: parsedData
      };

      if (resp.ok) {
        statusBadge.className = 'status-tag success';
        statusBadge.textContent = '✓ ' + resp.status + ' OK · ' + durationMs + ' ms';
      } else {
        statusBadge.className = 'status-tag error';
        statusBadge.textContent = '✕ ' + resp.status + ' Error · ' + durationMs + ' ms';
      }

      metaStatus.textContent = resp.status + ' ' + (resp.statusText || (resp.ok ? 'OK' : 'Error'));
      metaStatus.style.color = resp.ok ? 'var(--success)' : 'var(--error)';
      metaDuration.textContent = durationMs + ' ms';
      metaModel.textContent = currentModel.id + ' (v' + (currentModel.version || '1.0.0') + ')';

      const reqId = (parsedData && parsedData.meta && parsedData.meta.request_id) || '—';
      metaReqId.textContent = reqId;

      renderInspector();
      saveHistoryItem(currentModel.id, payload, resp.status, durationMs);
    } catch (err) {
      const durationMs = Math.round(performance.now() - startTime);
      lastResponse = {
        status: 0,
        statusText: 'Network / Client Error',
        ok: false,
        durationMs: durationMs,
        data: { error: { code: 'NETWORK_ERROR', message: err.message || 'Connection failed' } }
      };
      lastRawText = 'Error: ' + (err.message || 'Connection failed');

      statusBadge.className = 'status-tag error';
      statusBadge.textContent = '✕ Connection Failed · ' + durationMs + ' ms';

      metaStatus.textContent = 'Network Error';
      metaStatus.style.color = 'var(--error)';
      metaDuration.textContent = durationMs + ' ms';
      metaModel.textContent = currentModel.id;
      metaReqId.textContent = '—';

      renderInspector();
    } finally {
      runBtn.disabled = false;
      runBtn.innerHTML = '<span>Run inference</span><span class="kbd-tag">Ctrl + Enter</span>';
    }
  }

  document.getElementById('btn-run').addEventListener('click', runInference);

  // Inspector Tabs Switching
  document.querySelectorAll('.tab-item').forEach(btn => {
    btn.addEventListener('click', (e) => {
      document.querySelectorAll('.tab-item').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.inspector-view').forEach(v => v.classList.remove('active'));

      e.target.classList.add('active');
      activeInspectorTab = e.target.getAttribute('data-tab');
      document.getElementById('view-' + activeInspectorTab).classList.add('active');
    });
  });

  // Render Inspector Views
  function renderInspector() {
    if (!lastResponse) return;

    // 1. Result View
    const idleState = document.getElementById('result-idle-state');
    const activeContent = document.getElementById('result-active-content');
    idleState.style.display = 'none';
    activeContent.style.display = 'flex';
    activeContent.style.flexDirection = 'column';

    const data = lastResponse.data;
    if (lastResponse.ok && data && data.data && data.data.value !== undefined) {
      const val = data.data.value;
      if (Array.isArray(val)) {
        // Embedding
        const preview = val.slice(0, 8).map(n => typeof n === 'number' ? n.toFixed(5) : n).join(', ');
        activeContent.innerHTML =
          '<div style="margin-bottom:10px; display:flex; align-items:center; gap:8px;">' +
            '<span class="pill pill-task" style="font-size:12px; padding:3px 8px;">Dimension: ' + val.length + '</span>' +
            '<span class="pill" style="color:var(--success);">float32 array</span>' +
          '</div>' +
          '<pre class="output-box" style="flex:1; margin:0; color:#38bdf8; font-family:var(--font-mono); font-size:12px;">' +
            '// Vector Preview (first 8 dimensions):\n[' + preview + (val.length > 8 ? ', ...' : '') + ']' +
          '</pre>';
      } else if (typeof val === 'object' && val !== null) {
        // Classification / Structured
        let html = '<div style="display:flex; flex-direction:column; gap:8px; flex:1;">';
        if (val.label !== undefined) {
          const scorePercent = val.score !== undefined ? (val.score * 100).toFixed(1) + '%' : null;
          html +=
            '<div style="display:flex; align-items:center; justify-content:space-between; background:var(--bg-input); border:1px solid var(--border); border-radius:6px; padding:12px 16px;">' +
              '<div>' +
                '<div style="font-size:10px; text-transform:uppercase; color:var(--text-muted); font-weight:700;">Predicted Class</div>' +
                '<div style="font-size:18px; font-weight:800; color:var(--accent); font-family:var(--font-mono);">' + escapeHtml(val.label) + '</div>' +
              '</div>' +
              (scorePercent ? '<div style="text-align:right;"><div style="font-size:10px; text-transform:uppercase; color:var(--text-muted); font-weight:700;">Confidence</div><div style="font-size:18px; font-weight:800; color:var(--success); font-family:var(--font-mono);">' + scorePercent + '</div></div>' : '') +
            '</div>';
        }
        html += '<pre class="output-box" style="flex:1; margin:0;">' + JSON.stringify(val, null, 2) + '</pre></div>';
        activeContent.innerHTML = html;
      } else {
        // Text output
        activeContent.innerHTML =
          '<div class="output-box" style="flex:1; margin:0;">' + escapeHtml(String(val)) + '</div>';
      }
    } else if (data && data.error) {
      activeContent.innerHTML =
        '<div style="background:var(--error-subtle); border:1px solid var(--error); border-radius:6px; padding:16px;">' +
          '<div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">' +
            '<span style="font-weight:800; color:var(--error); font-size:13px;">✕ Execution Failed</span>' +
            '<span class="pill" style="color:var(--error); font-family:var(--font-mono);">' + escapeHtml(data.error.code || 'ERROR') + '</span>' +
          '</div>' +
          '<div style="font-size:13px; color:var(--text-main); line-height:1.5;">' + escapeHtml(data.error.message || 'An error occurred.') + '</div>' +
        '</div>';
    } else {
      activeContent.innerHTML = '<pre class="output-box" style="flex:1; margin:0;">' + JSON.stringify(data, null, 2) + '</pre>';
    }

    // 2. JSON View
    document.getElementById('json-view-box').textContent = JSON.stringify(data, null, 2);

    // 3. Raw View
    document.getElementById('raw-view-box').textContent = lastRawText;
  }

  function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  // Code Generation
  function updateCodeSnippet() {
    const modelId = (currentModel && currentModel.id) || (document.getElementById('model-select').value) || 'model-id';
    const payload = getPayloadObject();
    const payloadJson = JSON.stringify(payload, null, 2);
    const origin = window.location.origin || (window.location.protocol + '//' + window.location.host);
    const url = origin + '/v1/models/' + encodeURIComponent(modelId) + '/predict';

    let code = '';
    if (activeCodeLang === 'curl') {
      code = 'curl -X POST \\\n' +
        '  ' + url + ' \\\n' +
        '  -H "Content-Type: application/json" \\\n' +
        '  -d \'' + payloadJson + '\'';
    } else if (activeCodeLang === 'python') {
      const pythonPayload = JSON.stringify(payload, null, 4)
        .replace(/: true/g, ': True')
        .replace(/: false/g, ': False')
        .replace(/: null/g, ': None');

      code = 'import requests\n\n' +
        'response = requests.post(\n' +
        '    "' + url + '",\n' +
        '    json=' + pythonPayload + '\n' +
        ')\n\n' +
        'result = response.json()\n' +
        'print(result["data"]["value"])';
    } else if (activeCodeLang === 'js') {
      code = 'const response = await fetch("' + url + '", {\n' +
        '  method: "POST",\n' +
        '  headers: { "Content-Type": "application/json" },\n' +
        '  body: JSON.stringify(' + payloadJson + ')\n' +
        '});\n\n' +
        'const result = await response.json();\n' +
        'console.log(result.data.value);';
    } else if (activeCodeLang === 'go') {
      const bt = String.fromCharCode(96);
      code = 'package main\n\n' +
        'import (\n' +
        '\t"bytes"\n' +
        '\t"fmt"\n' +
        '\t"io"\n' +
        '\t"net/http"\n' +
        ')\n\n' +
        'func main() {\n' +
        '\tbody := []byte(' + bt + payloadJson + bt + ')\n' +
        '\tresp, err := http.Post("' + url + '", "application/json", bytes.NewBuffer(body))\n' +
        '\tif err != nil {\n' +
        '\t\tpanic(err)\n' +
        '\t}\n' +
        '\tdefer resp.Body.Close()\n\n' +
        '\tout, _ := io.ReadAll(resp.Body)\n' +
        '\tfmt.Println(string(out))\n' +
        '}';
    }

    document.getElementById('code-output-box').textContent = code;
  }

  document.querySelectorAll('.lang-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      document.querySelectorAll('.lang-btn').forEach(b => b.classList.remove('active'));
      e.target.classList.add('active');
      activeCodeLang = e.target.getAttribute('data-lang');
      updateCodeSnippet();
    });
  });

  document.getElementById('btn-copy-snippet').addEventListener('click', () => {
    const text = document.getElementById('code-output-box').textContent;
    navigator.clipboard.writeText(text).then(() => {
      const btn = document.getElementById('btn-copy-snippet');
      btn.textContent = 'Copied ✓';
      setTimeout(() => { btn.textContent = 'Copy Code'; }, 2000);
    });
  });

  // History System
  function saveHistoryItem(modelId, payload, status, durationMs) {
    const previewText = (payload.input && payload.input.text) ? payload.input.text : '';
    const item = {
      model: modelId,
      text: previewText.length > 50 ? previewText.substring(0, 50) + '...' : previewText,
      payload: payload,
      status: status,
      durationMs: durationMs,
      timestamp: new Date().toLocaleTimeString()
    };

    recentRequests.unshift(item);
    if (recentRequests.length > 10) recentRequests.pop();
    localStorage.setItem('runtime_recent_requests', JSON.stringify(recentRequests));
    renderHistoryList();
  }

  function renderHistoryList() {
    const container = document.getElementById('history-container');
    if (recentRequests.length === 0) {
      container.innerHTML = '<div style="font-size:12px; color:var(--text-dim); text-align:center; padding:24px;">No executions logged in this session yet.</div>';
      return;
    }

    container.innerHTML = '';
    recentRequests.forEach(item => {
      const row = document.createElement('div');
      row.className = 'history-row';
      const isOk = item.status >= 200 && item.status < 300;
      row.innerHTML =
        '<div style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">' +
          '<strong style="color:var(--accent); font-family:var(--font-mono);">' + escapeHtml(item.model) + ':</strong> ' +
          '<span style="color:var(--text-main);">' + escapeHtml(item.text) + '</span>' +
        '</div>' +
        '<div style="display:flex; align-items:center; gap:6px; flex-shrink:0;">' +
          '<span class="pill" style="color:' + (isOk ? 'var(--success)' : 'var(--error)') + '; font-weight:700;">' + item.status + '</span>' +
          '<span style="font-size:11px; color:var(--text-dim); font-family:var(--font-mono);">' + item.durationMs + 'ms</span>' +
        '</div>';

      row.addEventListener('click', () => {
        const select = document.getElementById('model-select');
        select.value = item.model;
        onModelSelected(item.model);

        if (item.payload) {
          if (item.payload.input && item.payload.input.text !== undefined) {
            document.getElementById('form-input-text').value = item.payload.input.text;
          }
          if (item.payload.options) {
            document.getElementById('opt-temp').value = item.payload.options.temperature !== undefined ? item.payload.options.temperature : '';
            document.getElementById('opt-tokens').value = item.payload.options.max_new_tokens !== undefined ? item.payload.options.max_new_tokens : '';
            document.getElementById('opt-topp').value = item.payload.options.top_p !== undefined ? item.payload.options.top_p : '';
            document.getElementById('opt-topk').value = item.payload.options.top_k !== undefined ? item.payload.options.top_k : '';
          }
          syncFormToJson();
          updateCodeSnippet();
        }
      });

      container.appendChild(row);
    });
  }

  document.getElementById('btn-clear-history').addEventListener('click', () => {
    recentRequests = [];
    localStorage.removeItem('runtime_recent_requests');
    renderHistoryList();
  });

  // Shortcuts: Ctrl+Enter / Cmd+Enter
  window.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      runInference();
    }
  });

  // Initialization
  window.addEventListener('DOMContentLoaded', () => {
    loadModels();
    renderHistoryList();
    updateCodeSnippet();

    if (window.SwaggerUIBundle) {
      window.ui = SwaggerUIBundle({
        url: "/swagger/openapi.json",
        dom_id: '#swagger-ui-container',
        deepLinking: true,
        displayRequestDuration: true,
        tryItOutEnabled: true,
        defaultModelsExpandDepth: -1,
        defaultModelExpandDepth: -1,
        docExpansion: "none",
        filter: true,
        showExtensions: false,
        showCommonExtensions: false,
        presets: [
          SwaggerUIBundle.presets.apis,
          SwaggerUIStandalonePreset
        ],
        plugins: [
          SwaggerUIBundle.plugins.DownloadUrl
        ],
        layout: "BaseLayout"
      });
    }
  });
</script>
</body>
</html>`

// OpenAPISpecJSON defines the canonical universal OpenAPI 3.0.3 specification with dynamic example metadata.
var OpenAPISpecJSON = []byte(`{
  "openapi": "3.0.3",
  "info": {
    "title": "AI Model Runtime API",
    "description": "Universal model-agnostic inference API console. High-performance Pure-Go runtime supporting SafeTensors and ONNX architectures.",
    "version": "1.0.0",
    "contact": {
      "name": "AI Model Runtime Engineering"
    }
  },
  "servers": [
    {
      "url": "/",
      "description": "Current AI Model Runtime Instance"
    }
  ],
  "x-runtime-examples": [
    {
      "id": "turkish-question",
      "name": "🇹🇷 Turkish Question",
      "category": "Text Generation",
      "task": "text-generation",
      "description": "General knowledge question in Turkish",
      "input": {
        "text": "Türkiye'nin başkenti nedir?"
      },
      "options": {
        "max_new_tokens": 64,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50
      }
    },
    {
      "id": "greeting",
      "name": "👋 Simple Greeting",
      "category": "Text Generation",
      "task": "text-generation",
      "description": "Conversational greeting prompt",
      "input": {
        "text": "Merhaba, nasılsın?"
      },
      "options": {
        "max_new_tokens": 32,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50
      }
    },
    {
      "id": "translation",
      "name": "🌍 Translation",
      "category": "Text Generation",
      "task": "text-generation",
      "description": "Translate sentence from Turkish to English",
      "input": {
        "text": "Bugün hava çok güzel. Bu cümleyi İngilizceye çevir:"
      },
      "options": {
        "max_new_tokens": 64,
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 50
      }
    },
    {
      "id": "short-explanation",
      "name": "💡 Short Explanation",
      "category": "Text Generation",
      "task": "text-generation",
      "description": "Concise concept explanation",
      "input": {
        "text": "Yapay zeka nedir? Kısaca açıkla."
      },
      "options": {
        "max_new_tokens": 64,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50
      }
    },
    {
      "id": "creative-writing",
      "name": "✍️ Creative Writing",
      "category": "Text Generation",
      "task": "text-generation",
      "description": "Short creative story generation",
      "input": {
        "text": "Kırmızı bir araba hakkında kısa bir hikaye yaz."
      },
      "options": {
        "max_new_tokens": 128,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 50
      }
    },
    {
      "id": "summarization",
      "name": "📝 Summarization",
      "category": "Text Generation",
      "task": "text-generation",
      "description": "Summarize given Turkish passage",
      "input": {
        "text": "Yapay zeka, makinelerin karmaşık problemleri insanlar gibi çözmesini sağlayan bir bilgisayar bilimi dalıdır. Bu metni tek cümleyle özetle:"
      },
      "options": {
        "max_new_tokens": 48,
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 50
      }
    },
    {
      "id": "embedding-sentence",
      "name": "🔢 Sentence Embedding",
      "category": "Embedding",
      "task": "embedding",
      "description": "Generate dense vector embeddings for semantic search",
      "input": {
        "text": "Yapay zeka teknolojileri son yıllarda hızla gelişmektedir."
      },
      "options": {}
    },
    {
      "id": "embedding-similarity",
      "name": "🔍 Semantic Similarity",
      "category": "Embedding",
      "task": "embedding",
      "description": "Vectorize sentence for semantic comparison",
      "input": {
        "text": "İstanbul Türkiye'nin en kalabalık ve tarihi şehridir."
      },
      "options": {}
    },
    {
      "id": "sentiment-positive",
      "name": "😊 Positive Sentiment",
      "category": "Classification",
      "task": "text-classification",
      "description": "Classify sentiment of positive customer review",
      "input": {
        "text": "Ürün harika, teslimat çok hızlı gerçekleşti ve paketleme mükemmeldi!"
      },
      "options": {}
    },
    {
      "id": "sentiment-negative",
      "name": "🙁 Negative Sentiment",
      "category": "Classification",
      "task": "text-classification",
      "description": "Classify sentiment of negative customer review",
      "input": {
        "text": "Hiç memnun kalmadım, ürün kırık geldi ve müşteri hizmetleri ilgilenmedi."
      },
      "options": {}
    },
    {
      "id": "berturk-ai-detector",
      "name": "🛡️ BERTurk AI Text Detector",
      "category": "Classification",
      "task": "text-classification",
      "description": "Detect whether Turkish text was generated by AI or written by a human",
      "input": {
        "text": "Bu araştırmada, makine öğrenmesi algoritmalarının veri analizi süreçlerindeki doğruluğu kapsamlı deneysel yöntemlerle karşılaştırılmıştır."
      },
      "options": {}
    },
    {
      "id": "qa-document",
      "name": "❓ Question Answering",
      "category": "Question Answering",
      "task": "question-answering",
      "description": "Answer question based on knowledge",
      "input": {
        "text": "Türkiye Cumhuriyeti hangi yılda ilan edilmiştir?"
      },
      "options": {
        "max_new_tokens": 32,
        "temperature": 0.3
      }
    }
  ],
  "tags": [
    {
      "name": "Inference",
      "description": "Universal model inference operations"
    },
    {
      "name": "Models",
      "description": "Model discovery and metadata"
    },
    {
      "name": "Jobs",
      "description": "Asynchronous background inference jobs"
    },
    {
      "name": "System",
      "description": "System health and readiness checks"
    }
  ],
  "paths": {
    "/v1/models/{modelID}/predict": {
      "post": {
        "tags": ["Inference"],
        "summary": "Run inference using the selected model",
        "description": "Run inference using the selected model.",
        "operationId": "predictDefaultVersion",
        "parameters": [
          {
            "name": "modelID",
            "in": "path",
            "required": true,
            "description": "Select the target model to execute inference against",
            "schema": {
              "type": "string"
            }
          }
        ],
        "requestBody": {
          "required": true,
          "description": "Canonical universal request payload",
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/InferenceRequest"
              },
              "example": {
                "input": {
                  "text": "Türkiye'nin başkenti nedir?"
                },
                "options": {
                  "max_new_tokens": 64
                }
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Inference execution succeeded",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/InferenceResponse"
                },
                "example": {
                  "data": {
                    "value": "Ankara..."
                  },
                  "meta": {
                    "request_id": "req-1789902920323-e17728e90192f868",
                    "model": {
                      "id": "text-generation-tr-gpt2",
                      "version": "1.0.0"
                    },
                    "duration_ms": 18
                  }
                }
              }
            }
          },
          "400": { "$ref": "#/components/responses/ErrorResponse" },
          "401": { "$ref": "#/components/responses/ErrorResponse" },
          "404": { "$ref": "#/components/responses/ErrorResponse" },
          "409": { "$ref": "#/components/responses/ErrorResponse" },
          "422": { "$ref": "#/components/responses/ErrorResponse" },
          "429": { "$ref": "#/components/responses/ErrorResponse" },
          "500": { "$ref": "#/components/responses/ErrorResponse" },
          "503": { "$ref": "#/components/responses/ErrorResponse" }
        }
      }
    },
    "/v1/models/{modelID}/versions/{version}/predict": {
      "post": {
        "tags": ["Inference"],
        "summary": "Run inference using a specific model version",
        "description": "Run inference using a specific model version.",
        "operationId": "predictSpecificVersion",
        "parameters": [
          {
            "name": "modelID",
            "in": "path",
            "required": true,
            "description": "Target model identifier",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "version",
            "in": "path",
            "required": true,
            "description": "Target model semantic version",
            "schema": {
              "type": "string",
              "example": "1.0.0"
            }
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/InferenceRequest"
              },
              "example": {
                "input": {
                  "text": "Türkiye'nin başkenti nedir?"
                },
                "options": {
                  "max_new_tokens": 64
                }
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Inference execution succeeded",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/InferenceResponse"
                }
              }
            }
          },
          "400": { "$ref": "#/components/responses/ErrorResponse" },
          "404": { "$ref": "#/components/responses/ErrorResponse" },
          "500": { "$ref": "#/components/responses/ErrorResponse" },
          "503": { "$ref": "#/components/responses/ErrorResponse" }
        }
      }
    },
    "/v1/models": {
      "get": {
        "tags": ["Models"],
        "summary": "Available inference models",
        "description": "Returns list of all available inference models discovered and loaded by the runtime.",
        "operationId": "listModels",
        "responses": {
          "200": {
            "description": "List of available models",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ListModelsResponse"
                },
                "example": {
                  "models": [
                    {
                      "id": "text-generation-tr-gpt2",
                      "version": "1.0.0",
                      "name": "Turkish GPT-2 Medium Instruct",
                      "status": "READY"
                    }
                  ]
                }
              }
            }
          }
        }
      }
    },
    "/v1/models/{modelID}": {
      "get": {
        "tags": ["Models"],
        "summary": "Get model details",
        "description": "Retrieve metadata and lifecycle state for a specific model.",
        "operationId": "getModel",
        "parameters": [
          {
            "name": "modelID",
            "in": "path",
            "required": true,
            "description": "Target model identifier",
            "schema": {
              "type": "string"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Model metadata details",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ModelInfo"
                }
              }
            }
          },
          "404": {
            "$ref": "#/components/responses/ErrorResponse"
          }
        }
      }
    },
    "/v1/jobs": {
      "post": {
        "tags": ["Jobs"],
        "summary": "Submit an asynchronous inference job",
        "description": "Enqueues an inference request for background worker processing.",
        "operationId": "createJob",
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/CreateJobRequest"
              },
              "example": {
                "model_id": "text-generation-tr-gpt2",
                "input": {
                  "text": "Türkiye'nin başkenti nedir?"
                },
                "options": {
                  "max_new_tokens": 64
                }
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Job accepted and queued",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/CreateJobResponse"
                }
              }
            }
          },
          "400": { "$ref": "#/components/responses/ErrorResponse" },
          "500": { "$ref": "#/components/responses/ErrorResponse" }
        }
      }
    },
    "/v1/jobs/{jobID}": {
      "get": {
        "tags": ["Jobs"],
        "summary": "Get job status and output",
        "description": "Inspect the status and retrieve output for an asynchronous job.",
        "operationId": "getJob",
        "parameters": [
          {
            "name": "jobID",
            "in": "path",
            "required": true,
            "description": "Unique job identifier",
            "schema": {
              "type": "string"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Job status and output payload",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Job"
                }
              }
            }
          },
          "404": {
            "$ref": "#/components/responses/ErrorResponse"
          }
        }
      }
    },
    "/health": {
      "get": {
        "tags": ["System"],
        "summary": "Liveness probe",
        "description": "Returns HTTP 200 if the server process is responsive.",
        "operationId": "getHealth",
        "responses": {
          "200": {
            "description": "Service is live",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/HealthResponse"
                },
                "example": {
                  "status": "ok",
                  "timestamp": "2026-09-20T14:30:00Z"
                }
              }
            }
          }
        }
      }
    },
    "/ready": {
      "get": {
        "tags": ["System"],
        "summary": "Readiness probe",
        "description": "Verifies that all models and subsystems are ready to serve inference.",
        "operationId": "getReady",
        "responses": {
          "200": {
            "description": "Service is ready",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ReadyResponse"
                },
                "example": {
                  "status": "ready",
                  "dependencies": {
                    "models": "ready",
                    "queue": "ready",
                    "job_store": "ready"
                  }
                }
              }
            }
          },
          "503": {
            "description": "One or more subsystems are not ready",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ReadyResponse"
                }
              }
            }
          }
        }
      }
    },
    "/predict": {
      "post": {
        "tags": ["Inference"],
        "summary": "Legacy inference endpoint",
        "description": "Backwards compatibility prediction endpoint.",
        "operationId": "predictLegacy",
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/InferenceRequest"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Inference succeeded",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/InferenceResponse"
                }
              }
            }
          }
        }
      }
    }
  },
  "components": {
    "schemas": {
      "InferenceRequest": {
        "type": "object",
        "required": ["input"],
        "properties": {
          "input": {
            "$ref": "#/components/schemas/InferenceInput"
          },
          "options": {
            "$ref": "#/components/schemas/InferenceOptions"
          }
        }
      },
      "InferenceInput": {
        "type": "object",
        "required": ["text"],
        "properties": {
          "text": {
            "type": "string",
            "description": "Input text prompt or query",
            "example": "Türkiye'nin başkenti nedir?"
          }
        }
      },
      "InferenceOptions": {
        "type": "object",
        "description": "Generic inference and decoding options",
        "properties": {
          "max_new_tokens": {
            "type": "integer",
            "description": "Maximum number of new tokens to generate",
            "example": 64
          },
          "temperature": {
            "type": "number",
            "description": "Sampling temperature",
            "example": 0.7
          },
          "top_p": {
            "type": "number",
            "description": "Top-p nucleus sampling probability cutoff",
            "example": 0.9
          },
          "top_k": {
            "type": "integer",
            "description": "Top-k sampling cutoff",
            "example": 50
          },
          "do_sample": {
            "type": "boolean",
            "description": "Whether to use multinomial sampling",
            "example": false
          }
        }
      },
      "InferenceResponse": {
        "type": "object",
        "required": ["data", "meta"],
        "properties": {
          "data": {
            "$ref": "#/components/schemas/ResponseData"
          },
          "meta": {
            "$ref": "#/components/schemas/ResponseMeta"
          }
        }
      },
      "ResponseData": {
        "type": "object",
        "required": ["value"],
        "properties": {
          "value": {
            "description": "Model inference result (string for text generation, float vector for embeddings, or structured output)",
            "example": "Ankara..."
          }
        }
      },
      "ResponseMeta": {
        "type": "object",
        "required": ["request_id", "model", "duration_ms"],
        "properties": {
          "request_id": {
            "type": "string",
            "example": "req-1789902920323-e17728e90192f868"
          },
          "model": {
            "type": "object",
            "required": ["id", "version"],
            "properties": {
              "id": {
                "type": "string",
                "example": "text-generation-tr-gpt2"
              },
              "version": {
                "type": "string",
                "example": "1.0.0"
              }
            }
          },
          "duration_ms": {
            "type": "integer",
            "example": 18
          }
        }
      },
      "ModelInfo": {
        "type": "object",
        "required": ["id", "version", "status"],
        "properties": {
          "id": {
            "type": "string",
            "example": "text-generation-tr-gpt2"
          },
          "version": {
            "type": "string",
            "example": "1.0.0"
          },
          "name": {
            "type": "string",
            "example": "Turkish GPT-2 Medium Instruct"
          },
          "status": {
            "type": "string",
            "example": "READY"
          },
          "task": {
            "type": "string",
            "example": "text-generation"
          },
          "languages": {
            "type": "array",
            "items": { "type": "string" },
            "example": ["tr"]
          }
        }
      },
      "ListModelsResponse": {
        "type": "object",
        "required": ["models"],
        "properties": {
          "models": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/ModelInfo"
            }
          }
        }
      },
      "ErrorDetail": {
        "type": "object",
        "required": ["code", "message"],
        "properties": {
          "code": {
            "type": "string",
            "example": "MODEL_NOT_READY"
          },
          "message": {
            "type": "string",
            "example": "Model is not ready."
          }
        }
      },
      "ErrorMeta": {
        "type": "object",
        "required": ["request_id"],
        "properties": {
          "request_id": {
            "type": "string",
            "example": "req-error-12345"
          }
        }
      },
      "ErrorResponse": {
        "type": "object",
        "required": ["error", "meta"],
        "properties": {
          "error": {
            "$ref": "#/components/schemas/ErrorDetail"
          },
          "meta": {
            "$ref": "#/components/schemas/ErrorMeta"
          }
        }
      },
      "CreateJobRequest": {
        "type": "object",
        "required": ["model_id", "input"],
        "properties": {
          "model_id": {
            "type": "string",
            "example": "text-generation-tr-gpt2"
          },
          "version": {
            "type": "string",
            "example": "1.0.0"
          },
          "input": {
            "$ref": "#/components/schemas/InferenceInput"
          },
          "options": {
            "$ref": "#/components/schemas/InferenceOptions"
          }
        }
      },
      "CreateJobResponse": {
        "type": "object",
        "required": ["job_id", "state", "created_at"],
        "properties": {
          "job_id": {
            "type": "string",
            "example": "job-1789903500-a1b2c3d4"
          },
          "state": {
            "type": "string",
            "example": "queued"
          },
          "created_at": {
            "type": "string",
            "format": "date-time",
            "example": "2026-09-20T14:30:00Z"
          }
        }
      },
      "Job": {
        "type": "object",
        "required": ["id", "model_id", "state", "created_at"],
        "properties": {
          "id": {
            "type": "string",
            "example": "job-1789903500-a1b2c3d4"
          },
          "model_id": {
            "type": "string",
            "example": "text-generation-tr-gpt2"
          },
          "version": {
            "type": "string",
            "example": "1.0.0"
          },
          "state": {
            "type": "string",
            "enum": ["queued", "running", "completed", "failed"],
            "example": "completed"
          },
          "created_at": {
            "type": "string",
            "format": "date-time",
            "example": "2026-09-20T14:30:00Z"
          },
          "data": {
            "$ref": "#/components/schemas/ResponseData"
          },
          "meta": {
            "$ref": "#/components/schemas/ResponseMeta"
          },
          "error": {
            "$ref": "#/components/schemas/ErrorDetail"
          }
        }
      },
      "HealthResponse": {
        "type": "object",
        "required": ["status"],
        "properties": {
          "status": {
            "type": "string",
            "example": "ok"
          }
        }
      },
      "ReadyResponse": {
        "type": "object",
        "required": ["status", "dependencies"],
        "properties": {
          "status": {
            "type": "string",
            "example": "ready"
          },
          "dependencies": {
            "type": "object",
            "additionalProperties": {
              "type": "string"
            },
            "example": {
              "models": "ready",
              "queue": "ready",
              "job_store": "ready"
            }
          }
        }
      }
    },
    "responses": {
      "ErrorResponse": {
        "description": "Standardized canonical API error envelope",
        "content": {
          "application/json": {
            "schema": {
              "$ref": "#/components/schemas/ErrorResponse"
            },
            "example": {
              "error": {
                "code": "MODEL_NOT_READY",
                "message": "Model is not ready."
              },
              "meta": {
                "request_id": "req-error-12345"
              }
            }
          }
        }
      }
    }
  }
}`)

// RegisterSwaggerRoutes registers the Swagger UI and OpenAPI JSON endpoints on the provided ServeMux.
func RegisterSwaggerRoutes(mux *http.ServeMux) {
	// 1. Swagger UI Redirect: /swagger -> /swagger/
	mux.HandleFunc("GET /swagger", func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "/swagger/", http.StatusMovedPermanently)
	})

	// 2. Swagger UI HTML
	mux.HandleFunc("GET /swagger/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/swagger/" && r.URL.Path != "/swagger/index.html" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Header().Set("Cache-Control", "no-cache, no-store, must-revalidate")
		w.Header().Set("Pragma", "no-cache")
		w.Header().Set("Expires", "0")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(SwaggerUIHTML))
	})

	// 3. OpenAPI 3.0.3 Spec JSON
	openAPIHandler := func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		w.Header().Set("Cache-Control", "no-cache, no-store, must-revalidate")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(OpenAPISpecJSON)
	}

	mux.HandleFunc("GET /swagger/openapi.json", openAPIHandler)
	mux.HandleFunc("GET /openapi.json", openAPIHandler)
}

// ValidateOpenAPISpec parses OpenAPISpecJSON to verify syntax correctness.
func ValidateOpenAPISpec() error {
	var v any
	return json.Unmarshal(OpenAPISpecJSON, &v)
}
