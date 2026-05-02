from __future__ import annotations

import copy
import html
import json
import os
import re
from datetime import date
from pathlib import Path
from typing import Any

import httpx


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config" / "app_config.json"
DEFAULT_RISK_DISCLAIMER_ZH = "以上内容仅基于系统检索到的证据生成，不构成投资建议或确定性买卖结论。"
DEFAULT_RISK_DISCLAIMER_EN = (
    "This answer is based only on evidence retrieved by the system and is not investment advice "
    "or a deterministic buy/sell conclusion."
)
DEFAULT_RISK_DISCLAIMER = DEFAULT_RISK_DISCLAIMER_ZH

DEFAULT_CHATBOT_CONFIG: dict[str, Any] = {
    "server": {
        "host": "127.0.0.1",
        "port": 8765,
    },
    "ui": {
        "title": "Financial Chatbot by Group 4.2",
        "input_placeholder": "Ask a financial question, e.g. What do you think about Ping An Insurance (601318.SH)?",
        "submit_text": "Submit",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "chat_path": "/chat/completions",
        "model": "deepseek-chat",
        "api_key": "",
        "timeout_seconds": 30,
    },
    "live_data": {
        "enabled": True,
    },
}


class DeepSeekError(RuntimeError):
    pass


def load_chatbot_config(
    config_path: str | Path | None = None,
    *,
    load_env_file: bool = True,
) -> dict[str, Any]:
    if load_env_file:
        _load_dotenv(ROOT / ".env")

    path = Path(os.getenv("FINANCIAL_CHATBOT_CONFIG") or config_path or DEFAULT_CONFIG_PATH)
    config = copy.deepcopy(DEFAULT_CHATBOT_CONFIG)
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"chatbot config must be a JSON object: {path}")
        _deep_merge(config, loaded)

    _apply_env_overrides(config)
    _coerce_config_types(config)
    return config


def apply_live_data_env(config: dict[str, Any]) -> None:
    enabled = bool((config.get("live_data") or {}).get("enabled", True))
    value = "1" if enabled else "0"
    for name in (
        "QI_USE_LIVE_MARKET",
        "QI_USE_LIVE_NEWS",
        "QI_USE_LIVE_ANNOUNCEMENT",
        "QI_USE_LIVE_MACRO",
    ):
        os.environ.setdefault(name, value)


def render_index_html(config: dict[str, Any]) -> str:
    ui = config.get("ui") or {}
    title = html.escape(str(ui.get("title") or DEFAULT_CHATBOT_CONFIG["ui"]["title"]))
    placeholder = html.escape(str(ui.get("input_placeholder") or DEFAULT_CHATBOT_CONFIG["ui"]["input_placeholder"]))
    submit_text = html.escape(str(ui.get("submit_text") or DEFAULT_CHATBOT_CONFIG["ui"]["submit_text"]))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: "Aptos Display", "Segoe UI Variable", "Trebuchet MS", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(79, 70, 229, 0.18), transparent 32rem),
        radial-gradient(circle at bottom right, rgba(14, 165, 233, 0.16), transparent 30rem),
        linear-gradient(135deg, #f2f6ff 0%, #e9eefc 48%, #f8fbff 100%);
      color: #172033;
      --primary: #4f46e5;
      --primary-dark: #4338ca;
      --primary-soft: #eef2ff;
      --surface: rgba(255, 255, 255, 0.92);
      --chat-bg: #f8fafd;
      --text-muted: #64748b;
      --border: #e5eaf2;
      --shadow-lg: 0 28px 70px rgba(30, 41, 59, 0.18);
      --shadow-sm: 0 8px 24px rgba(30, 41, 59, 0.08);
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      min-height: 100vh;
      display: flex;
      justify-content: center;
      align-items: center;
      padding: 24px;
    }}
    .chat-app {{
      width: min(920px, 100%);
      height: min(860px, 92vh);
      background: var(--surface);
      border: 1px solid rgba(255, 255, 255, 0.72);
      border-radius: 28px;
      box-shadow: var(--shadow-lg);
      display: flex;
      flex-direction: column;
      overflow: hidden;
      backdrop-filter: blur(16px);
      animation: panelIn 0.38s ease-out;
    }}
    @keyframes panelIn {{
      from {{ opacity: 0; transform: translateY(18px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
    .chat-header {{
      padding: 20px 28px;
      border-bottom: 1px solid var(--border);
      background: rgba(255, 255, 255, 0.9);
      display: flex;
      align-items: center;
      gap: 14px;
      flex-shrink: 0;
    }}
    .brand-mark {{
      width: 40px;
      height: 40px;
      border-radius: 14px;
      background: linear-gradient(135deg, var(--primary), #0ea5e9);
      box-shadow: 0 10px 24px rgba(79, 70, 229, 0.26);
      display: grid;
      place-items: center;
      color: white;
      font-weight: 800;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(20px, 3vw, 28px);
      letter-spacing: -0.02em;
      line-height: 1.1;
    }}
    .status-pill {{
      margin-left: auto;
      color: var(--text-muted);
      background: #f1f5f9;
      border-radius: 999px;
      padding: 6px 14px;
      font-size: 13px;
      font-weight: 650;
      white-space: nowrap;
    }}
    .status-pill.busy {{
      color: #4338ca;
      background: var(--primary-soft);
    }}
    .chat-messages {{
      flex: 1;
      overflow-y: auto;
      padding: 26px 26px 18px;
      background:
        linear-gradient(rgba(255,255,255,0.55), rgba(255,255,255,0.55)),
        var(--chat-bg);
      display: flex;
      flex-direction: column;
      gap: 18px;
      scroll-behavior: smooth;
    }}
    .chat-messages::-webkit-scrollbar {{
      width: 7px;
    }}
    .chat-messages::-webkit-scrollbar-thumb {{
      background: #cbd5e1;
      border-radius: 999px;
    }}
    .msg-row {{
      display: flex;
      flex-direction: column;
      max-width: min(780px, 88%);
      animation: messageIn 0.24s ease-out;
    }}
    @keyframes messageIn {{
      from {{ opacity: 0; transform: translateY(8px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
    .msg-user {{
      align-self: flex-end;
      align-items: flex-end;
    }}
    .msg-bot {{
      align-self: flex-start;
      align-items: flex-start;
      width: 100%;
    }}
    .bubble {{
      position: relative;
      border-radius: 20px;
      padding: 15px 18px;
      line-height: 1.7;
      font-size: 15px;
      word-break: break-word;
      box-shadow: var(--shadow-sm);
    }}
    .bubble-user {{
      color: white;
      background: linear-gradient(135deg, var(--primary), var(--primary-dark));
      border-bottom-right-radius: 8px;
      padding-right: 42px;
    }}
    .bubble-bot {{
      color: #172033;
      background: white;
      border: 1px solid var(--border);
      border-bottom-left-radius: 8px;
      padding-right: 42px;
      width: 100%;
    }}
    .copy-btn {{
      position: absolute;
      top: 9px;
      right: 9px;
      border: 0;
      border-radius: 999px;
      padding: 4px 8px;
      color: #64748b;
      background: rgba(241, 245, 249, 0.9);
      font-size: 11px;
      font-weight: 700;
      cursor: pointer;
      opacity: 0;
      transition: opacity 0.18s ease, background 0.18s ease;
    }}
    .bubble:hover .copy-btn {{
      opacity: 1;
    }}
    .copy-btn:hover {{
      background: #e2e8f0;
    }}
    .answer-text {{
      white-space: pre-wrap;
    }}
    .section-title {{
      margin: 18px 0 8px;
      font-weight: 800;
      color: #334155;
      font-size: 14px;
      letter-spacing: 0.01em;
    }}
    .key-points {{
      margin: 0;
      padding-left: 20px;
    }}
    .key-points li {{
      margin-bottom: 5px;
    }}
    .disclaimer {{
      margin-top: 16px;
      color: var(--text-muted);
      font-size: 13px;
      border-left: 3px solid #cbd5e1;
      padding: 10px 12px;
      background: #f8fafc;
      border-radius: 0 12px 12px 0;
    }}
    .evidence-list {{
      list-style: none;
      padding: 0;
      margin: 0;
      display: grid;
      gap: 10px;
    }}
    .evidence-list li {{
      padding: 12px 14px;
      background: #f8fafc;
      border: 1px solid #eef2f7;
      border-radius: 14px;
    }}
    .evidence-title {{
      font-weight: 750;
      margin-bottom: 4px;
    }}
    .evidence-list a {{
      color: var(--primary);
      text-decoration: none;
      font-weight: 750;
      background: var(--primary-soft);
      border-radius: 999px;
      padding: 4px 10px;
      display: inline-block;
    }}
    .evidence-list a:hover {{
      background: #dfe7ff;
    }}
    .evidence-meta {{
      color: var(--text-muted);
      font-size: 12px;
      margin-right: 8px;
    }}
    .typing-dots {{
      display: inline-flex;
      gap: 5px;
      align-items: center;
      min-height: 24px;
    }}
    .typing-dots span {{
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--primary);
      animation: typing 1s infinite ease-in-out;
    }}
    .typing-dots span:nth-child(2) {{
      animation-delay: 0.15s;
    }}
    .typing-dots span:nth-child(3) {{
      animation-delay: 0.3s;
    }}
    @keyframes typing {{
      0%, 80%, 100% {{ opacity: 0.35; transform: translateY(0); }}
      40% {{ opacity: 1; transform: translateY(-4px); }}
    }}
    .error-box {{
      color: #991b1b;
      background: #fef2f2;
      border: 1px solid #fecaca;
      border-radius: 16px;
      padding: 12px 14px;
      line-height: 1.5;
    }}
    .composer {{
      display: flex;
      gap: 12px;
      align-items: stretch;
      padding: 18px;
      border-top: 1px solid var(--border);
      background: rgba(255, 255, 255, 0.92);
    }}
    input {{
      flex: 1;
      font-size: 16px;
      padding: 14px 16px;
      border: 2px solid var(--border);
      border-radius: 16px;
      outline: none;
      background: #f8fafc;
      transition: border-color 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
    }}
    input:focus {{
      border-color: var(--primary);
      background: white;
      box-shadow: 0 0 0 4px rgba(79, 70, 229, 0.11);
    }}
    button {{
      border: 0;
      border-radius: 16px;
      padding: 0 24px;
      background: linear-gradient(135deg, var(--primary), var(--primary-dark));
      color: #fff;
      font-size: 15px;
      font-weight: 800;
      cursor: pointer;
      white-space: nowrap;
      transition: transform 0.18s ease, box-shadow 0.18s ease;
    }}
    button:hover:not(:disabled) {{
      transform: translateY(-1px);
      box-shadow: 0 10px 24px rgba(79, 70, 229, 0.25);
    }}
    button:disabled {{
      background: #94a3b8;
      cursor: not-allowed;
      transform: none;
      box-shadow: none;
    }}
    @media (max-width: 640px) {{
      body {{ padding: 12px; }}
      .chat-app {{ height: calc(100vh - 24px); border-radius: 22px; }}
      .chat-header {{ padding: 16px; }}
      .brand-mark {{ display: none; }}
      .status-pill {{ display: none; }}
      .chat-messages {{ padding: 18px 14px 12px; }}
      .msg-row {{ max-width: 95%; }}
      .composer {{ flex-direction: column; padding: 14px; }}
      button {{ min-height: 48px; }}
    }}
  </style>
</head>
<body>
  <main class="chat-app">
    <header class="chat-header">
      <div class="brand-mark">F</div>
      <h1>{title}</h1>
      <div class="status-pill" id="status-pill">Ready</div>
    </header>
    <section class="chat-messages" id="chat-messages" aria-live="polite">
      <div class="msg-row msg-bot">
        <div class="bubble bubble-bot">
          <div class="answer-text">Hello. I am your financial chatbot. Ask a question and I will analyze it using retrieved evidence, live data when available, and a risk-aware summary.</div>
        </div>
      </div>
    </section>
    <form class="composer" id="chat-form">
      <input id="query-input" name="query" type="text" autocomplete="off" placeholder="{placeholder}" />
      <button id="submit-button" type="submit">{submit_text}</button>
    </form>
  </main>
  <script>
    const form = document.getElementById("chat-form");
    const input = document.getElementById("query-input");
    const button = document.getElementById("submit-button");
    const statusPill = document.getElementById("status-pill");
    const messagesEl = document.getElementById("chat-messages");
    const submitText = button.textContent;

    function escapeHtml(value) {{
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }}

    function scrollToBottom() {{
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }}

    function setStatus(text, busy = false) {{
      statusPill.textContent = text;
      statusPill.classList.toggle("busy", busy);
    }}

    async function copyBubbleText(buttonEl) {{
      const bubble = buttonEl.closest(".bubble");
      if (!bubble) return;
      const clone = bubble.cloneNode(true);
      const copyButton = clone.querySelector(".copy-btn");
      if (copyButton) copyButton.remove();
      const text = clone.innerText.trim();
      try {{
        await navigator.clipboard.writeText(text);
        buttonEl.textContent = "Copied";
        setTimeout(() => {{ buttonEl.textContent = "Copy"; }}, 1200);
      }} catch (_) {{
        buttonEl.textContent = "Copy failed";
        setTimeout(() => {{ buttonEl.textContent = "Copy"; }}, 1200);
      }}
    }}

    function attachCopyButton(bubble) {{
      const copyButton = document.createElement("button");
      copyButton.type = "button";
      copyButton.className = "copy-btn";
      copyButton.textContent = "Copy";
      copyButton.addEventListener("click", () => copyBubbleText(copyButton));
      bubble.appendChild(copyButton);
    }}

    function renderKeyPoints(items) {{
      if (!items || !items.length) return "";
      return `<ul class="key-points">${{items.map(item => `<li>${{escapeHtml(item)}}</li>`).join("")}}</ul>`;
    }}

    function renderEvidenceSources(items) {{
      if (!items || !items.length) return "";
      const rows = items.map(item => {{
        const title = escapeHtml(item.title || item.source_name || item.source_type || "Unknown source");
        const metaParts = [item.source_name, item.source_type].filter(Boolean).map(escapeHtml);
        const meta = metaParts.length ? `<span class="evidence-meta">${{metaParts.join(" · ")}}</span>` : "";
        const url = item.source_url ? String(item.source_url) : "";
        const link = url
          ? `<a href="${{escapeHtml(url)}}" target="_blank" rel="noopener noreferrer">Open webpage</a>`
          : `<span class="evidence-meta">No web link</span>`;
        return `<li><div class="evidence-title">${{title}}</div><div>${{meta}} ${{link}}</div></li>`;
      }}).join("");
      return `<ul class="evidence-list">${{rows}}</ul>`;
    }}

    function buildBotContent(data) {{
      const answer = `<div class="answer-text">${{escapeHtml(data.answer || "")}}</div>`;
      const keyPoints = data.key_points && data.key_points.length
        ? `<div class="section-title">Key Points</div>${{renderKeyPoints(data.key_points)}}`
        : "";
      const evidence = data.evidence_sources && data.evidence_sources.length
        ? `<div class="section-title">Evidence Sources</div>${{renderEvidenceSources(data.evidence_sources)}}`
        : "";
      const disclaimer = data.risk_disclaimer
        ? `<div class="disclaimer">${{escapeHtml(data.risk_disclaimer)}}</div>`
        : "";
      return `${{answer}}${{keyPoints}}${{evidence}}${{disclaimer}}`;
    }}

    function appendMessage(kind, htmlContent, options = {{}}) {{
      const rawText = Boolean(options.rawText);
      const row = document.createElement("div");
      row.className = `msg-row ${{kind === "user" ? "msg-user" : "msg-bot"}}`;
      const bubble = document.createElement("div");
      bubble.className = `bubble ${{kind === "user" ? "bubble-user" : "bubble-bot"}}`;
      bubble.innerHTML = rawText ? escapeHtml(htmlContent) : htmlContent;
      attachCopyButton(bubble);
      row.appendChild(bubble);
      messagesEl.appendChild(row);
      scrollToBottom();
      return row;
    }}

    function appendTyping() {{
      const row = document.createElement("div");
      row.className = "msg-row msg-bot";
      row.id = "typing-row";
      row.innerHTML = '<div class="bubble bubble-bot"><div class="typing-dots"><span></span><span></span><span></span></div></div>';
      messagesEl.appendChild(row);
      scrollToBottom();
      return row;
    }}

    function removeTyping() {{
      const typing = document.getElementById("typing-row");
      if (typing) typing.remove();
    }}

    function appendError(message) {{
      const row = document.createElement("div");
      row.className = "msg-row msg-bot";
      row.innerHTML = `<div class="error-box">${{escapeHtml(message)}}</div>`;
      messagesEl.appendChild(row);
      scrollToBottom();
    }}

    function setLoading(loading) {{
      button.disabled = loading;
      input.disabled = loading;
      button.textContent = loading ? "Analyzing..." : submitText;
      setStatus(loading ? "Analyzing" : "Ready", loading);
    }}

    form.addEventListener("submit", async (event) => {{
      event.preventDefault();
      const query = input.value.trim();
      if (!query) {{
        appendError("Please enter a question.");
        return;
      }}
      input.value = "";
      appendMessage("user", query, {{ rawText: true }});
      appendTyping();
      setLoading(true);
      try {{
        const res = await fetch("/chat", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{ query }})
        }});
        const data = await res.json();
        removeTyping();
        if (!res.ok) {{
          throw new Error(data.detail || "Request failed");
        }}
        appendMessage("bot", buildBotContent(data));
        if (data.llm && data.llm.status === "fallback") {{
          appendMessage("bot", '<div class="answer-text">Using structured-summary fallback.</div>');
        }}
        setStatus("Complete");
      }} catch (error) {{
        removeTyping();
        const message = error instanceof TypeError && String(error.message || "").toLowerCase().includes("fetch")
          ? "Cannot reach the local server. Keep the start.bat command window open, then refresh http://127.0.0.1:8765/."
          : (error.message || error);
        appendError(message);
        setStatus("Error");
      }} finally {{
        setLoading(false);
        input.focus();
      }}
    }});

    const welcomeBubble = messagesEl.querySelector(".bubble");
    if (welcomeBubble) attachCopyButton(welcomeBubble);
    input.focus();
  </script>
</body>
</html>"""


class DeepSeekClient:
    def __init__(self, config: dict[str, Any], *, http_client: Any | None = None) -> None:
        deepseek = config.get("deepseek") or {}
        self.base_url = str(deepseek.get("base_url") or DEFAULT_CHATBOT_CONFIG["deepseek"]["base_url"]).rstrip("/")
        self.chat_path = str(deepseek.get("chat_path") or DEFAULT_CHATBOT_CONFIG["deepseek"]["chat_path"])
        self.model = str(deepseek.get("model") or DEFAULT_CHATBOT_CONFIG["deepseek"]["model"])
        self.api_key = str(deepseek.get("api_key") or "")
        self.timeout_seconds = int(deepseek.get("timeout_seconds") or DEFAULT_CHATBOT_CONFIG["deepseek"]["timeout_seconds"])
        self.http_client = http_client

    def generate(self, record: dict[str, Any]) -> dict[str, Any]:
        if not self._has_api_key():
            raise DeepSeekError("DeepSeek API key is not configured")

        payload = compact_evidence_payload(record)
        query = str(payload.get("query") or "")
        response = self._post_chat_completion(make_answer_messages(payload))
        content = response["choices"][0]["message"]["content"]
        parsed = _parse_json_object(content)
        if not answer_matches_language(parsed, query):
            response = self._post_chat_completion(make_answer_language_repair_messages(query, parsed))
            content = response["choices"][0]["message"]["content"]
            parsed = _parse_json_object(content)
            if not answer_matches_language(parsed, query):
                raise DeepSeekError("DeepSeek response language did not match the query language")
        return normalize_llm_answer(parsed, record, model=self.model)

    def _post_chat_completion(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        url = f"{self.base_url}{self.chat_path if self.chat_path.startswith('/') else '/' + self.chat_path}"
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        client = self.http_client or httpx.Client(timeout=self.timeout_seconds)
        close_client = self.http_client is None
        try:
            result = client.post(url, headers=headers, json=body)
            result.raise_for_status()
            data = result.json()
        except Exception as exc:  # noqa: BLE001
            raise DeepSeekError(f"DeepSeek API request failed: {exc}") from exc
        finally:
            if close_client:
                client.close()
        if not isinstance(data, dict) or not data.get("choices"):
            raise DeepSeekError("DeepSeek API response is missing choices")
        return data

    def _has_api_key(self) -> bool:
        stripped = self.api_key.strip()
        return bool(stripped and stripped.lower() not in {"your_deepseek_api_key_here", "changeme"})


def build_chatbot_response(
    *,
    query: str,
    pipeline_result: dict[str, Any],
    deepseek_client: DeepSeekClient,
    progress: Any | None = None,
) -> dict[str, Any]:
    record = {
        "status": "ok",
        "query": query,
        "nlu_result": pipeline_result["nlu_result"],
        "retrieval_result": pipeline_result["retrieval_result"],
    }
    try:
        if progress:
            progress("DeepSeek: sending compact evidence for response polishing...")
        answer = deepseek_client.generate(record)
        if progress:
            progress("DeepSeek: response received and normalized.")
        llm_status = {"provider": "deepseek", "model": deepseek_client.model, "status": "ok", "error": None}
    except Exception as exc:  # noqa: BLE001
        if progress:
            progress(f"DeepSeek: unavailable; using structured-summary fallback. reason={exc}")
        answer = template_answer(record, fallback_reason=str(exc))
        llm_status = {"provider": "deepseek", "model": deepseek_client.model, "status": "fallback", "error": str(exc)}
    if progress:
        progress("Step 3/3: applying market freshness guard and formatting evidence sources...")
    answer = apply_market_freshness_guard(answer, record)
    evidence_sources = build_evidence_sources(record, answer.get("evidence_used") or [])
    if progress:
        progress(f"Step 3/3 complete: evidence_sources={len(evidence_sources)}")
    return {
        **answer,
        "evidence_sources": evidence_sources,
        "llm": llm_status,
        "nlu_result": pipeline_result["nlu_result"],
        "retrieval_result": pipeline_result["retrieval_result"],
    }


def compact_evidence_payload(record: dict[str, Any]) -> dict[str, Any]:
    retrieval = record.get("retrieval_result") or {}
    documents = retrieval.get("documents") or []
    structured_data = retrieval.get("structured_data") or []
    query = record.get("query") or (record.get("nlu_result") or {}).get("raw_query")
    response_language = detect_query_language(str(query or ""))
    return {
        "query": query,
        "response_language": response_language,
        "nlu_result": {
            "question_style": (record.get("nlu_result") or {}).get("question_style"),
            "product_type": (record.get("nlu_result") or {}).get("product_type"),
            "intent_labels": (record.get("nlu_result") or {}).get("intent_labels"),
            "topic_labels": (record.get("nlu_result") or {}).get("topic_labels"),
            "entities": (record.get("nlu_result") or {}).get("entities"),
            "risk_flags": (record.get("nlu_result") or {}).get("risk_flags"),
        },
        "retrieval_result": {
            "retrieval_confidence": retrieval.get("retrieval_confidence"),
            "warnings": retrieval.get("warnings") or [],
            "coverage": retrieval.get("coverage") or {},
            "analysis_summary": retrieval.get("analysis_summary") or {},
            "structured_data": [_compact_structured_item(item) for item in structured_data[:10]],
            "documents": [_compact_document(item) for item in documents[:8]],
        },
        "output_contract": {
            "answer": "string",
            "key_points": ["string"],
            "risk_disclaimer": "string",
            "evidence_used": ["evidence_id"],
        },
    }


def make_answer_messages(payload: dict[str, Any]) -> list[dict[str, str]]:
    query = str(payload.get("query") or "")
    target_language = _target_language_name(detect_query_language(query))
    return [
        {
            "role": "system",
            "content": (
                "You are the response-polishing layer for a financial chatbot. "
                "Answer only from the JSON evidence provided by the user. Do not invent market prices, "
                "financial data, news, macro facts, statistics, or investment conclusions. "
                "Return exactly one strict JSON object with these keys: "
                "answer, key_points, risk_disclaimer, evidence_used. "
                "Preserve evidence_used IDs exactly and do not add IDs that are absent from the evidence. "
                f"All natural-language strings in answer, key_points, and risk_disclaimer must be in {target_language}. "
                "If the target language is Chinese, use Simplified Chinese. If it is English, write fluent English "
                "even when company names or source names in the evidence are Chinese."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        },
    ]


def answer_matches_language(output: dict[str, Any], query: str) -> bool:
    expected_language = detect_query_language(query)
    text_values: list[str] = []
    for key in ("answer", "risk_disclaimer"):
        value = str(output.get(key) or "").strip()
        if value:
            text_values.append(value)
    for key in ("key_points", "limitations"):
        values = output.get(key)
        if isinstance(values, list):
            text_values.extend(str(item).strip() for item in values if str(item).strip())
    return all(_text_matches_language(value, expected_language) for value in text_values)


def make_answer_language_repair_messages(query: str, output: dict[str, Any]) -> list[dict[str, str]]:
    target_language = _target_language_name(detect_query_language(query))
    return [
        {
            "role": "system",
            "content": (
                "You rewrite answer-generation JSON into the user's language. "
                "Return only one valid JSON object with exactly these keys: "
                "answer, key_points, risk_disclaimer, evidence_used. "
                "Preserve the financial meaning, risk caution, and evidence_used IDs. "
                f"All natural-language strings must be in {target_language}."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "query": query,
                    "target_language": target_language,
                    "current_output": output,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    ]


def detect_query_language(text: str) -> str:
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", text or ""))
    latin_count = len(re.findall(r"[A-Za-z]", text or ""))
    if cjk_count and cjk_count >= max(2, latin_count * 0.4):
        return "zh"
    if latin_count:
        return "en"
    if cjk_count:
        return "zh"
    return "zh"


def _target_language_name(language: str) -> str:
    return "Chinese" if language == "zh" else "English"


def _default_risk_disclaimer(record: dict[str, Any]) -> str:
    query = str(record.get("query") or (record.get("nlu_result") or {}).get("raw_query") or "")
    return DEFAULT_RISK_DISCLAIMER_EN if detect_query_language(query) == "en" else DEFAULT_RISK_DISCLAIMER_ZH


def _text_matches_language(text: str, expected_language: str) -> bool:
    signal = _language_signal(text)
    if signal in {"neutral", "mixed"}:
        return True
    if expected_language == "en":
        return signal != "zh"
    return signal != "en"


def _language_signal(text: str) -> str:
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", text or ""))
    latin_count = len(re.findall(r"[A-Za-z]", text or ""))
    if cjk_count == 0 and latin_count < 4:
        return "neutral"
    if cjk_count >= max(2, int(latin_count * 0.2)):
        return "zh"
    if latin_count >= max(4, cjk_count * 4):
        return "en"
    return "mixed"


def normalize_llm_answer(output: dict[str, Any], record: dict[str, Any], *, model: str) -> dict[str, Any]:
    answer = str(output.get("answer") or "").strip()
    if not answer:
        raise DeepSeekError(f"{model} returned an empty answer")
    key_points_raw = output.get("key_points") or []
    key_points = [str(item).strip() for item in key_points_raw if str(item).strip()] if isinstance(key_points_raw, list) else []
    evidence_used_raw = output.get("evidence_used") or []
    allowed_ids = _evidence_ids(record)
    evidence_used = [
        str(item).strip()
        for item in evidence_used_raw
        if str(item).strip() and (not allowed_ids or str(item).strip() in allowed_ids)
    ]
    return {
        "answer": answer,
        "key_points": key_points[:6],
        "risk_disclaimer": str(output.get("risk_disclaimer") or _default_risk_disclaimer(record)).strip(),
        "evidence_used": evidence_used[:8],
    }


def _template_key_points(
    *,
    language: str,
    analysis_summary: dict[str, Any],
    structured_data: list[Any],
    documents: list[Any],
    warnings: list[Any],
) -> list[str]:
    key_points: list[str] = []
    if analysis_summary:
        readiness = analysis_summary.get("data_readiness") or {}
        ready_labels = [label for label, ready in readiness.items() if isinstance(ready, bool) and ready]
        if language == "en":
            coverage = ", ".join(ready_labels) if ready_labels else "basic evidence"
            key_points.append(f"Generated a structured analysis summary covering: {coverage}.")
        else:
            key_points.append(f"已生成结构化分析摘要，覆盖：{', '.join(ready_labels) if ready_labels else '基础证据'}。")
    if structured_data:
        if language == "en":
            key_points.append(f"Loaded {len(structured_data)} structured data item(s).")
        else:
            key_points.append(f"已读取 {len(structured_data)} 条结构化数据。")
    if documents:
        source_types = sorted({str(doc.get("source_type") or "document") for doc in documents if isinstance(doc, dict)})
        if language == "en":
            source_text = ", ".join(source_types) if source_types else "documents"
            key_points.append(f"Retrieved {len(documents)} text evidence item(s), including source types: {source_text}.")
        else:
            key_points.append(f"已检索 {len(documents)} 条文本证据，来源类型包括：{', '.join(source_types)}。")
    if warnings:
        if language == "en":
            key_points.append("Data warnings are present; inspect retrieval_result.warnings for details.")
        else:
            key_points.append(f"数据提示：{'; '.join(str(item) for item in warnings[:3])}。")
    if not key_points:
        if language == "en":
            key_points.append("Available evidence is limited; consider adding a clearer target, time range, or data source.")
        else:
            key_points.append("当前可用证据有限，建议补充更明确的标的、时间范围或数据源。")
    return key_points


def template_answer(record: dict[str, Any], *, fallback_reason: str | None = None) -> dict[str, Any]:
    query = str(record.get("query") or "")
    retrieval = record.get("retrieval_result") or {}
    documents = retrieval.get("documents") or []
    structured_data = retrieval.get("structured_data") or []
    warnings = retrieval.get("warnings") or []
    analysis_summary = retrieval.get("analysis_summary") or {}

    language = detect_query_language(query)
    key_points = _template_key_points(
        language=language,
        analysis_summary=analysis_summary,
        structured_data=structured_data,
        documents=documents,
        warnings=warnings,
    )

    if language == "en":
        prefix = "Model refinement failed; the following is a structured summary."
        if fallback_reason and "API key" in fallback_reason:
            prefix = "DeepSeek API is not configured; the following is a structured summary."
        answer = (
            f'{prefix}\n\nFor "{query}", the system completed financial question understanding '
            f"and evidence retrieval. {' '.join(key_points)}"
        )
    else:
        prefix = "模型润色失败，以下为结构化摘要。"
        if fallback_reason and "API key" in fallback_reason:
            prefix = "DeepSeek API 未配置，以下为结构化摘要。"
        answer = f"针对“{query}”，系统已完成金融问题理解和证据检索。{''.join(key_points)}"
        answer = f"{prefix}\n\n{answer}"
    return {
        "answer": answer,
        "key_points": key_points,
        "risk_disclaimer": _default_risk_disclaimer(record),
        "evidence_used": sorted(_evidence_ids(record))[:8],
    }


def apply_market_freshness_guard(answer: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    query = str(record.get("query") or (record.get("nlu_result") or {}).get("raw_query") or "")
    if not _asks_for_current_market_data(query):
        return answer

    language = detect_query_language(query)
    today_date = date.today()
    today = today_date.isoformat()
    market_item = _first_market_item(record)
    if _is_known_non_trading_day(today_date):
        return _non_trading_day_market_answer(
            answer=answer,
            market_item=market_item,
            language=language,
            today=today,
        )
    if not market_item:
        guarded = dict(answer)
        if language == "en":
            guarded["answer"] = (
                f"Unable to retrieve today's ({today}) real-time market data, so I cannot determine "
                "whether it rose or fell today. The current response can only provide background "
                "based on non-real-time evidence."
            )
            key_point = f"Today's ({today}) real-time quote was not retrieved."
        else:
            guarded["answer"] = (
                f"未获取到今日（{today}）实时行情，因此不能判断今天是否上涨或下跌。"
                "当前回复仅能基于非实时证据做背景说明。"
            )
            key_point = f"今日（{today}）实时行情获取失败。"
        guarded["key_points"] = _prepend_unique(
            answer.get("key_points") if isinstance(answer.get("key_points"), list) else [],
            key_point,
        )
        guarded["evidence_used"] = []
        return guarded

    payload = market_item.get("payload") if isinstance(market_item.get("payload"), dict) else {}
    trade_date = str(payload.get("trade_date") or market_item.get("as_of") or "").strip()
    if not trade_date or trade_date[:10] == today:
        return answer

    symbol = payload.get("symbol") or market_item.get("evidence_id") or "该标的"
    close = payload.get("close") if payload.get("close") is not None else payload.get("price")
    pct_change = payload.get("pct_change_1d")
    guarded = dict(answer)
    if language == "en":
        close_text = f"; latest available price/close is {close}" if close is not None else ""
        pct_text = f"; percent change is {pct_change}%" if pct_change is not None else ""
        guarded["answer"] = (
            f"Unable to retrieve today's ({today}) real-time quote; the latest available market date "
            f"is {trade_date[:10]}. {symbol}{close_text}{pct_text}. Therefore, I cannot use this data "
            f"to determine whether it rose or fell today ({today})."
        )
        key_point = f"Today's quote was not retrieved; the latest available market date is {trade_date[:10]}."
    else:
        close_text = f"，最新可用价格/收盘价为 {close}" if close is not None else ""
        pct_text = f"，涨跌幅为 {pct_change}%" if pct_change is not None else ""
        guarded["answer"] = (
            f"未获取到今日（{today}）实时行情；系统最新可用行情日期是 {trade_date[:10]}。"
            f"{symbol}{close_text}{pct_text}。"
            f"因此不能据此判断今天（{today}）是否上涨或下跌。"
        )
        key_point = f"今日行情未获取成功，最新可用行情日期为 {trade_date[:10]}。"
    guarded["key_points"] = _prepend_unique(
        answer.get("key_points") if isinstance(answer.get("key_points"), list) else [],
        key_point,
    )
    guarded["evidence_used"] = [str(market_item.get("evidence_id"))] if market_item.get("evidence_id") else []
    return guarded


def _non_trading_day_market_answer(
    *,
    answer: dict[str, Any],
    market_item: dict[str, Any] | None,
    language: str,
    today: str,
) -> dict[str, Any]:
    guarded = dict(answer)
    if not market_item:
        if language == "en":
            guarded["answer"] = (
                f"Today ({today}) is not a regular A-share trading day, so no same-day market move "
                "is expected. I also could not retrieve a recent trading-day quote for this request."
            )
            key_point = f"Today ({today}) is not a regular A-share trading day."
        else:
            guarded["answer"] = (
                f"今天（{today}）不是 A 股常规交易日，因此没有当日涨跌行情。"
                "本次请求也未获取到最近交易日行情。"
            )
            key_point = f"今天（{today}）不是 A 股常规交易日。"
        guarded["key_points"] = _prepend_unique(
            answer.get("key_points") if isinstance(answer.get("key_points"), list) else [],
            key_point,
        )
        guarded["evidence_used"] = []
        return guarded

    payload = market_item.get("payload") if isinstance(market_item.get("payload"), dict) else {}
    trade_date = str(payload.get("trade_date") or market_item.get("as_of") or "").strip()
    symbol = payload.get("symbol") or market_item.get("evidence_id") or ("the target" if language == "en" else "该标的")
    close = payload.get("close") if payload.get("close") is not None else payload.get("price")
    pct_change = payload.get("pct_change_1d")
    if language == "en":
        date_text = f" The latest available trading-day quote is from {trade_date[:10]}." if trade_date else ""
        close_text = f" Latest available price/close: {close}." if close is not None else ""
        pct_text = f" Change on that trading day: {pct_change}%." if pct_change is not None else ""
        guarded["answer"] = (
            f"Today ({today}) is not a regular A-share trading day, so there is no same-day trading move. "
            f"{symbol}.{date_text}{close_text}{pct_text}"
        )
        key_point = f"Today ({today}) is not a regular A-share trading day; using the latest available trading-day quote."
    else:
        date_text = f"最新可用交易日行情日期为 {trade_date[:10]}。" if trade_date else ""
        close_text = f"最新可用价格/收盘价为 {close}。" if close is not None else ""
        pct_text = f"该交易日涨跌幅为 {pct_change}%。" if pct_change is not None else ""
        guarded["answer"] = (
            f"今天（{today}）不是 A 股常规交易日，因此没有当日涨跌行情。"
            f"{symbol}。{date_text}{close_text}{pct_text}"
        )
        key_point = f"今天（{today}）不是 A 股常规交易日，已使用最新可用交易日行情。"
    guarded["key_points"] = _prepend_unique(
        answer.get("key_points") if isinstance(answer.get("key_points"), list) else [],
        key_point,
    )
    guarded["evidence_used"] = [str(market_item.get("evidence_id"))] if market_item.get("evidence_id") else []
    return guarded


def build_evidence_sources(record: dict[str, Any], evidence_used: list[str] | None = None, *, limit: int = 8) -> list[dict[str, Any]]:
    retrieval = record.get("retrieval_result") or {}
    items = (retrieval.get("documents") or []) + (retrieval.get("structured_data") or [])
    by_id = {
        str(item.get("evidence_id") or ""): item
        for item in items
        if str(item.get("evidence_id") or "").strip()
    }
    selected: list[dict[str, Any]] = []
    for evidence_id in evidence_used or []:
        item = by_id.get(str(evidence_id))
        if item:
            selected.append(item)
    if not selected:
        selected = items[:limit]
    return [_source_display_item(item) for item in selected[:limit]]


def _asks_for_current_market_data(query: str) -> bool:
    lower_query = query.lower()
    return any(term in query for term in ("今天", "今日", "现在", "当前", "实时", "最新")) or any(
        term in lower_query
        for term in (
            "today",
            "current",
            "right now",
            "now",
            "real-time",
            "realtime",
            "latest",
        )
    )


def _is_known_non_trading_day(day: date) -> bool:
    return day.weekday() >= 5 or (day.month, day.day) in {
        (1, 1),
        (5, 1),
        (10, 1),
        (10, 2),
        (10, 3),
        (10, 4),
        (10, 5),
        (10, 6),
        (10, 7),
    }


def _first_market_item(record: dict[str, Any]) -> dict[str, Any] | None:
    retrieval = record.get("retrieval_result") or {}
    for item in retrieval.get("structured_data") or []:
        if item.get("source_type") == "market_api":
            return item
    return None


def _prepend_unique(items: list[Any], item: str) -> list[str]:
    normalized = [str(value) for value in items if str(value).strip()]
    return [item] + [value for value in normalized if value != item]


def _source_display_item(item: dict[str, Any]) -> dict[str, Any]:
    source_name = item.get("source_name") or item.get("provider") or item.get("source_type")
    title = item.get("title") or item.get("summary") or item.get("evidence_id") or source_name
    source_url = item.get("source_url")
    payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
    if not source_url:
        source_url = payload.get("source_url") or payload.get("url")
    return {
        "evidence_id": item.get("evidence_id"),
        "source_type": item.get("source_type"),
        "source_name": source_name,
        "title": title,
        "source_url": source_url,
    }


def _compact_document(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_id": item.get("evidence_id"),
        "source_type": item.get("source_type"),
        "source_name": item.get("source_name"),
        "title": item.get("title"),
        "summary": item.get("summary"),
        "text_excerpt": item.get("text_excerpt"),
        "publish_time": item.get("publish_time"),
        "source_url": item.get("source_url"),
    }


def _compact_structured_item(item: dict[str, Any]) -> dict[str, Any]:
    payload = item.get("payload") or {}
    if isinstance(payload, dict):
        payload = {
            key: value
            for key, value in payload.items()
            if key not in {"history", "raw", "rows"} and not str(key).startswith("_debug")
        }
    return {
        "evidence_id": item.get("evidence_id"),
        "source_type": item.get("source_type"),
        "source_name": item.get("source_name"),
        "provider": item.get("provider"),
        "as_of": item.get("as_of"),
        "quality_flags": item.get("quality_flags") or [],
        "payload": payload,
    }


def _evidence_ids(record: dict[str, Any]) -> set[str]:
    retrieval = record.get("retrieval_result") or {}
    ids: set[str] = set()
    for item in (retrieval.get("documents") or []) + (retrieval.get("structured_data") or []):
        evidence_id = str(item.get("evidence_id") or "").strip()
        if evidence_id:
            ids.add(evidence_id)
    return ids


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise DeepSeekError("DeepSeek response did not contain a JSON object") from None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise DeepSeekError(f"DeepSeek response JSON parse failed: {exc}") from exc
    if not isinstance(parsed, dict):
        raise DeepSeekError("DeepSeek response JSON must be an object")
    return parsed


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _apply_env_overrides(config: dict[str, Any]) -> None:
    mappings = {
        "CHATBOT_HOST": ("server", "host"),
        "CHATBOT_PORT": ("server", "port"),
        "CHATBOT_TITLE": ("ui", "title"),
        "CHATBOT_INPUT_PLACEHOLDER": ("ui", "input_placeholder"),
        "CHATBOT_SUBMIT_TEXT": ("ui", "submit_text"),
        "DEEPSEEK_BASE_URL": ("deepseek", "base_url"),
        "DEEPSEEK_CHAT_PATH": ("deepseek", "chat_path"),
        "DEEPSEEK_MODEL": ("deepseek", "model"),
        "DEEPSEEK_API_KEY": ("deepseek", "api_key"),
        "DEEPSEEK_TIMEOUT_SECONDS": ("deepseek", "timeout_seconds"),
        "CHATBOT_LIVE_DATA": ("live_data", "enabled"),
    }
    for env_name, path in mappings.items():
        if env_name in os.environ:
            section, key = path
            config.setdefault(section, {})[key] = os.environ[env_name]


def _coerce_config_types(config: dict[str, Any]) -> None:
    config["server"]["port"] = int(config["server"]["port"])
    config["deepseek"]["timeout_seconds"] = int(config["deepseek"]["timeout_seconds"])
    config["live_data"]["enabled"] = _parse_bool(config["live_data"].get("enabled", True))


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
