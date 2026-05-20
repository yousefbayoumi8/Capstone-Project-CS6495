"""
SecureBank Chat Interface — FastAPI + Custom HTML/CSS/JS
Replaces webui.py (Gradio) with a proper web interface.

Usage:
    python webui_new.py                    # default model (qwen)
    python webui_new.py --model llama
    python webui_new.py --port 7860

Then open http://localhost:7860 in your browser.
Your laptop can access it via your Tailscale IP on the same port.
"""

import argparse
import gc
import json
import sys
import threading
import time
import uuid
from pathlib import Path
from threading import Thread
from typing import List, Dict, Optional, Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import TextIteratorStreamer

# ── your existing modules ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from src.model_loader import load_model, generate_response, render_chat, MODELS
from src.target_model import SYSTEM_PROMPT

# ── app setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="SecureBank Assistant")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

MODEL = None
TOKENIZER = None
MODEL_NAME = "qwen"

# Serializes model swaps + generation. One GPU = one inference at a time anyway.
_model_lock = threading.Lock()

def _ensure_model(name: str):
    """Swap to `name` if it isn't the currently loaded model. Caller holds _model_lock."""
    global MODEL, TOKENIZER, MODEL_NAME
    if name == MODEL_NAME and MODEL is not None:
        return
    if name not in MODELS:
        raise HTTPException(status_code=400,
                            detail=f"Unknown model '{name}'. Available: {list(MODELS)}")
    # Free current model from VRAM before loading the next one
    MODEL = None
    TOKENIZER = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    MODEL, TOKENIZER = load_model(name)
    MODEL_NAME = name

# ── request / response models ─────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []

class ChatResponse(BaseModel):
    response: str

# ── API endpoints ─────────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    messages = req.history + [{"role": "user", "content": req.message}]
    reply = generate_response(MODEL, TOKENIZER, SYSTEM_PROMPT, messages)
    return ChatResponse(response=reply)

@app.get("/model-info")
async def model_info():
    return {"model": MODEL_NAME}

@app.get("/reset")
async def reset():
    return {"status": "ok"}

# ── OpenAI-compatible API (for Open WebUI, LibreChat, etc.) ───────────────────
class OAIMessage(BaseModel):
    role: str
    content: Any

class OAIChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[OAIMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: Optional[bool] = False

def _content_to_text(content: Any) -> str:
    # OpenAI allows content to be a string OR a list of parts ({type, text})
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
            elif isinstance(p, str):
                parts.append(p)
        return "".join(parts)
    return str(content)

@app.get("/v1/models")
async def list_models():
    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "model", "created": created, "owned_by": "local"}
            for name in MODELS
        ],
    }

def _build_streamer_inputs(system_prompt: str, convo: List[Dict[str, str]]):
    full_messages = [{"role": "system", "content": system_prompt}] + convo
    return render_chat(TOKENIZER, full_messages).to(MODEL.device)

def _sse_stream(requested: str, system_prompt: str, convo: List[Dict[str, str]],
                max_new: int, temperature: float):
    """SSE generator. Acquires _model_lock for the whole stream."""
    with _model_lock:
        _ensure_model(requested)

        inputs = _build_streamer_inputs(system_prompt, convo)
        streamer = TextIteratorStreamer(
            TOKENIZER, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=temperature,
            pad_token_id=TOKENIZER.eos_token_id,
            streamer=streamer,
        )
        thread = Thread(target=MODEL.generate, kwargs=gen_kwargs)
        thread.start()

        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        def chunk(delta: dict, finish: Optional[str] = None):
            return "data: " + json.dumps({
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_NAME,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }) + "\n\n"

        yield chunk({"role": "assistant"})
        try:
            for token in streamer:
                if token:
                    yield chunk({"content": token})
        finally:
            thread.join()
        yield chunk({}, finish="stop")
        yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def chat_completions(req: OAIChatRequest):
    # Pull out any system messages the client sent; merge with our SYSTEM_PROMPT
    system_parts = [SYSTEM_PROMPT]
    convo: List[Dict[str, str]] = []
    for m in req.messages:
        text = _content_to_text(m.content)
        if m.role == "system":
            system_parts.append(text)
        else:
            convo.append({"role": m.role, "content": text})

    system_prompt = "\n\n".join(p for p in system_parts if p)
    max_new = req.max_tokens or 512
    temperature = req.temperature if req.temperature is not None else 0.7
    requested = req.model or MODEL_NAME

    if req.stream:
        return StreamingResponse(
            _sse_stream(requested, system_prompt, convo, max_new, temperature),
            media_type="text/event-stream",
        )

    with _model_lock:
        _ensure_model(requested)
        reply = generate_response(MODEL, TOKENIZER, system_prompt, convo,
                                  max_new_tokens=max_new, temperature=temperature)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": reply},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

# ── HTML frontend ─────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SecureBank Assistant</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  /* ── reset & variables ───────────────────────────────────────── */
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:        #0a0e17;
    --surface:   #111827;
    --surface2:  #1a2234;
    --border:    #1f2d45;
    --accent:    #2563eb;
    --accent2:   #3b82f6;
    --danger:    #ef4444;
    --text:      #e2e8f0;
    --text-muted:#64748b;
    --user-bg:   #1e3a5f;
    --bot-bg:    #131d2e;
    --radius:    12px;
    --font:      'Sora', sans-serif;
    --mono:      'JetBrains Mono', monospace;
  }

  html, body {
    height: 100%;
    background: var(--bg);
    color: var(--text);
    font-family: var(--font);
    font-size: 15px;
    line-height: 1.6;
    overflow: hidden;
  }

  /* ── layout ──────────────────────────────────────────────────── */
  .shell {
    display: grid;
    grid-template-columns: 260px 1fr;
    grid-template-rows: 100vh;
    height: 100vh;
  }

  /* ── sidebar ─────────────────────────────────────────────────── */
  .sidebar {
    background: var(--surface);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    padding: 24px 20px;
    gap: 24px;
    overflow-y: auto;
  }

  .logo {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .logo-icon {
    width: 38px; height: 38px;
    background: var(--accent);
    border-radius: 10px;
    display: grid;
    place-items: center;
    font-size: 18px;
    flex-shrink: 0;
  }

  .logo-text {
    display: flex;
    flex-direction: column;
  }

  .logo-name {
    font-size: 15px;
    font-weight: 600;
    letter-spacing: -0.3px;
    color: var(--text);
  }

  .logo-sub {
    font-size: 11px;
    color: var(--text-muted);
    letter-spacing: 0.5px;
    text-transform: uppercase;
  }

  .divider {
    height: 1px;
    background: var(--border);
  }

  .section-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 10px;
  }

  .status-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .status-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 12px;
  }

  .status-key { color: var(--text-muted); }

  .status-val {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text);
    background: var(--border);
    padding: 2px 8px;
    border-radius: 4px;
  }

  .badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 11px;
    font-weight: 500;
    padding: 3px 9px;
    border-radius: 20px;
  }

  .badge-green { background: #14532d44; color: #4ade80; }
  .badge-blue  { background: #1e3a5f66; color: #60a5fa; }

  .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: currentColor;
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
  }

  /* system prompt accordion */
  .accordion {
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
  }

  .accordion-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 14px;
    background: var(--surface2);
    cursor: pointer;
    font-size: 12px;
    font-weight: 500;
    user-select: none;
    transition: background 0.15s;
  }

  .accordion-header:hover { background: var(--border); }

  .accordion-arrow {
    transition: transform 0.2s;
    font-size: 10px;
    color: var(--text-muted);
  }

  .accordion.open .accordion-arrow { transform: rotate(180deg); }

  .accordion-body {
    display: none;
    padding: 12px 14px;
    background: var(--bg);
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-muted);
    line-height: 1.7;
    white-space: pre-wrap;
    border-top: 1px solid var(--border);
  }

  .accordion.open .accordion-body { display: block; }

  /* reset button */
  .btn-reset {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 10px;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: transparent;
    color: var(--text-muted);
    font-family: var(--font);
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s;
    letter-spacing: 0.3px;
  }

  .btn-reset:hover {
    background: var(--surface2);
    color: var(--text);
    border-color: var(--accent);
  }

  .sidebar-footer {
    margin-top: auto;
    font-size: 11px;
    color: var(--text-muted);
    text-align: center;
    line-height: 1.8;
  }

  /* ── main chat area ───────────────────────────────────────────── */
  .main {
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
  }

  /* top bar */
  .topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 28px;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
    flex-shrink: 0;
  }

  .topbar-left {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .topbar-title {
    font-size: 15px;
    font-weight: 600;
  }

  .topbar-sub {
    font-size: 11px;
    color: var(--text-muted);
  }

  /* messages */
  .messages {
    flex: 1;
    overflow-y: auto;
    padding: 28px;
    display: flex;
    flex-direction: column;
    gap: 20px;
    scroll-behavior: smooth;
  }

  .messages::-webkit-scrollbar { width: 4px; }
  .messages::-webkit-scrollbar-track { background: transparent; }
  .messages::-webkit-scrollbar-thumb {
    background: var(--border);
    border-radius: 4px;
  }

  /* welcome state */
  .welcome {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    flex: 1;
    gap: 16px;
    text-align: center;
    color: var(--text-muted);
    padding: 40px;
  }

  .welcome-icon {
    font-size: 40px;
    opacity: 0.4;
  }

  .welcome h2 {
    font-size: 18px;
    font-weight: 500;
    color: var(--text);
  }

  .welcome p {
    font-size: 13px;
    max-width: 380px;
    line-height: 1.7;
  }

  /* message bubbles */
  .msg {
    display: flex;
    gap: 12px;
    animation: fadeUp 0.2s ease;
    max-width: 820px;
  }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .msg.user { align-self: flex-end; flex-direction: row-reverse; }
  .msg.bot  { align-self: flex-start; }

  .avatar {
    width: 32px; height: 32px;
    border-radius: 9px;
    display: grid;
    place-items: center;
    font-size: 14px;
    flex-shrink: 0;
    margin-top: 2px;
  }

  .msg.user .avatar { background: var(--accent); }
  .msg.bot  .avatar { background: var(--surface2); border: 1px solid var(--border); }

  .bubble {
    padding: 12px 16px;
    border-radius: var(--radius);
    font-size: 14px;
    line-height: 1.65;
    max-width: 680px;
  }

  .msg.user .bubble {
    background: var(--user-bg);
    border: 1px solid #2563eb44;
    border-bottom-right-radius: 4px;
  }

  .msg.bot .bubble {
    background: var(--bot-bg);
    border: 1px solid var(--border);
    border-bottom-left-radius: 4px;
  }

  /* typing indicator */
  .typing-dots {
    display: flex;
    gap: 4px;
    align-items: center;
    padding: 4px 0;
  }

  .typing-dots span {
    width: 6px; height: 6px;
    background: var(--text-muted);
    border-radius: 50%;
    animation: bounce 1.2s infinite;
  }

  .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
  .typing-dots span:nth-child(3) { animation-delay: 0.4s; }

  @keyframes bounce {
    0%, 60%, 100% { transform: translateY(0); }
    30%           { transform: translateY(-6px); }
  }

  /* ── input area ───────────────────────────────────────────────── */
  .input-area {
    padding: 20px 28px 24px;
    border-top: 1px solid var(--border);
    background: var(--surface);
    flex-shrink: 0;
  }

  .input-row {
    display: flex;
    gap: 10px;
    align-items: flex-end;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 10px 12px;
    transition: border-color 0.15s;
  }

  .input-row:focus-within { border-color: var(--accent); }

  textarea {
    flex: 1;
    background: transparent;
    border: none;
    outline: none;
    color: var(--text);
    font-family: var(--font);
    font-size: 14px;
    resize: none;
    min-height: 24px;
    max-height: 120px;
    line-height: 1.5;
    padding: 2px 0;
  }

  textarea::placeholder { color: var(--text-muted); }

  .send-btn {
    width: 36px; height: 36px;
    border-radius: 9px;
    background: var(--accent);
    border: none;
    color: white;
    cursor: pointer;
    display: grid;
    place-items: center;
    flex-shrink: 0;
    transition: background 0.15s, transform 0.1s;
    font-size: 15px;
  }

  .send-btn:hover  { background: var(--accent2); }
  .send-btn:active { transform: scale(0.94); }
  .send-btn:disabled { background: var(--border); cursor: not-allowed; }

  .input-hint {
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 8px;
    padding-left: 4px;
  }
</style>
</head>
<body>
<div class="shell">

  <!-- ── sidebar ──────────────────────────────────────────────── -->
  <aside class="sidebar">
    <div class="logo">
      <div class="logo-icon">🏦</div>
      <div class="logo-text">
        <span class="logo-name">SecureBank</span>
      </div>
    </div>

    <div class="divider"></div>

    <div>
      <div class="section-label">System Status</div>
      <div class="status-card">
        <div class="status-row">
          <span class="status-key">Model</span>
          <span class="status-val" id="model-badge">loading…</span>
        </div>
        <div class="status-row">
          <span class="status-key">Status</span>
          <span class="badge badge-green"><span class="dot"></span>Online</span>
        </div>
        <div class="status-row">
          <span class="status-key">Mode</span>
          <span class="badge badge-blue">Attack Target</span>
        </div>
      </div>
    </div>

    <div>
      <div class="section-label">System Prompt</div>
      <div class="accordion" id="prompt-accordion">
        <div class="accordion-header" onclick="toggleAccordion()">
          <span>View active prompt</span>
          <span class="accordion-arrow">▼</span>
        </div>
        <div class="accordion-body" id="prompt-body">Loading…</div>
      </div>
    </div>

    <button class="btn-reset" onclick="resetChat()">
      ↺ Reset conversation
    </button>

    <div class="sidebar-footer">
      Capstone Project
    </div>
  </aside>

  <!-- ── main ─────────────────────────────────────────────────── -->
  <main class="main">
    <div class="topbar">
      <div class="topbar-left">
        <span class="topbar-title">SecureBank Assistant</span>
        <span class="topbar-sub">Customer Service Interface</span>
      </div>
      <span class="badge badge-green"><span class="dot"></span>Ready</span>
    </div>

    <div class="messages" id="messages">
      <div class="welcome" id="welcome">
        <div class="welcome-icon">🏦</div>
        <h2>Welcome to SecureBank</h2>
        <p>This chatbot is a prompt injection research target.
           Send a normal banking message — or an attack prompt —
           and observe how the model responds.</p>
      </div>
    </div>

    <div class="input-area">
      <div class="input-row">
        <textarea id="input"
                  placeholder="Type a message or attack prompt…"
                  rows="1"
                  onkeydown="handleKey(event)"
                  oninput="autoResize(this)"></textarea>
        <button class="send-btn" id="send-btn" onclick="sendMessage()">➤</button>
      </div>
      <div class="input-hint">Enter to send · Shift+Enter for new line</div>
    </div>
  </main>

</div>

<script>
  let history = [];
  const messagesEl = document.getElementById("messages");
  const inputEl    = document.getElementById("input");
  const sendBtn    = document.getElementById("send-btn");
  const welcomeEl  = document.getElementById("welcome");

  // ── load model info & system prompt on page load ───────────────
  (async () => {
    try {
      const r = await fetch("/model-info");
      const d = await r.json();
      document.getElementById("model-badge").textContent = d.model;
    } catch {}

    // Inline the system prompt from the server
    // We embed it via template substitution below
    document.getElementById("prompt-body").textContent = SYSTEM_PROMPT_JS;
  })();

  function toggleAccordion() {
    document.getElementById("prompt-accordion").classList.toggle("open");
  }

  // ── auto-resize textarea ───────────────────────────────────────
  function autoResize(el) {
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 120) + "px";
  }

  // ── keyboard handler ───────────────────────────────────────────
  function handleKey(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  // ── add a message bubble ───────────────────────────────────────
  function addMessage(role, content) {
    if (welcomeEl) welcomeEl.remove();

    const msg = document.createElement("div");
    msg.className = `msg ${role}`;

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = role === "user" ? "👤" : "🏦";

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = content;

    msg.appendChild(avatar);
    msg.appendChild(bubble);
    messagesEl.appendChild(msg);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return bubble;
  }

  // ── typing indicator ───────────────────────────────────────────
  function addTyping() {
    const msg = document.createElement("div");
    msg.className = "msg bot";
    msg.id = "typing";

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = "🏦";

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.innerHTML = '<div class="typing-dots"><span></span><span></span><span></span></div>';

    msg.appendChild(avatar);
    msg.appendChild(bubble);
    messagesEl.appendChild(msg);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function removeTyping() {
    const el = document.getElementById("typing");
    if (el) el.remove();
  }

  // ── send a message ─────────────────────────────────────────────
  async function sendMessage() {
    const text = inputEl.value.trim();
    if (!text) return;

    inputEl.value = "";
    inputEl.style.height = "auto";
    sendBtn.disabled = true;

    addMessage("user", text);
    addTyping();

    try {
      const res = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, history })
      });
      const data = await res.json();

      removeTyping();
      addMessage("bot", data.response);

      history.push({ role: "user",      content: text });
      history.push({ role: "assistant", content: data.response });

    } catch (err) {
      removeTyping();
      addMessage("bot", "⚠ Error connecting to the model server.");
    }

    sendBtn.disabled = false;
    inputEl.focus();
  }

  // ── reset conversation ─────────────────────────────────────────
  function resetChat() {
    history = [];
    messagesEl.innerHTML = `
      <div class="welcome" id="welcome">
        <div class="welcome-icon">🏦</div>
        <h2>Conversation reset</h2>
        <p>History cleared. Start a new interaction below.</p>
      </div>`;
  }
</script>

<!-- inject system prompt text safely via JSON encoding -->
<script>
  const SYSTEM_PROMPT_JS = SYSTEM_PROMPT_PLACEHOLDER;
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    # Safely embed the system prompt into the page
    prompt_json = json.dumps(SYSTEM_PROMPT)
    html = HTML.replace("SYSTEM_PROMPT_PLACEHOLDER", prompt_json)
    return HTMLResponse(html)

# ── entry point ───────────────────────────────────────────────────────────────
def main():
    global MODEL, TOKENIZER, MODEL_NAME

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen",
                        choices=["qwen", "llama", "gemma"])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    MODEL_NAME = args.model
    MODEL, TOKENIZER = load_model(args.model)

    print(f"\n✓ Model loaded: {args.model}")
    print(f"✓ Interface: http://localhost:{args.port}\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")

if __name__ == "__main__":
    main()