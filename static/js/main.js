/**
 * main.js — SmartStudy AI Global JS
 * Shared utilities: toast, chat, markdown renderer, auto-resize
 */

// ── Toast ────────────────────────────────────────────────────

function showToast(msg, type = 'info', ms = 3500) {
  const t = document.getElementById('toast');
  if (!t) return;
  const icons = { success: '✅', error: '❌', warning: '⚠️', info: 'ℹ️' };
  t.innerHTML = `<span class="toast-icon">${icons[type] || 'ℹ️'}</span> ${msg}`;
  t.className = `toast ${type}`;
  t.classList.remove('hidden');
  clearTimeout(t._tid);
  t._tid = setTimeout(() => t.classList.add('hidden'), ms);
}

// ── Markdown renderer ────────────────────────────────────────

function renderMd(text) {
  if (!text) return '';
  let h = text
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  h = h
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`(.+?)`/g, '<code>$1</code>')
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm,  '<h2>$1</h2>')
    .replace(/^# (.+)$/gm,   '<h1>$1</h1>')
    .replace(/^- (.+)$/gm,   '<li>$1</li>')
    .replace(/(<li>[\s\S]*?<\/li>)\n*/g, m => '<ul style="margin:.4rem 0 .4rem 1.2rem">' + m + '</ul>')
    .replace(/\n\n+/g, '</p><p>')
    .replace(/\n/g, '<br>');
  if (!h.match(/^<[hpu]/)) h = '<p>' + h + '</p>';
  return h;
}

// ── Chat (index page) ────────────────────────────────────────

async function sendChatMessage() {
  const input = document.getElementById('chatInput');
  const btn   = document.getElementById('chatSend');
  const msgs  = document.getElementById('chatMessages');
  if (!input || !msgs) return;

  const text = input.value.trim();
  if (!text) return;

  appendMsg('user', text);
  input.value = '';
  input.style.height = 'auto';

  const typId = showTyping();
  if (btn) btn.disabled = true;

  try {
    const r = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
    });
    const d = await r.json();
    removeTyping(typId);
    if (d.error) {
      appendMsg('assistant', `❌ ${d.error}`);
    } else {
      appendMsg('assistant', d.response || 'Done!', d.tool_calls);
    }
  } catch (err) {
    removeTyping(typId);
    appendMsg('assistant', '❌ Connection error. Check your GROQ_API_KEY in .env');
  } finally {
    if (btn) btn.disabled = false;
    if (input) input.focus();
  }
}

function appendMsg(role, text, toolCalls) {
  const c = document.getElementById('chatMessages');
  if (!c) return;

  const wrap = document.createElement('div');
  wrap.className = `chat-msg ${role}`;

  const avi = document.createElement('div');
  avi.className = 'msg-avi';
  avi.textContent = role === 'user' ? '🧑' : '🎓';

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';
  bubble.innerHTML = renderMd(text);

  if (toolCalls && toolCalls.length) {
    const pills = document.createElement('div');
    pills.style.cssText = 'display:flex;flex-wrap:wrap;gap:4px;margin-top:8px;';
    toolCalls.forEach(tc => {
      const p = document.createElement('span');
      p.className = 'tool-pill';
      p.textContent = `🔧 ${tc.tool}`;
      pills.appendChild(p);
    });
    bubble.appendChild(pills);
  }

  wrap.appendChild(avi);
  wrap.appendChild(bubble);
  c.appendChild(wrap);
  c.scrollTop = c.scrollHeight;
}

function showTyping() {
  const c = document.getElementById('chatMessages');
  if (!c) return null;
  const id  = `typ-${Date.now()}`;
  const row = document.createElement('div');
  row.id = id;
  row.className = 'typing-row';
  row.innerHTML = `
    <div class="msg-avi">🎓</div>
    <div class="typing-bubble">
      <span class="td"></span><span class="td"></span><span class="td"></span>
    </div>`;
  c.appendChild(row);
  c.scrollTop = c.scrollHeight;
  return id;
}

function removeTyping(id) {
  if (id) document.getElementById(id)?.remove();
}

function clearChat() {
  const c = document.getElementById('chatMessages');
  if (!c) return;
  c.innerHTML = `
    <div class="chat-msg assistant">
      <div class="msg-avi">🎓</div>
      <div class="msg-bubble"><p>Chat cleared! What would you like to study?</p></div>
    </div>`;
  fetch('/api/chat/reset', { method: 'POST' }).catch(() => {});
}

// ── Auto-resize textarea & keyboard shortcuts ────────────────

document.addEventListener('DOMContentLoaded', () => {
  const inp = document.getElementById('chatInput');
  if (inp) {
    inp.addEventListener('input', function () {
      this.style.height = 'auto';
      this.style.height = Math.min(this.scrollHeight, 120) + 'px';
    });
    inp.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChatMessage(); }
    });
  }
});
