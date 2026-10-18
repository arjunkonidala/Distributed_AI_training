const state = { username: '', record: null, conversationId: null, category: 'symptoms' };
const lookupView = document.querySelector('#lookup-view');
const chatView = document.querySelector('#chat-view');
const lookupForm = document.querySelector('#lookup-form');
const createForm = document.querySelector('#create-form');
const usernameInput = document.querySelector('#username');
const status = document.querySelector('#lookup-status');
const messages = document.querySelector('#messages');
const conversationList = document.querySelector('#conversation-list');

async function request(url, options = {}) {
  const response = await fetch(url, { headers: { 'Content-Type': 'application/json' }, ...options });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || 'Something went wrong.');
  return payload;
}

function showLookup() { chatView.classList.add('hidden'); lookupView.classList.remove('hidden'); usernameInput.focus(); }
function showChat(record) {
  state.record = record;
  state.username = record.profile.username;
  state.conversationId = record.active_conversation_id;
  lookupView.classList.add('hidden');
  chatView.classList.remove('hidden');
  document.querySelector('#patient-name').textContent = record.profile.name;
  document.querySelector('#patient-meta').textContent = [record.profile.date_of_birth && `DOB ${record.profile.date_of_birth}`, record.profile.height, record.profile.weight].filter(Boolean).join(' · ') || 'Care record started today';
  renderConversationList();
  renderConversation(record.conversation);
  renderRecord();
  document.querySelector('#chat-input').focus();
}
function renderConversation(conversation) {
  messages.innerHTML = '';
  if (conversation?.messages?.length) conversation.messages.forEach(addMessage);
  else addMessage({ role: 'assistant', text: `Welcome, ${state.record.profile.name.split(' ')[0]}. I’m here to help keep track of what matters to you. What would you like me to know?` });
}
function renderTableHtml(rows) {
  if (!rows || !rows.length) return '';
  const header = rows[0];
  const bodyRows = rows.slice(1);
  let html = '<div class="bubble-table-wrapper"><table><thead><tr>';
  header.forEach(h => { html += `<th>${h}</th>`; });
  html += '</tr></thead><tbody>';
  bodyRows.forEach(r => {
    html += '<tr>';
    r.forEach(c => { html += `<td>${c}</td>`; });
    html += '</tr>';
  });
  html += '</tbody></table></div>';
  return html;
}

function formatMarkdown(text) {
  if (!text) return '';
  let escaped = escapeHtml(text);
  const rawLines = escaped.split('\n');
  let inTable = false;
  let tableRows = [];
  let processedLines = [];
  
  for (let i = 0; i < rawLines.length; i++) {
    const line = rawLines[i].trim();
    if (line.startsWith('|') && line.endsWith('|')) {
      if (line.match(/^\|(\s*:?-+:?\s*\|)+$/)) continue;
      inTable = true;
      const cells = line.slice(1, -1).split('|').map(c => c.trim());
      tableRows.push(cells);
    } else {
      if (inTable) {
        processedLines.push(renderTableHtml(tableRows));
        tableRows = [];
        inTable = false;
      }
      processedLines.push(line);
    }
  }
  if (inTable && tableRows.length) {
    processedLines.push(renderTableHtml(tableRows));
  }
  
  let formatted = processedLines.join('\n');
  formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  formatted = formatted.replace(/^### (.*?)$/gm, '<h4>$1</h4>');
  formatted = formatted.replace(/^## (.*?)$/gm, '<h3>$1</h3>');
  formatted = formatted.replace(/^\s*[-*•]\s+(.*?)$/gm, '<li>$1</li>');
  formatted = formatted.replace(/(<li>.*?<\/li>(\s*<li>.*?<\/li>)*)/gs, '<ul>$1</ul>');
  
  // Ensure block elements are cleanly separated so they never get wrapped in <p>
  formatted = formatted.replace(/(<(?:h[1-6]|div class="bubble-table-wrapper"|ul)[^>]*>)/g, '\n\n$1');
  formatted = formatted.replace(/(<\/(?:h[1-6]|div|ul)>)/g, '$1\n\n');
  
  const blocks = formatted.split(/\n\n+/);
  return blocks.map(b => {
    b = b.trim();
    if (!b) return '';
    if (b.startsWith('<div class="bubble-table-wrapper"') || b.startsWith('<h') || b.startsWith('<ul>')) return b;
    return `<p>${b.replace(/\n/g, '<br>')}</p>`;
  }).filter(Boolean).join('');
}

function addMessage(message) {
  const line = document.createElement('div');
  line.className = `message ${message.role}`;
  const categories = message.categories || (message.category ? [message.category] : []);
  const label = categories.length ? `<div class="category-badge">Updated ${escapeHtml(categories.join(', '))}</div>` : '';
  const content = message.role === 'assistant' ? formatMarkdown(message.text) : `<p>${escapeHtml(message.text)}</p>`;
  line.innerHTML = `<div class="avatar">${message.role === 'assistant' ? '+' : 'Y'}</div><div class="bubble">${label}${content}</div>`;
  messages.append(line);
  messages.scrollTop = messages.scrollHeight;
}
function escapeHtml(value) { const node = document.createElement('div'); node.textContent = value ?? ''; return node.innerHTML; }
function friendly(key) { return key.replaceAll('_', ' ').replace(/\b\w/g, letter => letter.toUpperCase()); }
function hasValue(value) { return Array.isArray(value) ? value.length : value !== null && value !== undefined && value !== false && value !== ''; }
function renderRecord() {
  const forms = state.record?.forms;
  if (!forms || typeof forms !== 'object') return;
  Object.entries(forms).forEach(([name, form]) => {
    const counter = document.querySelector(`#${name}-count`);
    const count = name === 'treatment' && Array.isArray(form)
      ? form.filter(entry => Object.values(entry || {}).some(hasValue)).length
      : Object.values(form || {}).filter(hasValue).length;
    if (counter) counter.textContent = count;
  });
  if (!forms[state.category]) state.category = Object.keys(forms)[0] || 'symptoms';
  document.querySelectorAll('.record-tab').forEach(tab => tab.classList.toggle('active', tab.dataset.category === state.category));
  const selectedForm = forms[state.category];
  if (state.category === 'treatment' && Array.isArray(selectedForm)) {
    const treatments = selectedForm.filter(entry => Object.values(entry || {}).some(hasValue));
    document.querySelector('#form-details').innerHTML = treatments.length ? treatments.map((treatment, index) => {
      const title = treatment.intervention_name || `Intervention ${index + 1}`;
      const details = Object.entries(treatment).filter(([key, value]) => key !== 'intervention_name' && hasValue(value));
      return `<section class="treatment-entry"><p class="treatment-name">${escapeHtml(title)}</p>${details.map(([key, value]) => `<div class="field"><span class="field-label">${friendly(key)}</span><p class="field-value">${escapeHtml(Array.isArray(value) ? value.join(', ') : value)}</p></div>`).join('')}</section>`;
    }).join('') : '<p class="empty">Details shared in chat will appear here.</p>';
    return;
  }
  if (state.category === 'timeline' && Array.isArray(selectedForm)) {
    const events = selectedForm.filter(entry => Object.values(entry || {}).some(hasValue));
    document.querySelector('#form-details').innerHTML = events.length ? events.map(event => {
      return `<section class="treatment-entry"><div class="field"><span class="field-label">${escapeHtml(event.timestamp)}</span><p class="field-value">${escapeHtml(event.event_description)}</p></div></section>`;
    }).join('') : '<p class="empty">No timeline events recorded yet.</p>';
    return;
  }
  const entries = Object.entries(selectedForm || {}).filter(([, value]) => hasValue(value));
  document.querySelector('#form-details').innerHTML = entries.length ? entries.map(([key, value]) => `<div class="field"><span class="field-label">${friendly(key)}</span><p class="field-value">${escapeHtml(Array.isArray(value) ? value.join(', ') : value)}</p></div>`).join('') : '<p class="empty">Details shared in chat will appear here.</p>';
}
function renderConversationList() {
  conversationList.innerHTML = '';
  state.record.conversations.forEach(conversation => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `conversation-item${conversation.id === state.conversationId ? ' active' : ''}`;
    button.dataset.conversationId = conversation.id;
    button.textContent = conversation.title || 'New chat';
    button.title = conversation.title || 'New chat';
    conversationList.append(button);
  });
}

lookupForm.addEventListener('submit', async event => {
  event.preventDefault(); status.textContent = ''; createForm.classList.add('hidden');
  const username = usernameInput.value.trim();
  try {
    const result = await request(`/api/profile?username=${encodeURIComponent(username)}`);
    if (result.found) showChat(result.record);
    else { status.textContent = 'No profile found for that username. Create one below to begin.'; createForm.classList.remove('hidden'); document.querySelector('#new-name').focus(); }
  } catch (error) { status.textContent = error.message; }
});

createForm.addEventListener('submit', async event => {
  event.preventDefault();
  try {
    const result = await request('/api/profile', { method: 'POST', body: JSON.stringify({ username: usernameInput.value.trim(), name: document.querySelector('#new-name').value, date_of_birth: document.querySelector('#new-dob').value, height: document.querySelector('#new-height').value, weight: document.querySelector('#new-weight').value }) });
    showChat(result.record);
  } catch (error) { status.textContent = error.message; }
});

document.querySelector('#new-conversation').addEventListener('click', async () => {
  try {
    const result = await request('/api/conversations', { method: 'POST', body: JSON.stringify({ username: state.username }) });
    showChat(result.record);
  } catch (error) { addMessage({ role: 'assistant', text: error.message }); }
});

conversationList.addEventListener('click', async event => {
  const button = event.target.closest('[data-conversation-id]');
  if (!button || button.dataset.conversationId === state.conversationId) return;
  try {
    const result = await request('/api/conversations/select', { method: 'POST', body: JSON.stringify({ username: state.username, conversation_id: button.dataset.conversationId }) });
    showChat(result.record);
  } catch (error) { addMessage({ role: 'assistant', text: error.message }); }
});

async function handleSendMessage(event) {
  if (event) event.preventDefault();
  const input = document.querySelector('#chat-input');
  if (!input) return;
  const text = input.value.trim();
  if (!text || input.disabled) return;
  
  addMessage({ role: 'user', text });
  input.value = '';
  input.style.height = 'auto'; // reset textarea height
  input.disabled = true;
  
  // Show a typing indicator
  const typingEl = document.createElement('div');
  typingEl.className = 'message assistant typing-indicator';
  typingEl.innerHTML = '<div class="avatar">+</div><div class="bubble"><span class="dot"></span><span class="dot"></span><span class="dot"></span></div>';
  messages.append(typingEl);
  messages.scrollTop = messages.scrollHeight;
  
  try {
    const result = await request('/api/chat', { 
      method: 'POST', 
      body: JSON.stringify({ 
        username: state.username, 
        conversation_id: state.conversationId, 
        message: text 
      }) 
    });
    state.record = result.record;
    state.conversationId = result.record.active_conversation_id;
    const updatedCategories = result.categories || (result.category ? [result.category] : []);
    if (!state.record.forms?.[state.category] && updatedCategories.length) state.category = updatedCategories[0];
    
    // Remove typing indicator, then append just the AI response (not re-render all)
    typingEl.remove();
    const conv = result.record.conversation;
    if (conv?.messages?.length) {
      const lastMsg = conv.messages[conv.messages.length - 1];
      if (lastMsg && lastMsg.role === 'assistant') addMessage(lastMsg);
    }
    
    renderConversationList();
    renderRecord();
  } catch (error) { 
    typingEl.remove();
    addMessage({ role: 'assistant', text: error.message }); 
  } finally { 
    input.disabled = false; 
    input.focus(); 
  }
}

document.querySelector('#chat-form').addEventListener('submit', handleSendMessage);

const chatInput = document.querySelector('#chat-input');
if (chatInput) {
  // Enter = send, Shift+Enter = newline
  chatInput.addEventListener('keydown', event => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      if (!chatInput.disabled) handleSendMessage(null);
    }
  });
  // Auto-resize textarea as user types
  chatInput.addEventListener('input', () => {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 150) + 'px';
  });
}

document.querySelectorAll('.record-tab').forEach(tab => tab.addEventListener('click', () => { state.category = tab.dataset.category; renderRecord(); }));
document.querySelector('#switch-profile').addEventListener('click', showLookup);
