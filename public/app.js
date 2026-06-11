// Global State
let currentTab = 'dashboard';
let statsInterval = null;
let finetuneInterval = null;
let downloadInterval = null;
let isFinetuningActive = false;
let isDownloadingActive = false;
let chatMessages = [
    { role: 'assistant', content: 'Hello! Welcome to the Lumina Studio playground. Choose a provider and model above and start chatting.' }
];

const providerModels = {
    openrouter: [
        { id: 'meta-llama/llama-3-8b-instruct:free', name: 'Llama 3 8B Instruct (Free)' },
        { id: 'mistralai/mistral-7b-instruct:free', name: 'Mistral 7B Instruct (Free)' },
        { id: 'microsoft/phi-3-mini-128k-instruct:free', name: 'Phi 3 Mini Instruct (Free)' },
        { id: 'google/gemma-2-9b-it:free', name: 'Gemma 2 9B IT (Free)' },
        { id: 'qwen/qwen-2-7b-instruct:free', name: 'Qwen 2 7B Instruct (Free)' }
    ],
    openai: [
        { id: 'gpt-4o-mini', name: 'GPT-4o Mini' },
        { id: 'gpt-4o', name: 'GPT-4o' },
        { id: 'gpt-3.5-turbo', name: 'GPT-3.5 Turbo' }
    ],
    gemini: [
        { id: 'gemini-1.5-flash', name: 'Gemini 1.5 Flash' },
        { id: 'gemini-1.5-pro', name: 'Gemini 1.5 Pro' }
    ],
    anthropic: [
        { id: 'claude-3-5-sonnet-20240620', name: 'Claude 3.5 Sonnet' },
        { id: 'claude-3-haiku-20240307', name: 'Claude 3 Haiku' }
    ],
    cohere: [
        { id: 'command-r', name: 'Command R' },
        { id: 'command-r-plus', name: 'Command R+' }
    ],
    local: [
        { id: 'local-base', name: 'Local Base Model' },
        { id: 'local-adapter', name: 'Local Model + Fine-tuned Adapter' }
    ]
};

// Document Ready
document.addEventListener('DOMContentLoaded', () => {
    // Initial loads
    fetchStats();
    fetchConfig();
    checkHuggingFaceLogin();
    loadDatasets();
    loadFinetuneHistory();
    loadLocalModels();
    loadActiveDataset();
    
    // Poll stats every 4 seconds
    statsInterval = setInterval(fetchStats, 4000);
    
    // Check if tasks were already running on server start
    checkFinetuneStatus(true);
    checkDownloadStatus(true);
    
    // Initialize playground selectors
    onPlaygroundProviderChange('single');
    onPlaygroundProviderChange('a');
    onPlaygroundProviderChange('b');
    
    // Set initial active button in side nav
    switchTab('dashboard');
});

// Tab Navigation
const tabMeta = {
    dashboard: {
        title: 'Dashboard Overview',
        desc: 'Monitor system resource usage, configure API credentials, and view system specs.'
    },
    downloader: {
        title: 'Model Downloader Workspace',
        desc: 'Download weights snapshots from Hugging Face or direct HTTP/HTTPS URLs.'
    },
    playground: {
        title: 'Lumina Model Playground',
        desc: 'Interactive chat playground supporting cross-provider comparisons and local model inference.'
    },
    huggingface: {
        title: 'Hugging Face Hub Integration',
        desc: 'Validate write permissions, explore models and datasets, and push local model adapters.'
    },
    dataset: {
        title: 'Dataset Builder Studio',
        desc: 'Interactively construct, validate, and manage prompt-response datasets for fine-tuning.'
    },
    finetune: {
        title: 'Fine-Tuning Studio',
        desc: 'Run parameter-efficient fine-tuning (LoRA) on pre-trained models using custom datasets.'
    }
};

function switchTab(tabId) {
    currentTab = tabId;
    
    // Update headers
    document.getElementById('current-tab-title').innerText = tabMeta[tabId].title;
    document.getElementById('current-tab-desc').innerText = tabMeta[tabId].desc;
    
    // Update buttons active class
    document.querySelectorAll('.nav-item').forEach(btn => {
        btn.classList.remove('active');
    });
    const activeBtn = document.getElementById(`tab-btn-${tabId}`);
    if (activeBtn) activeBtn.classList.add('active');
    
    // Update panels active class
    document.querySelectorAll('.tab-panel').forEach(panel => {
        panel.classList.remove('active');
    });
    const activePanel = document.getElementById(`panel-${tabId}`);
    if (activePanel) activePanel.classList.add('active');
    
    // Tab-specific adjustments
    if (tabId === 'playground') {
        scrollToBottom('chat-messages-box');
    } else if (tabId === 'finetune') {
        scrollToBottom('ft-console-box');
    } else if (tabId === 'downloader') {
        scrollToBottom('dl-console-box');
        loadLocalModels();
    }
}

// Helper: Toggle Password/Key Visibility
function togglePasswordVisibility(inputId) {
    const el = document.getElementById(inputId);
    if (el.type === 'password') {
        el.type = 'text';
    } else {
        el.type = 'password';
    }
}

// Helper: Scroll to bottom of an element
function scrollToBottom(elemId) {
    const el = document.getElementById(elemId);
    if (el) {
        el.scrollTop = el.scrollHeight;
    }
}

// 1. Fetch & Update System Stats
async function fetchStats() {
    try {
        const response = await fetch('/api/stats');
        if (!response.ok) throw new Error('Failed to fetch stats');
        const data = await response.json();
        
        // Update stats summary in header
        if (data.cpuUsage) {
            document.getElementById('cpu-val').innerText = data.cpuUsage;
            document.getElementById('cpu-bar').style.width = data.cpuUsage;
        }
        
        if (data.memory) {
            document.getElementById('mem-val').innerText = `${data.memory.used} / ${data.memory.total}`;
            document.getElementById('mem-bar').style.width = data.memory.percentage;
        }
        
        // Update dashboard details card
        if (currentTab === 'dashboard') {
            document.getElementById('sys-hostname').innerText = data.hostname || '-';
            document.getElementById('sys-platform').innerText = data.platform || '-';
            document.getElementById('sys-cpu').innerText = data.cpu || '-';
            document.getElementById('sys-cores').innerText = data.cpuCount || '-';
            document.getElementById('sys-disk').innerText = data.disk || '-';
            
            // Format uptime
            const uptimeSecs = data.uptime;
            const hours = Math.floor(uptimeSecs / 3600);
            const minutes = Math.floor((uptimeSecs % 3600) / 60);
            document.getElementById('sys-uptime').innerText = `${hours}h ${minutes}m`;
        }
        
        // Update online indicator
        document.getElementById('status-pulse').className = 'pulse-indicator green';
        document.getElementById('status-text').innerText = 'Server: Online';
    } catch (err) {
        console.error('Stats error:', err);
        document.getElementById('status-pulse').className = 'pulse-indicator red';
        document.getElementById('status-text').innerText = 'Server: Disconnected';
    }
}

// 2. Load & Save Settings
async function fetchConfig() {
    try {
        const response = await fetch('/api/config');
        const data = await response.json();
        if (data.success && data.config) {
            if (data.config.HF_TOKEN) document.getElementById('hf-token-input').value = data.config.HF_TOKEN;
            if (data.config.OPENROUTER_API_KEY) document.getElementById('openrouter-key-input').value = data.config.OPENROUTER_API_KEY;
            if (data.config.OPENAI_API_KEY) document.getElementById('openai-key-input').value = data.config.OPENAI_API_KEY;
            if (data.config.GEMINI_API_KEY) document.getElementById('gemini-key-input').value = data.config.GEMINI_API_KEY;
            if (data.config.ANTHROPIC_API_KEY) document.getElementById('anthropic-key-input').value = data.config.ANTHROPIC_API_KEY;
            if (data.config.COHERE_API_KEY) document.getElementById('cohere-key-input').value = data.config.COHERE_API_KEY;
        }
    } catch (err) {
        console.error('Failed to load settings:', err);
    }
}

async function saveSettings(event) {
    event.preventDefault();
    const hfToken = document.getElementById('hf-token-input').value.trim();
    const openrouterKey = document.getElementById('openrouter-key-input').value.trim();
    const openaiKey = document.getElementById('openai-key-input').value.trim();
    const geminiKey = document.getElementById('gemini-key-input').value.trim();
    const anthropicKey = document.getElementById('anthropic-key-input').value.trim();
    const cohereKey = document.getElementById('cohere-key-input').value.trim();
    
    const messageEl = document.getElementById('settings-message');
    messageEl.className = 'alert-message';
    messageEl.innerText = '';
    
    const btn = document.getElementById('btn-save-settings');
    btn.disabled = true;
    btn.innerText = 'Saving Settings...';
    
    try {
        const response = await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                HF_TOKEN: hfToken,
                OPENROUTER_API_KEY: openrouterKey,
                OPENAI_API_KEY: openaiKey,
                GEMINI_API_KEY: geminiKey,
                ANTHROPIC_API_KEY: anthropicKey,
                COHERE_API_KEY: cohereKey
            })
        });
        
        const data = await response.json();
        if (data.success) {
            messageEl.classList.add('success');
            messageEl.innerText = 'API Credentials saved and loaded successfully!';
            await fetchConfig();
            await checkHuggingFaceLogin();
        } else {
            throw new Error(data.error || 'Unknown error saving settings');
        }
    } catch (err) {
        messageEl.classList.add('error');
        messageEl.innerText = `Failed to save: ${err.message}`;
    } finally {
        btn.disabled = false;
        btn.innerText = 'Save API Settings';
    }
}

// 3. Multi-API Chat Playground
let playgroundMode = 'single'; // State helper

function setPlaygroundMode(mode) {
    playgroundMode = mode;
    const btnSingle = document.getElementById('btn-mode-single');
    const btnCompare = document.getElementById('btn-mode-compare');
    const singleControls = document.getElementById('single-model-controls');
    const compareControls = document.getElementById('compare-model-controls');
    const singleBox = document.getElementById('chat-messages-box');
    const compareBox = document.getElementById('chat-compare-box');
    
    if (mode === 'single') {
        btnSingle.classList.add('active');
        btnSingle.style.background = 'var(--primary)';
        btnSingle.style.color = '#fff';
        btnCompare.classList.remove('active');
        btnCompare.style.background = 'transparent';
        btnCompare.style.color = 'var(--text-secondary)';
        
        singleControls.classList.remove('hidden');
        compareControls.classList.add('hidden');
        singleBox.classList.remove('hidden');
        compareBox.classList.add('hidden');
    } else {
        btnSingle.classList.remove('active');
        btnSingle.style.background = 'transparent';
        btnSingle.style.color = 'var(--text-secondary)';
        btnCompare.classList.add('active');
        btnCompare.style.background = 'var(--primary)';
        btnCompare.style.color = '#fff';
        
        singleControls.classList.add('hidden');
        compareControls.classList.remove('hidden');
        singleBox.classList.add('hidden');
        compareBox.classList.remove('hidden');
        
        scrollToBottom('chat-messages-a');
        scrollToBottom('chat-messages-b');
    }
}

function onPlaygroundProviderChange(pane) {
    const providerSelectId = pane === 'single' ? 'playground-provider-select' : `playground-provider-${pane}-select`;
    const modelSelectId = pane === 'single' ? 'playground-model-select' : `playground-model-${pane}-select`;
    
    const providerSelect = document.getElementById(providerSelectId);
    if (!providerSelect) return;
    
    const provider = providerSelect.value;
    const modelSelect = document.getElementById(modelSelectId);
    if (!modelSelect) return;
    
    modelSelect.innerHTML = '';
    
    const models = providerModels[provider] || [];
    models.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.id;
        opt.innerText = m.name;
        modelSelect.appendChild(opt);
    });
    
    if (pane === 'single') {
        toggleLocalModelInputs();
    } else {
        toggleCompareLocalInputs(pane);
    }
}

function toggleLocalModelInputs() {
    const val = document.getElementById('playground-model-select').value;
    const isLocal = val.startsWith('local-');
    const container = document.getElementById('local-model-custom-inputs');
    if (isLocal) {
        container.classList.remove('hidden');
    } else {
        container.classList.add('hidden');
    }
}

function toggleCompareLocalInputs(paneId) {
    const select = document.getElementById(`playground-model-${paneId}-select`);
    if (!select) return;
    const val = select.value;
    const isLocal = val.startsWith('local-');
    const container = document.getElementById(`local-model-${paneId}-custom-inputs`);
    if (container) {
        if (isLocal) {
            container.classList.remove('hidden');
        } else {
            container.classList.add('hidden');
        }
    }
}

function updateSliderVal(sliderId, displayId) {
    const val = document.getElementById(sliderId).value;
    document.getElementById(displayId).innerText = val;
}

function handleChatSubmit(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendChatPrompt();
    }
}

function renderChat() {
    const box = document.getElementById('chat-messages-box');
    box.innerHTML = '';
    
    chatMessages.forEach(msg => {
        const bubble = document.createElement('div');
        bubble.className = `chat-bubble ${msg.role}`;
        
        if (msg.role === 'system-error') {
            bubble.innerText = msg.content;
        } else {
            bubble.innerHTML = formatChatText(msg.content);
        }
        
        box.appendChild(bubble);
    });
    scrollToBottom('chat-messages-box');
}

function formatChatText(text) {
    let escaped = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
    
    const codeBlockRegex = /```(?:[a-zA-Z0-9]+)?\n([\s\S]*?)\n```/g;
    escaped = escaped.replace(codeBlockRegex, (match, code) => {
        return `<pre><code>${code.trim()}</code></pre>`;
    });

    escaped = escaped.replace(/`([^`]+)`/g, '<code>$1</code>');

    const parts = escaped.split(/(<pre>[\s\S]*?<\/pre>)/);
    const formattedParts = parts.map(part => {
        if (part.startsWith('<pre>')) return part;
        return part.replace(/\n/g, '<br>');
    });
    
    return formattedParts.join('');
}

async function sendChatPrompt() {
    const textarea = document.getElementById('chat-input');
    const prompt = textarea.value.trim();
    if (!prompt) return;
    
    textarea.value = '';
    textarea.disabled = true;
    const sendBtn = document.getElementById('btn-send-chat');
    sendBtn.disabled = true;
    
    const temp = parseFloat(document.getElementById('temp-slider').value);
    const maxTokens = parseInt(document.getElementById('tokens-slider').value);
    
    if (playgroundMode === 'compare') {
        const userBubbleA = document.createElement('div');
        userBubbleA.className = 'chat-bubble user';
        userBubbleA.innerHTML = formatChatText(prompt);
        document.getElementById('chat-messages-a').appendChild(userBubbleA);

        const userBubbleB = document.createElement('div');
        userBubbleB.className = 'chat-bubble user';
        userBubbleB.innerHTML = formatChatText(prompt);
        document.getElementById('chat-messages-b').appendChild(userBubbleB);
        
        const loaderA = document.createElement('div');
        loaderA.className = 'chat-bubble assistant loader-bubble';
        loaderA.innerHTML = '<p>Thinking...</p>';
        document.getElementById('chat-messages-a').appendChild(loaderA);

        const loaderB = document.createElement('div');
        loaderB.className = 'chat-bubble assistant loader-bubble';
        loaderB.innerHTML = '<p>Thinking...</p>';
        document.getElementById('chat-messages-b').appendChild(loaderB);

        scrollToBottom('chat-messages-a');
        scrollToBottom('chat-messages-b');

        async function fetchPaneCompletion(paneId) {
            const provider = document.getElementById(`playground-provider-${paneId}-select`).value;
            const select = document.getElementById(`playground-model-${paneId}-select`);
            const modelVal = select.value;
            const isLocal = provider === 'local';
            const endpoint = isLocal ? '/api/local/chat' : '/api/playground/chat';
            
            let modelName = modelVal;
            if (isLocal) {
                modelName = document.getElementById(`playground-local-model-${paneId}-name`).value.trim();
            }
            
            const bodyPayload = isLocal ? {
                model: modelName,
                adapter: modelVal === 'local-adapter' ? 'model_output' : null,
                prompt: prompt,
                max_tokens: maxTokens,
                temperature: temp
            } : {
                provider: provider,
                model: modelVal,
                messages: [{ role: 'user', content: prompt }],
                temperature: temp,
                max_tokens: maxTokens
            };

            const startTime = Date.now();
            try {
                const response = await fetch(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(bodyPayload)
                });
                
                const data = await response.json();
                const durationSecs = ((Date.now() - startTime) / 1000).toFixed(2);
                
                const paneBox = document.getElementById(`chat-messages-${paneId}`);
                const loaders = paneBox.querySelectorAll('.loader-bubble');
                loaders.forEach(l => l.remove());
                
                if (response.ok) {
                    let text = '';
                    if (isLocal) {
                        text = data.text;
                    } else if (data.choices && data.choices[0]) {
                        text = data.choices[0].message.content;
                    } else {
                        throw new Error('No choices returned from model');
                    }
                    
                    const bubble = document.createElement('div');
                    bubble.className = 'chat-bubble assistant';
                    bubble.innerHTML = formatChatText(text);
                    paneBox.appendChild(bubble);
                    
                    const tokenCount = Math.round(text.trim().split(/\s+/).length * 1.35);
                    const metricsContainer = document.getElementById(`metrics-${paneId}`);
                    metricsContainer.innerHTML = `
                        <span class="metric-badge latency">${durationSecs}s</span>
                        <span class="metric-badge tokens">${tokenCount} tokens</span>
                    `;
                } else {
                    throw new Error(data.error || 'Failed to generate content');
                }
            } catch (err) {
                const paneBox = document.getElementById(`chat-messages-${paneId}`);
                const loaders = paneBox.querySelectorAll('.loader-bubble');
                loaders.forEach(l => l.remove());
                
                const bubble = document.createElement('div');
                bubble.className = 'chat-bubble system-error';
                bubble.innerText = `Error: ${err.message}`;
                paneBox.appendChild(bubble);
            } finally {
                scrollToBottom(`chat-messages-${paneId}`);
            }
        }

        try {
            await Promise.all([fetchPaneCompletion('a'), fetchPaneCompletion('b')]);
        } catch (err) {
            console.error('Parallel comparison completion error:', err);
        } finally {
            textarea.disabled = false;
            sendBtn.disabled = false;
            textarea.focus();
        }
        return;
    }
    
    // Single Mode execution path
    chatMessages.push({ role: 'user', content: prompt });
    renderChat();
    
    chatMessages.push({ role: 'assistant', content: 'Thinking...' });
    renderChat();
    
    const provider = document.getElementById('playground-provider-select').value;
    const selectedModel = document.getElementById('playground-model-select').value;
    const isLocal = provider === 'local';
    const endpoint = isLocal ? '/api/local/chat' : '/api/playground/chat';
    
    const bodyPayload = isLocal ? {
        model: document.getElementById('playground-local-model-name').value.trim(),
        adapter: selectedModel === 'local-adapter' ? 'model_output' : null,
        prompt: prompt,
        max_tokens: maxTokens,
        temperature: temp
    } : {
        provider: provider,
        model: selectedModel,
        messages: chatMessages.filter(m => m.role === 'user' || m.role === 'assistant').slice(0, -1),
        temperature: temp,
        max_tokens: maxTokens
    };
    
    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(bodyPayload)
        });
        
        const data = await response.json();
        chatMessages.pop();
        
        if (response.ok) {
            let content = '';
            if (isLocal) {
                content = data.text;
            } else if (data.choices && data.choices[0]) {
                content = data.choices[0].message.content;
            } else {
                throw new Error('No choices returned from model');
            }
            chatMessages.push({
                role: 'assistant',
                content: content
            });
        } else {
            throw new Error(data.error || 'Failed to generate content');
        }
    } catch (err) {
        if (chatMessages[chatMessages.length - 1].content === 'Thinking...') {
            chatMessages.pop();
        }
        chatMessages.push({
            role: 'system-error',
            content: `Error calling API: ${err.message}`
        });
    } finally {
        textarea.disabled = false;
        sendBtn.disabled = false;
        renderChat();
        textarea.focus();
    }
}

function clearChat() {
    if (playgroundMode === 'single') {
        chatMessages = [
            { role: 'assistant', content: 'Chat history cleared. Choose a model and start a new conversation!' }
        ];
        renderChat();
    } else {
        document.getElementById('chat-messages-a').innerHTML = `
            <div class="chat-bubble assistant">
                <p>Model A ready. Submit a prompt to generate side-by-side completions.</p>
            </div>
        `;
        document.getElementById('chat-messages-b').innerHTML = `
            <div class="chat-bubble assistant">
                <p>Model B ready. Submit a prompt to generate side-by-side completions.</p>
            </div>
        `;
        document.getElementById('metrics-a').innerHTML = '';
        document.getElementById('metrics-b').innerHTML = '';
    }
}

// 4. Hugging Face Integrations
async function checkHuggingFaceLogin() {
    const unauthSection = document.getElementById('hf-profile-unauthorized');
    const authSection = document.getElementById('hf-profile-authorized');
    
    try {
        const response = await fetch('/api/hf/whoami');
        const data = await response.json();
        
        if (data.success && data.username) {
            unauthSection.classList.add('hidden');
            authSection.classList.remove('hidden');
            
            document.getElementById('hf-avatar').src = data.avatarUrl || 'https://huggingface.co/avatars/default.png';
            document.getElementById('hf-username').innerText = data.fullname ? `${data.fullname} (@${data.username})` : `@${data.username}`;
            document.getElementById('hf-email').innerText = data.email || 'Email: hidden/unavailable';
            document.getElementById('hf-orgs').innerText = data.orgs.length > 0 ? `Organizations: ${data.orgs.join(', ')}` : 'Organizations: Personal Account';
            document.getElementById('hf-auth-type').innerText = data.authType;
            
            // Auto fill upload repo-id prefix
            document.getElementById('hf-upload-repoid').placeholder = `e.g. ${data.username}/my-fine-tuned-model`;
        } else {
            unauthSection.classList.remove('hidden');
            authSection.classList.add('hidden');
        }
    } catch (e) {
        unauthSection.classList.remove('hidden');
        authSection.classList.add('hidden');
    }
}

async function searchHuggingFace() {
    const query = document.getElementById('hf-search-input').value.trim();
    if (!query) return;
    
    const type = document.getElementById('hf-search-type').value;
    const container = document.getElementById('hf-search-results');
    
    container.innerHTML = '<p class="placeholder-text">Searching Hugging Face Hub...</p>';
    
    try {
        const endpoint = type === 'models' ? '/api/hf/search-models' : '/api/hf/search-datasets';
        const response = await fetch(`${endpoint}?query=${encodeURIComponent(query)}&limit=15`);
        const data = await response.json();
        
        container.innerHTML = '';
        
        if (data.success) {
            const list = type === 'models' ? data.models : data.datasets;
            
            if (list.length === 0) {
                container.innerHTML = `<p class="placeholder-text">No ${type} found matching "${query}"</p>`;
                return;
            }
            
            list.forEach(item => {
                const card = document.createElement('div');
                card.className = 'repo-card';
                card.onclick = () => {
                    // Clicking model/dataset copies ID to clipboard
                    navigator.clipboard.writeText(item.id);
                    // Also auto fill fine tuning base model input if clicking a model
                    if (type === 'models' && currentTab === 'finetune') {
                        document.getElementById('ft-model-select').innerHTML += `<option value="${item.id}" selected>${item.id}</option>`;
                        alert(`Model ID "${item.id}" copied and set as base model!`);
                    } else {
                        alert(`Copied ID to clipboard: ${item.id}`);
                    }
                };
                
                const title = document.createElement('div');
                title.className = 'repo-title';
                title.innerText = item.id;
                
                const meta = document.createElement('div');
                meta.className = 'repo-meta';
                
                // Downloads
                const downloads = document.createElement('span');
                downloads.innerHTML = `📥 ${item.downloads.toLocaleString()}`;
                
                // Likes
                const likes = document.createElement('span');
                likes.innerHTML = `❤️ ${item.likes.toLocaleString()}`;
                
                meta.appendChild(downloads);
                meta.appendChild(likes);
                
                if (item.pipeline_tag) {
                    const tag = document.createElement('span');
                    tag.innerHTML = `🏷️ ${item.pipeline_tag}`;
                    meta.appendChild(tag);
                }
                
                card.appendChild(title);
                card.appendChild(meta);
                container.appendChild(card);
            });
        } else {
            throw new Error(data.error || 'Failed to search Hub');
        }
    } catch (err) {
        container.innerHTML = `<p class="placeholder-text text-danger">Search failed: ${err.message}</p>`;
    }
}

async function uploadModelToHub() {
    const repoId = document.getElementById('hf-upload-repoid').value.trim();
    const folder = document.getElementById('hf-upload-localpath').value.trim();
    
    const messageEl = document.getElementById('hf-upload-message');
    messageEl.className = 'alert-message';
    messageEl.innerText = '';
    
    if (!repoId) {
        messageEl.classList.add('error');
        messageEl.innerText = 'Repository ID is required.';
        return;
    }
    
    const btn = document.getElementById('btn-upload-model');
    btn.disabled = true;
    btn.innerText = 'Uploading Adapter...';
    
    try {
        const response = await fetch('/api/hf/upload-model', {
            // Wait, we need a post route for upload in the backend. Let's make sure it handles this POST
            // Oh, wait, in server.js we did not define /api/hf/upload-model POST, only whoami and search!
            // Wait, let's write a route wrapper for upload in server.js? 
            // Yes! But wait, let's check what routes are in server.js. Let's look at server.js we wrote.
            // Oh! In server.js we only have whoami, search-models, search-datasets, and start/stop/status of fine-tuning.
            // Wait! In server.js did we implement upload?
            // Ah! We didn't define POST /api/hf/upload-model. Let's look at the hf_helper wrapper.
            // We have handle_upload_model in hf_helper.py, but not the backend route!
            // Wait, let me add /api/hf/upload-model POST to server.js.
            // Let's check my task list or proceed. First, let's finish the js file then I can make a quick contiguous block replacement in server.js to add this route!
        });
        
        // Wait, since we need to write the JS code for upload call first:
        const uploadResponse = await fetch('/api/hf/upload-model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ repo_id: repoId, folder: folder })
        });
        
        const data = await uploadResponse.json();
        if (data.success) {
            messageEl.classList.add('success');
            messageEl.innerHTML = `Success! Adapter uploaded to Hub: <a href="${data.repoUrl}" target="_blank" class="help-link">${repoId}</a>`;
        } else {
            throw new Error(data.error || 'Failed to upload folder');
        }
    } catch (err) {
        messageEl.classList.add('error');
        messageEl.innerText = `Upload failed: ${err.message}`;
    } finally {
        btn.disabled = false;
        btn.innerText = 'Upload Adapter Weights';
    }
}

// 5. Fine-Tuning Studio State
function clearConsole() {
    document.getElementById('ft-console-box').innerHTML = '';
}

function appendConsoleLine(text, type = '') {
    const box = document.getElementById('ft-console-box');
    const line = document.createElement('div');
    line.className = `console-line ${type}`;
    line.innerText = text;
    box.appendChild(line);
    
    // Auto scroll console
    box.scrollTop = box.scrollHeight;
}

async function startFineTuning(event) {
    event.preventDefault();
    if (isFinetuningActive) return;
    
    const model = document.getElementById('ft-model-select').value;
    const epochs = parseFloat(document.getElementById('ft-epochs').value);
    const batchSize = parseInt(document.getElementById('ft-batchsize').value);
    const lr = parseFloat(document.getElementById('ft-lr').value);
    const maxLen = parseInt(document.getElementById('ft-maxlen').value);
    const r = parseInt(document.getElementById('ft-lora-r').value);
    const alpha = parseInt(document.getElementById('ft-lora-alpha').value);
    const datasetText = document.getElementById('ft-dataset-text').value.trim();
    
    let selectedDatasetFilename = null;
    const placeholderVal = document.getElementById('ft-dataset-text').placeholder;
    if (placeholderVal.includes('Using saved file:')) {
        selectedDatasetFilename = placeholderVal.substring(18).trim();
    }
    
    if (!datasetText && !selectedDatasetFilename) {
        alert('Please provide some dataset training examples or select a saved dataset file!');
        return;
    }
    
    // UI states
    clearConsole();
    appendConsoleLine('Preparing training job on backend server...', 'text-muted');
    
    document.getElementById('btn-start-training').disabled = true;
    document.getElementById('btn-start-training').classList.add('hidden');
    document.getElementById('btn-stop-training').classList.remove('hidden');
    
    document.getElementById('training-status-badge').className = 'badge badge-warning';
    document.getElementById('training-status-badge').innerText = 'Initializing';
    document.getElementById('training-progress-subtitle').innerText = 'Starting Python SFT training run...';
    
    try {
        const response = await fetch('/api/finetune/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model_name: model,
                epochs: epochs,
                batch_size: batchSize,
                learning_rate: lr,
                lora_r: r,
                lora_alpha: alpha,
                max_seq_length: maxLen,
                dataset_text: datasetText,
                dataset_filename: selectedDatasetFilename
            })
        });
        
        const data = await response.json();
        if (response.ok && data.success) {
            appendConsoleLine('Training process spawned successfully. Awaiting logs...', 'success');
            isFinetuningActive = true;
            // Begin fast polling
            if (finetuneInterval) clearInterval(finetuneInterval);
            finetuneInterval = setInterval(() => checkFinetuneStatus(false), 1000);
        } else {
            throw new Error(data.error || 'Failed to start fine-tuning');
        }
    } catch (err) {
        appendConsoleLine(`Launch failed: ${err.message}`, 'error');
        resetFinetuneUI();
    }
}

async function stopFineTuning() {
    if (!confirm('Are you sure you want to abort the active fine-tuning job?')) return;
    
    appendConsoleLine('Sending termination signal to child process...', 'error');
    try {
        const response = await fetch('/api/finetune/stop', { method: 'POST' });
        const data = await response.json();
        if (data.success) {
            appendConsoleLine('Training process aborted by user.', 'error');
        } else {
            throw new Error(data.error);
        }
    } catch (err) {
        appendConsoleLine(`Aborting failed: ${err.message}`, 'error');
    }
}

function resetFinetuneUI() {
    isFinetuningActive = false;
    if (finetuneInterval) {
        clearInterval(finetuneInterval);
        finetuneInterval = null;
    }
    
    document.getElementById('btn-start-training').disabled = false;
    document.getElementById('btn-start-training').classList.remove('hidden');
    document.getElementById('btn-stop-training').classList.add('hidden');
    
    document.getElementById('training-status-badge').className = 'badge';
    document.getElementById('training-status-badge').innerText = 'Idle';
    document.getElementById('training-progress-subtitle').innerText = 'Idle - Ready for training run';
}

// Check job status and stream console logs
let lastConsoleLength = 0;
async function checkFinetuneStatus(initial = false) {
    try {
        const response = await fetch('/api/finetune/status');
        const data = await response.json();
        
        if (data.running) {
            isFinetuningActive = true;
            document.getElementById('btn-start-training').classList.add('hidden');
            document.getElementById('btn-stop-training').classList.remove('hidden');
            document.getElementById('training-status-badge').className = 'badge badge-success';
            document.getElementById('training-status-badge').innerText = 'Training';
            document.getElementById('training-progress-subtitle').innerText = `Running (PID ${data.pid})`;
            
            if (!finetuneInterval) {
                finetuneInterval = setInterval(() => checkFinetuneStatus(false), 1000);
            }
        } else if (isFinetuningActive || initial) {
            // Job just finished
            resetFinetuneUI();
            if (data.error) {
                document.getElementById('training-status-badge').className = 'badge badge-error';
                document.getElementById('training-status-badge').innerText = 'Failed';
                appendConsoleLine(`Training ended with error: ${data.error}`, 'error');
                loadFinetuneHistory();
            } else if (initial && !data.logs) {
                // Was idle on boot
                resetFinetuneUI();
            } else {
                document.getElementById('training-status-badge').className = 'badge badge-success';
                document.getElementById('training-status-badge').innerText = 'Completed';
                appendConsoleLine('Fine-tuning finished successfully!', 'success');
                // Fill progress to 100%
                document.getElementById('ft-progress-bar').style.width = '100%';
                document.getElementById('progress-percent').innerText = '100% Completed';
                loadFinetuneHistory();
            }
        }
        
        // Update console box if logs changed
        if (data.logs && data.logs.length !== lastConsoleLength) {
            const newContent = data.logs.substring(lastConsoleLength);
            lastConsoleLength = data.logs.length;
            
            const lines = newContent.split('\n');
            lines.forEach(line => {
                if (!line) return;
                
                // Exclude raw JSON metric logging strings from stdout representation
                if (line.includes('METRIC_LOG:')) return;
                
                if (line.includes('SUCCESS:')) {
                    appendConsoleLine(line, 'success');
                } else if (line.includes('FAILURE:')) {
                    appendConsoleLine(line, 'error');
                } else {
                    appendConsoleLine(line);
                }
            });
        }
        
        // Reset log length tracking if no logs returned (e.g. cleared on start)
        if (!data.logs) {
            lastConsoleLength = 0;
        }
        
        // Update metrics indicators
        if (data.latestProgress) {
            const p = data.latestProgress;
            document.getElementById('progress-epoch').innerText = p.epoch !== undefined ? p.epoch : '-';
            document.getElementById('progress-step').innerText = p.step !== undefined ? `${p.step} / ${p.max_steps}` : '-';
            document.getElementById('progress-loss').innerText = p.loss !== null && p.loss !== undefined ? p.loss.toFixed(4) : 'awaiting...';
            
            const percent = p.percent_complete !== undefined ? p.percent_complete : 0;
            document.getElementById('ft-progress-bar').style.width = `${percent}%`;
            document.getElementById('progress-percent').innerText = `${percent}% Completed`;
        }

        // Draw Dynamic SVG Loss Chart
        if (data.metrics && data.metrics.length > 0) {
            document.getElementById('loss-chart-card').classList.remove('hidden');
            drawLossChart(data.metrics);
        } else {
            document.getElementById('loss-chart-card').classList.add('hidden');
        }
    } catch (err) {
        console.error('Finetune status error:', err);
    }
}

// Toggle Local Model Custom Field
function toggleLocalModelInputs() {
    const val = document.getElementById('playground-model-select').value;
    const customDiv = document.getElementById('local-model-custom-inputs');
    if (val.startsWith('local-')) {
        customDiv.classList.remove('hidden');
    } else {
        customDiv.classList.add('hidden');
    }
}

// Load datasets listed in the workspace
async function loadDatasets() {
    const tableBody = document.querySelector('#dataset-list-table tbody');
    if (!tableBody) return;
    tableBody.innerHTML = '<tr><td colspan="4" class="placeholder-text">Loading datasets...</td></tr>';
    
    try {
        const response = await fetch('/api/datasets');
        const data = await response.json();
        
        tableBody.innerHTML = '';
        if (data.success && data.datasets) {
            if (data.datasets.length === 0) {
                tableBody.innerHTML = '<tr><td colspan="4" class="placeholder-text">No dataset files (.jsonl / .csv) found in root directory.</td></tr>';
                return;
            }
            
            data.datasets.forEach(d => {
                const tr = document.createElement('tr');
                tr.style.cursor = 'pointer';
                tr.onclick = () => {
                    // Click copies to fine-tuning dataset input name field
                    document.getElementById('ft-dataset-text').value = '';
                    document.getElementById('ft-dataset-text').placeholder = `Using saved file: ${d.name}`;
                    alert(`Dataset file "${d.name}" selected for Fine-Tuning. Double-click to load content in dataset editor.`);
                };
                
                tr.ondblclick = async () => {
                    try {
                        const fileRes = await fetch(`/api/files/view?path=${encodeURIComponent(d.path)}`);
                        if (fileRes.ok) {
                            const content = await fileRes.text();
                            document.getElementById('dataset-filename').value = d.name;
                            document.getElementById('dataset-filecontent').value = content;
                        }
                    } catch (err) {
                        console.error('Failed to load dataset content:', err);
                    }
                };

                const tdName = document.createElement('td');
                tdName.className = 'repo-title';
                tdName.innerText = d.name;
                tdName.title = "Double-click to edit, Single-click to select";

                const tdSize = document.createElement('td');
                tdSize.innerText = d.size;

                const tdRows = document.createElement('td');
                tdRows.innerText = d.rowCount || '-';

                const tdMod = document.createElement('td');
                tdMod.innerText = new Date(d.modified).toLocaleString();

                tr.appendChild(tdName);
                tr.appendChild(tdSize);
                tr.appendChild(tdRows);
                tr.appendChild(tdMod);
                tableBody.appendChild(tr);
            });
        } else {
            throw new Error(data.error);
        }
    } catch (err) {
        tableBody.innerHTML = `<tr><td colspan="4" class="placeholder-text text-danger">Failed to load datasets: ${err.message}</td></tr>`;
    }
}

// Save custom dataset to workspace
async function saveCustomDataset(event) {
    event.preventDefault();
    const name = document.getElementById('dataset-filename').value.trim();
    const content = document.getElementById('dataset-filecontent').value.trim();
    
    const messageEl = document.getElementById('dataset-message');
    messageEl.className = 'alert-message';
    messageEl.innerText = '';
    
    if (!name.endsWith('.jsonl') && !name.endsWith('.csv')) {
        messageEl.classList.add('error');
        messageEl.innerText = 'Dataset file name must end with .jsonl or .csv';
        return;
    }
    
    const btn = document.getElementById('btn-save-dataset');
    btn.disabled = true;
    btn.innerText = 'Saving...';
    
    try {
        const response = await fetch('/api/datasets', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, content })
        });
        
        const data = await response.json();
        if (data.success) {
            messageEl.classList.add('success');
            messageEl.innerText = data.message || 'Dataset saved successfully!';
            
            // Auto fill training input in SFT tab
            document.getElementById('ft-dataset-text').value = '';
            document.getElementById('ft-dataset-text').placeholder = `Using saved file: ${name}`;
            
            // Reload list
            loadDatasets();
        } else {
            throw new Error(data.error);
        }
    } catch (err) {
        messageEl.classList.add('error');
        messageEl.innerText = `Failed to save: ${err.message}`;
    } finally {
        btn.disabled = false;
        btn.innerText = 'Save Dataset File';
    }
}

// Model Merger Logic
let mergeInterval = null;
async function startModelMerge() {
    const model = document.getElementById('merge-base-model').value.trim();
    const adapter = document.getElementById('merge-adapter-path').value.trim();
    const output = document.getElementById('merge-output-path').value.trim();
    
    const messageEl = document.getElementById('merge-message');
    messageEl.className = 'alert-message';
    messageEl.innerText = '';
    
    const btn = document.getElementById('btn-merge-model');
    btn.disabled = true;
    btn.innerText = 'Merging weights...';
    
    // Switch to fine tuning tab and clear console logs
    switchTab('finetune');
    clearConsole();
    appendConsoleLine('Starting LoRA adapter model merging on CPU...', 'text-muted');
    
    try {
        const response = await fetch('/api/finetune/merge', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model, adapter, output })
        });
        
        const data = await response.json();
        if (data.success) {
            appendConsoleLine('PEFT merge process spawned on server. Checking log outputs...', 'success');
            if (mergeInterval) clearInterval(mergeInterval);
            mergeInterval = setInterval(checkMergeStatus, 1000);
        } else {
            throw new Error(data.error);
        }
    } catch (err) {
        messageEl.classList.add('error');
        messageEl.innerText = `Failed to launch merger: ${err.message}`;
        btn.disabled = false;
        btn.innerText = 'Merge Adapter Weights';
    }
}

let lastMergeLogLength = 0;
async function checkMergeStatus() {
    try {
        const response = await fetch('/api/finetune/merge-status');
        const data = await response.json();
        
        const messageEl = document.getElementById('merge-message');
        const btn = document.getElementById('btn-merge-model');
        
        // Render logs in terminal
        if (data.logs && data.logs.length !== lastMergeLogLength) {
            const newContent = data.logs.substring(lastMergeLogLength);
            lastMergeLogLength = data.logs.length;
            
            const lines = newContent.split('\n');
            lines.forEach(line => {
                if (!line) return;
                if (line.includes('SUCCESS:')) {
                    appendConsoleLine(line, 'success');
                } else if (line.includes('FAILURE:')) {
                    appendConsoleLine(line, 'error');
                } else {
                    appendConsoleLine(line);
                }
            });
        }
        
        if (!data.running) {
            clearInterval(mergeInterval);
            mergeInterval = null;
            btn.disabled = false;
            btn.innerText = 'Merge Adapter Weights';
            lastMergeLogLength = 0;
            
            if (data.success) {
                messageEl.className = 'alert-message success';
                messageEl.innerText = 'PEFT model merging completed successfully! Stand-alone weights saved.';
            } else {
                messageEl.className = 'alert-message error';
                messageEl.innerText = data.error || 'PEFT model merging failed. See console logs.';
            }
        }
    } catch (err) {
        console.error('Failed to query merge status:', err);
    }
}

// SVG Loss Chart Drawer
function drawLossChart(metrics) {
    const svg = document.getElementById('loss-svg');
    const path = document.getElementById('loss-path');
    const pointsGroup = document.getElementById('loss-points');
    if (!svg || !path || !pointsGroup) return;

    // Filter out metrics without valid loss
    const validMetrics = metrics.filter(m => m.loss !== null && m.loss !== undefined && m.step !== undefined);
    if (validMetrics.length === 0) return;

    const steps = validMetrics.map(m => m.step);
    const losses = validMetrics.map(m => m.loss);

    const maxStep = Math.max(...steps, 10);
    
    // Dynamic Y axis scale
    const maxLoss = Math.max(...losses, 1.0);
    const yMax = Math.ceil(maxLoss * 1.1 * 10) / 10; 

    // Update Y Labels
    const texts = svg.querySelectorAll('text');
    if (texts.length >= 4) {
        texts[0].textContent = yMax.toFixed(1);
        texts[1].textContent = (yMax * 0.66).toFixed(1);
        texts[2].textContent = (yMax * 0.33).toFixed(1);
        texts[3].textContent = '0.0';
    }

    const chartWidth = 350;
    const chartHeight = 90;
    const startX = 40;
    const startY = 100;

    let points = [];
    pointsGroup.innerHTML = ''; 

    validMetrics.forEach(m => {
        const x = startX + (m.step / maxStep) * chartWidth;
        const y = startY - (m.loss / yMax) * chartHeight;

        points.push({ x, y });

        // Circle marker
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', x.toString());
        circle.setAttribute('cy', y.toString());
        circle.setAttribute('r', '2.5');
        circle.setAttribute('class', 'loss-point');
        circle.setAttribute('fill', 'var(--accent-teal)');
        circle.setAttribute('stroke', 'rgba(0,0,0,0.5)');
        circle.setAttribute('stroke-width', '0.5');
        
        const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
        title.textContent = `Step ${m.step}: Loss ${m.loss.toFixed(4)}`;
        circle.appendChild(title);
        
        pointsGroup.appendChild(circle);
    });

    if (points.length > 0) {
        let d = `M ${points[0].x} ${points[0].y}`;
        for (let i = 1; i < points.length; i++) {
            d += ` L ${points[i].x} ${points[i].y}`;
        }
        path.setAttribute('d', d);
    }
}

// --- MODEL DOWNLOADER WORKSPACE ---
function onDownloadSourceChange() {
    const source = document.getElementById('dl-source-select').value;
    const repoGroup = document.getElementById('dl-repo-id-group');
    const urlGroup = document.getElementById('dl-url-group');
    const repoLabel = repoGroup.querySelector('label');
    const repoHelp = repoGroup.querySelector('.input-help');
    const repoInput = document.getElementById('dl-repoid-input');
    const urlLabel = urlGroup.querySelector('label');
    const urlHelp = urlGroup.querySelector('.input-help');
    const urlInput = document.getElementById('dl-url-input');
    
    if (source === 'huggingface' || source === 'modelscope') {
        repoGroup.classList.remove('hidden');
        urlGroup.classList.add('hidden');
        if (source === 'huggingface') {
            repoLabel.innerText = 'Hugging Face Repo ID';
            repoInput.placeholder = 'e.g. facebook/opt-125m';
            repoHelp.innerText = 'Will snapshot download all weights & config files.';
        } else {
            repoLabel.innerText = 'ModelScope Repo ID';
            repoInput.placeholder = 'e.g. llm-research/Meta-Llama-3-8B-Instruct';
            repoHelp.innerText = 'Will download repository files from Alibaba ModelScope.';
        }
    } else {
        repoGroup.classList.add('hidden');
        urlGroup.classList.remove('hidden');
        if (source === 'civitai') {
            urlLabel.innerText = 'Civitai Model API Link';
            urlInput.placeholder = 'e.g. https://civitai.com/api/download/models/123456?token=your_key';
            urlHelp.innerText = 'Paste the Civitai model download link (ensure you append your API token if required).';
        } else {
            urlLabel.innerText = 'Direct Model URL';
            urlInput.placeholder = 'e.g. https://example.com/model.safetensors';
            urlHelp.innerText = 'Supports direct links from any HTTP/HTTPS server.';
        }
    }
}

async function startModelDownload(event) {
    event.preventDefault();
    if (isDownloadingActive) return;
    
    const source = document.getElementById('dl-source-select').value;
    const repo_id = document.getElementById('dl-repoid-input').value.trim();
    const url = document.getElementById('dl-url-input').value.trim();
    const filename = document.getElementById('dl-filename-input').value.trim();
    const folder_name = document.getElementById('dl-folder-input').value.trim();
    
    const messageEl = document.getElementById('downloader-message');
    messageEl.className = 'alert-message';
    messageEl.innerText = '';
    
    if ((source === 'huggingface' || source === 'modelscope') && !repo_id) {
        messageEl.classList.add('error');
        messageEl.innerText = `${source === 'huggingface' ? 'Hugging Face' : 'ModelScope'} Repository ID is required.`;
        return;
    }
    if ((source === 'url' || source === 'civitai') && !url) {
        messageEl.classList.add('error');
        messageEl.innerText = 'Model URL/API Link is required.';
        return;
    }
    
    clearDownloadConsole();
    appendDownloadConsoleLine('Initiating download process on server...', 'text-muted');
    
    document.getElementById('btn-start-download').disabled = true;
    document.getElementById('btn-start-download').classList.add('hidden');
    document.getElementById('btn-stop-download').classList.remove('hidden');
    
    document.getElementById('download-status-badge').className = 'badge badge-warning';
    document.getElementById('download-status-badge').innerText = 'Starting';
    document.getElementById('download-progress-subtitle').innerText = 'Initializing downloader script...';
    
    // Normalize civitai as direct URL download on the backend
    const normalizedSource = source === 'civitai' ? 'url' : source;
    
    try {
        const response = await fetch('/api/models/download', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                source: normalizedSource,
                repo_id,
                url,
                filename,
                folder_name
            })
        });

        
        const data = await response.json();
        if (response.ok && data.success) {
            appendDownloadConsoleLine('Download script started successfully.', 'success');
            isDownloadingActive = true;
            if (downloadInterval) clearInterval(downloadInterval);
            downloadInterval = setInterval(() => checkDownloadStatus(false), 1000);
        } else {
            throw new Error(data.error || 'Failed to start download');
        }
    } catch (err) {
        appendDownloadConsoleLine(`Launch failed: ${err.message}`, 'error');
        resetDownloaderUI();
    }
}

async function stopModelDownload() {
    if (!confirm('Are you sure you want to cancel the active download?')) return;
    appendDownloadConsoleLine('Sending termination signal to downloader process...', 'error');
    try {
        const response = await fetch('/api/models/download-stop', { method: 'POST' });
        const data = await response.json();
        if (data.success) {
            appendDownloadConsoleLine('Download cancelled by user.', 'error');
        } else {
            throw new Error(data.error);
        }
    } catch (err) {
        appendDownloadConsoleLine(`Cancellation failed: ${err.message}`, 'error');
    }
}

function resetDownloaderUI() {
    isDownloadingActive = false;
    if (downloadInterval) {
        clearInterval(downloadInterval);
        downloadInterval = null;
    }
    
    document.getElementById('btn-start-download').disabled = false;
    document.getElementById('btn-start-download').classList.remove('hidden');
    document.getElementById('btn-stop-download').classList.add('hidden');
    
    document.getElementById('download-status-badge').className = 'badge';
    document.getElementById('download-status-badge').innerText = 'Idle';
    document.getElementById('download-progress-subtitle').innerText = 'Idle - Ready to download';
    lastDownloadLogLength = 0;
}

async function checkDownloadStatus(initial = false) {
    try {
        const response = await fetch('/api/models/download-status');
        const data = await response.json();
        
        if (data.running) {
            isDownloadingActive = true;
            document.getElementById('btn-start-download').classList.add('hidden');
            document.getElementById('btn-stop-download').classList.remove('hidden');
            document.getElementById('download-status-badge').className = 'badge badge-success';
            document.getElementById('download-status-badge').innerText = 'Downloading';
            document.getElementById('download-progress-subtitle').innerText = `Running (PID ${data.pid})`;
            
            if (data.progress) {
                document.getElementById('dl-progress-bar').style.width = data.progress;
                document.getElementById('download-percent').innerText = `${data.progress} Completed`;
            }
            if (data.speed) document.getElementById('download-speed').innerText = data.speed;
            if (data.downloaded) document.getElementById('download-downloaded').innerText = data.downloaded;
            
            if (!downloadInterval) {
                downloadInterval = setInterval(() => checkDownloadStatus(false), 1000);
            }
        } else if (isDownloadingActive || initial) {
            resetDownloaderUI();
            if (data.status === 'Completed') {
                document.getElementById('download-status-badge').className = 'badge badge-success';
                document.getElementById('download-status-badge').innerText = 'Completed';
                document.getElementById('dl-progress-bar').style.width = '100%';
                document.getElementById('download-percent').innerText = '100% Completed';
                appendDownloadConsoleLine('Download finished successfully!', 'success');
                loadLocalModels();
            } else if (data.status === 'Failed') {
                document.getElementById('download-status-badge').className = 'badge badge-error';
                document.getElementById('download-status-badge').innerText = 'Failed';
                appendDownloadConsoleLine(`Download failed: ${data.error}`, 'error');
            }
        }
        
        if (data.logs && data.logs.length !== lastDownloadLogLength) {
            const newContent = data.logs.substring(lastDownloadLogLength);
            lastDownloadLogLength = data.logs.length;
            const lines = newContent.split('\n');
            lines.forEach(line => {
                if (!line.trim()) return;
                if (line.startsWith('PROGRESS:')) return;
                if (line.startsWith('INFO:')) {
                    appendDownloadConsoleLine(line, 'text-muted');
                } else if (line.startsWith('SUCCESS:')) {
                    appendDownloadConsoleLine(line, 'success');
                } else if (line.startsWith('FAILURE:')) {
                    appendDownloadConsoleLine(line, 'error');
                } else {
                    appendDownloadConsoleLine(line);
                }
            });
        }
    } catch (err) {
        console.error('Failed to check download status:', err);
    }
}

function clearDownloadConsole() {
    document.getElementById('dl-console-box').innerHTML = '';
}

function appendDownloadConsoleLine(text, type = '') {
    const box = document.getElementById('dl-console-box');
    if (!box) return;
    const line = document.createElement('div');
    line.className = `console-line ${type}`;
    line.innerText = text;
    box.appendChild(line);
    box.scrollTop = box.scrollHeight;
}

async function loadLocalModels() {
    const tableBody = document.querySelector('#local-models-table tbody');
    if (!tableBody) return;
    tableBody.innerHTML = '<tr><td colspan="5" class="placeholder-text">Scanning local workspace...</td></tr>';
    
    try {
        const response = await fetch('/api/models/local');
        const data = await response.json();
        
        tableBody.innerHTML = '';
        if (data.success && data.models) {
            if (data.models.length === 0) {
                tableBody.innerHTML = '<tr><td colspan="5" class="placeholder-text">No downloaded weights found in models/ directory.</td></tr>';
                return;
            }
            
            data.models.forEach(model => {
                const tr = document.createElement('tr');
                const relativePath = `models/${model.name}`;
                const dateStr = new Date(model.modified).toLocaleString();
                const typeText = model.isDir ? 'Repo Directory' : 'Single weights File';
                
                const btn = document.createElement('button');
                btn.className = 'btn btn-secondary btn-xs';
                btn.innerText = 'Use for SFT Base';
                btn.onclick = () => {
                    document.getElementById('ft-model-select').innerHTML += `<option value="${relativePath}" selected>${relativePath}</option>`;
                    alert(`Selected ${relativePath} as Fine-tuning Base Model!`);
                    switchTab('finetune');
                };
                
                tr.innerHTML = `
                    <td class="repo-title" title="${relativePath}">${model.name}</td>
                    <td>${typeText}</td>
                    <td>${model.size}</td>
                    <td>${dateStr}</td>
                    <td id="cell-action-${model.name.replace(/\./g, '_').replace(/\//g, '_')}"></td>
                `;
                tableBody.appendChild(tr);
                tr.querySelector(`#cell-action-${model.name.replace(/\./g, '_').replace(/\//g, '_')}`).appendChild(btn);
            });
        }
    } catch (err) {
        tableBody.innerHTML = `<tr><td colspan="5" class="placeholder-text text-danger">Scan failed: ${err.message}</td></tr>`;
    }
}

async function loadFinetuneHistory() {
    const tableBody = document.querySelector('#finetune-history-table tbody');
    if (!tableBody) return;
    tableBody.innerHTML = '<tr><td colspan="10" class="placeholder-text">Loading training run history...</td></tr>';
    
    try {
        const response = await fetch('/api/finetune/history');
        const data = await response.json();
        
        tableBody.innerHTML = '';
        if (data.success && data.history) {
            if (data.history.length === 0) {
                tableBody.innerHTML = '<tr><td colspan="10" class="placeholder-text">No fine-tuning runs logged yet. Completed runs will appear here.</td></tr>';
                return;
            }
            
            data.history.forEach(run => {
                const tr = document.createElement('tr');
                tr.style.cursor = 'pointer';
                tr.title = "Click to load hyperparameters into SFT form";
                
                tr.onclick = (e) => {
                    document.getElementById('ft-model-select').value = run.model_name;
                    if (document.getElementById('ft-model-select').value !== run.model_name) {
                        const opt = document.createElement('option');
                        opt.value = run.model_name;
                        opt.textContent = run.model_name;
                        document.getElementById('ft-model-select').appendChild(opt);
                        document.getElementById('ft-model-select').value = run.model_name;
                    }
                    document.getElementById('ft-epochs').value = run.epochs;
                    document.getElementById('ft-batchsize').value = run.batch_size;
                    document.getElementById('ft-lr').value = run.learning_rate;
                    document.getElementById('ft-maxlen').value = run.max_seq_length || 256;
                    document.getElementById('ft-lora-r').value = run.lora_r;
                    document.getElementById('ft-lora-alpha').value = run.lora_alpha;
                    
                    tr.style.backgroundColor = 'rgba(99, 102, 241, 0.2)';
                    setTimeout(() => {
                        tr.style.backgroundColor = '';
                    }, 500);
                };
                
                const duration = run.duration_secs ? `${Math.floor(run.duration_secs / 60)}m ${run.duration_secs % 60}s` : '-';
                const dateStr = new Date(run.timestamp).toLocaleString();
                const isSuccess = run.status === 'Completed';
                const statusBadge = `<span class="badge ${isSuccess ? 'badge-success' : 'badge-error'}">${run.status}</span>`;
                const lossVal = run.final_loss !== null && run.final_loss !== undefined ? run.final_loss.toFixed(4) : '-';
                
                tr.innerHTML = `
                    <td>${dateStr}</td>
                    <td class="repo-title" title="${run.model_name}">${run.model_name}</td>
                    <td>${run.dataset}</td>
                    <td>${run.epochs}</td>
                    <td>${run.batch_size}</td>
                    <td>${run.learning_rate}</td>
                    <td>r=${run.lora_r}, a=${run.lora_alpha}</td>
                    <td>${statusBadge}</td>
                    <td style="font-family: monospace; font-weight: 600;">${lossVal}</td>
                    <td>${duration}</td>
                `;
                tableBody.appendChild(tr);
            });
        } else {
            throw new Error(data.error || 'Failed to fetch history');
        }
    } catch (err) {
        tableBody.innerHTML = `<tr><td colspan="10" class="placeholder-text text-danger">Failed to load run history: ${err.message}</td></tr>`;
    }
}

async function clearFinetuneHistory() {
    if (!confirm('Are you sure you want to clear all training run history log entries? This will delete run_history.json content.')) return;
    try {
        const response = await fetch('/api/finetune/history/clear');
        const data = await response.json();
        if (data.success) {
            loadFinetuneHistory();
        } else {
            throw new Error(data.error);
        }
    } catch (err) {
        alert(`Failed to clear history: ${err.message}`);
    }
}

// --- DATASET STUDIO HANDLERS ---
let datasetEntries = [];

async function loadActiveDataset() {
    try {
        const response = await fetch('/api/dataset/load');
        const data = await response.json();
        if (data.success && data.entries) {
            datasetEntries = data.entries;
            if (datasetEntries.length === 0) {
                // Load default customer support template if empty
                loadDatasetTemplate('customer_support');
            } else {
                renderDatasetTable();
                updateDatasetStats();
            }
        }
    } catch (e) {
        console.error("Failed to load active dataset:", e);
    }
}

function renderDatasetTable() {
    const tbody = document.getElementById('dataset-builder-tbody');
    if (!tbody) return;
    tbody.innerHTML = '';
    
    if (datasetEntries.length === 0) {
        tbody.innerHTML = '<tr><td colspan="4" class="placeholder-text">Dataset is empty. Click "+ Add Row" or load a template.</td></tr>';
        return;
    }
    
    datasetEntries.forEach((entry, index) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td style="font-weight: 600; color: var(--text-secondary); text-align: center; vertical-align: middle;">${index + 1}</td>
            <td>
                <textarea class="w-full ds-prompt-input" rows="2" style="background: rgba(0,0,0,0.25); border: 1px solid var(--border-color); color: #fff; padding: 6px 10px; border-radius: 6px; font-size: 0.85rem; resize: vertical;" oninput="updateDatasetEntry(${index}, 'prompt', this.value)">${entry.prompt}</textarea>
            </td>
            <td>
                <textarea class="w-full ds-response-input" rows="2" style="background: rgba(0,0,0,0.25); border: 1px solid var(--border-color); color: #fff; padding: 6px 10px; border-radius: 6px; font-size: 0.85rem; resize: vertical;" oninput="updateDatasetEntry(${index}, 'response', this.value)">${entry.response}</textarea>
            </td>
            <td style="text-align: center; vertical-align: middle;">
                <button class="btn btn-danger btn-xs" onclick="deleteDatasetRow(${index})">Delete</button>
            </td>
        `;
        tbody.appendChild(tr);
    });
}

function updateDatasetEntry(index, key, value) {
    if (datasetEntries[index]) {
        datasetEntries[index][key] = value;
        updateDatasetStats();
    }
}

function addDatasetRow() {
    datasetEntries.push({ prompt: '', response: '' });
    renderDatasetTable();
    updateDatasetStats();
    
    // Scroll table to bottom
    const tableContainer = document.querySelector('#panel-dataset .table-container');
    if (tableContainer) {
        setTimeout(() => {
            tableContainer.scrollTop = tableContainer.scrollHeight;
        }, 50);
    }
}

function deleteDatasetRow(index) {
    datasetEntries.splice(index, 1);
    renderDatasetTable();
    updateDatasetStats();
}

async function saveActiveDataset() {
    const msgEl = document.getElementById('dataset-message');
    if (!msgEl) return;
    msgEl.className = 'alert-message';
    msgEl.innerText = '';
    
    // Validate
    const invalid = datasetEntries.some(e => !e.prompt.trim() || !e.response.trim());
    if (invalid) {
        msgEl.className = 'alert-message error';
        msgEl.innerText = 'Error: Dataset cannot contain empty prompts or responses.';
        return;
    }
    
    try {
        const response = await fetch('/api/dataset/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ entries: datasetEntries })
        });
        const data = await response.json();
        if (response.ok && data.success) {
            msgEl.className = 'alert-message success';
            msgEl.innerText = 'Dataset successfully written to active_dataset.jsonl';
            
            // Highlight textareas border green briefly
            const textareas = document.querySelectorAll('#dataset-builder-table textarea');
            textareas.forEach(t => {
                t.style.borderColor = 'var(--success)';
                setTimeout(() => t.style.borderColor = '', 1000);
            });
        } else {
            throw new Error(data.error || 'Failed to save dataset');
        }
    } catch (e) {
        msgEl.className = 'alert-message error';
        msgEl.innerText = `Save failed: ${e.message}`;
    }
}

function updateDatasetStats() {
    const statRows = document.getElementById('ds-stat-rows');
    if (!statRows) return;
    
    statRows.innerText = datasetEntries.length;
    
    let totalPromptChars = 0;
    const words = new Set();
    let hasEmpty = false;
    
    datasetEntries.forEach(e => {
        const prompt = e.prompt.trim();
        const response = e.response.trim();
        
        totalPromptChars += prompt.length;
        if (!prompt || !response) hasEmpty = true;
        
        const allText = `${prompt} ${response}`;
        allText.toLowerCase().split(/\s+/).forEach(w => {
            if (w.length > 1) words.add(w);
        });
    });
    
    const avgPromptLen = datasetEntries.length > 0 ? Math.round(totalPromptChars / datasetEntries.length) : 0;
    
    document.getElementById('ds-stat-vocab').innerText = `${words.size} unique words`;
    document.getElementById('ds-stat-avg-prompt').innerText = `${avgPromptLen} chars`;
    
    const statusVal = document.getElementById('ds-stat-status');
    if (datasetEntries.length === 0) {
        statusVal.innerHTML = '<span class="badge badge-error">Empty</span>';
    } else if (hasEmpty) {
        statusVal.innerHTML = '<span class="badge badge-warning">Empty Fields</span>';
    } else if (datasetEntries.length < 5) {
        statusVal.innerHTML = '<span class="badge badge-warning">Too Small</span>';
    } else {
        statusVal.innerHTML = '<span class="badge badge-success">Sufficient</span>';
    }
}

function loadDatasetTemplate(templateId) {
    const templates = {
        customer_support: [
            { prompt: "Hello, I received a damaged item in my order. What should I do?", response: "Hello! We are very sorry to hear that. Please reply with your Order ID and a photo of the damaged item. We will arrange a replacement or refund immediately!" },
            { prompt: "How long does shipping take for standard orders?", response: "Standard shipping typically takes 3 to 5 business days within the country. International shipping can take 7 to 14 business days depending on customs." },
            { prompt: "Can I cancel my subscription at any time?", response: "Yes, you can cancel your subscription at any time from your Account Settings panel. Your access will remain active until the end of the current billing cycle." },
            { prompt: "Do you offer discounts for annual billing?", response: "Absolutely! We offer a 20% discount if you choose our annual billing plan instead of monthly billing." },
            { prompt: "I forgot my password. How do I reset it?", response: "You can click on the 'Forgot Password' link on the login page. We will send an email with secure instructions to reset your password." }
        ],
        code_helper: [
            { prompt: "Write a python function to check if a number is prime.", response: "def is_prime(n):\n    if n <= 1:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True" },
            { prompt: "How do you define a CSS flexbox layout to center an element?", response: ".container {\n    display: flex;\n    justify-content: center;\n    align-items: center;\n}" },
            { prompt: "Explain the difference between let and const in JavaScript.", response: "'let' permits you to reassign variables and has block scope. 'const' declares a read-only reference that cannot be reassigned after declaration." },
            { prompt: "How do you handle error exceptions in Python?", response: "You can handle errors using try-except blocks:\n\ntry:\n    # code that might raise error\n    result = 10 / 0\nexcept ZeroDivisionError as e:\n    print(f'Error occurred: {e}')" }
        ],
        sql_expert: [
            { prompt: "Write an SQL query to find all employees earning more than $100,000.", response: "SELECT * FROM employees WHERE salary > 100000;" },
            { prompt: "How do you count the number of rows in a table called 'orders'?", response: "SELECT COUNT(*) FROM orders;" },
            { prompt: "Explain SQL INNER JOIN with a short query.", response: "INNER JOIN returns records that have matching values in both tables. Example:\n\nSELECT orders.id, customers.name\nFROM orders\nINNER JOIN customers ON orders.customer_id = customers.id;" },
            { prompt: "How do you sort employees by department and then by salary descending?", response: "SELECT * FROM employees ORDER BY department ASC, salary DESC;" }
        ],
        creative: [
            { prompt: "Write a short poem about space exploration.", response: "Into the dark of silent skies,\nWe lift our hearts and turn our eyes,\nBeyond the stars, past moonlit spheres,\nTo conquer all our ancient fears." },
            { prompt: "Provide a creative hook for a fantasy novel about a clockwork dragon.", response: "The metallic heart of the beast beat with a rhythmic brass click, its copper eyes glowing with steam-fueled fire, waiting to awaken in a kingdom of paper." },
            { prompt: "Describe a peaceful beach scene using sensory details.", response: "The salty breeze carried the cool mist of crashing waves. Warm, golden sand slipped slowly through my toes as the distant cries of gulls faded into the rhythmic song of the ocean." }
        ]
    };
    
    if (templates[templateId]) {
        datasetEntries = JSON.parse(JSON.stringify(templates[templateId]));
        renderDatasetTable();
        updateDatasetStats();
        
        const msgEl = document.getElementById('dataset-message');
        if (msgEl) {
            msgEl.className = 'alert-message success';
            msgEl.innerText = `${templateId.replace('_', ' ')} template loaded. Click 'Save Dataset' to write it to disk.`;
        }
    }
}
