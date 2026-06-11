const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const os = require('os');
const { spawn, exec } = require('child_process');
require('dotenv').config({ path: path.join(__dirname, '.env') });

const app = express();
const PORT = 8500;
const ENV_PATH = path.join(__dirname, '.env');
const VENV_PYTHON = path.join(__dirname, '.venv', 'bin', 'python');
const HISTORY_PATH = path.join(__dirname, 'run_history.json');

let pythonInfo = {
    pythonVersion: 'Checking...',
    torchVersion: 'Checking...',
    device: 'Checking...'
};

function checkPythonEnv() {
    const checkScript = `import sys
import torch
device = "CPU"
if torch.cuda.is_available():
    device = "CUDA GPU"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "Apple Silicon GPU (MPS)"
print(f"{sys.version.split()[0]} | {torch.__version__} | {device}", end="")`;
    const proc = spawn(VENV_PYTHON, ['-c', checkScript]);
    let stdout = '';
    proc.stdout.on('data', data => stdout += data.toString());
    proc.on('close', code => {
        if (code === 0 && stdout.trim()) {
            const parts = stdout.trim().split(' | ');
            if (parts.length >= 3) {
                pythonInfo.pythonVersion = parts[0];
                pythonInfo.torchVersion = parts[1];
                pythonInfo.device = parts[2];
                console.log(`Python Environment Check: Python ${pythonInfo.pythonVersion}, PyTorch ${pythonInfo.torchVersion}, Device: ${pythonInfo.device}`);
            }
        } else {
            console.error("Python environment check failed on startup");
            pythonInfo.pythonVersion = 'Unavailable';
            pythonInfo.torchVersion = 'Unavailable';
            pythonInfo.device = 'CPU (Fallback)';
        }
    });
}
checkPythonEnv();


// Helper: Log completed training run to local JSON history file
function logTrainingRun(run) {
    let history = [];
    if (fs.existsSync(HISTORY_PATH)) {
        try {
            history = JSON.parse(fs.readFileSync(HISTORY_PATH, 'utf8'));
        } catch (e) {
            console.error("Failed to parse run history, resetting:", e);
        }
    }
    history.unshift(run);
    fs.writeFileSync(HISTORY_PATH, JSON.stringify(history, null, 2), 'utf8');
}

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// Global state for active training job
let activeTrainingProcess = null;
let trainingStatus = {
    running: false,
    pid: null,
    metrics: [],
    latestProgress: null,
    error: null,
    logFile: path.join(__dirname, 'fine_tune.log')
};

// Global state for active merge job
let mergeStatus = {
    running: false,
    pid: null,
    success: false,
    error: null,
    logFile: path.join(__dirname, 'merge.log')
};

// Global state for model downloader
const MODELS_DIR = path.join(__dirname, 'models');
if (!fs.existsSync(MODELS_DIR)) {
    fs.mkdirSync(MODELS_DIR, { recursive: true });
}
let activeDownloadProcess = null;
let downloadStatus = {
    running: false,
    pid: null,
    progress: '0%',
    speed: '0 KB/s',
    downloaded: '0 MB / 0 MB',
    status: 'Idle',
    error: null,
    logFile: path.join(__dirname, 'download.log')
};

// Helper: Read environment variables
function getEnvConfig() {
    if (fs.existsSync(ENV_PATH)) {
        const envContent = fs.readFileSync(ENV_PATH, 'utf8');
        const config = {};
        envContent.split('\n').forEach(line => {
            const match = line.match(/^\s*([\w.-]+)\s*=\s*(.*)?$/);
            if (match) {
                let val = match[2] ? match[2].trim() : '';
                if ((val.startsWith('"') && val.endsWith('"')) || (val.startsWith("'") && val.endsWith("'"))) {
                    val = val.substring(1, val.length - 1);
                }
                config[match[1]] = val;
            }
        });
        return config;
    }
    return {};
}

// Helper: Save environment variables
function saveEnvConfig(config) {
    const lines = [];
    for (const [key, val] of Object.entries(config)) {
        if (val !== undefined && val !== null) {
            lines.push(`${key}="${val.replace(/"/g, '\\"')}"`);
        }
    }
    fs.writeFileSync(ENV_PATH, lines.join('\n'), 'utf8');
    require('dotenv').config({ path: ENV_PATH });
}

// 1. Config Endpoints
app.get('/api/config', (req, res) => {
    const config = getEnvConfig();
    const masked = {};
    for (const [key, val] of Object.entries(config)) {
        if (val) {
            masked[key] = val.substring(0, 4) + '...' + val.substring(Math.max(0, val.length - 4));
        } else {
            masked[key] = '';
        }
    }
    res.json({ success: true, config: masked });
});

app.post('/api/config', (req, res) => {
    try {
        const { HF_TOKEN, OPENROUTER_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY, COHERE_API_KEY } = req.body;
        const current = getEnvConfig();
        
        if (HF_TOKEN !== undefined && !HF_TOKEN.includes('...')) current.HF_TOKEN = HF_TOKEN;
        if (OPENROUTER_API_KEY !== undefined && !OPENROUTER_API_KEY.includes('...')) current.OPENROUTER_API_KEY = OPENROUTER_API_KEY;
        if (OPENAI_API_KEY !== undefined && !OPENAI_API_KEY.includes('...')) current.OPENAI_API_KEY = OPENAI_API_KEY;
        if (GEMINI_API_KEY !== undefined && !GEMINI_API_KEY.includes('...')) current.GEMINI_API_KEY = GEMINI_API_KEY;
        if (ANTHROPIC_API_KEY !== undefined && !ANTHROPIC_API_KEY.includes('...')) current.ANTHROPIC_API_KEY = ANTHROPIC_API_KEY;
        if (COHERE_API_KEY !== undefined && !COHERE_API_KEY.includes('...')) current.COHERE_API_KEY = COHERE_API_KEY;
        
        saveEnvConfig(current);
        res.json({ success: true, message: 'Settings saved successfully' });
    } catch (err) {
        res.status(500).json({ success: false, error: err.message });
    }
});

// 2. Unified Multi-API Playground Proxy
app.post('/api/playground/chat', async (req, res) => {
    const config = getEnvConfig();
    const { provider, model, messages, temperature, max_tokens } = req.body;
    
    const temp = temperature !== undefined ? parseFloat(temperature) : 0.7;
    const tokens = max_tokens ? parseInt(max_tokens) : 1024;
    
    try {
        if (provider === 'openrouter') {
            const apiKey = config.OPENROUTER_API_KEY;
            if (!apiKey) return res.status(400).json({ error: 'OpenRouter API key is not configured.' });
            
            const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${apiKey}`,
                    'HTTP-Referer': 'http://localhost:8500',
                    'X-Title': 'Lumina Studio'
                },
                body: JSON.stringify({
                    model: model || 'meta-llama/llama-3-8b-instruct:free',
                    messages: messages || [],
                    temperature: temp,
                    max_tokens: tokens
                })
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error?.message || `OpenRouter returned status ${response.status}`);
            return res.json(data);
            
        } else if (provider === 'openai') {
            const apiKey = config.OPENAI_API_KEY;
            if (!apiKey) return res.status(400).json({ error: 'OpenAI API key is not configured.' });
            
            const response = await fetch('https://api.openai.com/v1/chat/completions', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${apiKey}`
                },
                body: JSON.stringify({
                    model: model || 'gpt-4o-mini',
                    messages: messages || [],
                    temperature: temp,
                    max_tokens: tokens
                })
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error?.message || `OpenAI returned status ${response.status}`);
            return res.json(data);
            
        } else if (provider === 'gemini') {
            const apiKey = config.GEMINI_API_KEY;
            if (!apiKey) return res.status(400).json({ error: 'Gemini API key is not configured.' });
            
            // Convert OpenAI messages to Gemini format
            const contents = (messages || []).map(m => {
                let role = m.role;
                if (role === 'assistant') role = 'model';
                if (role === 'system') role = 'user';
                return {
                    role: role,
                    parts: [{ text: m.content }]
                };
            });
            
            const systemMessage = (messages || []).find(m => m.role === 'system');
            const reqBody = {
                contents: contents.filter(c => c.role !== 'system'),
                generationConfig: {
                    temperature: temp,
                    maxOutputTokens: tokens
                }
            };
            if (systemMessage) {
                reqBody.systemInstruction = {
                    parts: [{ text: systemMessage.content }]
                };
            }
            
            const modelName = model || 'gemini-1.5-flash';
            const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${modelName}:generateContent?key=${apiKey}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(reqBody)
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error?.message || `Gemini returned status ${response.status}`);
            
            // Map Gemini output back to OpenAI structure for frontend compatibility
            const text = data.candidates?.[0]?.content?.parts?.[0]?.text || '';
            return res.json({
                choices: [{
                    message: {
                        role: 'assistant',
                        content: text
                    }
                }]
            });
            
        } else if (provider === 'anthropic') {
            const apiKey = config.ANTHROPIC_API_KEY;
            if (!apiKey) return res.status(400).json({ error: 'Anthropic API key is not configured.' });
            
            const systemMessage = (messages || []).find(m => m.role === 'system');
            const userAssistantMessages = (messages || []).filter(m => m.role === 'user' || m.role === 'assistant');
            
            const reqBody = {
                model: model || 'claude-3-5-sonnet-20240620',
                max_tokens: tokens,
                messages: userAssistantMessages,
                temperature: temp
            };
            if (systemMessage) {
                reqBody.system = systemMessage.content;
            }
            
            const response = await fetch('https://api.anthropic.com/v1/messages', {
                method: 'POST',
                headers: {
                    'x-api-key': apiKey,
                    'anthropic-version': '2023-06-01',
                    'content-type': 'application/json'
                },
                body: JSON.stringify(reqBody)
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error?.message || `Anthropic returned status ${response.status}`);
            
            const text = data.content?.[0]?.text || '';
            return res.json({
                choices: [{
                    message: {
                        role: 'assistant',
                        content: text
                    }
                }]
            });
            
        } else if (provider === 'cohere') {
            const apiKey = config.COHERE_API_KEY;
            if (!apiKey) return res.status(400).json({ error: 'Cohere API key is not configured.' });
            
            const chatHistory = (messages || []).slice(0, -1).map(m => ({
                role: m.role === 'assistant' ? 'CHATBOT' : 'USER',
                message: m.content
            }));
            const latestMessage = messages && messages.length > 0 ? messages[messages.length - 1].content : '';
            
            const response = await fetch('https://api.cohere.com/v1/chat', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${apiKey}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: latestMessage,
                    model: model || 'command-r-plus',
                    temperature: temp,
                    chat_history: chatHistory
                })
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.message || `Cohere returned status ${response.status}`);
            
            return res.json({
                choices: [{
                    message: {
                        role: 'assistant',
                        content: data.text
                    }
                }]
            });
        } else {
            return res.status(400).json({ error: `Unsupported API Provider: ${provider}` });
        }
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// 2.5 Model Downloader Endpoints
app.get('/api/models/local', (req, res) => {
    try {
        if (!fs.existsSync(MODELS_DIR)) {
            return res.json({ success: true, models: [] });
        }
        const dirs = fs.readdirSync(MODELS_DIR, { withFileTypes: true });
        const models = [];
        dirs.forEach(entry => {
            const entryPath = path.join(MODELS_DIR, entry.name);
            const stats = fs.statSync(entryPath);
            if (entry.isDirectory()) {
                let sizeBytes = 0;
                const files = fs.readdirSync(entryPath);
                files.forEach(f => {
                    try {
                        const fStats = fs.statSync(path.join(entryPath, f));
                        if (fStats.isFile()) sizeBytes += fStats.size;
                    } catch (e) {}
                });
                models.push({
                    name: entry.name,
                    isDir: true,
                    path: entryPath,
                    size: (sizeBytes / (1024 * 1024)).toFixed(2) + ' MB',
                    modified: stats.mtime
                });
            } else if (entry.isFile() && (entry.name.endsWith('.bin') || entry.name.endsWith('.safetensors') || entry.name.endsWith('.gguf') || entry.name.endsWith('.json'))) {
                models.push({
                    name: entry.name,
                    isDir: false,
                    path: entryPath,
                    size: (stats.size / (1024 * 1024)).toFixed(2) + ' MB',
                    modified: stats.mtime
                });
            }
        });
        res.json({ success: true, models });
    } catch (err) {
        res.status(500).json({ success: false, error: err.message });
    }
});

app.post('/api/models/download', (req, res) => {
    if (downloadStatus.running) {
        return res.status(400).json({ error: 'A download is already running.' });
    }
    
    const { source, repo_id, url, filename, folder_name } = req.body;
    const config = getEnvConfig();
    
    downloadStatus.running = true;
    downloadStatus.progress = '0%';
    downloadStatus.speed = '0 KB/s';
    downloadStatus.downloaded = '0 MB / 0 MB';
    downloadStatus.status = 'Downloading';
    downloadStatus.error = null;
    
    const logStream = fs.createWriteStream(downloadStatus.logFile, { flags: 'w' });
    logStream.write(`=== Download Started: ${new Date().toISOString()} ===\n`);
    logStream.write(`Source: ${source}\n`);
    if (source === 'huggingface') logStream.write(`Repo: ${repo_id}\n`);
    else logStream.write(`URL: ${url}\n`);
    logStream.write(`Filename: ${filename || 'Default'}\n`);
    logStream.write(`Folder: ${folder_name || 'Default'}\n\n`);
    
    let targetFolder = folder_name;
    if (!targetFolder) {
        if (source === 'huggingface') {
            targetFolder = repo_id.replace(/\//g, '_');
        } else {
            targetFolder = 'downloads';
        }
    }
    const outputDir = path.join(MODELS_DIR, targetFolder);
    
    const args = [
        path.join(__dirname, 'download_model.py'),
        '--source', source,
        '--output-dir', outputDir
    ];
    
    if (source === 'huggingface') {
        args.push('--repo-id', repo_id);
        if (config.HF_TOKEN) {
            args.push('--token', config.HF_TOKEN);
        }
        if (filename) {
            args.push('--filename', filename);
        }
    } else {
        args.push('--url', url);
        if (filename) {
            args.push('--filename', filename);
        }
    }
    
    const proc = spawn(VENV_PYTHON, args);
    activeDownloadProcess = proc;
    downloadStatus.pid = proc.pid;
    
    proc.stdout.on('data', data => {
        const chunk = data.toString();
        logStream.write(chunk);
        
        const lines = chunk.split('\n');
        lines.forEach(line => {
            if (line.startsWith('PROGRESS:')) {
                const parts = line.substring(9).split(' | ');
                if (parts.length >= 3) {
                    downloadStatus.progress = parts[0].trim();
                    downloadStatus.speed = parts[1].replace('SPEED:', '').trim();
                    downloadStatus.downloaded = parts[2].replace('DOWNLOADED:', '').trim();
                } else if (parts.length >= 2) {
                    downloadStatus.progress = parts[0].trim();
                    downloadStatus.speed = parts[1].replace('SPEED:', '').trim();
                } else {
                    downloadStatus.progress = parts[0].trim();
                }
            }
        });
    });
    
    proc.stderr.on('data', data => {
        logStream.write(data.toString());
    });
    
    proc.on('close', code => {
        logStream.write(`\n=== Download Finished: ${new Date().toISOString()} with exit code ${code} ===\n`);
        logStream.end();
        
        downloadStatus.running = false;
        downloadStatus.pid = null;
        activeDownloadProcess = null;
        
        if (code === 0) {
            downloadStatus.status = 'Completed';
            downloadStatus.progress = '100%';
        } else {
            downloadStatus.status = 'Failed';
            downloadStatus.error = `Download failed with exit code ${code}. See download.log.`;
        }
    });
    
    res.json({ success: true, message: 'Model download process started.' });
});

app.get('/api/models/download-status', (req, res) => {
    let logs = '';
    if (fs.existsSync(downloadStatus.logFile)) {
        logs = fs.readFileSync(downloadStatus.logFile, 'utf8');
    }
    res.json({
        running: downloadStatus.running,
        pid: downloadStatus.pid,
        progress: downloadStatus.progress,
        speed: downloadStatus.speed,
        downloaded: downloadStatus.downloaded,
        status: downloadStatus.status,
        error: downloadStatus.error,
        logs: logs
    });
});

app.post('/api/models/download-stop', (req, res) => {
    if (activeDownloadProcess && downloadStatus.running) {
        try {
            activeDownloadProcess.kill('SIGTERM');
            res.json({ success: true, message: 'Download cancellation signal sent.' });
        } catch (err) {
            res.status(500).json({ success: false, error: err.message });
        }
    } else {
        res.status(400).json({ error: 'No active download process to stop.' });
    }
});

// 3. Hugging Face CLI Wrappers
app.get('/api/hf/whoami', (req, res) => {
    const config = getEnvConfig();
    const token = config.HF_TOKEN;
    if (!token) {
        return res.json({ success: false, error: 'Hugging Face Token is not set.' });
    }

    const args = [path.join(__dirname, 'hf_helper.py'), 'whoami', '--token', token];
    const proc = spawn(VENV_PYTHON, args);
    let stdout = '';
    let stderr = '';

    proc.stdout.on('data', data => stdout += data.toString());
    proc.stderr.on('data', data => stderr += data.toString());

    proc.on('close', code => {
        if (code !== 0) {
            return res.json({ success: false, error: stderr || `Exit code ${code}` });
        }
        try {
            const parsed = JSON.parse(stdout.trim());
            res.json(parsed);
        } catch (e) {
            res.json({ success: false, error: 'Failed to parse response: ' + stdout });
        }
    });
});

app.get('/api/hf/search-models', (req, res) => {
    const { query, limit } = req.query;
    if (!query) return res.status(400).json({ error: 'Query parameter is required' });

    const config = getEnvConfig();
    const args = [path.join(__dirname, 'hf_helper.py'), 'search-models', '--query', query];
    if (limit) args.push('--limit', limit);
    if (config.HF_TOKEN) args.push('--token', config.HF_TOKEN);

    const proc = spawn(VENV_PYTHON, args);
    let stdout = '';
    let stderr = '';

    proc.stdout.on('data', data => stdout += data.toString());
    proc.stderr.on('data', data => stderr += data.toString());

    proc.on('close', code => {
        if (code !== 0) return res.json({ success: false, error: stderr });
        try {
            res.json(JSON.parse(stdout.trim()));
        } catch (e) {
            res.json({ success: false, error: 'Failed to parse response: ' + stdout });
        }
    });
});

app.get('/api/hf/search-datasets', (req, res) => {
    const { query, limit } = req.query;
    if (!query) return res.status(400).json({ error: 'Query parameter is required' });

    const config = getEnvConfig();
    const args = [path.join(__dirname, 'hf_helper.py'), 'search-datasets', '--query', query];
    if (limit) args.push('--limit', limit);
    if (config.HF_TOKEN) args.push('--token', config.HF_TOKEN);

    const proc = spawn(VENV_PYTHON, args);
    let stdout = '';
    let stderr = '';

    proc.stdout.on('data', data => stdout += data.toString());
    proc.stderr.on('data', data => stderr += data.toString());

    proc.on('close', code => {
        if (code !== 0) return res.json({ success: false, error: stderr });
        try {
            res.json(JSON.parse(stdout.trim()));
        } catch (e) {
            res.json({ success: false, error: 'Failed to parse response: ' + stdout });
        }
    });
});

app.post('/api/hf/upload-model', (req, res) => {
    const { repo_id, folder } = req.body;
    if (!repo_id || !folder) return res.status(400).json({ error: 'repo_id and folder are required' });

    const config = getEnvConfig();
    const token = config.HF_TOKEN;
    if (!token) {
        return res.status(400).json({ error: 'Hugging Face Token is not set. Please configure it in Settings.' });
    }

    const targetFolder = path.resolve(__dirname, folder);
    if (!fs.existsSync(targetFolder)) {
        return res.status(400).json({ error: `Local folder not found: ${folder}` });
    }

    const args = [
        path.join(__dirname, 'hf_helper.py'), 
        'upload-model', 
        '--repo-id', repo_id, 
        '--folder', targetFolder, 
        '--token', token
    ];
    
    const proc = spawn(VENV_PYTHON, args);
    let stdout = '';
    let stderr = '';

    proc.stdout.on('data', data => stdout += data.toString());
    proc.stderr.on('data', data => stderr += data.toString());

    proc.on('close', code => {
        if (code !== 0) return res.json({ success: false, error: stderr || `Exit code ${code}` });
        try {
            res.json(JSON.parse(stdout.trim()));
        } catch (e) {
            res.json({ success: false, error: 'Failed to parse response: ' + stdout });
        }
    });
});

// 4. Fine-Tuning Controller
app.post('/api/finetune/start', (req, res) => {
    if (trainingStatus.running) {
        return res.status(400).json({ error: 'A fine-tuning process is already running.' });
    }

    const config = getEnvConfig();
    const startTime = Date.now();
    const {
        model_name,
        epochs,
        batch_size,
        learning_rate,
        lora_r,
        lora_alpha,
        max_seq_length,
        dataset_text,
        dataset_jsonl,
        dataset_filename
    } = req.body;

    // Reset status
    trainingStatus.running = true;
    trainingStatus.metrics = [];
    trainingStatus.latestProgress = null;
    trainingStatus.error = null;

    // Prepare dataset file
    let datasetPath = '';
    if (dataset_filename) {
        datasetPath = path.join(__dirname, dataset_filename);
    } else if (dataset_jsonl) {
        datasetPath = path.join(__dirname, 'active_dataset.jsonl');
        fs.writeFileSync(datasetPath, dataset_jsonl, 'utf8');
    } else if (dataset_text) {
        // Parse raw text and write to a JSONL dataset format with 'text' column
        datasetPath = path.join(__dirname, 'active_dataset.jsonl');
        const lines = dataset_text.trim().split('\n\n');
        const jsonlContent = lines.map(line => JSON.stringify({ text: line.trim() })).join('\n');
        fs.writeFileSync(datasetPath, jsonlContent, 'utf8');
    }

    // Write hyperparameter JSON config
    const ftConfig = {
        model_name: model_name || 'facebook/opt-125m',
        dataset_path: datasetPath,
        output_dir: path.join(__dirname, 'model_output'),
        lora_r: parseInt(lora_r) || 8,
        lora_alpha: parseInt(lora_alpha) || 16,
        epochs: parseFloat(epochs) || 1.0,
        batch_size: parseInt(batch_size) || 1,
        learning_rate: parseFloat(learning_rate) || 2e-4,
        max_seq_length: parseInt(max_seq_length) || 512,
        hf_token: config.HF_TOKEN
    };

    const configPath = path.join(__dirname, 'active_training_config.json');
    fs.writeFileSync(configPath, JSON.stringify(ftConfig, null, 2), 'utf8');

    // Create log stream
    const logStream = fs.createWriteStream(trainingStatus.logFile, { flags: 'w' });
    logStream.write(`=== Fine-Tuning Started: ${new Date().toISOString()} ===\n`);
    logStream.write(`Config: ${JSON.stringify(ftConfig, null, 2)}\n\n`);

    // Spawn Python Process
    const args = [path.join(__dirname, 'fine_tune.py'), '--config', configPath];
    const proc = spawn(VENV_PYTHON, args);
    activeTrainingProcess = proc;
    trainingStatus.pid = proc.pid;

    console.log(`Spawned fine-tuning process PID ${proc.pid}`);

    proc.stdout.on('data', data => {
        const chunk = data.toString();
        logStream.write(chunk);
        
        // Parse metric logs (prefixed with "METRIC_LOG:")
        const lines = chunk.split('\n');
        lines.forEach(line => {
            if (line.startsWith('METRIC_LOG:')) {
                try {
                    const metricsData = JSON.parse(line.substring(11).trim());
                    trainingStatus.metrics.push(metricsData);
                    trainingStatus.latestProgress = metricsData;
                } catch (err) {
                    console.error('Failed to parse metric log line:', line);
                }
            }
        });
    });

    proc.stderr.on('data', data => {
        const chunk = data.toString();
        logStream.write(chunk);
    });

    proc.on('close', code => {
        logStream.write(`\n=== Fine-Tuning Finished: ${new Date().toISOString()} with exit code ${code} ===\n`);
        logStream.end();
        
        trainingStatus.running = false;
        trainingStatus.pid = null;
        activeTrainingProcess = null;
        
        if (code !== 0) {
            trainingStatus.error = `Process exited with error code ${code}. Check logs for details.`;
        }
        console.log(`Fine-tuning process PID finished with code ${code}`);

        // Calculate run metrics and log run to history
        const durationSecs = Math.round((Date.now() - startTime) / 1000);
        let finalLoss = null;
        if (trainingStatus.metrics.length > 0) {
            const validMetrics = trainingStatus.metrics.filter(m => m.loss !== null && m.loss !== undefined);
            if (validMetrics.length > 0) {
                finalLoss = validMetrics[validMetrics.length - 1].loss;
            }
        }
        
        logTrainingRun({
            id: Date.now().toString(),
            timestamp: new Date().toISOString(),
            model_name: ftConfig.model_name,
            epochs: ftConfig.epochs,
            batch_size: ftConfig.batch_size,
            learning_rate: ftConfig.learning_rate,
            lora_r: ftConfig.lora_r,
            lora_alpha: ftConfig.lora_alpha,
            dataset: dataset_filename || (dataset_text ? 'inline_text' : 'synthetic'),
            status: code === 0 ? 'Completed' : 'Failed',
            final_loss: finalLoss,
            duration_secs: durationSecs
        });
    });

    res.json({ success: true, message: 'Fine-tuning process started successfully.' });
});

app.get('/api/finetune/status', (req, res) => {
    let logs = '';
    if (fs.existsSync(trainingStatus.logFile)) {
        // Read last 100 lines or complete file if small
        logs = fs.readFileSync(trainingStatus.logFile, 'utf8');
    }
    res.json({
        running: trainingStatus.running,
        pid: trainingStatus.pid,
        metrics: trainingStatus.metrics,
        latestProgress: trainingStatus.latestProgress,
        error: trainingStatus.error,
        logs: logs
    });
});

app.get('/api/finetune/history', (req, res) => {
    let history = [];
    if (fs.existsSync(HISTORY_PATH)) {
        try {
            history = JSON.parse(fs.readFileSync(HISTORY_PATH, 'utf8'));
        } catch (e) {
            console.error("Failed to read history file:", e);
        }
    }
    res.json({ success: true, history });
});

app.get('/api/finetune/history/clear', (req, res) => {
    try {
        fs.writeFileSync(HISTORY_PATH, JSON.stringify([], null, 2), 'utf8');
        res.json({ success: true, message: 'History cleared successfully.' });
    } catch (e) {
        res.status(500).json({ success: false, error: e.message });
    }
});

app.post('/api/finetune/stop', (req, res) => {
    if (activeTrainingProcess && trainingStatus.running) {
        try {
            // Kill child process and all its descendants
            activeTrainingProcess.kill('SIGTERM');
            res.json({ success: true, message: 'Fine-tuning process stop signal sent.' });
        } catch (err) {
            res.status(500).json({ success: false, error: err.message });
        }
    } else {
        res.status(400).json({ error: 'No active fine-tuning process to stop.' });
    }
});

// 4.5 Advanced API Routes (Local Chat, Dataset Management, PEFT Merging)
app.get('/api/dataset/load', (req, res) => {
    const datasetPath = path.join(__dirname, 'active_dataset.jsonl');
    if (!fs.existsSync(datasetPath)) {
        return res.json({ success: true, entries: [] });
    }
    try {
        const fileContent = fs.readFileSync(datasetPath, 'utf8');
        const lines = fileContent.trim().split('\n').filter(line => line.length > 0);
        const entries = [];
        lines.forEach(line => {
            try {
                const parsed = JSON.parse(line);
                const text = parsed.text || '';
                // Parse "### User: <prompt>\n### Assistant: <response>"
                const userMatch = text.match(/###\s*User:\s*([\s\S]*?)\n###\s*Assistant:\s*([\s\S]*)/i);
                if (userMatch) {
                    entries.push({
                        prompt: userMatch[1].trim(),
                        response: userMatch[2].trim()
                    });
                } else {
                    entries.push({
                        prompt: text,
                        response: ''
                    });
                }
            } catch (e) {
                entries.push({ prompt: line, response: '' });
            }
        });
        res.json({ success: true, entries });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.post('/api/dataset/save', (req, res) => {
    const { entries } = req.body;
    if (!Array.isArray(entries)) {
        return res.status(400).json({ error: 'Entries must be an array' });
    }
    const datasetPath = path.join(__dirname, 'active_dataset.jsonl');
    try {
        const jsonlContent = entries.map(entry => {
            const formattedText = `### User: ${entry.prompt.trim()}\n### Assistant: ${entry.response.trim()}`;
            return JSON.stringify({ text: formattedText });
        }).join('\n');
        fs.writeFileSync(datasetPath, jsonlContent, 'utf8');
        res.json({ success: true, message: 'Dataset saved successfully.' });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.post('/api/local/chat', (req, res) => {
    const { model, adapter, prompt, max_tokens, temperature } = req.body;
    if (!prompt) return res.status(400).json({ error: 'Prompt is required' });

    const args = [
        path.join(__dirname, 'local_inference.py'),
        '--model', model || 'facebook/opt-125m',
        '--prompt', prompt
    ];
    if (adapter) {
        const adapterPath = path.resolve(__dirname, adapter);
        if (fs.existsSync(adapterPath)) {
            args.push('--adapter', adapterPath);
        }
    }
    if (max_tokens) args.push('--max-tokens', max_tokens.toString());
    if (temperature !== undefined) args.push('--temperature', temperature.toString());

    console.log(`Spawning local inference: python3 ${args.join(' ')}`);
    const proc = spawn(VENV_PYTHON, args);
    let stdout = '';
    let stderr = '';

    proc.stdout.on('data', data => stdout += data.toString());
    proc.stderr.on('data', data => stderr += data.toString());

    proc.on('close', code => {
        if (code !== 0) {
            return res.status(500).json({ error: stderr || `Process failed with exit code ${code}` });
        }
        res.json({ success: true, text: stdout });
    });
});

app.post('/api/finetune/merge', (req, res) => {
    if (mergeStatus.running) {
        return res.status(400).json({ error: 'A model merge process is already running.' });
    }

    const { model, adapter, output } = req.body;
    
    mergeStatus.running = true;
    mergeStatus.success = false;
    mergeStatus.error = null;

    const baseModelName = model || 'facebook/opt-125m';
    const adapterPath = path.resolve(__dirname, adapter || 'model_output');
    const outputPath = path.resolve(__dirname, output || 'model_output_merged');

    const logStream = fs.createWriteStream(mergeStatus.logFile, { flags: 'w' });
    logStream.write(`=== Model Merging Started: ${new Date().toISOString()} ===\n`);
    logStream.write(`Base Model: ${baseModelName}\n`);
    logStream.write(`Adapter: ${adapterPath}\n`);
    logStream.write(`Output: ${outputPath}\n\n`);

    const args = [
        path.join(__dirname, 'merge_peft.py'),
        '--model', baseModelName,
        '--adapter', adapterPath,
        '--output', outputPath
    ];

    const proc = spawn(VENV_PYTHON, args);
    mergeStatus.pid = proc.pid;

    console.log(`Spawned merge process PID ${proc.pid}`);

    proc.stdout.on('data', data => logStream.write(data.toString()));
    proc.stderr.on('data', data => logStream.write(data.toString()));

    proc.on('close', code => {
        logStream.write(`\n=== Merging Finished: ${new Date().toISOString()} with exit code ${code} ===\n`);
        logStream.end();
        
        mergeStatus.running = false;
        mergeStatus.pid = null;
        
        if (code === 0) {
            mergeStatus.success = true;
        } else {
            mergeStatus.success = false;
            mergeStatus.error = `Merge failed with exit code ${code}. Check merge.log.`;
        }
        console.log(`Merge process PID finished with code ${code}`);
    });

    res.json({ success: true, message: 'Model merge process started.' });
});

app.get('/api/finetune/merge-status', (req, res) => {
    let logs = '';
    if (fs.existsSync(mergeStatus.logFile)) {
        logs = fs.readFileSync(mergeStatus.logFile, 'utf8');
    }
    res.json({
        running: mergeStatus.running,
        success: mergeStatus.success,
        error: mergeStatus.error,
        logs: logs
    });
});

app.get('/api/datasets', (req, res) => {
    try {
        const files = fs.readdirSync(__dirname);
        const datasets = [];
        files.forEach(file => {
            if (file.endsWith('.jsonl') || file.endsWith('.csv')) {
                const fullPath = path.join(__dirname, file);
                const stats = fs.statSync(fullPath);
                
                let rowCount = 0;
                if (file.endsWith('.jsonl')) {
                    const content = fs.readFileSync(fullPath, 'utf8');
                    rowCount = content.trim().split('\n').filter(Boolean).length;
                }

                datasets.push({
                    name: file,
                    path: fullPath,
                    size: (stats.size / 1024).toFixed(2) + ' KB',
                    rowCount: rowCount,
                    modified: stats.mtime
                });
            }
        });
        res.json({ success: true, datasets });
    } catch (err) {
        res.status(500).json({ success: false, error: err.message });
    }
});

app.post('/api/datasets', (req, res) => {
    const { name, content } = req.body;
    if (!name || !content) {
        return res.status(400).json({ error: 'Dataset name and content are required' });
    }
    
    const safeName = path.basename(name);
    if (!safeName.endsWith('.jsonl') && !safeName.endsWith('.csv')) {
        return res.status(400).json({ error: 'File name must end with .jsonl or .csv' });
    }

    try {
        const targetPath = path.join(__dirname, safeName);
        fs.writeFileSync(targetPath, content, 'utf8');
        res.json({ success: true, message: `Dataset ${safeName} saved successfully.` });
    } catch (err) {
        res.status(500).json({ success: false, error: err.message });
    }
});

app.get('/api/files/view', (req, res) => {
    const filePath = req.query.path;
    if (!filePath) return res.status(400).json({ error: 'Path parameter is required' });

    const resolvedPath = path.resolve(filePath);
    if (!resolvedPath.startsWith(__dirname)) {
        return res.status(403).json({ error: 'Access denied: path must be inside project workspace' });
    }

    try {
        if (!fs.existsSync(resolvedPath)) {
            return res.status(404).json({ error: 'File not found' });
        }
        const content = fs.readFileSync(resolvedPath, 'utf8');
        res.setHeader('Content-Type', 'text/plain');
        res.send(content);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// Helper: Calculate CPU usage cross-platform
function getCPUUsage() {
    return new Promise((resolve) => {
        const first = os.cpus();
        setTimeout(() => {
            const second = os.cpus();
            let userDiff = 0;
            let sysDiff = 0;
            let idleDiff = 0;
            for (let i = 0; i < first.length; i++) {
                if (!first[i] || !second[i]) continue;
                const t1 = first[i].times;
                const t2 = second[i].times;
                userDiff += (t2.user - t1.user) + (t2.nice - t1.nice);
                sysDiff += t2.sys - t1.sys;
                idleDiff += t2.idle - t1.idle;
            }
            const total = userDiff + sysDiff + idleDiff;
            const usage = total > 0 ? ((userDiff + sysDiff) / total) * 100 : 0;
            resolve(usage.toFixed(1) + '%');
        }, 150);
    });
}

// 5. System Stats Endpoint
app.get('/api/stats', async (req, res) => {
    try {
        const cpuUsage = await getCPUUsage();
        
        exec('df -h /', (err, dfStdout) => {
            let diskUsage = 'Unknown';
            if (!err && dfStdout) {
                const lines = dfStdout.trim().split('\n');
                if (lines.length > 1) {
                    const parts = lines[1].replace(/\s+/g, ' ').split(' ');
                    if (parts.length >= 4) {
                        // Normalize indices for standard Unix / macOS output
                        const size = parts[1];
                        const used = parts[2];
                        const avail = parts[3];
                        diskUsage = `${used} used of ${size} (${avail} free)`;
                    }
                }
            }

            const totalMem = os.totalmem();
            const freeMem = os.freemem();
            const usedMem = totalMem - freeMem;

            res.json({
                hostname: os.hostname(),
                platform: os.platform(),
                uptime: Math.floor(os.uptime()),
                cpu: os.cpus()[0] ? os.cpus()[0].model : 'Unknown',
                cpuCount: os.cpus().length,
                cpuUsage: cpuUsage,
                memory: {
                    total: (totalMem / (1024 ** 3)).toFixed(2) + ' GB',
                    used: (usedMem / (1024 ** 3)).toFixed(2) + ' GB',
                    percentage: ((usedMem / totalMem) * 100).toFixed(1) + '%'
                },
                disk: diskUsage,
                pythonInfo: pythonInfo
            });
        });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.listen(PORT, '0.0.0.0', () => {
    console.log(`AI Studio Server running at http://localhost:${PORT}`);
});
