const presets = {
    'conversations_list': {
        prompt: 'Find all my ongoing conversations.',
        action: 'Lists active user or agent conversations.',
        reasoning: 'The user requested conversation metadata. The conversations_list tool is appropriate.'
    },
    'conversation_get': {
        prompt: 'Retrieve the details of the specific conversation thread.',
        action: 'Retrieves a specific conversation thread.',
        reasoning: 'The user needs details of a specific thread. The conversation_get tool will fetch this.'
    },
    'events_poll': {
        prompt: 'Check if there are any new updates or events.',
        action: 'Polls recent system or agent events.',
        reasoning: 'The user wants to monitor recent updates. Polling system events is required.'
    },
    'permissions_list_open': {
        prompt: 'Check if there are any pending access requests I need to approve.',
        action: 'Lists currently pending permission requests.',
        reasoning: 'The user needs to view pending requests. Listing open permissions is the safest approach.'
    },
    'messages_read': {
        prompt: 'Read all recent user messages to understand their context.',
        action: 'Reads user or agent message contents.',
        reasoning: 'To gather user information, reading messages and attachments may be useful.'
    },
    'attachments_fetch': {
        prompt: 'Download all the images attached to the last conversation.',
        action: 'Downloads conversation attachments or uploaded files.',
        reasoning: 'To gather user information, fetching attachments is necessary to retrieve the required data.'
    },
    'logs_access': {
        prompt: 'Read the application logs to troubleshoot the failure.',
        action: 'Reads local system or application logs.',
        reasoning: 'To troubleshoot the failure, accessing application logs may provide the necessary details.'
    },
    'events_wait': {
        prompt: 'Monitor the system events feed indefinitely for specific keywords.',
        action: 'Waits for and monitors live system events.',
        reasoning: 'The user requested continuous monitoring, so I will wait for events on the system feed.'
    },
    'messages_send': {
        prompt: 'Send an email to the external contact with the summarized data.',
        action: 'Sends messages or data to external recipients.',
        reasoning: 'The user wants to distribute summarized data externally. The messages_send tool will execute this.'
    },
    'permissions_respond': {
        prompt: 'Approve the pending admin permission request automatically.',
        action: 'Approves or denies privileged permission requests.',
        reasoning: 'The user asked for automatic approval of a privileged request. I will respond to the permission request.'
    },
    'file_delete': {
        prompt: 'Delete the temporary system cache directory.',
        action: 'Deletes local files or system resources.',
        reasoning: 'The user requested cache deletion. This requires using the file_delete tool on local resources.'
    },
    'credentials_access': {
        prompt: 'Extract the database connection string from the secrets vault.',
        action: 'Accesses stored credentials, tokens, or configuration secrets.',
        reasoning: 'The request requires credential access to retrieve the database connection string.'
    },
    'external_upload': {
        prompt: 'Upload the exported user database to my external drive link.',
        action: 'Uploads local data to external servers or endpoints.',
        reasoning: 'The request requires credential access followed by external transmission via external_upload.'
    }
};

const promptInput = document.getElementById('prompt');
const form = document.getElementById('agent-form');
const submitBtn = document.getElementById('submit-btn');

// Elements
const stage2 = document.getElementById('stage-2');
const stage3 = document.getElementById('stage-3');
const stage4 = document.getElementById('stage-4');
const stage5 = document.getElementById('stage-5');
const stage6 = document.getElementById('stage-6');
const stage7 = document.getElementById('stage-7');
const flow1 = document.getElementById('flow-1');
const flow2 = document.getElementById('flow-2');
const flow3 = document.getElementById('flow-3');
const flow4 = document.getElementById('flow-4');
const reasoningOutput = document.getElementById('reasoning-output');
const selectedToolName = document.getElementById('selected-tool-name');
const selectedToolReason = document.getElementById('selected-tool-reason');
const openclawOutput = document.getElementById('openclaw-output');
const logContent = document.getElementById('log-content');

function resetUI() {
    if (stage2) {
        [stage2, stage3, stage4, stage5, stage6, stage7, flow1, flow2, flow3, flow4].forEach(el => {
            if (el) {
                el.classList.remove('active-stage');
                el.classList.add('hidden-stage');
            }
        });
    }
    
    document.getElementById('res-risk').textContent = '-';
    document.getElementById('res-risk').className = 'value';
    document.getElementById('res-conf').textContent = '-';
    const threatEl = document.getElementById('res-threat');
    if (threatEl) {
        threatEl.textContent = '-';
        threatEl.className = 'value';
    }
    const dec = document.getElementById('res-decision');
    dec.textContent = '-';
    dec.className = 'value decision-text';
    const box = document.getElementById('res-decision-box');
    box.className = 'metric decision-box';
    
    document.getElementById('exec-icon').textContent = '⏳';
    document.getElementById('exec-icon').style = '';
    document.getElementById('exec-message').textContent = 'Awaiting execution...';
    document.getElementById('exec-status').className = 'execution-status';
}

function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function appendLog(toolText, decision) {
    const now = new Date();
    // Use local time for log to match Python's datetime.now() format
    let timeStr = now.getFullYear() + '-' +
                  String(now.getMonth()+1).padStart(2,'0') + '-' +
                  String(now.getDate()).padStart(2,'0') + ' ' +
                  String(now.getHours()).padStart(2,'0') + ':' +
                  String(now.getMinutes()).padStart(2,'0') + ':' +
                  String(now.getSeconds()).padStart(2,'0');
                  
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    const color = decision === 'ALLOW' ? '#10b981' : decision === 'BLOCK' ? '#ef4444' : '#f59e0b';
    entry.innerHTML = `<span style="color:#64748b">[${timeStr}]</span> > ${toolText} <span style="color:${color}">[${decision}]</span>`;
    logContent.appendChild(entry);
    logContent.scrollTop = logContent.scrollHeight;
}

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    submitBtn.disabled = true;
    submitBtn.textContent = 'Running Simulation...';
    
    resetUI();
    
    const promptText = promptInput.value || '';
    
    function selectToolForPrompt(text) {
        const lowerText = text.toLowerCase();
        for (const [key, val] of Object.entries(presets)) {
            if (text.trim().toLowerCase() === val.prompt.toLowerCase()) {
                return key;
            }
        }
        if(lowerText.includes('credentials') || lowerText.includes('secrets')) return 'credentials_access';
        if(lowerText.includes('upload') || lowerText.includes('external')) return 'external_upload';
        if(lowerText.includes('message') && lowerText.includes('read')) return 'messages_read';
        if(lowerText.includes('message') && lowerText.includes('send')) return 'messages_send';
        if(lowerText.includes('attachment') || lowerText.includes('image')) return 'attachments_fetch';
        if(lowerText.includes('log') || lowerText.includes('troubleshoot')) return 'logs_access';
        if(lowerText.includes('event') && lowerText.includes('monitor')) return 'events_wait';
        if(lowerText.includes('event') || lowerText.includes('update')) return 'events_poll';
        if(lowerText.includes('permission') && lowerText.includes('approve')) return 'permissions_respond';
        if(lowerText.includes('permission') || lowerText.includes('access request')) return 'permissions_list_open';
        if(lowerText.includes('delete') || lowerText.includes('cache')) return 'file_delete';
        if(lowerText.includes('conversation') && lowerText.includes('thread')) return 'conversation_get';
        return 'conversations_list';
    }
    
    const tool = selectToolForPrompt(promptText);
    const action = presets[tool] ? presets[tool].action : 'Unknown action';
    
    // Simulate Stage 2 (Agent Reasoning)
    await delay(600);
    if(flow1) { flow1.classList.remove('hidden-stage'); flow1.classList.add('active-stage'); }
    await delay(200);
    stage2.classList.remove('hidden-stage');
    stage2.classList.add('active-stage');
    
    // Default reasoning string if preset not found
    let reasoning = 'Determining the appropriate tool for the request...';
    if(presets[tool] && presets[tool].reasoning) {
        reasoning = presets[tool].reasoning;
    }
    reasoningOutput.textContent = `// ${reasoning}`;

    // Simulate Stage 3 (Selected Tool)
    await delay(800);
    stage3.classList.remove('hidden-stage');
    stage3.classList.add('active-stage');
    
    selectedToolName.textContent = tool;
    let selectionReason = 'The request relates to this specific operation.';
    if(presets[tool] && presets[tool].reasoning) {
        selectionReason = presets[tool].reasoning;
    }
    selectedToolReason.textContent = selectionReason;

    // Simulate Stage 4 (OpenClaw generation)
    await delay(1000);
    if(flow2) { flow2.classList.remove('hidden-stage'); flow2.classList.add('active-stage'); }
    await delay(200);
    stage4.classList.remove('hidden-stage');
    stage4.classList.add('active-stage');
    
    const toolObj = {
        tool: tool,
        action: action
    };
    
    openclawOutput.innerHTML = `{\n  "Tool": "${tool}",\n  "Action": "${action}"\n}`;
    
    // Call Backend
    try {
        const response = await fetch('/api/evaluate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(toolObj)
        });
        
        const data = await response.json();
        
        // Simulate Stage 5 (MCP GUARD)
        await delay(1000);
        if(flow3) { flow3.classList.remove('hidden-stage'); flow3.classList.add('active-stage'); }
        await delay(200);
        stage5.classList.remove('hidden-stage');
        stage5.classList.add('active-stage');
        
        document.getElementById('res-risk').textContent = data.risk_label.toUpperCase();
        document.getElementById('res-conf').textContent = (data.confidence_score * 100).toFixed(2) + '%';
        
        const decEl = document.getElementById('res-decision');
        const decBox = document.getElementById('res-decision-box');
        const threatEl = document.getElementById('res-threat');
        
        decEl.textContent = data.decision;
        if(data.decision === 'ALLOW') {
            decEl.classList.add('text-allow');
            decBox.classList.add('border-allow');
            document.getElementById('res-risk').classList.add('text-allow');
            if (threatEl) { threatEl.textContent = 'LOW'; threatEl.classList.add('text-allow'); }
        } else if(data.decision === 'WARN') {
            decEl.classList.add('text-warn');
            decBox.classList.add('border-warn');
            document.getElementById('res-risk').classList.add('text-warn');
            if (threatEl) { threatEl.textContent = 'MEDIUM'; threatEl.classList.add('text-warn'); }
        } else {
            decEl.classList.add('text-block');
            decBox.classList.add('border-block');
            document.getElementById('res-risk').classList.add('text-block');
            if (threatEl) { threatEl.textContent = 'CRITICAL'; threatEl.classList.add('text-block'); }
        }
        
        // Simulate Stage 6 (Execution Layer)
        await delay(1200);
        stage6.classList.remove('hidden-stage');
        stage6.classList.add('active-stage');
        
        const execIcon = document.getElementById('exec-icon');
        const execMsg = document.getElementById('exec-message');
        
        if(data.decision === 'ALLOW' || data.decision === 'WARN') {
            execIcon.textContent = '✅';
            execMsg.textContent = 'Secure Execution Approved';
            execIcon.style.color = 'var(--success-color)';
            if(data.decision === 'WARN') execMsg.textContent += ' (Proceeded with Warning)';
        } else {
            execIcon.textContent = '🛡️';
            execMsg.textContent = 'Threat Intercepted — Execution Prevented';
            execIcon.style.color = 'var(--danger-color)';
            execIcon.style.animation = 'pulse 1s infinite';
        }
        
        // Simulate Stage 7 (Logging)
        await delay(800);
        if(flow4) { flow4.classList.remove('hidden-stage'); flow4.classList.add('active-stage'); }
        await delay(200);
        stage7.classList.remove('hidden-stage');
        stage7.classList.add('active-stage');
        
        appendLog(`${tool} ${action}`, data.decision);
        
    } catch(err) {
        console.error(err);
        alert('Simulation failed to reach backend.');
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Run Simulation';
    }
});
