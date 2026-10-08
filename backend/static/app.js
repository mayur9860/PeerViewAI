/**
 * PeerReview AI — Frontend Application
 *
 * Vanilla JS, no framework, no bundler.
 * Communicates with the FastAPI backend via fetch().
 */

// ==========================================================
// State
// ==========================================================

const state = {
    userId: null,
    userEmail: null,
    drafts: [],
    currentDraftId: null,
    currentDraft: null,
    sections: {},        // { section_type: { id, content, ... } }
    archetype: null,
    rankedSections: [],
    currentCritiqueSection: null,
};

// ==========================================================
// DOM References
// ==========================================================

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const DOM = {
    userSetupCard: $('#user-setup-card'),
    inputEmail: $('#input-email'),
    btnCreateUser: $('#btn-create-user'),
    userEmail: $('#user-email'),

    btnNewDraft: $('#btn-new-draft'),
    newDraftCard: $('#new-draft-card'),
    inputTitle: $('#input-title'),
    selectField: $('#select-field'),
    btnSubmitDraft: $('#btn-submit-draft'),
    btnCancelDraft: $('#btn-cancel-draft'),
    draftList: $('#draft-list'),

    sectionEditor: $('#section-editor'),
    sectionEditorTitle: $('#section-editor-title'),
    selectSectionType: $('#select-section-type'),
    textareaSection: $('#textarea-section'),
    btnSaveSection: $('#btn-save-section'),
    btnAnalyze: $('#btn-analyze'),

    inputSearch: $('#input-search'),
    btnSearch: $('#btn-search'),
    searchResults: $('#search-results'),

    archetypeDisplay: $('#archetype-display'),
    archetypeBadge: $('#archetype-badge'),
    archetypeLabel: $('#archetype-label'),
    archetypeConfidence: $('#archetype-confidence'),
    archetypeDesc: $('#archetype-desc'),

    rankedSections: $('#ranked-sections'),
    featuresDisplay: $('#features-display'),
    featuresGrid: $('#features-grid'),

    chatContainer: $('#chat-container'),
    critiqueStatus: $('#critique-status'),
    critiqueHistory: $('#critique-history'),
    historyList: $('#history-list'),

    loadingOverlay: $('#loading-overlay'),
    loadingText: $('#loading-text'),
    toastContainer: $('#toast-container'),
};

// ==========================================================
// API Helper
// ==========================================================

const API_BASE = '';

async function api(method, path, body = null) {
    const opts = {
        method,
        headers: { 'Content-Type': 'application/json' },
    };
    if (body) opts.body = JSON.stringify(body);

    const res = await fetch(`${API_BASE}${path}`, opts);
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || `API error ${res.status}`);
    }
    return res.json();
}

// ==========================================================
// UI Helpers
// ==========================================================

function showLoading(text = 'Processing...') {
    DOM.loadingText.textContent = text;
    DOM.loadingOverlay.classList.remove('hidden');
}

function hideLoading() {
    DOM.loadingOverlay.classList.add('hidden');
}

function toast(message, type = 'info') {
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.textContent = message;
    DOM.toastContainer.appendChild(el);
    setTimeout(() => {
        el.classList.add('hiding');
        setTimeout(() => el.remove(), 300);
    }, 4000);
}

function show(el) { el.classList.remove('hidden'); }
function hide(el) { el.classList.add('hidden'); }

function renderKaTeX(container) {
    if (typeof renderMathInElement === 'function') {
        renderMathInElement(container, {
            delimiters: [
                { left: '$$', right: '$$', display: true },
                { left: '$', right: '$', display: false },
            ],
            throwOnError: false,
        });
    }
}

const SECTION_LABELS = {
    intro: 'Introduction',
    related_work: 'Related Work',
    methods: 'Methods',
    results: 'Results',
    discussion: 'Discussion',
    conclusion: 'Conclusion',
};

const ARCHETYPE_DESCRIPTIONS = {
    'The Overclaimer': 'Tends to use strong, unqualified assertions where evidence warrants more caution.',
    'The Undersold Contributor': 'Buries novel findings in hedged language, underselling contributions.',
    'The Wall-of-Citations Writer': 'Lists citations without synthesis or narrative connection.',
    'The Structure-Drifter': 'Content migrates between sections, blurring structural boundaries.',
    'The Jargon-Heavy Novice': 'High density of specialized terms that may obscure rather than clarify.',
    'The Methodical Reviser': 'Revises steadily; may benefit from higher-level structural feedback.',
    'The Late-Stage Panic-Editor': 'Many rapid recent revisions after long inactivity, suggesting deadline crunch.',
    'The Data-Rich Narrative-Poor Writer': 'Results dense with data but lacking narrative interpretation.',
};

// ==========================================================
// User Management
// ==========================================================

DOM.btnCreateUser.addEventListener('click', async () => {
    const email = DOM.inputEmail.value.trim();
    if (!email) return toast('Please enter an email', 'error');

    try {
        showLoading('Creating account...');
        const user = await api('POST', '/users', { email });
        state.userId = user.id;
        state.userEmail = email;
        DOM.userEmail.textContent = email;
        hide(DOM.userSetupCard);
        show(DOM.sectionEditor);
        toast('Account created!', 'success');
    } catch (e) {
        toast(e.message, 'error');
    } finally {
        hideLoading();
    }
});

// ==========================================================
// Draft Management
// ==========================================================

DOM.btnNewDraft.addEventListener('click', () => {
    if (!state.userId) return toast('Create an account first', 'error');
    show(DOM.newDraftCard);
});

DOM.btnCancelDraft.addEventListener('click', () => {
    hide(DOM.newDraftCard);
});

DOM.btnSubmitDraft.addEventListener('click', async () => {
    const title = DOM.inputTitle.value.trim();
    const field = DOM.selectField.value;
    if (!title) return toast('Enter a title', 'error');

    try {
        showLoading('Creating draft...');
        const draft = await api('POST', '/drafts', {
            title,
            field,
            user_id: state.userId,
        });
        state.drafts.push(draft);
        renderDraftList();
        hide(DOM.newDraftCard);
        DOM.inputTitle.value = '';
        selectDraft(draft.id);
        toast(`Draft "${title}" created!`, 'success');
    } catch (e) {
        toast(e.message, 'error');
    } finally {
        hideLoading();
    }
});

function renderDraftList() {
    if (state.drafts.length === 0) {
        DOM.draftList.innerHTML = '<p class="empty-state">No drafts yet</p>';
        return;
    }

    const fieldIcons = { cs: '💻', biology: '🧬', economics: '📈' };

    DOM.draftList.innerHTML = state.drafts.map(d => `
        <div class="draft-item ${d.id === state.currentDraftId ? 'active' : ''}"
             data-id="${d.id}" onclick="selectDraft(${d.id})">
            <div class="draft-icon">${fieldIcons[d.field] || '📄'}</div>
            <div class="draft-info">
                <div class="draft-title">${d.title}</div>
                <div class="draft-field">${d.field}</div>
            </div>
        </div>
    `).join('');
}

// Make selectDraft global so onclick works
window.selectDraft = async function(draftId) {
    state.currentDraftId = draftId;
    state.currentDraft = state.drafts.find(d => d.id === draftId);
    state.sections = {};

    renderDraftList();
    show(DOM.sectionEditor);
    DOM.sectionEditorTitle.textContent = `Edit: ${state.currentDraft.title}`;

    // Reset panels
    DOM.archetypeDisplay.classList.add('hidden');
    DOM.featuresDisplay.classList.add('hidden');
    DOM.rankedSections.innerHTML = `
        <div class="empty-state">
            <div class="empty-icon">📋</div>
            <p>Analyze this draft to see revision priorities</p>
        </div>`;
    resetChat();

    toast(`Selected: ${state.currentDraft.title}`, 'info');
};

// ==========================================================
// Section Editor
// ==========================================================

DOM.btnSaveSection.addEventListener('click', async () => {
    if (!state.currentDraftId) return toast('Select a draft first', 'error');

    const sectionType = DOM.selectSectionType.value;
    const content = DOM.textareaSection.value.trim();
    if (!content) return toast('Enter section content', 'error');

    try {
        showLoading('Saving section...');
        const section = await api('POST', `/drafts/${state.currentDraftId}/sections`, {
            section_type: sectionType,
            content,
        });
        state.sections[sectionType] = section;
        toast(`${SECTION_LABELS[sectionType]} saved!`, 'success');
    } catch (e) {
        toast(e.message, 'error');
    } finally {
        hideLoading();
    }
});

// ==========================================================
// Analysis
// ==========================================================

DOM.btnAnalyze.addEventListener('click', async () => {
    if (!state.currentDraftId) return toast('Select a draft first', 'error');

    try {
        showLoading('Analyzing draft... Computing features & classifying archetype');

        const result = await api('POST', `/drafts/${state.currentDraftId}/analyze`);

        state.archetype = {
            id: result.archetype_id,
            label: result.archetype_label,
            confidence: result.archetype_confidence,
        };

        // Show archetype
        show(DOM.archetypeDisplay);
        DOM.archetypeLabel.textContent = result.archetype_label;
        DOM.archetypeConfidence.textContent = `${(result.archetype_confidence * 100).toFixed(1)}%`;
        DOM.archetypeDesc.textContent = ARCHETYPE_DESCRIPTIONS[result.archetype_label] || '';

        // Show features
        show(DOM.featuresDisplay);
        DOM.featuresGrid.innerHTML = Object.entries(result.features).map(([name, val]) => `
            <div class="feature-item">
                <span class="feature-name" title="${name}">${name.replace(/_/g, ' ')}</span>
                <span class="feature-value">${val.toFixed(3)}</span>
            </div>
        `).join('');

        toast(`Archetype: ${result.archetype_label}`, 'success');

        // Auto-fetch recommendations
        await fetchRecommendations();

    } catch (e) {
        toast(e.message, 'error');
    } finally {
        hideLoading();
    }
});

async function fetchRecommendations() {
    try {
        const result = await api('GET', `/drafts/${state.currentDraftId}/recommend`);
        state.rankedSections = result.ranked_sections;
        renderRankedSections(result);
    } catch (e) {
        toast(`Recommendations: ${e.message}`, 'error');
    }
}

function renderRankedSections(result) {
    if (!result.ranked_sections || result.ranked_sections.length === 0) {
        DOM.rankedSections.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">✅</div>
                <p>All sections recently revised. Nothing to prioritize.</p>
            </div>`;
        return;
    }

    DOM.rankedSections.innerHTML = result.ranked_sections.map(s => `
        <div class="section-rank-item" onclick="startCritique(${s.section_id}, '${s.section_type}')">
            <div class="rank-number">${s.rank}</div>
            <div class="rank-info">
                <div class="rank-section-type">${SECTION_LABELS[s.section_type] || s.section_type}</div>
                <div class="rank-priority">
                    <div class="priority-bar">
                        <div class="priority-fill" style="width: ${s.priority_score * 100}%"></div>
                    </div>
                    <span class="priority-score">${(s.priority_score * 100).toFixed(0)}%</span>
                </div>
            </div>
        </div>
    `).join('');
}

// ==========================================================
// Critique
// ==========================================================

function resetChat() {
    DOM.chatContainer.innerHTML = `
        <div class="empty-state">
            <div class="empty-icon">🎓</div>
            <p>Click a section in the recommendations panel to start a critique session</p>
        </div>`;
    DOM.critiqueStatus.textContent = '';
    DOM.critiqueStatus.className = 'critique-status';
    hide(DOM.critiqueHistory);
}

window.startCritique = async function(sectionId, sectionType) {
    state.currentCritiqueSection = { id: sectionId, type: sectionType };

    DOM.chatContainer.innerHTML = '';
    DOM.critiqueStatus.textContent = 'Running...';
    DOM.critiqueStatus.className = 'critique-status info';

    try {
        showLoading(`Running Socratic critique on ${SECTION_LABELS[sectionType] || sectionType}...`);

        const result = await api('POST', `/sections/${sectionId}/critique`);

        // Render transcript
        DOM.chatContainer.innerHTML = '';

        for (const turn of result.transcript) {
            appendChatMessage(turn.role, turn.content);
        }

        // Final message (highlighted)
        if (result.final_message) {
            appendChatMessage('final', result.final_message);
        }

        // Status badge
        DOM.critiqueStatus.textContent = result.outcome;
        DOM.critiqueStatus.className = `critique-status ${result.outcome}`;

        // Render KaTeX
        renderKaTeX(DOM.chatContainer);

        toast(`Critique complete: ${result.outcome}`, 'success');

        // Load history
        await loadCritiqueHistory(sectionId);

    } catch (e) {
        toast(`Critique failed: ${e.message}`, 'error');
        DOM.critiqueStatus.textContent = 'Error';
        DOM.critiqueStatus.className = 'critique-status abandoned';
    } finally {
        hideLoading();
    }
};

function appendChatMessage(role, content) {
    const div = document.createElement('div');
    div.className = `chat-message role-${role}`;

    const roleLabels = {
        critic: '🔍 Critic',
        coach: '🎓 Coach',
        reviewer: '🛡️ Reviewer',
        system: '⚙️ System',
        final: '✨ Final Coaching Response',
    };

    div.innerHTML = `
        <div class="msg-role">${roleLabels[role] || role}</div>
        <div class="msg-content">${escapeHtml(content)}</div>
    `;

    DOM.chatContainer.appendChild(div);
    DOM.chatContainer.scrollTop = DOM.chatContainer.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

async function loadCritiqueHistory(sectionId) {
    if (!state.currentDraftId) return;

    try {
        const history = await api('GET', `/drafts/${state.currentDraftId}/sections/${sectionId}/history`);

        if (history && history.length > 0) {
            show(DOM.critiqueHistory);
            DOM.historyList.innerHTML = history.map(h => `
                <div class="history-item" onclick="viewHistorySession(${JSON.stringify(h.transcript).replace(/"/g, '&quot;')}, '${h.outcome}')">
                    <div class="history-meta">
                        <span class="history-date">${new Date(h.created_at).toLocaleString()}</span>
                        <span class="history-outcome ${h.outcome}">${h.outcome}</span>
                    </div>
                </div>
            `).join('');
        }
    } catch (e) {
        // History load failure is non-fatal
    }
}

window.viewHistorySession = function(transcript, outcome) {
    DOM.chatContainer.innerHTML = '';
    for (const turn of transcript) {
        appendChatMessage(turn.role, turn.content);
    }
    DOM.critiqueStatus.textContent = outcome;
    DOM.critiqueStatus.className = `critique-status ${outcome}`;
    renderKaTeX(DOM.chatContainer);
};

// ==========================================================
// Search
// ==========================================================

DOM.btnSearch.addEventListener('click', async () => {
    const query = DOM.inputSearch.value.trim();
    if (!query) return toast('Enter a search query', 'error');

    try {
        showLoading('Searching craft guidance...');
        const result = await api('POST', '/search', { query });

        DOM.searchResults.innerHTML = '';

        if (!result.results || result.results.length === 0) {
            DOM.searchResults.innerHTML = '<p class="empty-state" style="padding:12px">No results found</p>';
            return;
        }

        for (const r of result.results) {
            const div = document.createElement('div');
            div.className = 'search-result-item';
            div.innerHTML = `
                <div class="result-content">${escapeHtml(r.content.substring(0, 300))}${r.content.length > 300 ? '...' : ''}</div>
                <div class="result-meta">
                    ${r.section_type ? `<span class="result-tag">${r.section_type}</span>` : ''}
                    ${r.field ? `<span class="result-tag">${r.field}</span>` : ''}
                    ${r.difficulty ? `<span class="result-tag">${r.difficulty}</span>` : ''}
                    <span class="result-score">Score: ${r.relevance_score.toFixed(3)}</span>
                </div>
            `;
            DOM.searchResults.appendChild(div);
        }

        renderKaTeX(DOM.searchResults);
        toast(`Found ${result.results.length} results`, 'success');

    } catch (e) {
        toast(`Search failed: ${e.message}`, 'error');
    } finally {
        hideLoading();
    }
});

// Allow Enter key in search
DOM.inputSearch.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') DOM.btnSearch.click();
});

// ==========================================================
// Initialization
// ==========================================================

document.addEventListener('DOMContentLoaded', () => {
    // Check if user already exists in localStorage
    const savedUserId = localStorage.getItem('peerreview_userId');
    const savedEmail = localStorage.getItem('peerreview_userEmail');

    if (savedUserId && savedEmail) {
        state.userId = parseInt(savedUserId);
        state.userEmail = savedEmail;
        DOM.userEmail.textContent = savedEmail;
        hide(DOM.userSetupCard);
        show(DOM.sectionEditor);
    }

    // KaTeX auto-render on load
    if (typeof renderMathInElement === 'function') {
        renderMathInElement(document.body, {
            delimiters: [
                { left: '$$', right: '$$', display: true },
                { left: '$', right: '$', display: false },
            ],
            throwOnError: false,
        });
    }
});

// Save user to localStorage on creation
const origCreateUser = DOM.btnCreateUser.onclick;
DOM.btnCreateUser.addEventListener('click', () => {
    // Wait for state to update, then save
    setTimeout(() => {
        if (state.userId) {
            localStorage.setItem('peerreview_userId', state.userId);
            localStorage.setItem('peerreview_userEmail', state.userEmail);
        }
    }, 1000);
});
