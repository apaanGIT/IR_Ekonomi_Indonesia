/**
 * main.js — Interaksi UI & Keyword Highlighting
 */

function clearSearch() {
    const input = document.getElementById("search-input");
    if (input) {
        input.value = "";
        input.focus();
    }
}

/**
 * Highlight keywords dalam konten artikel
 * @param {string} elementId - id DOM container artikel
 * @param {string} queryStr - kata kunci query asli
 * @param {Array<string>} tokensArr - token hasil preprocessing
 */
function highlightKeywords(elementId, queryStr, tokensArr) {
    const container = document.getElementById(elementId);
    if (!container) return;

    // Kumpulkan kata kunci dari query mentah dan tokens
    let keywords = new Set();

    if (queryStr) {
        const words = queryStr.toLowerCase().split(/\s+/);
        words.forEach(w => {
            const clean = w.replace(/[^a-z0-9]/gi, '');
            if (clean.length > 2) keywords.add(clean);
        });
    }

    if (Array.isArray(tokensArr)) {
        tokensArr.forEach(t => {
            if (t && t.length > 2) keywords.add(t.toLowerCase());
        });
    }

    if (keywords.size === 0) return;

    let text = container.textContent;

    // Urutkan kata dari yang terpanjang ke terpendek agar matching optimal
    const sortedKeywords = Array.from(keywords).sort((a, b) => b.length - a.length);
    const escaped = sortedKeywords.map(k => k.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
    const regex = new RegExp(`\\b(${escaped.join('|')})\\b`, 'gi');

    // Ganti dengan tag mark
    const highlighted = text.replace(regex, '<mark class="highlight">$1</mark>');
    container.innerHTML = highlighted;
}
