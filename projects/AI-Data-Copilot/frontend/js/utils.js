const AppState = {
    profile: null,
    preview: [],
    charts: [],
    chartInstances: {}
};

function $(id) {
    return document.getElementById(id);
}

function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function setStatus(id, message, type = "") {
    const el = $(id);
    if (!el) return;
    el.textContent = message;
    el.className = `status ${type}`.trim();
}

function requireSession() {
    if (!API.sessionId) {
        throw new Error("Upload a dataset first.");
    }
}

function saveBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
}
