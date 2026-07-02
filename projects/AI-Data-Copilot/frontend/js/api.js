const API = {
    base: window.location.protocol === "file:" ? "http://127.0.0.1:8000/api" : "/api",
    sessionId: null,
    async request(path, options = {}) {
        let response;
        try {
            response = await fetch(`${this.base}${path}`, options);
        } catch (error) {
            throw new Error("Could not reach the FastAPI server. Start it with: uvicorn backend.main:app --reload");
        }
        if (!response.ok) {
            let detail = response.statusText;
            try {
                const body = await response.json();
                detail = body.detail || detail;
            } catch (_) {}
            throw new Error(detail);
        }
        const contentType = response.headers.get("content-type") || "";
        if (contentType.includes("application/json")) return response.json();
        return response.blob();
    },
    upload(file) {
        const form = new FormData();
        form.append("file", file);
        return this.request("/upload", { method: "POST", body: form });
    },
    summary() {
        return this.request(`/${this.sessionId}/summary`);
    },
    clean() {
        return this.request("/clean", { method: "POST", headers: jsonHeaders(), body: JSON.stringify({ session_id: this.sessionId }) });
    },
    automl(target) {
        return this.request("/automl", { method: "POST", headers: jsonHeaders(), body: JSON.stringify({ session_id: this.sessionId, target }) });
    },
    forecast() {
        return this.request("/forecast", { method: "POST", headers: jsonHeaders(), body: JSON.stringify({ session_id: this.sessionId, periods: 12 }) });
    },
    chat(message) {
        return this.request("/chat", { method: "POST", headers: jsonHeaders(), body: JSON.stringify({ session_id: this.sessionId, message }) });
    },
    report(type) {
        return this.request("/report", { method: "POST", headers: jsonHeaders(), body: JSON.stringify({ session_id: this.sessionId, report_type: type }) });
    },
    download() {
        return this.request(`/${this.sessionId}/download`);
    }
};

function jsonHeaders() {
    return { "Content-Type": "application/json" };
}
