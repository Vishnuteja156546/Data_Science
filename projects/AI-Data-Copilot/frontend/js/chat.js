function initChat() {
    $("chatBtn").addEventListener("click", sendChat);
    $("chatInput").addEventListener("keydown", event => {
        if (event.key === "Enter") sendChat();
    });
    addMessage("ai", "Upload a dataset, then ask me for summaries, risks, model ideas, or charts.");
}

async function sendChat() {
    const input = $("chatInput");
    const message = input.value.trim();
    if (!message) return;
    try {
        requireSession();
        input.value = "";
        addMessage("user", message);
        const response = await API.chat(message);
        addMessage("ai", response.answer);
        if (response.chart) {
            const container = document.createElement("div");
            container.className = "chart-container";
            $("chatHistory").appendChild(container);
            renderChartConfig(container, response.chart, "chatIntentChart");
        }
    } catch (error) {
        addMessage("ai", error.message);
    }
}

function addMessage(role, text) {
    const wrap = document.createElement("div");
    wrap.className = `message ${role === "user" ? "user" : "ai"}`;
    wrap.innerHTML = `<div class="avatar ${role === "user" ? "user" : "ai"}">${role === "user" ? "You" : "AI"}</div><div class="bubble">${escapeHtml(text).replaceAll("\n", "<br>")}</div>`;
    $("chatHistory").appendChild(wrap);
    $("chatHistory").scrollTop = $("chatHistory").scrollHeight;
}
