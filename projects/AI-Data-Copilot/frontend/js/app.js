document.addEventListener("DOMContentLoaded", () => {
    initNavigation();
    initUpload();
    initAutoml();
    initChat();
    initActions();
});

function initNavigation() {
    document.querySelectorAll(".nav-btn").forEach(button => {
        button.addEventListener("click", () => {
            document.querySelectorAll(".nav-btn").forEach(btn => btn.classList.remove("active"));
            button.classList.add("active");
            const target = $(button.dataset.section);
            if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
        });
    });
}

function initActions() {
    $("cleanBtn").addEventListener("click", async () => {
        try {
            requireSession();
            $("cleanLog").innerHTML = "<p>Cleaning dataset...</p>";
            const data = await API.clean();
            renderDashboard(data);
            $("cleanLog").innerHTML = `<ul class="list">${data.log.map(item => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
        } catch (error) {
            $("cleanLog").innerHTML = `<p class="status error">${escapeHtml(error.message)}</p>`;
        }
    });
    $("summaryBtn").addEventListener("click", async () => {
        try {
            requireSession();
            $("summaryOutput").innerHTML = "<p>Generating insights...</p>";
            const data = await API.summary();
            renderSummary(data);
        } catch (error) {
            $("summaryOutput").innerHTML = `<p class="status error">${escapeHtml(error.message)}</p>`;
        }
    });
    $("downloadBtn").addEventListener("click", downloadProcessed);
    $("processedCsvBtn").addEventListener("click", downloadProcessed);
    document.querySelectorAll("[data-report]").forEach(button => {
        button.addEventListener("click", async () => {
            try {
                requireSession();
                const type = button.dataset.report;
                const blob = await API.report(type);
                saveBlob(blob, `${type}-report.md`);
            } catch (error) {
                alert(error.message);
            }
        });
    });
}

function renderSummary(data) {
    const summary = Array.isArray(data.summary) ? data.summary : [JSON.stringify(data.summary || data)];
    const recommendations = data.recommendations || [];
    $("summaryOutput").innerHTML = `
        <div class="insight-card">
            <h3>Summary</h3>
            <ul class="list">${summary.map(item => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
            <h3>Recommendations</h3>
            <ul class="list">${recommendations.map(item => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
        </div>
    `;
}

async function downloadProcessed() {
    try {
        requireSession();
        const blob = await API.download();
        saveBlob(blob, "processed_dataset.csv");
    } catch (error) {
        alert(error.message);
    }
}
