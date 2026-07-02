function initAutoml() {
    $("runModelBtn").addEventListener("click", runAutoml);
    $("runForecastBtn").addEventListener("click", runForecast);
}

async function runAutoml() {
    try {
        requireSession();
        $("modelOutput").innerHTML = "<p>Training RandomForest model...</p>";
        const result = await API.automl($("targetSelect").value);
        const metrics = Object.entries(result.metrics).map(([key, value]) => `<li>${escapeHtml(key)}: ${Number(value).toFixed(4)}</li>`).join("");
        const features = result.feature_importance.map(item => `
            <div class="feature-item">
                <div class="feature-top"><span class="feature-name">${escapeHtml(item.feature)}</span><span class="feature-value">${item.importance.toFixed(4)}</span></div>
                <div class="feature-bar"><div class="feature-fill" style="width:${Math.min(item.importance * 100, 100)}%"></div></div>
            </div>
        `).join("");
        $("modelOutput").innerHTML = `<div class="insight-card"><h3>${escapeHtml(result.task)} model for ${escapeHtml(result.target)}</h3><ul>${metrics}</ul><div class="feature-list">${features}</div></div>`;
    } catch (error) {
        $("modelOutput").innerHTML = `<p class="status error">${escapeHtml(error.message)}</p>`;
    }
}

async function runForecast() {
    try {
        requireSession();
        const result = await API.forecast();
        renderForecast(result);
    } catch (error) {
        $("forecastTable").innerHTML = `<p class="status error">${escapeHtml(error.message)}</p>`;
    }
}
