function destroyChart(canvasId) {
    if (AppState.chartInstances[canvasId]) {
        AppState.chartInstances[canvasId].destroy();
        delete AppState.chartInstances[canvasId];
    }
}

function renderChartConfig(container, config, canvasId) {
    if (!config || !config.type) return;
    if (config.type === "heatmap") {
        container.innerHTML = `<h3>${escapeHtml(config.title)}</h3>${renderHeatmap(config)}`;
        return;
    }
    container.innerHTML = `<div class="chart-header"><div><div class="chart-title">${escapeHtml(config.title)}</div></div></div><div class="chart-wrap medium"><canvas id="${canvasId}"></canvas></div>`;
    destroyChart(canvasId);
    const ctx = $(canvasId);
    AppState.chartInstances[canvasId] = new Chart(ctx, {
        type: config.type,
        data: { labels: config.labels, datasets: config.datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { labels: { color: "#eef4f8" } } },
            scales: {
                x: { ticks: { color: "#9eb0bf" }, grid: { color: "rgba(255,255,255,.06)" } },
                y: { ticks: { color: "#9eb0bf" }, grid: { color: "rgba(255,255,255,.06)" } }
            }
        }
    });
}

function renderAllCharts(charts) {
    const container = $("chartsContainer");
    container.innerHTML = "";
    if (!charts || charts.length === 0) {
        container.innerHTML = `<div class="chart-empty">No charts available for this dataset.</div>`;
        return;
    }
    charts.forEach((chart, index) => {
        const card = document.createElement("div");
        card.className = "chart-container";
        container.appendChild(card);
        renderChartConfig(card, chart, `edaChart${index}`);
    });
}

function renderHeatmap(config) {
    const headers = config.labels.map(label => `<th>${escapeHtml(label)}</th>`).join("");
    const rows = config.rows.map(row => {
        const cells = row.values.map(value => {
            const alpha = Math.min(Math.abs(Number(value || 0)), 1);
            return `<td style="background:rgba(242,184,75,${alpha})">${escapeHtml(value)}</td>`;
        }).join("");
        return `<tr><th>${escapeHtml(row.column)}</th>${cells}</tr>`;
    }).join("");
    return `<div class="table-wrapper"><table class="preview-table"><thead><tr><th></th>${headers}</tr></thead><tbody>${rows}</tbody></table></div>`;
}

function renderForecast(result) {
    const history = result.history || [];
    const forecast = result.forecast || [];
    destroyChart("forecastChart");
    AppState.chartInstances.forecastChart = new Chart($("forecastChart"), {
        type: "line",
        data: {
            labels: [...history.map(row => row.date), ...forecast.map(row => row.date)],
            datasets: [
                { label: "History", data: [...history.map(row => row.value), ...forecast.map(() => null)], borderColor: "#58d68d" },
                { label: "Forecast", data: [...history.map(() => null), ...forecast.map(row => row.value)], borderColor: "#f2b84b" }
            ]
        },
        options: { responsive: true, maintainAspectRatio: false }
    });
    $("forecastTable").innerHTML = renderTable(forecast);
}
