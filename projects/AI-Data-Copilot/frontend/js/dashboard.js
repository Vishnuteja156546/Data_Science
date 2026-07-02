function renderDashboard(data) {
    AppState.profile = data.profile;
    AppState.preview = data.preview;
    AppState.charts = data.charts;
    renderOverview(data.profile);
    renderPreview(data.preview);
    renderProfiles(data.profile);
    renderAllCharts(data.charts);
    populateTargetSelect(data.profile);
}

function renderOverview(profile) {
    $("overviewCards").innerHTML = [
        statBox("Rows", profile.rows),
        statBox("Columns", profile.column_count),
        statBox("Missing Cells", profile.missing_cells),
        statBox("Duplicate Rows", profile.duplicate_rows)
    ].join("");
    $("healthScore").innerHTML = `<div class="health-meter"><div class="health-fill" style="width:${profile.health_score}%"></div></div><p>${profile.health_score}/100 health score</p>`;
}

function statBox(label, value) {
    return `<div class="stat-box"><span>${escapeHtml(label)}</span><h2>${escapeHtml(value)}</h2></div>`;
}

function renderPreview(rows) {
    $("previewTable").innerHTML = renderTable(rows);
}

function renderTable(rows) {
    if (!rows || rows.length === 0) return "<p>No rows to display.</p>";
    const columns = Object.keys(rows[0]);
    const head = columns.map(col => `<th>${escapeHtml(col)}</th>`).join("");
    const body = rows.map(row => `<tr>${columns.map(col => `<td>${escapeHtml(row[col])}</td>`).join("")}</tr>`).join("");
    return `<div class="table-wrapper"><table class="preview-table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function renderProfiles(profile) {
    $("profileGrid").innerHTML = profile.columns.map(column => `
        <div class="profile-card">
            <h3>${escapeHtml(column.name)}</h3>
            <div class="meta">
                Type: ${escapeHtml(column.semantic_type)}<br>
                Missing: ${escapeHtml(column.missing_count)} (${escapeHtml(column.missing_percent)}%)<br>
                Unique: ${escapeHtml(column.unique_count)}
            </div>
        </div>
    `).join("");
}

function populateTargetSelect(profile) {
    $("targetSelect").innerHTML = profile.columns
        .filter(column => column.unique_count > 1)
        .map(column => `<option value="${escapeHtml(column.name)}">${escapeHtml(column.name)}</option>`)
        .join("");
}
