const fmt = (value) => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  const num = Number(value);
  if (Math.abs(num) >= 1000) return num.toLocaleString();
  if (Math.abs(num) >= 1) return num.toFixed(3);
  return num.toFixed(4);
};

function renderMetrics(target, entries) {
  const el = document.getElementById(target);
  el.innerHTML = entries.map(([label, value]) => `
    <div class="metric">
      <div class="label">${label}</div>
      <div class="value">${value}</div>
    </div>`).join("");
}

function renderTable(id, rows, columns) {
  const table = document.getElementById(id);
  if (!rows || !rows.length) {
    table.innerHTML = "<tr><td>No data yet.</td></tr>";
    return;
  }
  const cols = columns || Object.keys(rows[0]);
  const thead = `<thead><tr>${cols.map(c => `<th>${c}</th>`).join("")}</tr></thead>`;
  const tbody = `<tbody>${rows.map(row => `<tr>${cols.map(c => `<td>${row[c] ?? ""}</td>`).join("")}</tr>`).join("")}</tbody>`;
  table.innerHTML = thead + tbody;
}

async function loadOverview() {
  const [overview, status, warnings, downloads, themes, importance, validation, tod, fp] = await Promise.all([
    fetch('/api/overview').then(r => r.json()),
    fetch('/api/status').then(r => r.json()),
    fetch('/api/bias-warnings').then(r => r.json()),
    fetch('/api/downloads').then(r => r.json()),
    fetch('/api/themes').then(r => r.json()),
    fetch('/api/indicator-importance').then(r => r.json()),
    fetch('/api/validation').then(r => r.json()),
    fetch('/api/time-of-day').then(r => r.json()),
    fetch('/api/false-positives').then(r => r.json()),
  ]);

  const summary = overview.summary || {};
  const build = summary.data_build_stats || {};
  renderMetrics('overviewGrid', [
    ['Interval', overview.data_interval],
    ['Lookback months', overview.lookback_months],
    ['Target', `${(overview.target_pct * 100).toFixed(2)}%`],
    ['Eligible rows', build.eligible_rows ?? '—'],
    ['Symbols', build.symbols ?? '—'],
    ['Themes found', (summary.themes || []).length],
  ]);

  renderMetrics('statusMeta', [
    ['Phase', status.phase],
    ['Running', status.is_running ? 'Yes' : 'No'],
    ['Progress', `${Math.round((status.progress || 0) * 100)}%`],
  ]);

  document.getElementById('logs').textContent = (status.log_lines || []).join('\n') || 'No run logs yet.';
  document.getElementById('warnings').innerHTML = (warnings.warnings || []).map(w => `<li>${w}</li>`).join('');

  renderTable('themesTable', (themes.themes || []).map(t => ({
    theme_name: t.theme_name,
    conditions: t.conditions,
    train_support: t.train_support,
    validation_lift: fmt(t.validation_lift),
    test_lift: fmt(t.test_lift),
    precision: fmt(t.precision),
    recall: fmt(t.recall),
    stability_score: fmt(t.stability_score),
    robustness_score: fmt(t.robustness_score),
  })));

  renderTable('featureTable', (importance.feature_importance || []).slice(0, 20).map(r => ({ feature: r.feature, importance: fmt(r.importance) })));
  renderTable('interactionTable', (importance.interaction_importance || []).slice(0, 20).map(r => ({ interaction: r.interaction, robustness_score: fmt(r.robustness_score), test_lift: fmt(r.test_lift) })));

  const validationRows = [];
  for (const [modelName, splits] of Object.entries(validation.metrics || {})) {
    for (const [splitName, m] of Object.entries(splits)) {
      validationRows.push({
        model: modelName,
        split: splitName,
        prevalence: fmt(m.prevalence),
        precision: fmt(m.precision),
        recall: fmt(m.recall),
        roc_auc: fmt(m.roc_auc),
        average_precision: fmt(m.average_precision),
        brier: fmt(m.brier),
      });
    }
  }
  renderTable('validationTable', validationRows);
  renderTable('todTable', (tod.rows || []).slice(0, 40).map(r => ({ theme_name: r.theme_name, bucket: r.bucket, observations: r.observations, hit_rate: fmt(r.hit_rate) })));
  renderTable('fpTable', (fp.rows || []).slice(0, 20).map(r => ({ feature: r.feature, true_positive_mean: fmt(r.true_positive_mean), false_positive_mean: fmt(r.false_positive_mean), difference: fmt(r.difference) })));

  const links = [];
  if (downloads.report_path) links.push(`<li><a href="/api/download/report">Download report</a></li>`);
  for (const name of Object.keys(downloads.artifacts || {})) {
    links.push(`<li><a href="/api/download/${name}">${name}</a></li>`);
  }
  document.getElementById('downloads').innerHTML = links.join('') || '<li>No artifacts yet.</li>';
}

document.getElementById('runBtn').addEventListener('click', async () => {
  const response = await fetch('/api/run-analysis', { method: 'POST' });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    alert(payload.detail || 'Failed to start analysis');
    return;
  }
  loadOverview();
});

loadOverview();
setInterval(loadOverview, 5000);
