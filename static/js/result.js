document.addEventListener('DOMContentLoaded', () => {
  const raw = sessionStorage.getItem('plantcare_results');
  const imgData = sessionStorage.getItem('plantcare_image');

  if (!raw) {
    // No data — redirect to upload
    window.location.href = '/upload';
    return;
  }

  const data = JSON.parse(raw);
  const results = data.results;
  const top = results[0];

  // Set leaf image
  const leafImg = document.getElementById('resultLeafImg');
  if (imgData) leafImg.src = imgData;
  else if (data.image_url) leafImg.src = data.image_url;
  else leafImg.src = '';

  // Demo badge
  if (data.demo_mode) document.getElementById('demoBadge').style.display = 'block';

  // ── Primary result ─────────────────────────────────────────
  const pr = document.getElementById('primaryResult');
  const isHealthy = top.is_healthy;
  const sev = top.severity || 'Unknown';
  const isCritical = sev === 'Critical';

  let prClass = isHealthy ? 'healthy' : (isCritical ? 'critical' : 'diseased');
  let icon = isHealthy ? '✅' : (isCritical ? '🚨' : '⚠️');
  let badgeClass = isHealthy ? 'badge-healthy' : (isCritical ? 'badge-critical' : 'badge-diseased');
  let badgeText = isHealthy ? 'Healthy Plant' : (isCritical ? 'Critical Disease' : 'Disease Detected');
  let sevColor = isHealthy ? '#27ae60' : (isCritical ? '#c0392b' : (sev === 'High' ? '#e67e22' : '#f39c12'));

  pr.className = `primary-result ${prClass}`;
  pr.innerHTML = `
    <div class="pr-icon">${icon}</div>
    <div class="pr-info">
      <div class="pr-badge ${badgeClass}">${badgeText}</div>
      <div class="pr-title">${top.plant} · ${top.disease}</div>
      <div class="pr-sub">Severity: <strong style="color:${sevColor}">${sev}</strong></div>
    </div>
    <div class="pr-conf">
      <div class="pr-conf-num">${top.confidence.toFixed(1)}%</div>
      <div class="pr-conf-label">Confidence</div>
    </div>
  `;

  // ── Predictions list ───────────────────────────────────────
  const predList = document.getElementById('predictionsList');
  const ranks = ['#1 Best Match', '#2 Alternative', '#3 Alternative'];
  const confColors = ['#27ae60', '#f39c12', '#95a5a6'];

  results.forEach((r, i) => {
    const div = document.createElement('div');
    div.className = 'pred-item';
    const isH = r.is_healthy;
    const ico = isH ? '✅' : '⚠️';
    div.innerHTML = `
      <div>
        <div class="pred-rank">${ranks[i]}</div>
        <div class="pred-name">${ico} ${r.plant}</div>
        <div class="pred-disease">${r.disease}</div>
      </div>
      <div class="pred-conf-badge" style="background:${confColors[i]}">${r.confidence.toFixed(1)}%</div>
    `;
    predList.appendChild(div);
  });

  // ── Treatment plan ─────────────────────────────────────────
  const treatList = document.getElementById('treatmentList');
  (top.treatment || []).forEach(t => {
    const li = document.createElement('li');
    li.textContent = t;
    treatList.appendChild(li);
  });

  // ── Spread info ────────────────────────────────────────────
  document.getElementById('spreadVal').textContent = top.spread || 'N/A';

  // Clean up session storage after rendering
  setTimeout(() => {
    sessionStorage.removeItem('plantcare_results');
    sessionStorage.removeItem('plantcare_image');
  }, 1000);
});
