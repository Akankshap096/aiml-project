const uploadCard = document.getElementById('uploadCard');
const uploadZone = document.getElementById('uploadZone');
const previewZone = document.getElementById('previewZone');
const fileInput = document.getElementById('fileInput');
const previewImg = document.getElementById('previewImg');
const loadingZone = document.getElementById('loadingZone');
const errorZone = document.getElementById('errorZone');
const analyzeBtn = document.getElementById('analyzeBtn');

let selectedFile = null;

// Drag and drop
uploadCard.addEventListener('dragover', e => { e.preventDefault(); uploadCard.classList.add('drag-over'); });
uploadCard.addEventListener('dragleave', () => uploadCard.classList.remove('drag-over'));
uploadCard.addEventListener('drop', e => {
  e.preventDefault();
  uploadCard.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) handleFile(file);
  else showError('Please drop an image file (JPG, PNG, WEBP).');
});

fileInput.addEventListener('change', e => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
});

function handleFile(file) {
  if (file.size > 16 * 1024 * 1024) { showError('File too large. Max 16MB.'); return; }
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = ev => {
    previewImg.src = ev.target.result;
    document.getElementById('pmFilename').textContent = file.name;
    document.getElementById('pmSize').textContent = `${(file.size / 1024).toFixed(1)} KB · ${file.type.split('/')[1].toUpperCase()}`;
    uploadZone.style.display = 'none';
    previewZone.style.display = 'flex';
    errorZone.style.display = 'none';
  };
  reader.readAsDataURL(file);
}

function reset() {
  selectedFile = null;
  fileInput.value = '';
  previewImg.src = '';
  previewZone.style.display = 'none';
  uploadZone.style.display = 'block';
  loadingZone.style.display = 'none';
  errorZone.style.display = 'none';
}

function showError(msg) {
  errorZone.innerHTML = `⚠️ <strong>Error:</strong> ${msg}`;
  errorZone.style.display = 'block';
  loadingZone.style.display = 'none';
}

async function analyze() {
  if (!selectedFile) return;

  loadingZone.style.display = 'block';
  uploadCard.style.display = 'none';
  errorZone.style.display = 'none';
  analyzeBtn.disabled = true;

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const res = await fetch('/predict', { method: 'POST', body: formData });
    const data = await res.json();

    if (data.error) {
      loadingZone.style.display = 'none';
      uploadCard.style.display = 'block';
      analyzeBtn.disabled = false;
      showError(data.error);
      return;
    }

    // Store results for result page
    sessionStorage.setItem('plantcare_results', JSON.stringify(data));
    // Store image preview
    const reader = new FileReader();
    reader.onload = ev => {
      sessionStorage.setItem('plantcare_image', ev.target.result);
      window.location.href = '/result';
    };
    reader.readAsDataURL(selectedFile);

  } catch (err) {
    loadingZone.style.display = 'none';
    uploadCard.style.display = 'block';
    analyzeBtn.disabled = false;
    showError('Network error: ' + err.message);
  }
}
