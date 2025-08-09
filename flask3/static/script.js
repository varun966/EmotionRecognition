// static/script.js
let cameraRunning = true;

function selectModel(modelName) {
  const stream = document.getElementById('video-stream');
  const modelLabel = document.getElementById('current-model');
  const toggleBtn = document.getElementById('toggle-camera');

  stream.src = `/video_feed?model=${modelName}`;

  let displayName = '';
  if (modelName === 'mobilenet') displayName = 'MobileNet';
  else if (modelName === 'efficientnet') displayName = 'EfficientNet';
  else if (modelName === 'customcnn') displayName = 'CustomCNN';

  modelLabel.textContent = `Currently using: ${displayName}`;

  // Restart camera automatically if it was stopped
  if (!cameraRunning) {
    cameraRunning = true;
    toggleBtn.textContent = "Stop Camera";
  }
}

function selectEnsemble() {
  const model1 = document.getElementById('model1').value;
  const model2 = document.getElementById('model2').value;
  const toggleBtn = document.getElementById('toggle-camera');

  if (model1 === model2) {
    alert('Please select two different models for ensemble.');
    return;
  }
  const stream = document.getElementById('video-stream');
  const modelLabel = document.getElementById('current-model');

  stream.src = `/video_feed?model=ensemble&model1=${model1}&model2=${model2}`;
  modelLabel.textContent = `Currently using: Ensemble (${model1} + ${model2})`;

  // Restart camera automatically if it was stopped
  if (!cameraRunning) {
    cameraRunning = true;
    toggleBtn.textContent = "Stop Camera";
  }
}

function toggleCamera() {
  const stream = document.getElementById('video-stream');
  const toggleBtn = document.getElementById('toggle-camera');
  const currentModel = document.getElementById('current-model').textContent.split(': ')[1];

  if (cameraRunning) {
    // Stop the feed
    stream.src = "";
    toggleBtn.textContent = "Start Camera";
    cameraRunning = false;
  } else {
    // Resume feed
    if (currentModel.toLowerCase().startsWith("ensemble")) {
      const m1 = document.getElementById('model1').value;
      const m2 = document.getElementById('model2').value;
      stream.src = `/video_feed?model=ensemble&model1=${m1}&model2=${m2}`;
    } else {
      stream.src = `/video_feed?model=${currentModel.toLowerCase()}`;
    }
    toggleBtn.textContent = "Stop Camera";
    cameraRunning = true;
  }
}
