// static/script.js
let cameraRunning = true;

// --- NEW: capture & upload state ---
let mediaStream = null;
let sendIntervalId = null;
let canvas = null;
let ctx = null;
const targetFps = 10; // ~10 FPS upload to server

async function startCamera() {
  const video = document.getElementById('cam');

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = mediaStream;

    // lazy-create canvas
    if (!canvas) {
      canvas = document.createElement('canvas');
      ctx = canvas.getContext('2d');
    }

    // start sending frames
    if (!sendIntervalId) {
      sendIntervalId = setInterval(async () => {
        const w = video.videoWidth || 640;
        const h = video.videoHeight || 480;
        if (!w || !h) return;

        canvas.width = w;
        canvas.height = h;
        ctx.drawImage(video, 0, 0, w, h);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.7);

        // POST to Flask (/upload_frame)
        try {
          await fetch('/upload_frame', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: dataUrl })
          });
        } catch (e) {
          // ignore transient errors
        }
      }, 1000 / targetFps);
    }
  } catch (err) {
    alert('Could not access camera. Please allow permission or use HTTPS if remote.');
    console.error(err);
  }
}

function stopCamera() {
  // stop sending frames
  if (sendIntervalId) {
    clearInterval(sendIntervalId);
    sendIntervalId = null;
  }
  // stop media tracks
  if (mediaStream) {
    mediaStream.getTracks().forEach(t => t.stop());
    mediaStream = null;
  }
}

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

  // If camera was stopped, restart it (so server keeps receiving frames)
  if (!cameraRunning) {
    cameraRunning = true;
    toggleBtn.textContent = "Stop Camera";
    startCamera();
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

  if (!cameraRunning) {
    cameraRunning = true;
    toggleBtn.textContent = "Stop Camera";
    startCamera();
  }
}

function toggleCamera() {
  const stream = document.getElementById('video-stream');
  const toggleBtn = document.getElementById('toggle-camera');
  const currentModel = document.getElementById('current-model').textContent.split(': ')[1];

  if (cameraRunning) {
    // Stop both: server stream and uploads
    stream.src = "";
    stopCamera();
    toggleBtn.textContent = "Start Camera";
    cameraRunning = false;
  } else {
    // Resume uploads
    startCamera();

    // Resume server stream
    if (currentModel.toLowerCase().startsWith("ensemble")) {
      const m1 = document.getElementById('model1').value;
      const m2 = document.getElementById('model2').value;
      stream.src = `/video_feed?model=ensemble&model1=${m1}&model2=${m2}`;
    } else {
      // map label back to key
      const key = currentModel.toLowerCase();
      stream.src = `/video_feed?model=${key}`;
    }
    toggleBtn.textContent = "Stop Camera";
    cameraRunning = true;
  }
}

// NEW: kick off on load so server starts receiving frames
window.addEventListener('load', startCamera);
