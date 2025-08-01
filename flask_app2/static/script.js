function selectModel(modelName) {
  const stream = document.getElementById('video-stream');
  const modelLabel = document.getElementById('current-model');

  stream.src = `/video_feed?model=${modelName}`;

  let displayName = '';
  if (modelName === 'mobilenet') displayName = 'MobileNet';
  else if (modelName === 'efficientnet') displayName = 'EfficientNet';
  else if (modelName === 'ensemble') displayName = 'Ensemble';

  modelLabel.textContent = `Currently using: ${displayName}`;
}
