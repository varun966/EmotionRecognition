from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
import mlflow.pyfunc
import mlflow
import os
import time
from collections import deque, Counter
from tensorflow.keras.applications.mobilenet import preprocess_input as preprocess_mobile

# ---- Flask App ----
app = Flask(__name__)
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# ---- DAGsHub / MLflow Setup ----
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "varun966"
repo_name = "EmotionRecognition"
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

model_name = "MobileNetV1"

def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["Staging"])
    if not latest_version:
        latest_version = client.get_latest_versions(model_name, stages=["None"])
    return latest_version[0].version if latest_version else None

model_version = get_latest_model_version(model_name)
model_uri = f'models:/{model_name}/{model_version}'
print(f"Fetching model from: {model_uri}")
model = mlflow.pyfunc.load_model(model_uri)

# ---- Face Detector ----
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ---- Emotion Memory & Frame Skipping ----
emotion_history = deque(maxlen=30)
FRAME_SKIP = 3

# ---- Video Stream Generator ----
def gen_frames():
    cap = cv2.VideoCapture(0)
    frame_count = 0

    while True:
        start_time = time.time()

        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(equalized, scaleFactor=1.3, minNeighbors=5)

        predictions_this_frame = []

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            try:
                face_img = cv2.resize(face_img, (224, 224)).astype("float32")
            except:
                continue

            face_img = preprocess_mobile(face_img)
            face_batch = np.expand_dims(face_img, axis=0)

            predictions = model.predict(face_batch)
            predicted_label = emotion_labels[np.argmax(predictions)]
            predictions_this_frame.append(predicted_label)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, predicted_label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Track overall emotion
        if predictions_this_frame:
            emotion_history.extend(predictions_this_frame)

        if emotion_history:
            overall_emotion = Counter(emotion_history).most_common(1)[0][0]
        else:
            overall_emotion = "Waiting..."

        # FPS calculation
        fps = 1.0 / (time.time() - start_time + 1e-5)

        # Overlay FPS and overall emotion
        overlay = np.zeros_like(frame)
        cv2.putText(overlay, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(overlay, f"Overall Emotion: {overall_emotion}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        frame = cv2.addWeighted(frame, 1.0, overlay, 0.4, 0)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# ---- Routes ----
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ---- Main ----
if __name__ == "__main__":
    app.run(debug=True)
