# flask_app/app.py

from flask import Flask, render_template, Response
import os

def create_app(testing=False):
    app = Flask(__name__)
    app.testing = testing

    if not testing:
        import cv2
        import numpy as np
        import time
        import mlflow.pyfunc
        from tensorflow.keras.applications.mobilenet import preprocess_input as preprocess_mobile
        from collections import deque, Counter

        # ---- Config ----
        emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
        dagshub_token = os.getenv("DAGSHUB_TOKEN", "dummy_token")
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        mlflow.set_tracking_uri(f"https://dagshub.com/varun966/EmotionRecognition.mlflow")

        model_name = "MobileNetV1"
        def get_latest_model_version(name):
            client = mlflow.MlflowClient()
            versions = client.get_latest_versions(name, stages=["Staging"]) or client.get_latest_versions(name)
            return versions[0].version if versions else None

        model_version = get_latest_model_version(model_name)
        model_uri = f"models:/{model_name}/{model_version}"
        model = mlflow.pyfunc.load_model(model_uri)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        emotion_history = deque(maxlen=30)
        FRAME_SKIP = 3

        def gen_frames():
            cap = cv2.VideoCapture(0)
            frame_count = 0
            while True:
                success, frame = cap.read()
                if not success:
                    break
                frame_count += 1
                if frame_count % FRAME_SKIP != 0:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                equalized = cv2.equalizeHist(gray)
                faces = face_cascade.detectMultiScale(equalized, 1.3, 5)

                for (x, y, w, h) in faces:
                    face_img = frame[y:y+h, x:x+w]
                    face_img = cv2.resize(face_img, (224, 224)).astype("float32")
                    face_img = preprocess_mobile(face_img)
                    face_batch = np.expand_dims(face_img, axis=0)
                    preds = model.predict(face_batch)
                    label = emotion_labels[np.argmax(preds)]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            cap.release()

        @app.route('/video_feed')
        def video_feed():
            return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    else:
        # For testing: dummy video stream
        @app.route('/video_feed')
        def video_feed():
            def dummy_frames():
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n"
                       b"TestImageData\r\n")
            return Response(dummy_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/')
    def index():
        return render_template('index.html')

    return app

# For running directly
if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
