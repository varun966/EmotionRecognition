# app.py
from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet import preprocess_input as preprocess_mobile
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficient
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import time
import os
import mlflow
from mlflow.tracking import MlflowClient
import tempfile
import traceback
from pathlib import Path
import pandas as pd

# ---- NEW: imports for browser-frame upload path ----
import base64
import threading
from collections import Counter  # NEW

# ---- NEW: global, thread-safe latest-frame buffer ----
_latest_frame = {"img": None, "lock": threading.Lock()}

def _set_latest_frame(img_bgr: np.ndarray):
    with _latest_frame["lock"]:
        _latest_frame["img"] = img_bgr

def _get_latest_frame():
    with _latest_frame["lock"]:
        if _latest_frame["img"] is None:
            return None
        return _latest_frame["img"].copy()

# ---- NEW: global, thread-safe overall emotion counters ----
_overall = {"counts": Counter(), "total": 0, "lock": threading.Lock()}

def _overall_increment(label: str):
    if label and label != "Unknown":
        with _overall["lock"]:
            _overall["counts"][label] += 1
            _overall["total"] += 1

def _overall_get():
    with _overall["lock"]:
        total = _overall["total"]
        counts = dict(_overall["counts"])
    if total > 0 and counts:
        top_label = max(counts, key=counts.get)
        top_pct = (counts[top_label] / total) * 100.0
    else:
        top_label, top_pct = "—", 0.0
    return top_label, top_pct, total, counts

def _overall_reset():
    with _overall["lock"]:
        _overall["counts"].clear()
        _overall["total"] = 0


def create_app(testing=False):
    app = Flask(__name__)
    app.testing = testing

    # Only run model-loading block when not testing
    if not testing:
        # -----------------------
        # Set up DagsHub + MLflow
        # -----------------------
        dagshub_token = os.getenv("DAGSHUB_TOKEN")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

        # mlflow username/password for basic auth to DagsHub MLflow endpoints
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "varun966"
        repo_name = "EmotionRecognition"
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

        client = MlflowClient()

        # small helper to get latest version (keeps behavior if stages deprecated)
        def get_latest_model_version(name):
            # returns version string (e.g. "1") or None
            versions = client.get_latest_versions(name, stages=["Production"]) or client.get_latest_versions(name)
            return versions[0].version if versions else None

        mobilenet_model_name = "MobileNetV1"
        effnet_model_name = "EfficientNetEmotionClassifier"
        custom_model_name = "CustomCNNEmotionClassifier"

        # storage for models and prediction wrappers
        models = {}

        # -----------------------
        # Generic predict wrapper
        # -----------------------
        def make_predict_fn(m):
            """
            Return a function predict_fn(input_array) -> np.array(preds).
            This function will try direct numpy input first; if that fails, it will
            flatten and convert to pandas.DataFrame before calling the model (pyfunc fallback).
            """
            def predict_fn(arr: np.ndarray):
                arr = np.asarray(arr)
                # ensure batch dim exists
                if arr.ndim == 3:
                    # e.g. (H,W,C) -> (1,H,W,C)
                    arr = np.expand_dims(arr, axis=0)
                try:
                    preds = m.predict(arr)
                    preds = np.asarray(preds)
                    return preds
                except Exception as e1:
                    # Try dataframe fallback (common for pyfunc)
                    try:
                        flat = arr.reshape((arr.shape[0], -1))
                        df = pd.DataFrame(flat)
                        preds = m.predict(df)
                        preds = np.asarray(preds)
                        return preds
                    except Exception as e2:
                        print(f"[ERROR] model.predict failed first attempt: {e1}")
                        print(f"[ERROR] model.predict failed fallback to DataFrame: {e2}")
                        raise
            return predict_fn

        # -----------------------
        # Load MobileNet & CustomCNN (prefer keras.load_model -> fallback to pyfunc)
        # -----------------------
        def load_model_keras_or_pyfunc(model_name):
            ver = get_latest_model_version(model_name)
            if not ver:
                raise RuntimeError(f"No model versions found for registered model: {model_name}")
            uri = f"models:/{model_name}/{ver}"
            try:
                m = mlflow.keras.load_model(uri)
                print(f"[INFO] Loaded {model_name} via mlflow.keras.load_model()")
                return m
            except Exception as e_keras:
                print(f"[WARN] mlflow.keras.load_model failed for {model_name}: {e_keras}")
                try:
                    m = mlflow.pyfunc.load_model(uri)
                    print(f"[INFO] Loaded {model_name} via mlflow.pyfunc.load_model() (pyfunc)")
                    return m
                except Exception as e_pyfunc:
                    print(f"[ERROR] Failed loading {model_name} via pyfunc: {e_pyfunc}")
                    raise

        print('Loading MobileNet Model')
        mobilenet_raw = load_model_keras_or_pyfunc(mobilenet_model_name)
        models['mobilenet'] = {
            'raw': mobilenet_raw,
            'predict': make_predict_fn(mobilenet_raw)
        }
        print('Loaded MobileNet Model')

        print('Loading CustomCNN Model')
        customcnn_raw = load_model_keras_or_pyfunc(custom_model_name)
        models['customcnn'] = {
            'raw': customcnn_raw,
            'predict': make_predict_fn(customcnn_raw)
        }
        print('Loaded CustomCNN Model')

        # -----------------------
        # EfficientNet: rebuild architecture and load weights from artifacts
        # -----------------------
        def build_custom_efficientnet(weights_path):
            EFFNET_IMG_SHAPE = (224, 224, 3)
            base_model = EfficientNetB0(weights=None, include_top=False, input_shape=EFFNET_IMG_SHAPE)
            base_model.trainable = True

            x = base_model.output
            x = GlobalAveragePooling2D(name='global_avg_pool')(x)
            x = Dropout(0.4, name='dropout_x')(x)
            x = Dense(128, activation='relu', kernel_regularizer=l2(0.001), name='dense_2')(x)
            x = Dropout(0.3, name='dropout_3')(x)
            outputs = Dense(7, activation='softmax', name='output', dtype='float32')(x)

            model = Model(inputs=base_model.input, outputs=outputs)
            model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])
            model.load_weights(str(weights_path))
            print(f"[INFO] Weights loaded from {weights_path}")
            return model

        def load_efficientnet_from_registry(model_name):
            client = mlflow.tracking.MlflowClient()
            versions = client.get_latest_versions(model_name, stages=["Production"]) or client.get_latest_versions(model_name)
            if not versions:
                raise RuntimeError(f"No versions found in registry for {model_name}")
            version = versions[0].version

            local_dir = Path(__file__).parent / "models" / f"{model_name}_v{version}"
            local_dir.mkdir(parents=True, exist_ok=True)

            artifacts_path = mlflow.artifacts.download_artifacts(
                artifact_uri=f"models:/{model_name}/{version}",
                dst_path=str(local_dir)
            )

            h5_candidates = list(local_dir.rglob("*.h5"))
            if not h5_candidates:
                possible_names = [
                    local_dir / "artifacts" / "effnet_weights_temp.h5",
                    local_dir / "effnet_weights_temp.h5",
                    local_dir / "artifacts" / "model.h5",
                    local_dir / "model.h5"
                ]
                found = None
                for p in possible_names:
                    if p.exists():
                        found = p
                        break
                if not found:
                    listing = [str(p) for p in local_dir.rglob("*")]
                    raise FileNotFoundError(f"Weights file not found under {local_dir}. Files found: {listing}")
                weights_path = found
            else:
                weights_path = h5_candidates[0]

            return build_custom_efficientnet(weights_path)

        print('Loading EfficientNet Model (rebuild + load weights from artifacts)')
        try:
            efficientnet_model = load_efficientnet_from_registry(effnet_model_name)
            models['efficientnet'] = {
                'raw': efficientnet_model,
                'predict': make_predict_fn(efficientnet_model)
            }
            print('Loaded EfficientNet Model')
        except Exception as e:
            print("[ERROR] Failed to load EfficientNet from registry:")
            traceback.print_exc()
            raise

        # -----------------------
        # Labels & cascade
        # -----------------------
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # -----------------------
        # Preprocessing helper
        # -----------------------
        def preprocess_for_model(face_gray, model_name):
            """
            face_gray: single-channel grayscale cropped face (np.ndarray)
            returns: preprocessed array ready for model.predict (batch dimension present)
            """
            if model_name == "customcnn":
                resized = cv2.resize(face_gray, (96, 96))
                arr = resized.astype(np.float32) / 255.0
                arr = np.expand_dims(arr, axis=-1)   # (96,96,1)
                arr = np.expand_dims(arr, axis=0)    # (1,96,96,1)
                return arr

            elif model_name == "mobilenet":
                rgb = cv2.cvtColor(face_gray, cv2.COLOR_GRAY2BGR)
                resized = cv2.resize(rgb, (224, 224)).astype(np.float32)
                arr = np.expand_dims(resized, axis=0)
                return preprocess_mobile(arr)

            elif model_name == "efficientnet":
                rgb = cv2.cvtColor(face_gray, cv2.COLOR_GRAY2BGR)
                resized = cv2.resize(rgb, (224, 224)).astype(np.float32)
                arr = np.expand_dims(resized, axis=0)
                return preprocess_efficient(arr)

            else:
                raise ValueError(f"Unknown model: {model_name}")

        # -----------------------
        # Prediction helper
        # -----------------------
        def predict_emotion(face_gray, model_param):
            """
            face_gray: grayscale crop (no batch dim).
            model_param: either 'mobilenet', 'efficientnet', 'customcnn', or
                        'ensemble' handled as query params (see generate_frames_stream).
            Returns predicted label string and raw preds if needed.
            """
            # apply histogram equalization
            face_eq = cv2.equalizeHist(face_gray)

            try:
                # Single model case (string names)
                if isinstance(model_param, str) and not model_param.startswith("ensemble:"):
                    input_arr = preprocess_for_model(face_eq, model_param)
                    predict_fn = models.get(model_param, {}).get('predict', None)
                    if predict_fn is None:
                        print(f"[ERROR] No predict function for model: {model_param}")
                        return "Unknown", None
                    preds = predict_fn(input_arr)

                else:
                    # Ensemble mode
                    if isinstance(model_param, str) and model_param.startswith("ensemble:"):
                        _, model_list_str = model_param.split(":", 1)
                        models_to_use = [m.strip() for m in model_list_str.split(",") if m.strip()]
                    elif isinstance(model_param, (list, tuple)):
                        models_to_use = model_param
                    else:
                        models_to_use = []

                    if not models_to_use:
                        return "Unknown", None

                    preds_list = []
                    for m in models_to_use:
                        arr = preprocess_for_model(face_eq, m)
                        predict_fn = models.get(m, {}).get('predict', None)
                        if predict_fn is None:
                            print(f"[WARN] model {m} not available for ensemble")
                            continue
                        p = predict_fn(arr)
                        p = np.asarray(p)
                        if p.ndim == 1:
                            p = np.expand_dims(p, axis=0)
                        preds_list.append(p)
                    if not preds_list:
                        return "Unknown", None
                    preds = np.mean(np.array(preds_list), axis=0)

                preds = np.asarray(preds)
                if preds.ndim == 2:
                    predicted_index = int(np.argmax(preds[0]))
                elif preds.ndim == 1:
                    predicted_index = int(np.argmax(preds))
                else:
                    predicted_index = int(np.argmax(preds.reshape(-1)))
                predicted_label = emotion_labels[predicted_index]
                return predicted_label, preds

            except Exception as e:
                print(f"[ERROR] predict_emotion failed: {e}")
                traceback.print_exc()
                return "Unknown", None

        # -----------------------
        # Frame generator & streaming
        # (NOW reads frames uploaded by browser instead of cv2.VideoCapture)
        # -----------------------
        def generate_frames_stream(model_query):
            # FPS calc
            time_start = time.time()
            frames = 0
            fps = 0.0

            while True:
                frame = _get_latest_frame()

                # If no frame yet, show a placeholder
                if frame is None:
                    placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
                    cv2.putText(placeholder, "Waiting for browser camera...",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    # encode and yield placeholder then loop again
                    ret2, buffer = cv2.imencode('.jpg', placeholder)
                    if ret2:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    time.sleep(0.05)  # small wait
                    continue

                gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray_full, scaleFactor=1.3, minNeighbors=5)

                for (x, y, w, h) in faces:
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
                    face_crop = gray_full[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        continue

                    try:
                        label, _ = predict_emotion(face_crop, model_query)
                    except Exception as e:
                        print(f"[WARN] prediction failed for a face: {e}")
                        label = "Unknown"

                    # ---- NEW: keep running counts for 'overall'
                    if label != "Unknown":
                        _overall_increment(label)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                # compute lightweight FPS every second
                frames += 1
                elapsed = time.time() - time_start
                if elapsed >= 1.0:
                    fps = frames / elapsed
                    frames = 0
                    time_start = time.time()
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # ---- NEW: draw overall on the stream
                top_label, top_pct, tot, _ = _overall_get()
                cv2.putText(frame, f"Overall: {top_label} {top_pct:.0f}% (n={tot})", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                ret2, buffer = cv2.imencode('.jpg', frame)
                if not ret2:
                    continue
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # -----------------------
        # Routes
        # -----------------------
        @app.route('/')
        def index():
            return render_template('index.html')

        @app.route('/video_feed')
        def video_feed():
            model = request.args.get('model', 'mobilenet')
            if model == 'ensemble':
                m1 = request.args.get('model1')
                m2 = request.args.get('model2')
                if not m1 or not m2:
                    return "Specify model1 and model2 for ensemble", 400
                model_query = f"ensemble:{m1},{m2}"
            else:
                model_query = model
            return Response(generate_frames_stream(model_query),
                            mimetype='multipart/x-mixed-replace; boundary=frame')

        # ---- browser uploads JPEG frames here ----
        @app.route('/upload_frame', methods=['POST'])
        def upload_frame():
            """
            Expects JSON: { "image": "data:image/jpeg;base64,...." }
            Optionally you can send raw base64 without a prefix; both are handled.
            """
            data = request.get_json(silent=True)
            if not data or "image" not in data:
                return jsonify({"ok": False, "error": "missing image"}), 400

            img_str = data["image"]
            # Strip data URL header if present
            if "," in img_str:
                img_str = img_str.split(",", 1)[1]
            try:
                jpg = base64.b64decode(img_str)
                arr = np.frombuffer(jpg, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    return jsonify({"ok": False, "error": "decode failed"}), 400

                _set_latest_frame(frame)

                # ✅ DEBUG LINE
                print("[DEBUG] Received frame from browser")

                return jsonify({"ok": True})
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 400

        # ---- NEW: overall emotion JSON + reset endpoints ----
        @app.route('/overall_json')
        def overall_json():
            top, pct, total, counts = _overall_get()
            # ensure consistent label order
            summary = []
            for lab in emotion_labels:
                c = counts.get(lab, 0)
                p = (c / total * 100.0) if total > 0 else 0.0
                summary.append({"label": lab, "count": c, "pct": round(p, 1)})
            return jsonify({"top": top, "top_pct": round(pct, 1), "total": total, "summary": summary})

        @app.route('/reset_overall', methods=['POST'])
        def reset_overall():
            _overall_reset()
            return jsonify({"ok": True})

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
