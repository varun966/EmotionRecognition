# app.py
from flask import Flask, render_template, Response, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input as preprocess_mobile
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficient
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import time
import os
import mlflow

def create_app(testing=False):
    app = Flask(__name__)
    app.testing = testing

    if not testing:

        app = Flask(__name__)

        # -----------------------
        # Load Models 
        # -----------------------
        # Set up DagsHub credentials for MLflow tracking
        dagshub_token = os.getenv("DAGSHUB_TOKEN")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "varun966"
        repo_name = "EmotionRecognition"

        # Set up MLflow tracking URI
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

        def get_latest_model_version(name):
            client = mlflow.MlflowClient()
            versions = client.get_latest_versions(name, stages=["Production"]) or client.get_latest_versions(name)
            return versions[0].version if versions else None
        

        mobilenet_model_name = "MobileNetV1"
        effnet_model_name = "EfficientNetEmotionClassifier"
        custom_model_name = "CustomCNNEmotionClassifier" 

        print('Loading MobileNet Model')
        mobile_version = get_latest_model_version(mobilenet_model_name)
        mobile_uri = f"models:/{mobilenet_model_name}/{mobile_version}"
        mobilenet_model = mlflow.pyfunc.load_model(mobile_uri)
        print('Loaded MobileNet Model')


        print('Loading EfficientNet Model')
        effnet_version = get_latest_model_version(effnet_model_name)
        effnet_uri = f"models:/{effnet_model_name}/{effnet_version}"
        efficientnet_model = mlflow.keras.load_model(effnet_uri)
        print('Loaded EfficientNet Model')

        print('Loading CustomCNN Model')
        custom_version = get_latest_model_version(custom_model_name)
        custom_uri = f"models:/{custom_model_name}/{custom_version}"
        customcnn_model = mlflow.pyfunc.load_model(custom_uri)
        print('Loaded CustomCNN Model')

        # # Rebuild EfficientNet B0 head (same as training)
        # base_model = EfficientNetB0(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
        # x = base_model.output
        # x = GlobalAveragePooling2D(name='global_avg_pool')(x)
        # x = Dropout(0.4, name='dropout_x')(x)
        # x = Dense(128, activation='relu', kernel_regularizer=None, name='dense_2')(x)
        # x = Dropout(0.3, name='dropout_3')(x)
        # outputs = Dense(7, activation='softmax', name='output', dtype='float32')(x)
        # efficientnet_model = Model(inputs=base_model.input, outputs=outputs)
        # efficientnet_model.compile(optimizer=Adam(learning_rate=1e-4),
        #                         loss='categorical_crossentropy', metrics=['accuracy'])
        # efficientnet_model.load_weights("./models/effnet_model_saved_weights.h5")

        # # Custom CNN (trained on 96x96x1, rescale 1./255, histogram eq in pipeline)
        # customcnn_model = load_model("./models/customcnn_model.h5")

        # Labels
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

        # Haar cascade
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
                # CustomCNN expects 96x96x1 and normalization 1./255
                resized = cv2.resize(face_gray, (96, 96))
                arr = resized.astype(np.float32) / 255.0
                arr = np.expand_dims(arr, axis=-1)   # (96,96,1)
                arr = np.expand_dims(arr, axis=0)    # (1,96,96,1)
                return arr

            elif model_name == "mobilenet":
                # MobileNet expects 224x224x3; preprocess_mobile handles scaling
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
                        'ensemble' handled as query params (see generate_frames).
            Returns predicted label string and raw preds if needed.
            """
            # apply histogram equalization (you said you used equalization in training)
            face_eq = cv2.equalizeHist(face_gray)

            # Single model case (string names)
            if isinstance(model_param, str) and not model_param.startswith("ensemble:"):
                input_arr = preprocess_for_model(face_eq, model_param)
                if model_param == "mobilenet":
                    preds = mobilenet_model.predict(input_arr, verbose=0)
                elif model_param == "efficientnet":
                    preds = efficientnet_model.predict(input_arr, verbose=0)
                elif model_param == "customcnn":
                    preds = customcnn_model.predict(input_arr, verbose=0)
                else:
                    return "Unknown", None
            else:
                # Ensemble mode: model_param expected to be like "ensemble:mobilenet,customcnn"
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
                    if m == "mobilenet":
                        p = mobilenet_model.predict(arr, verbose=0)
                    elif m == "efficientnet":
                        p = efficientnet_model.predict(arr, verbose=0)
                    elif m == "customcnn":
                        p = customcnn_model.predict(arr, verbose=0)
                    else:
                        continue
                    preds_list.append(p)
                # average predictions (works when all outputs are same shape)
                preds = np.mean(np.vstack(preds_list), axis=0)

            predicted_index = int(np.argmax(preds))
            return emotion_labels[predicted_index], preds

        # -----------------------
        # Frame generator & streaming
        # -----------------------
        def generate_frames_stream(model_query):
            """
            model_query: string. Examples:
            - 'mobilenet'
            - 'efficientnet'
            - 'customcnn'
            - 'ensemble:mobilenet,customcnn'
            """
            cap = cv2.VideoCapture(0)           # open camera
            # warm up
            time_start = time.time()
            frames = 0
            fps = 0.0

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = cv2.flip(frame, 1)
                    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    faces = face_cascade.detectMultiScale(gray_full, scaleFactor=1.3, minNeighbors=5)

                    for (x, y, w, h) in faces:
                        # handle boundary safety
                        x1, y1 = max(0, x), max(0, y)
                        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
                        face_crop = gray_full[y1:y2, x1:x2]
                        if face_crop.size == 0:
                            continue

                        label, _ = predict_emotion(face_crop, model_query)

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

                    # encode and yield
                    ret2, buffer = cv2.imencode('.jpg', frame)
                    if not ret2:
                        continue
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            finally:
                cap.release()

        # -----------------------
        # Routes
        # -----------------------
        @app.route('/')
        def index():
            return render_template('index.html')

        @app.route('/video_feed')
        def video_feed():
            # Two options:
            # 1) /video_feed?model=mobilenet
            # 2) /video_feed?model=ensemble&model1=mobilenet&model2=customcnn
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
        
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
