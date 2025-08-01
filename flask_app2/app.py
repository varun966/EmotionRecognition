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


import os
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, Activation, GlobalAveragePooling2D, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications import EfficientNetB0

from tensorflow.keras.applications.mobilenet import preprocess_input as preprocess_mobile
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficient


from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2 as cv
import time

app = Flask(__name__)

# Load models
mobilenet_model = load_model("./models/interim/mobilenet_model.h5")

# Rebuild same architecture
base_model = EfficientNetB0(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
base_model.trainable = True  # Or use .trainable = False if you want to freeze

x = base_model.output
x = GlobalAveragePooling2D(name='global_avg_pool')(x)
x = Dropout(0.4, name='dropout_x')(x)
# x = Dense(256, activation='relu', kernel_regularizer=l2(0.001), name='dense_1')(x)
# x = Dropout(0.3, name='dropout_2')(x)
x = Dense(128, activation='relu', kernel_regularizer=l2(0.001), name='dense_2')(x)
x = Dropout(0.3, name='dropout_3')(x)
outputs = Dense(7, activation='softmax', name='output', dtype='float32')(x)

efficientnet_model = Model(inputs=base_model.input, outputs=outputs)

# Compile the model again
efficientnet_model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Load weights
efficientnet_model.load_weights("./notebooks/effnet_model_saved_weights.h5")


emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def predict_emotion(face_img, model_name):
    # Apply histogram equalization to grayscale
    face_img = cv2.equalizeHist(face_img)

    # Convert back to 3-channel image (needed for MobileNet/EfficientNet)
    face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)

    resized_face = cv2.resize(face_img, (224, 224))
    input_face = np.expand_dims(resized_face, axis=0).astype(np.float32)

    if model_name == "mobilenet":
        input_face = preprocess_mobile(input_face)
        preds = mobilenet_model.predict(input_face, verbose=0)
    elif model_name == "efficientnet":
        input_face = preprocess_efficient(input_face)
        preds = efficientnet_model.predict(input_face, verbose=0)
    elif model_name == "ensemble":
        preds_mobile = mobilenet_model.predict(preprocess_mobile(input_face.copy()), verbose=0)
        preds_efficient = efficientnet_model.predict(preprocess_efficient(input_face.copy()), verbose=0)
        preds = (preds_mobile + preds_efficient) / 2.0
    else:
        return "Unknown"

    predicted_class = np.argmax(preds)
    return emotion_labels[predicted_class]


def generate_frames(model_name):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            if face_img.size == 0:
                continue

            emotion = predict_emotion(face_img, model_name)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    model_name = request.args.get('model', 'mobilenet')
    return Response(generate_frames(model_name),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
