import os
import sys 

from src.logger import logging
from src.exception import MyException
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact , ModelTrainerArtifact
from src.constants import *
from src.utils.main_utils import read_yaml_file

import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, Activation, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications import EfficientNetB0

from tensorflow.keras.applications.mobilenet import preprocess_input as preprocess_mobile
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficient
from tensorflow.keras.callbacks import LambdaCallback


from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2 as cv
import time



class ModelTrainer:

    def __init__(self, data_transformation_artifact:DataTransformationArtifact, model_trainer_config: ModelTrainerConfig):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config
        # ---------------------Get Params -------------------

        self.params = read_yaml_file('params.yaml')
        self.mobile_params = self.params.get("mobile_net_model",{})
        self.effnet_params = self.params.get("effnet_model",{})


        # -------------------- Callbacks --------------------
        self.lr_schedule = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)
        self.early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)



    def get_logging_callback(self):
        return LambdaCallback(
            on_epoch_end=lambda epoch, logs: logging.info(
                f"Epoch {epoch+1}: " + ", ".join([f"{k}={v:.4f}" for k, v in logs.items()])
            )
    )

    
    def MobileNetV1(self):

        logging.info("Entered MobileV1 Method of ModelTrainer Class")
        # Load base model
        base_model = MobileNet(weights='imagenet', input_shape=tuple(self.mobile_params["MOBILE_IMG_SHAPE"]))
        model = Sequential()

        for layer in base_model.layers[:self.mobile_params["MOBILE_DROP_LAYERS"]]:
            model.add(layer)

        if self.mobile_params["MOBILE_TRAINABLE_LAYERS"] == 0:
            model.trainable = False
        elif self.mobile_params["MOBILE_TRAINABLE_LAYERS"] == 1:
            model.trainable = True
        elif self.mobile_params["MOBILE_TRAINABLE_LAYERS"] < 0:
            for layer in model.layers[:self.mobile_params["MOBILE_TRAINABLE_LAYERS"]]:
                layer.trainable = False
            for layer in model.layers[self.mobile_params["MOBILE_TRAINABLE_LAYERS"]:]:
                layer.trainable = True

        # Custom Layers 
        model.add(GlobalAveragePooling2D(name='global_avg_pool'))
        model.add(Dropout(0.5, name='dropout_x'))
        model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001), name='dense_1'))
        model.add(Dropout(0.3, name='dropout_2'))
        model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001), name='dense_2'))
        model.add(Dropout(0.3, name='dropout_3'))
        model.add(Dense(7, activation='softmax', name='output', dtype='float32'))

        model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=1e-4),
        metrics=['accuracy']
        )

        train_generator, val_generator, class_weights = self.ImageGenerator(preprocess_mobile, self.mobile_params["MOBILE_BATCH_SIZE"])

        logging.info("Initiating MobileNet Model Training")

        history = model.fit(
            x=train_generator,
            validation_data=val_generator,
            epochs=self.mobile_params["MOBILE_EPOCHS"],
            verbose=self.mobile_params["MOBILE_VERBOSE"],
            class_weight=class_weights,
            callbacks=[self.lr_schedule, self.early_stop, self.get_logging_callback()]
        )
        model_save_path = os.path.join(self.model_trainer_config.model_path,'mobilenet_model.h5')
        model.save(model_save_path)
        logging.info("MobileNet Model Saved in the Interim Data Folder.")
        logging.info("Exited MobileV1 Method of ModelTrainer Class")


    def EfficientNetB0Model(self):
        logging.info("Entered EfficientNetB0Model Method of ModelTrainer Class")
        # Load base model
        # Load EfficientNetB0 as base model
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=self.effnet_params["EFFNET_IMG_SHAPE"])
        base_model.trainable = True

        # Build full model with custom head
        x = base_model.output
        x = GlobalAveragePooling2D(name='global_avg_pool')(x)
        x = Dropout(0.4, name='dropout_x')(x)
        # x = Dense(256, activation='relu', kernel_regularizer=l2(0.001), name='dense_1')(x)
        # x = Dropout(0.3, name='dropout_2')(x)
        x = Dense(128, activation='relu', kernel_regularizer=l2(0.001), name='dense_2')(x)
        x = Dropout(0.3, name='dropout_3')(x)
        outputs = Dense(7, activation='softmax', name='output', dtype='float32')(x)

        model = Model(inputs=base_model.input, outputs=outputs)

        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=1e-4),
            metrics=['accuracy']
        )

        logging.info('EfficientNetB0 Model Structure ready')
        logging.info("Exited EfficientNetB0Model Method of ModelTrainer Class")
        return model
    
    def train_efficientnet(self, model):

        logging.info("Entered train_efficientnet method of ModelTrainer class")

        train_generator, val_generator, class_weights = self.ImageGenerator(preprocess_efficient, self.effnet_params["EFFNET_BATCH_SIZE"])

        # -------------------- Training --------------------
        history = model.fit(
            x=train_generator,
            validation_data=val_generator,
            epochs=self.effnet_params["EFFNET_EPOCHS"],
            verbose=self.effnet_params["EFFNET_VERBOSE"],
            class_weight=class_weights,
            callbacks=[self.lr_schedule, self.early_stop]
        )

        model_save_path = os.path.join(self.model_trainer_config.model_path,'effnet_model_saved_weights.h5')
        model.save_weights(model_save_path)
        logging.info("EfficientNet Model Saved in the models data folder.")
        logging.info("Exited train_efficientnet method of ModelTrainer class")


    def ImageGenerator(self, image_processor, batch_size):

        logging.info("Entered ImageGenerator Method of ModelTrainer Class.")
        augmentation_params = {        # Move to constants in future
            "rotation_range": 10,
            "zoom_range": [0.1, 1.2],
            "width_shift_range": 0.1,
            "height_shift_range": 0.1,
            "shear_range": 0.1,
            "horizontal_flip": True,
            "fill_mode": 'nearest',
            "brightness_range": [0.8, 1.2],
            "channel_shift_range": 30.0
        }

        train_datagen = ImageDataGenerator(
            preprocessing_function=image_processor,
            validation_split=0.2,
            **augmentation_params
        )

        val_datagen = ImageDataGenerator(
            preprocessing_function=image_processor,
            validation_split=0.2
        )

        train_generator = train_datagen.flow_from_directory(
            directory=self.data_transformation_artifact.processed_train_path,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            seed=42
        )

        val_generator = val_datagen.flow_from_directory(
            directory=self.data_transformation_artifact.processed_train_path,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False,
            seed=42
        )
        print(train_generator.classes)
        class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes), y=train_generator.classes)
        class_weights = dict(enumerate(class_weights))
        logging.info("Entered ImageGenerator Method of ModelTrainer Class.")

        return train_generator, val_generator, class_weights



    def initiate_model_training(self):

        logging.info("Entered initiate_model_training of ModelTrainer Class.")
        os.makedirs(self.model_trainer_config.model_path, exist_ok=True)

        if self.mobile_params["TRAIN_MOBILE"] == True:
            self.MobileNetV1()

        if self.effnet_params["TRAIN_EFFNET"] == True:
            effnet_model = self.EfficientNetB0Model()
            self.train_efficientnet(effnet_model)


        logging.info("Exited initiate_model_training of ModelTrainer Class.")
        return ModelTrainerArtifact(model_path=self.model_trainer_config.model_path)