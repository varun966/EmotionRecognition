import os
import sys
import mlflow
import mlflow.keras
import dagshub
import numpy as np 
import json
from dotenv import load_dotenv

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

from src.logger import logging
from src.exception import MyException
from src.constants import *
from src.entity.artifact_entity import ModelEvaluationArtifact, ModelTrainerArtifact
from src.entity.config_entity import ModelEvaluationConfig

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.mobilenet import preprocess_input as preprocess_mobile
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficient

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score


class ModelEvaluation:

    # # Set up DagsHub credentials for MLflow tracking
    # load_dotenv(dotenv_path = DOT_ENV_PATH)

    # dagshub_token = os.getenv("DAGSHUB_TOKEN")
    # if not dagshub_token:
    #     raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

    # os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    # os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    # dagshub_url = "https://dagshub.com"
    # repo_owner = "varun966"
    # repo_name = "EmotionRecognition"

    # # Set up MLflow tracking URI
    # mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

    def __init__(self, model_trainer_artifact: ModelTrainerArtifact, model_evaluation_config: ModelEvaluationConfig):

        self.model_trainer_artifact = model_trainer_artifact
        self.model_evaluation_config = model_evaluation_config
        # Initialize DagsHub connection
        self._init_dagshub()


    def _init_dagshub(self):
        """Initialize DagsHub connection and MLflow tracking"""
        load_dotenv(dotenv_path=DOT_ENV_PATH)
        dagshub_token = os.getenv("DAGSHUB_TOKEN")
        
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")
        
        # Modern DagsHub initialization
        dagshub.init(
            repo_name="EmotionRecognition",
            repo_owner="varun966",
            mlflow=True
        )

            # Set credentials separately
        os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_token
        os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
        
        # Verify connection
        logging.info(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

    def ImageGenerator(self, process_function, test_path ):
            
        test_batches = ImageDataGenerator(preprocessing_function=process_function).flow_from_directory(
        directory=test_path,
        target_size=(224, 224),
        batch_size=10,
        shuffle=False
        )
        return test_batches

    def evaluate_model(self, model, process_function, model_path, test_path):


        test_batches = self.ImageGenerator(process_function, test_path)
        test_labels = test_batches.classes
        predictions = model.predict(x=test_batches, verbose=0)
        predicted_labels = np.argmax(predictions, axis=1)

        test_accuracy = accuracy_score(test_labels, predicted_labels)
        test_f1_weighted = f1_score(test_labels, predicted_labels, average='weighted')
        test_f1_macro = f1_score(test_labels, predicted_labels, average='macro')

        metrics_dict = {
        'accuracy': test_accuracy,
        'test_f1_weighted': test_f1_weighted,
        'test_f1_macro': test_f1_macro,

        }
        return metrics_dict

    def save_metrics(self, metrics: dict, file_path: str) -> None:
        """Save the evaluation metrics to a JSON file."""
        try:
            with open(file_path, 'w') as file:
                json.dump(metrics, file, indent=4)
            logging.info('Metrics saved to %s', file_path)
        except Exception as e:
            logging.error('Error occurred while saving the metrics: %s', e)
            raise

    def save_model_info(self, run_id: str, model_path: str, file_path: str) -> None:
        """Save the model run ID and path to a JSON file."""
        try:
            model_info = {'run_id': run_id, 'model_path': model_path}
            with open(file_path, 'w') as file:
                json.dump(model_info, file, indent=4)
            logging.debug('Model info saved to %s', file_path)
        except Exception as e:
            logging.error('Error occurred while saving the model info: %s', e)
            raise

    def initiate_model_evaluation(self)->ModelEvaluationArtifact:

        logging.info("Initiated Model Evaluation Component.")

        mlflow.set_experiment("MobileNet-Final Run")
        
        with mlflow.start_run() as run:  # Start an MLflow run
            try:
                logging.info("Starting Evaluation for MobileNet Model")

                model_path = "./models/mobilenet_model.h5"
                test_path = "./data/processed/test"
                process_function = preprocess_mobile
                model = load_model(model_path)
                metrics = self.evaluate_model(model, process_function, model_path, test_path )

                self.save_metrics(metrics, 'reports/mobile_metrics.json')

                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)

                # Log model to MLflow
               # mlflow.tensorflow.log_model(model, "mobilenet_model.h5")

                # Log model with proper signature
                mlflow.keras.log_model(
                    model=model,
                    artifact_path="mobilenet_emotion_model",
                    registered_model_name=None  
                )


                # Save model info
                self.save_model_info(run.info.run_id, "mobilenet_emotion_model", 'reports/mobile_experiment_info.json')


                # Log the metrics file to MLflow
                mlflow.log_artifact(local_path='reports/mobile_metrics.json')
            
                return ModelEvaluationArtifact(saved_model_info_path='reports', saved_metrics_path='reports')

            except Exception as e:
                raise MyException(e, sys) from e
            
        

        