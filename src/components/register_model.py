import json
import mlflow
import logging
from src.logger import logging
import os
import dagshub
from dotenv import load_dotenv
from src.utils.main_utils import read_yaml_file

from src.constants import *
from src.entity.artifact_entity import ModelEvaluationArtifact

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")


class RegisterModel:

    def __init__(self, model_evaluation_artifact: ModelEvaluationArtifact):
        self._init_dagshub()
        self.model_evaluation_artifact = model_evaluation_artifact

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

    def load_model_info(self, file_path: str) -> dict:

        """Load the model info from a JSON file."""
        try:
            with open(file_path, 'r') as file:
                model_info = json.load(file)
            logging.debug('Model info loaded from %s', file_path)
            return model_info
        except FileNotFoundError:
            logging.error('File not found: %s', file_path)
            raise
        except Exception as e:
            logging.error('Unexpected error occurred while loading the model info: %s', e)
            raise
    
    def register_model(self, model_name: str, model_info: dict):

        """Register the model to the MLflow Model Registry."""
        try:
            model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
            
            # Register the model
            print(model_uri)
            model_version = mlflow.register_model(model_uri, model_name)
            print('here')
            print(model_version)
            
            # Transition the model to "Staging" stage
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Staging"
            )
            
            logging.debug(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
        except Exception as e:
            logging.error('Error during model registration: %s', e)
            raise

    
    def initiate_model_registry(self):

        try:

            logging.info("Entered initiate_model_registry method of the RegisterModel class")
            #print(model_info)   

            params = read_yaml_file('params.yaml')
            mobile_params = params.get("mobile_net_model",{})
            effnet_params = params.get("effnet_model",{})
            customcnn_params = params.get("custom_model",{})

            if mobile_params["TRAIN_MOBILE"] == True:
            
                model_info_path = f'{self.model_evaluation_artifact.saved_model_info_path}/mobile_experiment_info.json'
                model_info = self.load_model_info(model_info_path)
                model_name = "MobileNetV1"
                self.register_model(model_name, model_info)

            if effnet_params["TRAIN_EFFNET"] == True:  

                effnet_info_path = f'{self.model_evaluation_artifact.saved_model_info_path}/efficientnet_experiment_info.json'
                effnet_model_info = self.load_model_info(effnet_info_path)

                effnet_model_name = "EfficientNetEmotionClassifier"
                self.register_model(effnet_model_name, effnet_model_info)

            if customcnn_params["TRAIN_CUSTOM"] == True:

                custom_info_path = f'{self.model_evaluation_artifact.saved_model_info_path}/customcnn_experiment_info.json'
                custom_model_info = self.load_model_info(custom_info_path)

                custom_model_name = "CustomCNNEmotionClassifier"
                self.register_model(custom_model_name, custom_model_info)

            logging.info("Exited initiate_model_registry method of the RegisterModel class")

        except Exception as e:
            logging.error('Failed to complete the model registration process: %s', e)
            print(f"Error: {e}")


