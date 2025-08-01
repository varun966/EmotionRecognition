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

from src.utils.main_utils import read_yaml_file
from src.logger import logging
from src.exception import MyException
from src.constants import *
from src.entity.artifact_entity import ModelEvaluationArtifact, ModelTrainerArtifact
from src.entity.config_entity import ModelEvaluationConfig, ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
from src.utils.main_utils import load_artifact, save_artifact

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.mobilenet import preprocess_input as preprocess_mobile
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficient

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score


import tempfile
import tensorflow as tf
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec


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

    def ImageGenerator(self, process_function, test_path ):
            
        test_batches = ImageDataGenerator(preprocessing_function=process_function).flow_from_directory(
        directory=test_path,
        target_size=(224, 224),
        batch_size=10,
        shuffle=False
        )
        return test_batches

    def evaluate_model(self, model, process_function, test_path):


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

        params = read_yaml_file('params.yaml')
        mobile_params = params.get("mobile_net_model",{})
        effnet_params = params.get("effnet_model",{})

        if mobile_params["TRAIN_MOBILE"] == True:

            mlflow.set_experiment("MobileNet-Final Run")
            
            with mlflow.start_run() as run:  # Start an MLflow run
                try:
                    logging.info("Starting Evaluation for MobileNet Model")

                    model_path = "./models/mobilenet_model.h5"
                    test_path = "./data/processed/test"
                    process_function = preprocess_mobile
                    model = load_model(model_path)
                    metrics = self.evaluate_model(model, process_function, test_path )

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

                except Exception as e:
                    raise MyException(e, sys) from e
                
        # ---------------------- Efficient Net B0 Evaluation -------------------------------------------------------------

        if effnet_params["TRAIN_EFFNET"] == True:  
              
            mlflow.set_experiment("EfficientNetB0-Final Run")
        
            with mlflow.start_run() as run:  # Start an MLflow run
                try:
                    logging.info("Starting Evaluation for EfficientNetBo Model")

                    model_weight = "./models/effnet_model_saved_weights.h5"
                    test_path = "./data/processed/test"
                    process_function = preprocess_efficient
                    transformation_artifact = load_artifact("artifacts/data_transformation.pkl")
                    model_class = ModelTrainer(transformation_artifact, ModelTrainerConfig())
                    model = model_class.EfficientNetB0Model()
                    model.load_weights(model_weight)
                    metrics = self.evaluate_model(model, process_function, test_path )

                    self.save_metrics(metrics, 'reports/efficientnet_metrics.json')

                    for metric_name, metric_value in metrics.items():
                        mlflow.log_metric(metric_name, metric_value)

                    # # Dummy input in NumPy format
                    # dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)

                    # # Optional: Create a signature if needed
                    # from mlflow.models.signature import infer_signature

                    # output = model.predict(dummy_input)
                    # signature = infer_signature(dummy_input, output if isinstance(output, np.ndarray) else output.numpy())


                    # # Log model with proper signature
                    # mlflow.keras.log_model(
                    #     model=model,
                    #     artifact_path="efficientnet_emotion_model",
                    #     input_example=dummy_input,
                    #     signature=signature
                    # )

    
                    # Create a wrapper class for the model
                    class EfficientNetWrapper(mlflow.pyfunc.PythonModel):
                        def load_context(self, context):
                            import tensorflow as tf
                            self.model = tf.keras.models.load_model(context.artifacts["model_path"])
                        
                        def predict(self, context, model_input):
                            return self.model.predict(model_input)

                    # Save the model weights only
                    weights_path = os.path.join("models", "effnet_weights_temp.h5")
                    model.save_weights(weights_path)
                    
                    # Define artifacts dictionary
                    artifacts = {
                        "model_path": weights_path
                    }

                    # Log the custom model
                    mlflow.pyfunc.log_model(
                        artifact_path="efficientnet_emotion_model",
                        python_model=EfficientNetWrapper(),
                        artifacts=artifacts,
                        registered_model_name="EfficientNetEmotionClassifier"
                    )

                    # Clean up temporary weights file
                    os.remove(weights_path)



                    # Save model info
                    self.save_model_info(run.info.run_id, "efficientnet_emotion_model", 'reports/efficientnet_experiment_info.json')


                    # Log the metrics file to MLflow
                    mlflow.log_artifact(local_path='reports/efficientnet_metrics.json')
                
                    return ModelEvaluationArtifact(saved_model_info_path='reports', saved_metrics_path='reports')

                except Exception as e:
                    raise MyException(e, sys) from e
                
        return ModelEvaluationArtifact(saved_model_info_path='reports', saved_metrics_path='reports')
        
    # using efficient Model 

    # # Load MobileNet model
    # mobile_model = mlflow.pyfunc.load_model("models:/MobileNetEmotionClassifier/latest")

    # # Load EfficientNet model
    # effnet_model = mlflow.pyfunc.load_model("models:/EfficientNetEmotionClassifier/latest")

    # # Make predictions
    # predictions = effnet_model.predict(your_input_data)
                
            

            