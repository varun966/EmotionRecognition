import os
from src.constants import *
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.register_model import RegisterModel
from src.entity.config_entity import DataValidationConfig, DataTransformationConfig, ModelEvaluationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, ModelTrainerArtifact

# ingestion = DataIngestion()
# ret = ingestion.initiate_data_ingestion()
# print(ret)
# val = DataValidation(ret, DataValidationConfig)

# ret2 = val.initiate_data_validation()

# trans = DataTransformation(ret, DataTransformationConfig, ret2)
# ret3 = trans.initiate_data_transformation()

# train = ModelTrainer(ret3)
# train.initiate_model_training()
ret4 = ModelTrainerArtifact(interim_model_path=os.path.join(SAVED_MODEL_PATH,'interim'))
eval = ModelEvaluation(ret4, ModelEvaluationConfig)
eval.initiate_model_evaluation()

register = RegisterModel()
register.initiate_model_registry()



