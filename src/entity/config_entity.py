import os
from src.constants import *
from dataclasses import dataclass 
from datetime import datetime
from from_root import from_root

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


@dataclass
class TrainingPiplelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    data_dir: str = DATA_DIR
    timestamp: str = TIMESTAMP


training_pipleine_config: TrainingPiplelineConfig = TrainingPiplelineConfig()


@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = training_pipleine_config.data_dir
    training_file_path: str = os.path.join(data_ingestion_dir, TRAIN_FILE_NAME)
    testing_file_path: str = os.path.join(data_ingestion_dir, TEST_FILE_NAME)
    train_val_split_ration: float = DATA_INGESTION_TRAIN_VAL_SPLIT_RATIO

@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(training_pipleine_config.data_dir, DATA_VALIDATION_DIR_NAME)
    validation_report_file_path: str = os.path.join(data_validation_dir, DATA_VALIDATION_REPORT_FILE_NAME)

@dataclass
class DataTransformationConfig:
    data_transformation_dir: str = os.path.join(training_pipleine_config.data_dir, )





