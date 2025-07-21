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
    training_file_path: str = os.path.join





