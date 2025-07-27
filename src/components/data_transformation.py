import os
import sys 

from src.logger import logging 
from src.exception import MyException
from src.constants import *
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact

class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_transformation_config: DataTransformationConfig, 
                 data_validation_artifact: DataValidationArtifact):