import os 
import sys 
import json

from src.exception import MyException
from src.logger import logging
from src.constants import *
from src.utils.main_utils import read_yaml_file
from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact 

class DataValidation:

    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_validation_config: configuration for data validation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path = SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)
        
    
    def validate_folder_structure(self, split_path):
        """
        Validates that expected class folder exists in the split directory
        """
        error_msg = ""

        actual_classes = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
        expected_classes = split_path["class_names"]

        if len(actual_classes) != len(expected_classes):
            error_msg += f"[{split_path.upper()}] expected {split_path['expected_classes']} classes, found {len(actual_classes)}.\n"

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Method Name: initiate_data_validation
        Description: This method initiates the data validation component for the pipeline
        
        Output: Returns bool value based on validation result
        On Failure: Write an exception log and then raise an exception 
        """
        try:
            validation_error_msg = ""
            logging.info("Starting Data Validation")
            