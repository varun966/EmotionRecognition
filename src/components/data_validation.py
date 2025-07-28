import os 
import sys 
import json
from tqdm import tqdm
import cv2 as cv

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
        
    
    def validate_folder_structure(self, split_path, split_config):
        """
        Validates that expected class folder exists in the split directory
        """
        error_msg = ""

        actual_classes = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
        expected_classes = split_config["class_names"]

        if len(actual_classes) != len(expected_classes):
            error_msg += f"[{split_path.upper()}] expected {split_path['expected_classes']} classes, found {len(actual_classes)}.\n"

        if set(actual_classes) != set(expected_classes):
            error_msg += f"[{split_path.upper()}] Class name mismatch. Expected: {set(expected_classes)}.\n"

        return error_msg
    
    def get_image_files(self, class_dir):
        return [f for f in os.listdir(class_dir) if f.split('.')[-1].lower() in self._schema_config["image"]["allowed_extensions"]]
    
    def validate_image_file(self, image_path, expected_shape, split_name, image_name):
        img = cv.imread(image_path, cv.IMREAD_UNCHANGED)
        error_msg = ""

        if img is None:
            if not self._schema_config["image"]["allow_corrupt_images"]:
                error_msg += f"[{split_name.upper()}] Corrupted image: {image_path}\n"
                return error_msg

        # Shape check
        if len(img.shape) == 2:  # Grayscale
            if img.shape != tuple(expected_shape):
                error_msg += f"[{split_name.upper()}] {image_name} - Shape {img.shape}, Expected: {expected_shape}\n"
        else:
            if img.shape[:2] != tuple(expected_shape):
                error_msg += f"[{split_name.upper()}] {image_name} - Shape {img.shape[:2]}, Expected: {expected_shape}\n"

        # Pixel range check
        if img.min() < self._schema_config["image"]["pixel_range"][0] or img.max() > self._schema_config["image"]["pixel_range"][1]:
            error_msg += f"[{split_name.upper()}] {image_name} - Pixel range out of bounds: [{img.min()}, {img.max()}]\n"

        # Grayscale check
        if self._schema_config["image"]["color_mode"] == "grayscale" and len(img.shape) != 2:
            error_msg += f"[{split_name.upper()}] {image_name} - Expected grayscale, got shape {img.shape}\n"

        return error_msg


    def validate_class_images(self, split_name,  split_path, split_config):
        error_msg = ""
        expected_shape = self._schema_config["image"]["shape"]

        for class_name in split_config["class_names"]:
            class_dir = os.path.join(split_path, class_name)
            if not os.path.exists(class_dir):
                error_msg += f"[{split_name.upper()}] Missing Folder: {class_name}.\n"
                continue

            image_files = self.get_image_files(class_dir)
            image_count = len(image_files)

            # Count check
            if image_count < split_config["min_images_per_class"] or image_count > split_config["max_images_per_class"]:
                error_msg += f"[{split_name.upper()}] {class_name}: Image Count {image_count} outside expected range [{split_config['min_images_per_class']}, {split_config['max_images_per_class']}].\n"

            # Image check
            for image_name in tqdm(image_files, desc=f"Validating {split_name}/{class_name}"):
                image_path = os.path.join(class_dir, image_name)
                error_msg += self.validate_image_file(image_path, expected_shape, split_name, image_name)

            logging.info(f'Image Validation completed for class: {split_name}/{class_name}')

        return error_msg


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

            for split_name in self._schema_config["splits"]:
                split_path = os.path.join(self.data_ingestion_artifact.raw_data_dir, split_name)

                # Class structure check
                validation_error_msg += self.validate_folder_structure(split_path=split_path, split_config=self._schema_config["splits"][split_name])

                # Image Count and file level validation
                validation_error_msg += self.validate_class_images(split_name, split_path, split_config = self._schema_config["splits"][split_name])

            if validation_error_msg:
                logging.error("Data Validation failed:\n"+validation_error_msg)
                raise MyException("Data Validation failed. check logs for details.")
            
            logging.info("Data Validation completed successfully.")
            return DataValidationArtifact(validation_status=True, message="Data Validation Successful.")

        except Exception as e:
            raise MyException(e, sys) from e
            