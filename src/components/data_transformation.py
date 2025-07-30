import os
import sys 
import cv2 as cv

from src.logger import logging 
from src.exception import MyException
from src.constants import *
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
from src.utils.main_utils import check_folder_exists

class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_transformation_config: DataTransformationConfig, 
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact

        except Exception as e:
            raise MyException(e, sys) from e
        
    def histogram_equalisation(self,img_path):
        img = cv.imread(img_path)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # as OpenCv is reading by default in BGR
        eq_img = cv.equalizeHist(gray)
        return eq_img


    def process_image(self,img_path):
        hist_eq_img = self.histogram_equalisation(img_path)
        # Insert any additional transformations as needed
        processed_img = hist_eq_img
        return processed_img
        
    def process_raw_data(self, raw_train, raw_test, target_train_path, target_test_path)->None:

        for filename in os.listdir(raw_train):
            os.makedirs(os.path.join(target_train_path, filename), exist_ok=True)
            for file in os.listdir(os.path.join(raw_train, filename)):
                if not os.path.exists(os.path.join(target_train_path, filename, file)):
                    img_path = os.path.join(raw_train, filename, file)
                    target_path = os.path.join(target_train_path, filename, file)
                    processed_img = self.process_image(img_path)
                    cv.imwrite(target_path, processed_img)
            logging.info(f'Image Transformation completed for train/{filename}')


        for filename in os.listdir(raw_test):
            os.makedirs(os.path.join(target_test_path, filename), exist_ok=True)
            for file in os.listdir(os.path.join(raw_test, filename)):
                if not os.path.exists(os.path.join(target_test_path, filename, file)):
                    img_path = os.path.join(raw_test, filename, file)
                    target_path = os.path.join(target_test_path, filename, file)
                    processed_img = self.process_image(img_path)
                    cv.imwrite(target_path, processed_img)
            
            logging.info(f'Image Transformation completed for test/{filename}')

    

    def initiate_data_transformation(self)->DataTransformationArtifact:
        
        logging.info("Entered initiate_data_transformation of DataTransformation Class")
        processed_train_path = self.data_transformation_config.processed_train_path
        #os.path.join(PROCESSED_FOLDER, TRAIN_FILE_NAME)
        processed_test_path = self.data_transformation_config.processed_test_path
        #os.path.join(PROCESSED_FOLDER,TEST_FILE_NAME)

        os.makedirs(processed_train_path, exist_ok=True)
        os.makedirs(processed_test_path, exist_ok=True)
        self.process_raw_data(self.data_ingestion_artifact.train_file_path, self.data_ingestion_artifact.test_file_path, processed_train_path, 
                                    processed_test_path)
        
        logging.info("Exited initiate_data_transformation of DataTransformation Class")
        return DataTransformationArtifact(processed_train_path=processed_train_path, proceseed_test_path=processed_test_path)
        
    