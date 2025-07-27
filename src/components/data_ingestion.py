
import os
import sys

from src.logger import logging
from src.exception import MyException

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.configuration.kaggle_connection import KaggleClient
from src.constants import *

class DataIngestion:

    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        """
        :param data_ingestion_config : Configuration for Data Ingestion
        """
        try:
            self.data_ingestion_config = data_ingestion_config
            self.client = KaggleClient().client
        except Exception as e:
            raise MyException(e, sys)
        
    def export_data_into_raw_store(self)->None:

        """
        Download the FER2013 Dataset to Raw Folder
        """
        if not os.listdir(DOWNLOAD_PATH):
            logging.info("Raw Data Store empty, starting dowload.")
            self.client.dataset_download_files(DATASET_NAME, path = DOWNLOAD_PATH, unzip=True)
            logging.info("Data Exported into the Raw Data Store.")
        else:
            logging.info("Files already present in Raw Data Store, skipping Download.")


    def initiate_data_ingestion(self): # -> DataIngestionArtifact:
        """
        Method Name: initiate_data_ingestion
        Description: This method initiates the data ingestion components of training pipeline
        
        Output: train set and test set are returned as the artifacts of data ingestion components 
        On Failure: Write an exception log and then raise an exception
        """

        logging.info("Entered initiate_data_ingestion method of import_dataset class")

        try:
            self.export_data_into_raw_store()

            logging.info("Exited initiate_data_ingestion method of import_dataset class")

            data_ingestion_artifact = DataIngestionArtifact(trained_file_path = self.data_ingestion_config.training_file_path, 
                                                            test_file_path = self.data_ingestion_config.testing_file_path)
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise MyException(e, sys) from e




