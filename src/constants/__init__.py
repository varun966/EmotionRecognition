import os 
from datetime import date 
from from_root import from_root

# Dot Env
DOT_ENV_PATH = from_root('.env')

# Kaggle Connection
DATASET_NAME = "msambare/fer2013"
DOWNLOAD_PATH = from_root('data/raw')
DATA_DIR = from_root('data')
KAGGLE_URL_KEY: str = "KAGGLE_USERNAME"
KAGGLE_KEY: str = "KAGGLE_KEY"


PIPELINE_NAME = ""
ARTIFACT_DIR: str = "artifact"



TRAIN_FILE_NAME: str = "train"
TEST_FILE_NAME: str = "test"

DATA_INGESTION_TRAIN_VAL_SPLIT_RATIO: float = 0.2


DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"


DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_REPORT_FILE_NAME: str = "report.yaml"

SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")


PROCESSED_FOLDER ='processed'

# MobileNet Training Constants
MOBILE_IMG_SHAPE = (224,224,3)
MOBILE_DROP_LAYERS = -5
MOBILE_TRAINABLE_LAYERS = 70
MOBILE_EPOCHS = 5
MOBILE_VERBOSE = 1
MOBILE_BATCH_SIZE = 16


SAVED_MODEL_PATH = from_root('models')







