from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.entity.config_entity import DataValidationConfig, DataTransformationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact

ingestion = DataIngestion()
ret = ingestion.initiate_data_ingestion()
print(ret)
val = DataValidation(ret, DataValidationConfig)

ret2 = val.initiate_data_validation()

trans = DataTransformation(ret, DataTransformationConfig, ret2)
trans.initiate_data_transformation()