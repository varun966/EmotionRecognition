from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    raw_data_dir: str
    train_file_path: str
    test_file_path: str

@dataclass
class DataValidationArtifact:
    validation_status: bool
    message: str 
    #validation_report_file_name: str 

@dataclass
class DataTransformationArtifact:
    processed_train_path: str
    proceseed_test_path: str

@dataclass
class ModelTrainerArtifact:
    interim_model_path: str

@dataclass 
class ModelEvaluationArtifact:
    trained_model_path: str 
    evaluation_report_file: str 
