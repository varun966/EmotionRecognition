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