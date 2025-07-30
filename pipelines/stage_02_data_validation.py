from src.components.data_validation import DataValidation
from src.entity.config_entity import DataValidationConfig
from src.utils.main_utils import load_artifact, save_artifact

if __name__ == "__main__":
    ingestion_artifact = load_artifact("artifacts/data_ingestion.pkl")
    validation = DataValidation(ingestion_artifact, DataValidationConfig())
    validation_artifact = validation.initiate_data_validation()

    save_artifact(validation_artifact, "artifacts/data_validation.pkl")
