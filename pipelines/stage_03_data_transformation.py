from src.components.data_transformation import DataTransformation
from src.entity.config_entity import DataTransformationConfig
from src.utils.main_utils import load_artifact, save_artifact

if __name__ == "__main__":
    ingestion_artifact = load_artifact("artifacts/data_ingestion.pkl")
    validation_artifact = load_artifact("artifacts/data_validation.pkl")

    transformation = DataTransformation(
        ingestion_artifact,
        DataTransformationConfig(),
        validation_artifact
    )
    transformation_artifact = transformation.initiate_data_transformation()

    save_artifact(transformation_artifact, "artifacts/data_transformation.pkl")
