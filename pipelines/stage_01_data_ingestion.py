from src.components.data_ingestion import DataIngestion
from src.utils.main_utils import save_artifact

if __name__ == "__main__":
    ingestion = DataIngestion()
    data_ingestion_artifact = ingestion.initiate_data_ingestion()

    save_artifact(data_ingestion_artifact, "artifacts/data_ingestion.pkl")
