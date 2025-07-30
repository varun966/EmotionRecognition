from src.components.model_trainer import ModelTrainer
from src.entity.config_entity import ModelTrainerConfig
from src.utils.main_utils import load_artifact, save_artifact

if __name__ == "__main__":
    transformation_artifact = load_artifact("artifacts/data_transformation.pkl")
    
    trainer = ModelTrainer(transformation_artifact, ModelTrainerConfig())
    model_trainer_artifact = trainer.initiate_model_training()

    save_artifact(model_trainer_artifact, "artifacts/model_trainer.pkl")
