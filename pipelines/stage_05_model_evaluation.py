from src.components.model_evaluation import ModelEvaluation
from src.entity.config_entity import ModelEvaluationConfig
from src.utils.main_utils import load_artifact, save_artifact

if __name__ == "__main__":
    model_trainer_artifact = load_artifact("artifacts/model_trainer.pkl")

    evaluation = ModelEvaluation(model_trainer_artifact, ModelEvaluationConfig())
    model_evaluation_artifact = evaluation.initiate_model_evaluation()

    save_artifact(model_evaluation_artifact, "artifacts/model_evaluation.pkl")

