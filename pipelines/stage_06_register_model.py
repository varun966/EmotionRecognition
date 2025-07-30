from src.components.register_model import RegisterModel
from src.entity.artifact_entity import ModelEvaluationArtifact
from src.utils.main_utils import load_artifact, save_artifact

if __name__ == "__main__":

    model_evaluation_artifact = load_artifact("artifacts/model_evaluation.pkl")

    register = RegisterModel(model_evaluation_artifact)
    register.initiate_model_registry()
