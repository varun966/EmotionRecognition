# promote model

import os
import mlflow
from src.utils.main_utils import read_yaml_file

# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "varun966"
repo_name = "EmotionRecognition"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

client = mlflow.MlflowClient()



def promote_model(model_name):

    #model_name = "MobileNetV1"
    # Get the latest version in staging
    #latest_version_staging = client.get_latest_versions(model_name, stages=["Staging"])[0].version
    latest_version_staging = client.get_latest_versions(model_name)[0].version

    # Archive the current production model
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    for version in prod_versions:
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Archived"
        )

    # Promote the new model to production
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version_staging,
        stage="Production"
    )
    print(f"{model_name} Model version {latest_version_staging} promoted to Production")

if __name__ == "__main__":

    params = read_yaml_file('params.yaml')
    mobile_params = params.get("mobile_net_model",{})
    effnet_params = params.get("effnet_model",{})
    custom_params = params.get("custom_model",{})


    if mobile_params["TRAIN_MOBILE"] == True:
        promote_model("MobileNetV1")
    if effnet_params["TRAIN_EFFNET"] == True:
        promote_model("EfficientNetEmotionClassifier")
    if custom_params["TRAIN_CUSTOM"] == True:
        promote_model("CustomCNNEmotionClassifier")
