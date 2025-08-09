import requests
import json

username = "varun966"
token = "b1fce1ff46a77f795627bbd2f2f6da4064833123"
url = f"https://dagshub.com/{username}/EmotionRecognition.mlflow/api/2.0/mlflow/runs/create"

payload = {
    "experiment_id": "0",
    "start_time": 1723197695000
}

response = requests.post(url, auth=(username, token), json=payload)

print("Status Code:", response.status_code)
print("Response Text:", response.text)
