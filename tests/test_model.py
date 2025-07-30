import unittest
import os
import mlflow
import mlflow.keras
import dagshub
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from dotenv import load_dotenv
import random

# Load .env and init DagsHub (from your _init_dagshub function)
load_dotenv()




os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

dagshub.init(
    repo_owner="varun966",
    repo_name="EmotionRecognition",
    mlflow=True
)

class TestEmotionModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model_name = "MobileNetV1"  # change if needed
        cls.stage = "Staging"
        cls.img_size = (224, 224)
        cls.test_data_dir = "data/processed/test"

        # Load model from DagsHub/MLflow Registry
        cls.model_uri = cls._get_model_uri(cls.model_name, cls.stage)
        cls.model = mlflow.keras.load_model(cls.model_uri)

        # Get one sample image path
        cls.image_path, cls.true_label = cls._get_random_image(cls.test_data_dir)

    @staticmethod
    def _get_model_uri(model_name, stage):
        client = mlflow.MlflowClient()
        versions = client.get_latest_versions(model_name, stages=[stage])
        if not versions:
            raise ValueError(f"No model in stage '{stage}' found for '{model_name}'")
        return f"models:/{model_name}/{versions[0].version}"

    @staticmethod
    def _get_random_image(base_path):
        emotion_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        selected_emotion = random.choice(emotion_dirs)
        image_files = os.listdir(os.path.join(base_path, selected_emotion))
        chosen_image = random.choice(image_files)
        return os.path.join(base_path, selected_emotion, chosen_image), selected_emotion

    def test_model_loads(self):
        self.assertIsNotNone(self.model)

    def test_prediction_shape_and_range(self):
        img = load_img(self.image_path, target_size=self.img_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet.preprocess_input(img_array)

        predictions = self.model.predict(img_array)

        # Check output shape
        self.assertEqual(predictions.shape, (1, 7), "Expected prediction shape (1, 7)")

        # Check probability sum
        prob_sum = np.sum(predictions[0])
        self.assertAlmostEqual(prob_sum, 1.0, places=2, msg="Probabilities should sum close to 1")

        # Check max probability class
        predicted_class = np.argmax(predictions[0])
        self.assertTrue(0 <= predicted_class < 7, "Predicted class index out of range")

    def test_image_loading(self):
        self.assertTrue(os.path.exists(self.image_path), f"Test image not found: {self.image_path}")
        img = load_img(self.image_path, target_size=self.img_size)
        self.assertEqual(img.size, self.img_size, "Image not resized correctly")

if __name__ == "__main__":
    unittest.main()
