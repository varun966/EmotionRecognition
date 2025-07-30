import os
import sys

from src.exception import MyException
from src.logger import logging 
from src.constants import KAGGLE_URL_KEY, KAGGLE_KEY


class KaggleClient:
    """
    Responsible for establishing connection with Kaggle.
    """
    client = None  # Shared Kaggle instance across all KaggleClient Instances 

    def __init__(self) -> None:
        """
        Initializes a connection 
        """
        try:
            logging.info("Entered the Kaggle Client Connection Class")

            if KaggleClient.client is None:
                # Read from environment variables (set locally or via GitHub secrets)
                os.environ["KAGGLE_USERNAME"] = os.getenv(KAGGLE_URL_KEY)
                os.environ["KAGGLE_KEY"] = os.getenv(KAGGLE_KEY)

                # Import after setting env vars so Kaggle API picks them up
                from kaggle.api.kaggle_api_extended import KaggleApi

                api = KaggleApi()
                api.authenticate()
                logging.info("API Initiated")

                KaggleClient.client = api

            self.client = KaggleClient.client
            logging.info("Kaggle Client Created")
            logging.info("Exited the Kaggle Client Connection Class")

        except Exception as e:
            raise MyException(e, sys)
