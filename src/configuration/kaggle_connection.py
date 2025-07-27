import os
import sys


from dotenv import load_dotenv
from src.exception import MyException
from src.logger import logging 
from src.constants import KAGGLE_URL_KEY, DOT_ENV_PATH, KAGGLE_KEY


class KaggleClient:
    """
    Responsible for establishing connection with Kaggle.
    """
    client = None  # Shared Kaggle instance across all KaggleClient Instances 
    def __init__(self)-> None:
        """
        Initializes a connection 
        """
        try:
            logging.info("Entered the Kaggle Client Connection Class")
            if KaggleClient.client is None:
                load_dotenv(dotenv_path=DOT_ENV_PATH)
                os.environ["KAGGLE_USERNAME"] = os.getenv(KAGGLE_URL_KEY)
                os.environ["KAGGLE_KEY"] = os.getenv(KAGGLE_KEY)

                # Import after initiating .env keys so it gets auto-picked
                from kaggle.api.kaggle_api_extended import KaggleApi

                # Authenticate with Kaggle API
                api = KaggleApi()
                api.authenticate()
                print('api created')

                # Save to class variable
                KaggleClient.client = api
            
            # Instance variable gets the shared client
            self.client = KaggleClient.client
            logging.info("Kaggle Client Created")
            logging.info("Exited the Kaggle Client Connection Class")

        except Exception as e:
            raise MyException(e, sys)
        