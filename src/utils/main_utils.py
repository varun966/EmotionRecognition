import os 
import sys 

import yaml 

from src.exception import MyException
from src.logger import logging

def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
        
    except Exception as e:
        raise MyException(e, sys) from e
    
def check_folder_exists(file_path) -> None:
    if not os.listdir.exists(file_path):
        os.makedirs(file_path)
