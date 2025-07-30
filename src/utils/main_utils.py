import os 
import sys 
import pickle
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

def save_artifact(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_artifact(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

