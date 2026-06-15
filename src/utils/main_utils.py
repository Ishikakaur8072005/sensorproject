import sys
import os
import pickle
import yaml
import boto3

from src.exception import CustomException
from src.logger import logging


class MainUtils:

    def read_yaml_file(self, filename: str) -> dict:
        try:
            with open(filename, "r", encoding="utf-8") as yaml_file:
                return yaml.safe_load(yaml_file)

        except Exception as e:
            raise CustomException(e, sys) from e

    def read_schema_config_file(self) -> dict:
        try:
            schema_config = self.read_yaml_file(
                os.path.join("config", "schema.yaml")
            )

            return schema_config

        except Exception as e:
            raise CustomException(e, sys) from e

    @staticmethod
    def save_object(file_path: str, obj: object) -> None:
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, "wb") as file_obj:
                pickle.dump(obj, file_obj)

        except Exception as e:
            raise CustomException(e, sys) from e

    @staticmethod
    def load_object(file_path: str) -> object:
        try:
            with open(file_path, "rb") as file_obj:
                return pickle.load(file_obj)

        except Exception as e:
            logging.info("Exception occurred in load_object function")
            raise CustomException(e, sys) from e