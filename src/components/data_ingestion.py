import sys
import os
import numpy as np
import pandas as pd
from pymongo import MongoClient
from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    artifact_folder: str = os.path.join(artifact_folder)

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.utils = MainUtils()

    def export_collection_as_dataframe(self, collection_name, db_name):
        try:
            logging.info("Connecting to MongoDB to export collection as DataFrame...")
            mongo_client = MongoClient(MONGO_DB_URL, serverSelectionTimeoutMS=3000)
            collection = mongo_client[db_name][collection_name]

            df = pd.DataFrame(list(collection.find()))
            if df.empty:
                raise Exception("MongoDB collection returned empty DataFrame.")

            if "_id" in df.columns:
                df = df.drop(columns=['_id'], axis=1)
            if "id" in df.columns:
                df = df.drop(columns=['id'], axis=1)

            df.replace({"na": np.nan}, inplace=True)
            return df
        
        except Exception as e:
            logging.warning(f"Could not fetch data from MongoDB ({e}). Falling back to local notebook dataset.")
            local_csv_path = os.path.join("notebooks", "wafer_23012020_041211.csv")
            if os.path.exists(local_csv_path):
                df = pd.read_csv(local_csv_path)
                if "Unnamed: 0" in df.columns:
                    df = df.drop(columns=["Unnamed: 0"])
                df.replace({"na": np.nan}, inplace=True)
                return df
            else:
                raise CustomException(f"MongoDB connection failed and local dataset not found: {e}", sys)
        
    def export_data_into_feature_store_file_path(self) -> str:
        try:
            logging.info("Exporting data into feature store file path...")    
            raw_file_path = self.data_ingestion_config.artifact_folder
            os.makedirs(raw_file_path, exist_ok=True)

            sensor_data = self.export_collection_as_dataframe(
                collection_name=MONGO_COLLECTION_NAME,
                db_name=MONGO_DATABASE_NAME
            )

            feature_store_file_path = os.path.join(raw_file_path, 'wafer_fault.csv')
            logging.info(f"Saving exported data into feature store: {feature_store_file_path}")
            sensor_data.to_csv(feature_store_file_path, index=False)

            return feature_store_file_path
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def intitiate_data_ingestion(self) -> str:
        logging.info("Entered initiate_data_ingestion method of DataIngestion class")
        try:
            feature_store_file_path = self.export_data_into_feature_store_file_path()
            logging.info("Data ingestion completed successfully")
            return feature_store_file_path
        except Exception as e:
            raise CustomException(e, sys) from e
