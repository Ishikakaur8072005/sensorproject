import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

class TrainPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.model_trainer = ModelTrainer()

    def run_pipeline(self):
        try:
            logging.info("Starting training pipeline...")
            feature_store_file_path = self.data_ingestion.intitiate_data_ingestion()
            logging.info(f"Data ingestion completed: {feature_store_file_path}")

            data_transformation = DataTransformation(feature_store_file_path=feature_store_file_path)
            train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation()
            logging.info(f"Data transformation completed: preprocessor saved at {preprocessor_path}")

            model_path = self.model_trainer.initiate_model_trainer(train_array=train_arr, test_array=test_arr)
            logging.info(f"Model training completed: model saved at {model_path}")

            return model_path
        except Exception as e:
            raise CustomException(e, sys) from e

if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()
