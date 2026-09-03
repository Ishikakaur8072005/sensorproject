import sys
import os
import pandas as pd
from dataclasses import dataclass
from src.constant import artifact_folder
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils

@dataclass
class PredictionPipelineConfig:
    model_file_path: str = os.path.join(artifact_folder, 'model.pkl')
    preprocessor_file_path: str = os.path.join(artifact_folder, 'preprocessor.pkl')

class PredictionPipeline:
    def __init__(self):
        self.prediction_pipeline_config = PredictionPipelineConfig()
        self.utils = MainUtils()

    def predict(self, features: pd.DataFrame):
        try:
            logging.info("Starting prediction pipeline execution...")
            model = self.utils.load_object(self.prediction_pipeline_config.model_file_path)
            preprocessor = self.utils.load_object(self.prediction_pipeline_config.preprocessor_file_path)

            # Drop unnecessary columns if present
            drop_cols = ["_id", "id", "Unnamed: 0", "Good/Bad", "quality"]
            for col in drop_cols:
                if col in features.columns:
                    features = features.drop(columns=[col])

            scaled_features = preprocessor.transform(features)
            preds = model.predict(scaled_features)
            return preds
        except Exception as e:
            raise CustomException(e, sys) from e

    def run_pipeline(self, request):
        try:
            if 'file' in request.files:
                file = request.files['file']
                df = pd.read_csv(file)
            else:
                df = pd.read_csv(request)
            return self.predict(df)
        except Exception as e:
            raise CustomException(e, sys) from e
