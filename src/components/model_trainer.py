import sys 
from typing import Dict
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.ensemble import EasyEnsembleClassifier

from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    artifact_folder = os.path.join(artifact_folder)
    trained_model_path = os.path.join(artifact_folder, 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.utils = MainUtils()    
        self.model = EasyEnsembleClassifier(random_state=42)

    def evaluate_model(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> Dict:
        try:
            self.model.fit(x_train, y_train)
            y_pred = self.model.predict(x_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)

            tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (cm[0, 0], 0, 0, 0)
            fa_rate = (fp / (tn + fp) * 100) if (tn + fp) > 0 else 0.0

            return {
                'accuracy': float(acc),
                'precision': float(prec),
                'recall': float(rec),
                'f1_score': float(f1),
                'confusion_matrix': cm,
                'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
                'false_alarm_rate': float(fa_rate)
            }
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_model_trainer(self, train_array: np.ndarray, test_array: np.ndarray) -> str:
        try:
            logging.info("Initiating model training with production model EasyEnsembleClassifier...")

            x_train, y_train = train_array[:, :-1], train_array[:, -1]
            x_test, y_test = test_array[:, :-1], test_array[:, -1]

            metrics = self.evaluate_model(x_train, y_train, x_test, y_test)

            print("\n==========================================================================================")
            print("PRODUCTION MODEL EVALUATION: EasyEnsembleClassifier")
            print("==========================================================================================")
            print(f"  Accuracy:          {metrics['accuracy']:.4f}")
            print(f"  Precision:         {metrics['precision']:.4f}")
            print(f"  Recall (Faults):   {metrics['recall']:.4f}")
            print(f"  F1-Score:          {metrics['f1_score']:.4f}")
            print(f"  Confusion Matrix [[TN, FP], [FN, TP]]:\n{metrics['confusion_matrix']}")
            total_faults = metrics['tp'] + metrics['fn']
            total_goods = metrics['tn'] + metrics['fp']
            print(f"  --> Faults Caught (True Positives): {metrics['tp']}/{total_faults} ({metrics['recall']*100:.1f}%)")
            print(f"  --> Missed Faults (False Negatives): {metrics['fn']}/{total_faults}")
            print(f"  --> False Alarms (False Positives):  {metrics['fp']}/{total_goods} ({metrics['false_alarm_rate']:.1f}%)")
            print("==========================================================================================\n")

            logging.info(
                f"EasyEnsemble Evaluation - Recall: {metrics['recall']:.4f}, Precision: {metrics['precision']:.4f}, "
                f"F1: {metrics['f1_score']:.4f}, False Alarms: {metrics['fp']}/{total_goods}"
            )

            # Recall Sanity Check: Flag degenerate models (0% recall or 100% false alarms)
            if metrics['recall'] == 0.0:
                raise CustomException("Recall sanity check failed: Model achieved 0.0 recall (failed to catch any faults).", sys)
            if metrics['recall'] == 1.0 and metrics['false_alarm_rate'] > 90.0:
                raise CustomException("Recall sanity check failed: Model predicted all positive (degenerate split/model).", sys)

            # Save final trained EasyEnsemble model object
            logging.info(f"Saving trained EasyEnsemble model to {self.model_trainer_config.trained_model_path}")
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_path), exist_ok=True)
            self.utils.save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=self.model
            )

            return self.model_trainer_config.trained_model_path

        except Exception as e:
            raise CustomException(e, sys) from e