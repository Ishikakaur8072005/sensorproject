import sys 
from typing import Generator, List, Tuple, Dict
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    artifact_folder = os.path.join(artifact_folder)
    trained_model_path = os.path.join(artifact_folder, 'model.pkl')
    expected_accuracy = 0.45
    model_config_file_path = os.path.join("config", 'model.yaml')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.utils = MainUtils()    

        self.models = {
            'XGBClassifier': XGBClassifier(scale_pos_weight=15, random_state=42, eval_metric='logloss'),
            'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42),
            'SVC': SVC(class_weight='balanced', random_state=42),
            'RandomForestClassifier': RandomForestClassifier(class_weight='balanced', random_state=42)
        }

    def evaluate_models_cv(self, X: np.array, y: np.array, models: Dict) -> Dict:
        try:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            report = {}

            for model_name, model in models.items():
                y_true_all, y_pred_all = [], []
                prec_list, rec_list, f1_list, acc_list = [], [], [], []

                for train_idx, val_idx in skf.split(X, y):
                    X_tr, X_val = X[train_idx], X[val_idx]
                    y_tr, y_val = y[train_idx], y[val_idx]

                    model.fit(X_tr, y_tr)
                    preds = model.predict(X_val)

                    y_true_all.extend(y_val)
                    y_pred_all.extend(preds)

                    prec_list.append(precision_score(y_val, preds, zero_division=0))
                    rec_list.append(recall_score(y_val, preds, zero_division=0))
                    f1_list.append(f1_score(y_val, preds, zero_division=0))
                    acc_list.append(accuracy_score(y_val, preds))

                cm = confusion_matrix(y_true_all, y_pred_all)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                else:
                    tn, fp, fn, tp = cm[0, 0], 0, 0, 0

                report[model_name] = {
                    'mean_acc': float(np.mean(acc_list)), 'std_acc': float(np.std(acc_list)),
                    'mean_precision': float(np.mean(prec_list)), 'std_precision': float(np.std(prec_list)),
                    'mean_recall': float(np.mean(rec_list)), 'std_recall': float(np.std(rec_list)),
                    'mean_f1': float(np.mean(f1_list)), 'std_f1': float(np.std(f1_list)),
                    'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
                    'cm': cm
                }

            return report

        except Exception as e:
            raise CustomException(e, sys)

    def finetune_best_model(self, best_model_object: object, best_model_name: str, x_train: np.array, y_train: np.array) -> object:
        try:
            model_config = self.utils.read_yaml_file(self.model_trainer_config.model_config_file_path)
            model_param_grid = model_config["model_selection"]["model"].get(best_model_name, {}).get("search_param_grid", {})

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            if model_param_grid:
                grid_search = GridSearchCV(
                    best_model_object, param_grid=model_param_grid, cv=skf, scoring='f1', n_jobs=-1, verbose=1
                )
                grid_search.fit(x_train, y_train)
                return grid_search.best_estimator_
            else:
                best_model_object.fit(x_train, y_train)
                return best_model_object
        except Exception as e:
            logging.warning(f"Finetuning failed or skipped for {best_model_name}: {e}. Returning base model.")
            best_model_object.fit(x_train, y_train)
            return best_model_object

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )  

            # Combine for full dataset 5-fold CV evaluation across all 100 rows
            X_full = np.vstack([x_train, x_test])
            y_full = np.hstack([y_train, y_test])

            logging.info("Running 5-Fold Stratified Cross-Validation across candidate models...")
            cv_report = self.evaluate_models_cv(X=X_full, y=y_full, models=self.models)

            print("\n==========================================================================================")
            print("PRE-TUNING 5-FOLD STRATIFIED CROSS-VALIDATION METRICS (Full Dataset - 100 rows, 6 Faults)")
            print("==========================================================================================")
            for m_name, res in cv_report.items():
                print(f"\nModel: {m_name}")
                print(f"  Accuracy:  {res['mean_acc']:.4f} +/- {res['std_acc']:.4f}")
                print(f"  Precision: {res['mean_precision']:.4f} +/- {res['std_precision']:.4f}")
                print(f"  Recall:    {res['mean_recall']:.4f} +/- {res['std_recall']:.4f}")
                print(f"  F1-Score:  {res['mean_f1']:.4f} +/- {res['std_f1']:.4f}")
                print(f"  Confusion Matrix [[TN, FP], [FN, TP]]:\n{res['cm']}")
                total_faults = res['tp'] + res['fn']
                total_goods = res['tn'] + res['fp']
                print(f"  --> Faults Caught (True Positives): {res['tp']}/{total_faults} ({res['tp']/total_faults*100:.1f}%)")
                print(f"  --> Missed Faults (False Negatives): {res['fn']}/{total_faults}")
                print(f"  --> False Alarms (False Positives):  {res['fp']}/{total_goods} ({res['fp']/total_goods*100:.1f}%)")

            # Select best model based on mean F1-score & recall
            best_model_name = max(cv_report.keys(), key=lambda k: (cv_report[k]['mean_f1'], cv_report[k]['mean_recall']))
            best_model_object = self.models[best_model_name]

            print(f"\nSelecting best model based on F1 / Recall: {best_model_name}")

            best_model = self.finetune_best_model(
                best_model_name=best_model_name,
                best_model_object=best_model_object,
                x_train=x_train,
                y_train=y_train
            )

            # Evaluate tuned best model across 5-fold CV
            tuned_cv_report = self.evaluate_models_cv(X=X_full, y=y_full, models={best_model_name: best_model})
            tuned_res = tuned_cv_report[best_model_name]

            print("\n==========================================================================================")
            print(f"POST-TUNING 5-FOLD CV METRICS FOR BEST MODEL ({best_model_name})")
            print("==========================================================================================")
            print(f"  Accuracy:  {tuned_res['mean_acc']:.4f} +/- {tuned_res['std_acc']:.4f}")
            print(f"  Precision: {tuned_res['mean_precision']:.4f} +/- {tuned_res['std_precision']:.4f}")
            print(f"  Recall:    {tuned_res['mean_recall']:.4f} +/- {tuned_res['std_recall']:.4f}")
            print(f"  F1-Score:  {tuned_res['mean_f1']:.4f} +/- {tuned_res['std_f1']:.4f}")
            print(f"  Confusion Matrix [[TN, FP], [FN, TP]]:\n{tuned_res['cm']}")
            total_faults = tuned_res['tp'] + tuned_res['fn']
            total_goods = tuned_res['tn'] + tuned_res['fp']
            print(f"  --> Faults Caught (True Positives): {tuned_res['tp']}/{total_faults} ({tuned_res['tp']/total_faults*100:.1f}%)")
            print(f"  --> Missed Faults (False Negatives): {tuned_res['fn']}/{total_faults}")
            print(f"  --> False Alarms (False Positives):  {tuned_res['fp']}/{total_goods} ({tuned_res['fp']/total_goods*100:.1f}%)")
            print("==========================================================================================\n")

            # Final fit on x_train and evaluation on held-out x_test
            best_model.fit(x_train, y_train)
            y_pred_test = best_model.predict(x_test)
            test_acc = accuracy_score(y_test, y_pred_test)
            test_prec = precision_score(y_test, y_pred_test, zero_division=0)
            test_rec = recall_score(y_test, y_pred_test, zero_division=0)
            test_f1 = f1_score(y_test, y_pred_test, zero_division=0)
            test_cm = confusion_matrix(y_test, y_pred_test)

            print("==========================================================================================")
            print(f"HELD-OUT TEST SET EVALUATION (20 rows: {int((y_test==0).sum())} Good, {int((y_test==1).sum())} Faulty)")
            print("==========================================================================================")
            print(f"  Test Accuracy:  {test_acc:.4f}")
            print(f"  Test Precision: {test_prec:.4f}")
            print(f"  Test Recall:    {test_rec:.4f}")
            print(f"  Test F1-Score:  {test_f1:.4f}")
            print(f"  Test Confusion Matrix [[TN, FP], [FN, TP]]:\n{test_cm}")
            print("==========================================================================================\n")

            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_path), exist_ok=True)
            self.utils.save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )

            return self.model_trainer_config.trained_model_path

        except Exception as e:
            raise CustomException(e, sys) from e