# Basic Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import os
import sys
# Modelling
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.utils import save_object,evaluate_models
from src.logger import logging
from src.exception import CustomException

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifact","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def InitiateModelTrainer(self,train_array,test_array):
        try:
            logging.info("split training and test input data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
                
            )

            models = {
                        "Ada-Boost Classifier": AdaBoostClassifier(algorithm='SAMME'),
                        "K-Nearest Neighbors": KNeighborsClassifier(),
                        "Decision Tree": DecisionTreeClassifier(),
                        "Random Forest": RandomForestClassifier(),
                        "XGBoost": XGBClassifier(),
                        "CatBoost": CatBoostClassifier(verbose=False)
                    }
            
            model_report:dict = evaluate_models(X_train= X_train, y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)

            accuracy = accuracy_score(y_test,predicted)
        
            return accuracy

        except Exception as e:
            raise CustomException(e,sys)
