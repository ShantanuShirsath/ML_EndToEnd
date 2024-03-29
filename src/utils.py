import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from src.exception import CustomException
from sklearn.metrics import accuracy_score


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for model_name, model_obj in models.items():
            model_obj.fit(X_train, y_train)  # Train model

            # Make predictions
            y_test_pred = model_obj.predict(X_test)
            
            # Evaluate the model
            accuracy = accuracy_score(y_test, y_test_pred)

            report[model_name] = accuracy
        
        return report

    except Exception as e:
        raise CustomException(e, sys)