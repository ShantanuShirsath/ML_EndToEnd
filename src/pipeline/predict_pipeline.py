import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifact","model.pkl")
            preprocessor_path=os.path.join('artifact','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            if preds == 0.0:
                return "Insomnia"
            if preds == 1.0:
                return "None"
            else:
                return "Sleep Apnea"
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        Gender: str,
        Age: int,
        Occupation : str,
        Sleep_Duration: float,
        Quality_of_Sleep: int,
        Physical_Activity_Level: int,
        Stress_Level: int,
        BMI_Category : str,
        Heart_Rate : int,
        Daily_Steps : int,
        systolic : int,
        diastolic : int ):

        self.Gender = Gender

        self.Age = Age

        self.Occupation = Occupation

        self.Sleep_Duration = Sleep_Duration

        self.Quality_of_Sleep = Quality_of_Sleep

        self.Physical_Activity_Level = Physical_Activity_Level

        self.BMI_Category = BMI_Category

        self.Heart_Rate = Heart_Rate

        self.Daily_Steps = Daily_Steps

        self.Stress_Level = Stress_Level

        self.systolic = systolic

        self.diastolic = diastolic

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Gender": [self.Gender],
                "Age": [self.Age],
                "Occupation": [self.Occupation],
                "Sleep_Duration": [self.Sleep_Duration],
                "Quality_of_Sleep": [self.Quality_of_Sleep],
                "Physical_Activity_Level": [self.Physical_Activity_Level],
                "BMI_Category": [self.BMI_Category],
                "Heart_Rate": [self.Heart_Rate],
                "Daily_Steps": [self.Daily_Steps],
                "Stress_Level": [self.Stress_Level],
                "systolic": [self.systolic],
                "diastolic": [self.diastolic]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)