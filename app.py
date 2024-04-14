from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

app=Flask(__name__)

## Route for a home page

@app.route('/')
def index():
    return render_template('home.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            Gender=request.form.get('Gender'),
            Age=int(request.form.get('Age')),
            Occupation=request.form.get('Occupation'),
            Sleep_Duration = int(request.form.get('Sleep_Duration')),
            Quality_of_Sleep=int(request.form.get('Quality_of_Sleep')),
            Physical_Activity_Level=int(request.form.get('Physical_Activity_Level')),
            Stress_Level=int(request.form.get('Stress_Level')),
            BMI_Category = request.form.get('BMI_Category'),
            Heart_Rate = int(request.form.get('Heart_Rate')),
            Daily_Steps = int(request.form.get('Daily_Steps')),
            systolic = int(request.form.get('systolic')),
            diastolic = int(request.form.get('diastolic'))
        )

        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results)
    

if __name__=="__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0",port = port)